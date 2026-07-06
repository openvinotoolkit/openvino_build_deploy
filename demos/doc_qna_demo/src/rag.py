"""
RAG 端到端：Embedder + ChromaStore + QwenLLM

提供 RAGPipeline.answer(question) → 答案 + 引用 + 各阶段耗时分解。

Prompt 设计要点：
  - system 明确"只用提供的上下文"，未命中要明确拒答（对应 q005/q015/q018 refusal）
  - context 每段顶 [doc_name p.N] 标注，让模型把引用塞进回答
  - 关闭 thinking，输出更紧凑
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from . import answer_grounding
from .embedding import EmbedTiming, OpenVINOEmbedder
from .entity_gate import EntityGate, check_grounding
from .llm import GenTiming, QwenLLM
from .reranker import OpenVINOReranker, RerankTiming
from .vector_store import ChromaStore, QueryResult, RetrievedChunk

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """你是一个严格基于给定文档作答的问答助手。规则：
1. 只能使用下方"参考资料"中明确出现的信息回答问题。如果原文有直接答案（哪怕只是一句话指引），照原文复述即可，不要因为答案简短就拒答。
2. 每条结论必须在末尾用方括号标注来源，格式为 [文档名 p.页码]，例如 [spec_with_tables p.1]，不要写成 [doc_name ...] 这种字面占位符。
3. 若参考资料里完全找不到相关事实，必须回答"文档中未提及"；严禁编造日期、数字、型号等任何具体细节，严禁基于常识或外部知识补全。
4. 回答用简洁中文，先给结论再给依据，避免长篇复述原文。
"""


def _build_context(hits: List[RetrievedChunk], max_chars: int = 3500) -> str:
    """把检索到的 chunks 拼成带来源标注的上下文，控制总长不爆 LLM context"""
    parts: List[str] = []
    used = 0
    for h in hits:
        tag = h.cite()
        block = f"{tag}\n{h.text.strip()}"
        if used + len(block) > max_chars and parts:
            break
        parts.append(block)
        used += len(block) + 2
    return "\n\n---\n\n".join(parts)


@dataclass
class RAGTiming:
    embed_query_ms: float = 0.0
    retrieve_ms: float = 0.0
    rerank_ms: float = 0.0
    llm_ms: float = 0.0
    total_ms: float = 0.0
    new_tokens: int = 0
    tokens_per_second: float = 0.0


@dataclass
class RAGAnswer:
    question: str
    answer: str
    hits: List[RetrievedChunk] = field(default_factory=list)
    timing: RAGTiming = field(default_factory=RAGTiming)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "citations": [
                {
                    "doc_name": h.metadata.get("doc_name"),
                    "page": h.metadata.get("page"),
                    "kind": h.metadata.get("kind"),
                    "score": round(h.score, 4),
                    "rerank_score": round(h.rerank_score, 4) if h.rerank_score is not None else None,
                    "chunk_id": h.chunk_id,
                    "preview": h.text[:120].replace("\n", " "),
                }
                for h in self.hits
            ],
            "timing": {
                "embed_query_ms": round(self.timing.embed_query_ms, 1),
                "retrieve_ms": round(self.timing.retrieve_ms, 1),
                "rerank_ms": round(self.timing.rerank_ms, 1),
                "llm_ms": round(self.timing.llm_ms, 1),
                "total_ms": round(self.timing.total_ms, 1),
                "new_tokens": self.timing.new_tokens,
                "tokens_per_second": round(self.timing.tokens_per_second, 1),
            },
        }


class RAGPipeline:
    """
    端到端 RAG 编排。embedder / store / llm 由调用方持有并复用。

    用法：
        embedder = OpenVINOEmbedder(device="CPU")
        store = ChromaStore("./chroma_db")
        llm = QwenLLM(device="CPU")
        rag = RAGPipeline(embedder, store, llm)
        ans = rag.answer("PaddleOCR-VL 总参数量是多少？", top_k=5)
        print(ans.answer)
    """

    def __init__(
        self,
        embedder: OpenVINOEmbedder,
        store: ChromaStore,
        llm: QwenLLM,
        top_k: int = 5,
        max_context_chars: int = 3500,
        max_new_tokens: int = 384,
        min_score: float = 0.0,
        reranker: Optional[OpenVINOReranker] = None,
        entity_gate: Optional[EntityGate] = None,
        retrieve_top_k: int = 20,
        rerank_min_score: float = 0.30,
        subject_grounding: bool = True,
        answer_grounding: bool = True,
    ):
        self.embedder = embedder
        self.store = store
        self.llm = llm
        self.top_k = top_k
        self.max_context_chars = max_context_chars
        self.max_new_tokens = max_new_tokens
        # bi-encoder 检索相似度下限（粗筛）：低于该值的命中不进后续流程。
        # 0.0 = 关闭。它拦得住明显域外题，但拦不住"域外实体 + 域内字段"的高相似度
        # 串台——那由 reranker + entity_gate 负责（见下）。
        self.min_score = min_score

        # 三级抗幻觉守卫（reranker 为主力，entity_gate 为确定性兜底）：
        #   1) entity_gate：query 提到"域内形态但语料没有"的伪实体（A500 / 未知标准号）→ 直接拒答
        #   2) reranker：cross-encoder 对 (query, passage) 联合打分，拦"语义主体串台"
        #      （火星探测器的额定功率 → sigmoid 0.036，远低于域内 0.98+）
        #   3) rerank_min_score：重排后 Top 全部低于该值 → 判定文档未覆盖，拒答
        self.reranker = reranker
        self.entity_gate = entity_gate
        # 主体接地 / 答案数值接地是**独立**守卫（各自用 module 函数、不依赖 entity_gate 对象），
        # 单独用开关控制——否则关掉实体码检查会连带把抗幻觉的答案接地也悄悄关了。
        self.subject_grounding = subject_grounding
        self.answer_grounding = answer_grounding
        # bi-encoder 先召回这么多候选，再交给 reranker 精排（召回宽、精排准）
        self.retrieve_top_k = max(retrieve_top_k, top_k)
        self.rerank_min_score = rerank_min_score

    def _refuse(self, question, answer, hits, timing, t_total) -> RAGAnswer:
        timing.total_ms = (time.perf_counter() - t_total) * 1000
        return RAGAnswer(question=question, answer=answer, hits=hits, timing=timing)

    def answer(
        self,
        question: str,
        top_k: Optional[int] = None,
        where: Optional[dict] = None,
    ) -> RAGAnswer:
        k = top_k or self.top_k
        timing = RAGTiming()
        t_total = time.perf_counter()

        # 0) 实体一致性守卫（最便宜、确定性）：query 提到域内形态但语料没有的伪实体，
        #    直接拒答，连 embedding 都不用跑。火星探测器/珠峰这类无实体码的题会放行。
        if self.entity_gate is not None:
            hit = self.entity_gate.check(question)
            if hit is not None:
                fam, ent = hit
                return self._refuse(
                    question, self.entity_gate.refusal_text(fam, ent), [], timing, t_total
                )

        # 1) embed query
        et = EmbedTiming()
        t0 = time.perf_counter()
        qvec = self.embedder.encode_queries([question], timing=et)[0]
        timing.embed_query_ms = (time.perf_counter() - t0) * 1000

        # 2) retrieve —— 有 reranker 时召回更宽（retrieve_top_k），交给精排收敛
        recall_k = self.retrieve_top_k if self.reranker is not None else k
        t0 = time.perf_counter()
        qr: QueryResult = self.store.query(question, qvec, top_k=recall_k, where=where)
        timing.retrieve_ms = (time.perf_counter() - t0) * 1000

        # 2.5) bi-encoder 粗筛：过滤明显低置信命中
        kept = [h for h in qr.hits if h.score >= self.min_score]
        if qr.hits and not kept:
            top = max(h.score for h in qr.hits)
            return self._refuse(
                question,
                f"文档中未提及。（检索最高相似度 {top:.3f} 低于阈值 "
                f"{self.min_score:.2f}，判定为文档未覆盖的问题）",
                qr.hits[:k],
                timing,
                t_total,
            )

        # 2.6) cross-encoder 精排 + 主体串台守卫
        if self.reranker is not None and kept:
            rt = RerankTiming()
            order = self.reranker.rerank(question, [h.text for h in kept], timing=rt)
            timing.rerank_ms = rt.total_ms
            reranked = []
            for idx, sc in order:
                kept[idx].rerank_score = sc
                reranked.append(kept[idx])
            # 重排后按精排分过滤：全部低于阈值 → 语义上没有真正相关的段落（串台/域外）
            survivors = [h for h in reranked if h.rerank_score >= self.rerank_min_score]
            if not survivors:
                top = reranked[0].rerank_score if reranked else 0.0
                return self._refuse(
                    question,
                    f"文档中未提及。（重排后最高相关度 {top:.3f} 低于阈值 "
                    f"{self.rerank_min_score:.2f}，判定为问题主体未被文档覆盖）",
                    reranked[:k],
                    timing,
                    t_total,
                )
            kept = survivors[:k]
        else:
            kept = kept[:k]

        # 2.7) 主体接地守卫：reranker 拦不住的强字段串台（特斯拉 Model 3 的工作温度，
        #      精排 0.77 甚至高过弱召回的域内题）——若 query 的主体词全都没出现在证据里，
        #      判定问的是文档没有的东西，拒答。极保守：只在"全未接地"时才拦。
        if self.subject_grounding and kept:
            ungrounded = check_grounding(question, [h.text for h in kept])
            if ungrounded is not None:
                return self._refuse(
                    question,
                    self.entity_gate.refusal_text("subject", ungrounded),
                    kept,
                    timing,
                    t_total,
                )

        # 3) LLM 生成
        context = _build_context(kept, max_chars=self.max_context_chars)
        if not context:
            # 检索无命中：直接返回拒答，不浪费 LLM
            timing.total_ms = (time.perf_counter() - t_total) * 1000
            return RAGAnswer(
                question=question,
                answer="文档中未提及。（检索未命中任何相关段落）",
                hits=qr.hits,
                timing=timing,
            )

        user = (
            f"问题：{question}\n\n"
            f"参考资料：\n{context}\n\n"
            f"请基于以上参考资料作答，每条结论后用 [doc_name p.页码] 标注来源。"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
        t0 = time.perf_counter()
        resp = self.llm.chat(messages, max_new_tokens=self.max_new_tokens)
        timing.llm_ms = (time.perf_counter() - t0) * 1000
        timing.new_tokens = resp.timing.new_tokens
        timing.tokens_per_second = resp.timing.tokens_per_second

        answer_text = resp.text.strip()

        # 4) 答案数值接地：模型可能从"主体对、内容不含答案"的 chunk 里编出具体日期/数字
        #    （Q5 实施日期即如此）——回答里的硬事实若无法在证据中核实，改判拒答。
        if self.answer_grounding:
            bad = answer_grounding.check_answer_grounding(answer_text, context, question)
            if bad is not None:
                answer_text = answer_grounding.refusal_text(bad)

        timing.total_ms = (time.perf_counter() - t_total) * 1000
        return RAGAnswer(
            question=question,
            answer=answer_text,
            hits=kept,
            timing=timing,
        )
