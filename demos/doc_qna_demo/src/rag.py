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

from .embedding import EmbedTiming, OpenVINOEmbedder
from .llm import GenTiming, QwenLLM
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
                    "chunk_id": h.chunk_id,
                    "preview": h.text[:120].replace("\n", " "),
                }
                for h in self.hits
            ],
            "timing": {
                "embed_query_ms": round(self.timing.embed_query_ms, 1),
                "retrieve_ms": round(self.timing.retrieve_ms, 1),
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
    ):
        self.embedder = embedder
        self.store = store
        self.llm = llm
        self.top_k = top_k
        self.max_context_chars = max_context_chars
        self.max_new_tokens = max_new_tokens

    def answer(
        self,
        question: str,
        top_k: Optional[int] = None,
        where: Optional[dict] = None,
    ) -> RAGAnswer:
        k = top_k or self.top_k
        timing = RAGTiming()
        t_total = time.perf_counter()

        # 1) embed query
        et = EmbedTiming()
        t0 = time.perf_counter()
        qvec = self.embedder.encode_queries([question], timing=et)[0]
        timing.embed_query_ms = (time.perf_counter() - t0) * 1000

        # 2) retrieve
        t0 = time.perf_counter()
        qr: QueryResult = self.store.query(question, qvec, top_k=k, where=where)
        timing.retrieve_ms = (time.perf_counter() - t0) * 1000

        # 3) LLM 生成
        context = _build_context(qr.hits, max_chars=self.max_context_chars)
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

        timing.total_ms = (time.perf_counter() - t_total) * 1000
        return RAGAnswer(
            question=question,
            answer=resp.text.strip(),
            hits=qr.hits,
            timing=timing,
        )
