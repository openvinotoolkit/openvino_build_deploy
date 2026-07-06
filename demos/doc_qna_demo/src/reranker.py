"""
OpenVINO Cross-Encoder Reranker 封装（bge-reranker-base-int8-ov）

为什么加 reranker：bi-encoder（Qwen3-Embedding）把整句压成一个向量，
"火星探测器的额定功率" 里 "额定功率" 主导相似度，会以 0.465 的高分命中
A300 功率表格 chunk——域外实体 + 域内字段的"串台"用单一 cosine 阈值拦不住
（域内 top-1 ∈ [0.72, 0.84] 与它只差 0.26，没有干净的分界）。

cross-encoder 把 (query, passage) 拼在一起联合编码，能"看见"query 里的主体
(火星探测器) 从没在 A300 功率行出现过，于是给出极低分。实测同一条串台题：
  - bi-encoder cosine  = 0.465（骗过 0.35 阈值）
  - reranker sigmoid   = 0.036（离域内 0.98+ 有 ~0.95 的余量）
分界一下子干净了，这正是 README Known Limitations #3 说的"彻底解法"。

选型：OpenVINO/bge-reranker-base-int8-ov —— 与 embedding.py 同理，全项目统一
用 OpenVINO 官方预转 IR 复现；bge-reranker-base 基于 XLM-RoBERTa，多语言
（中英文皆可），INT8，体积小。

加载路径：huggingface_hub.snapshot_download → 本地 IR → openvino.Core
推理流程：tokenizer(query, passage) 成对编码 → infer → logits[B,1] → sigmoid
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_RERANKER_ID = "OpenVINO/bge-reranker-base-int8-ov"


@dataclass
class RerankTiming:
    n_pairs: int = 0
    tokenize_ms: float = 0.0
    infer_ms: float = 0.0
    total_ms: float = 0.0
    extras: dict = field(default_factory=dict)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class OpenVINOReranker:
    """
    用 OpenVINO Runtime 直接加载 bge-reranker-base-int8-ov 的 IR，
    对 (query, passage) 成对打分，不依赖 optimum-intel。

    用法：
        rr = OpenVINOReranker(device="CPU")
        scores = rr.score("A300 额定功率？", ["...passage1...", "...passage2..."])
        order  = rr.rerank("A300 额定功率？", passages)  # [(idx, score), ...] 降序
    """

    def __init__(
        self,
        model_id: str = DEFAULT_RERANKER_ID,
        device: str = "CPU",
        cache_dir: Optional[str | Path] = None,
        local_dir: Optional[str | Path] = None,
        max_length: int = 512,
    ):
        self.model_id = model_id
        self.device = device
        self.max_length = max_length

        # 1) 本地 IR 目录（Windows 无开发者模式时 snapshot_download 会因 symlink 报权限错，
        #    这里统一下载到 local_dir，huggingface_hub 新版会拷贝真实文件、不建 symlink）
        if local_dir is not None:
            ir_dir = Path(local_dir)
            if not (ir_dir / "openvino_model.xml").exists():
                # local_dir 存在但还没内容 → 拉取到该目录
                from huggingface_hub import snapshot_download

                ir_dir = Path(
                    snapshot_download(repo_id=model_id, local_dir=str(ir_dir))
                )
        else:
            from huggingface_hub import snapshot_download

            logger.info(f"下载 / 复用 Reranker IR: {model_id}")
            ir_dir = Path(
                snapshot_download(
                    repo_id=model_id,
                    cache_dir=str(cache_dir) if cache_dir else None,
                )
            )
        self.ir_dir = ir_dir
        logger.info(f"Reranker IR: {ir_dir}")

        # 2) tokenizer（XLM-RoBERTa，需 sentencepiece）
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(str(ir_dir))

        # 3) 加载 OpenVINO IR
        import openvino as ov

        self.core = ov.Core()
        model = self.core.read_model(str(ir_dir / "openvino_model.xml"))
        self.compiled = self.core.compile_model(model, device)
        self.input_names = {p.get_any_name() for p in self.compiled.inputs}
        self.output_port = self.compiled.outputs[0]
        logger.info(
            f"Reranker compiled on {device}; inputs={sorted(self.input_names)}; "
            f"output_shape={self.output_port.partial_shape}"
        )

    def score(
        self,
        query: str,
        passages: List[str],
        batch_size: int = 8,
        timing: Optional[RerankTiming] = None,
    ) -> np.ndarray:
        """
        对 (query, 每个 passage) 打相关性分，返回 sigmoid 后的 [N] 概率（0~1，越大越相关）。
        """
        if not passages:
            return np.zeros((0,), dtype=np.float32)

        logits_out: List[np.ndarray] = []
        t_tok_total = 0.0
        t_inf_total = 0.0
        t0 = time.perf_counter()

        for i in range(0, len(passages), batch_size):
            batch = passages[i : i + batch_size]
            t_tok = time.perf_counter()
            enc = self.tokenizer(
                [query] * len(batch),
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="np",
            )
            input_ids = enc["input_ids"].astype(np.int64)
            attention_mask = enc["attention_mask"].astype(np.int64)
            t_tok_total += (time.perf_counter() - t_tok) * 1000

            inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
            # 部分转换流程会暴露 token_type_ids；XLM-R 不用它，补零即可
            if "token_type_ids" in self.input_names:
                inputs["token_type_ids"] = np.zeros_like(input_ids)
            inputs = {k: v for k, v in inputs.items() if k in self.input_names}

            t_inf = time.perf_counter()
            result = self.compiled(inputs)
            t_inf_total += (time.perf_counter() - t_inf) * 1000

            out = np.asarray(result[self.output_port])
            # 默认 bge-reranker 输出 [B,1] logits；若换成 [B,2] 双标签模型，取正类列，
            # 不能盲目 flatten（否则把 2B 个数当成 B 个 pair 的分数）
            if out.ndim == 2 and out.shape[1] > 1:
                logits = out[:, -1]
            else:
                logits = out.reshape(-1)
            logits_out.append(logits)

        logits = np.concatenate(logits_out, axis=0)
        scores = _sigmoid(logits).astype(np.float32)

        if timing is not None:
            timing.n_pairs = len(passages)
            timing.tokenize_ms = t_tok_total
            timing.infer_ms = t_inf_total
            timing.total_ms = (time.perf_counter() - t0) * 1000
        return scores

    def rerank(
        self,
        query: str,
        passages: List[str],
        batch_size: int = 8,
        timing: Optional[RerankTiming] = None,
    ) -> List[Tuple[int, float]]:
        """返回 [(原始索引, 分数), ...]，按分数降序。"""
        scores = self.score(query, passages, batch_size=batch_size, timing=timing)
        order = np.argsort(-scores)
        return [(int(i), float(scores[i])) for i in order]
