"""
OpenVINO Embedding 封装（Qwen3-Embedding-0.6B-int8-ov）

为什么不用 BGE-small-zh：BAAI/bge-small-zh-v1.5 在 OpenVINO 官方仓库
（huggingface.co/OpenVINO）下没有预转 IR；本项目其他模块（PaddleOCR-VL、
PP-DocLayoutV3、Qwen3-1.7B）都用官方预转 IR 复现，这里同样取
OpenVINO/Qwen3-Embedding-0.6B-int8-ov（多语言含中文，INT8）保持一致。

加载路径：huggingface_hub.snapshot_download → 本地缓存 → openvino.Core
推理流程：tokenize（左填充） → infer → last_token_pool → L2 normalize
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "OpenVINO/Qwen3-Embedding-0.6B-int8-ov"

# Qwen3-Embedding 官方推荐的任务指令（query 端加，document 端不加）
DEFAULT_QUERY_INSTRUCTION = (
    "Given a question about a product document, retrieve passages that answer the question"
)


@dataclass
class EmbedTiming:
    n_texts: int = 0
    tokenize_ms: float = 0.0
    infer_ms: float = 0.0
    total_ms: float = 0.0
    extras: dict = field(default_factory=dict)


def _last_token_pool(last_hidden: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """
    Qwen3-Embedding 用最后一个有效 token 的 hidden state 作为句子嵌入。
    tokenizer 用 padding_side=left，所以末位就是最后一个有效 token。
    这里同时兼容右填充：取每行 mask.sum()-1 位置。
    """
    if attention_mask[:, -1].sum() == attention_mask.shape[0]:
        return last_hidden[:, -1]
    seq_lens = attention_mask.sum(axis=1) - 1
    return last_hidden[np.arange(last_hidden.shape[0]), seq_lens]


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, eps)


class OpenVINOEmbedder:
    """
    用 OpenVINO Runtime 直接加载 Qwen3-Embedding-0.6B-int8-ov 的 IR，
    不依赖 optimum-intel。

    用法：
        emb = OpenVINOEmbedder(device="CPU")
        vecs = emb.encode(["产品说明...", "另一段..."])
        qvec = emb.encode_queries(["TDP 是多少？"])
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "CPU",
        cache_dir: Optional[str | Path] = None,
        max_length: int = 1024,
        local_dir: Optional[str | Path] = None,
        query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
    ):
        self.model_id = model_id
        self.device = device
        self.max_length = max_length
        self.query_instruction = query_instruction

        # 1) 拿到本地 IR 目录
        if local_dir is not None:
            ir_dir = Path(local_dir)
            if not (ir_dir / "openvino_model.xml").exists():
                raise FileNotFoundError(f"local_dir 不含 openvino_model.xml: {ir_dir}")
        else:
            from huggingface_hub import snapshot_download

            logger.info(f"下载 / 复用 IR: {model_id}")
            ir_dir = Path(
                snapshot_download(
                    repo_id=model_id,
                    cache_dir=str(cache_dir) if cache_dir else None,
                    local_dir_use_symlinks=False,
                )
            )
        self.ir_dir = ir_dir
        logger.info(f"Embedder IR: {ir_dir}")

        # 2) tokenizer：用 transformers（左填充，和 Qwen3 官方一致）
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(str(ir_dir), padding_side="left")

        # 3) 加载 OpenVINO IR
        import openvino as ov

        self.core = ov.Core()
        model = self.core.read_model(str(ir_dir / "openvino_model.xml"))
        self.compiled = self.core.compile_model(model, device)
        self.input_names = {p.get_any_name() for p in self.compiled.inputs}
        # 输出取第一个（last_hidden_state）
        self.output_port = self.compiled.outputs[0]
        logger.info(
            f"Embedder compiled on {device}; inputs={sorted(self.input_names)}; "
            f"output_shape={self.output_port.partial_shape}"
        )

    # ── 编码 ──────────────────────────────────────────────────────────────

    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        """单 batch 内部编码：返回 [B, D] L2-normalized embeddings"""
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )
        input_ids = enc["input_ids"].astype(np.int64)
        attention_mask = enc["attention_mask"].astype(np.int64)

        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        # 若模型还需要 position_ids（部分转换流程会显式暴露），按累加 mask 生成
        if "position_ids" in self.input_names:
            pos = np.cumsum(attention_mask, axis=1) - 1
            pos = np.clip(pos, 0, None).astype(np.int64)
            inputs["position_ids"] = pos
        # 仅传 IR 实际需要的输入
        inputs = {k: v for k, v in inputs.items() if k in self.input_names}

        result = self.compiled(inputs)
        last_hidden = result[self.output_port]
        sent_emb = _last_token_pool(last_hidden, attention_mask)
        return _l2_normalize(sent_emb).astype(np.float32)

    def encode(
        self,
        texts: List[str],
        batch_size: int = 8,
        timing: Optional[EmbedTiming] = None,
    ) -> np.ndarray:
        """编码文档（不加 instruction）。返回 [N, D] float32"""
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        chunks_out: List[np.ndarray] = []
        t_tok_total = 0.0
        t_inf_total = 0.0
        t0 = time.perf_counter()

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            t_tok = time.perf_counter()
            enc = self.tokenizer(
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
            if "position_ids" in self.input_names:
                pos = np.cumsum(attention_mask, axis=1) - 1
                pos = np.clip(pos, 0, None).astype(np.int64)
                inputs["position_ids"] = pos
            inputs = {k: v for k, v in inputs.items() if k in self.input_names}

            t_inf = time.perf_counter()
            result = self.compiled(inputs)
            t_inf_total += (time.perf_counter() - t_inf) * 1000

            last_hidden = result[self.output_port]
            sent_emb = _last_token_pool(last_hidden, attention_mask)
            chunks_out.append(_l2_normalize(sent_emb).astype(np.float32))

        out = np.concatenate(chunks_out, axis=0)
        total = (time.perf_counter() - t0) * 1000
        if timing is not None:
            timing.n_texts = len(texts)
            timing.tokenize_ms = t_tok_total
            timing.infer_ms = t_inf_total
            timing.total_ms = total
        return out

    def encode_queries(
        self,
        queries: List[str],
        batch_size: int = 8,
        instruction: Optional[str] = None,
        timing: Optional[EmbedTiming] = None,
    ) -> np.ndarray:
        """编码查询（按 Qwen3-Embedding 官方建议加 Instruct/Query 包装）"""
        instr = instruction if instruction is not None else self.query_instruction
        wrapped = [f"Instruct: {instr}\nQuery:{q}" for q in queries]
        return self.encode(wrapped, batch_size=batch_size, timing=timing)

    @property
    def dim(self) -> int:
        """嵌入维度（懒探测：跑一次单文本编码）"""
        if not hasattr(self, "_dim_cached"):
            v = self.encode(["dim probe"], batch_size=1)
            self._dim_cached = int(v.shape[1])
        return self._dim_cached
