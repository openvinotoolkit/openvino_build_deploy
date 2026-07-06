"""
ChromaDB 持久化向量库封装

将 Phase 2 chunks（{text, metadata}）+ Embedding 向量写入 ChromaDB persistent client，
对外暴露 add_chunks / query / count / reset 接口。

距离度量：cosine（与 OpenVINOEmbedder 的 L2 normalize 输出匹配）
集合名：默认 "doc_chunks"，可配置
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION = "doc_chunks"


@dataclass
class RetrievedChunk:
    text: str
    metadata: dict
    score: float  # cosine similarity (1 - cosine_distance)，bi-encoder 检索分
    chunk_id: str = ""
    rerank_score: Optional[float] = None  # cross-encoder 重排分（sigmoid 0~1），未重排则 None

    def cite(self) -> str:
        """生成简洁的来源标注 [doc_name p.N]"""
        d = self.metadata.get("doc_name", "unknown")
        p = self.metadata.get("page", "?")
        return f"[{d} p.{p}]"


@dataclass
class QueryResult:
    query: str
    hits: List[RetrievedChunk] = field(default_factory=list)


class ChromaStore:
    """
    持久化 ChromaDB 封装。

    用法：
        store = ChromaStore(persist_dir="./chroma_db")
        store.add_chunks(chunks=[{"text":..., "metadata":{...}}], embeddings=np.array([...]))
        result = store.query("问题", query_embedding=qvec, top_k=5)
    """

    def __init__(
        self,
        persist_dir: str | Path = "./chroma_db",
        collection_name: str = DEFAULT_COLLECTION,
        distance: str = "cosine",
    ):
        import chromadb
        from chromadb.config import Settings

        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        # get_or_create + hnsw:space 控制距离度量
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": distance},
        )
        logger.info(
            f"ChromaStore ready: {self.persist_dir} / {collection_name} "
            f"(distance={distance}, existing={self.collection.count()})"
        )

    # ── 写入 ──────────────────────────────────────────────────────────────

    def add_chunks(
        self,
        chunks: List[dict],
        embeddings: np.ndarray,
        id_prefix: str = "",
        batch_size: int = 256,
    ) -> List[str]:
        """
        chunks: [{"text": str, "metadata": dict}, ...]
        embeddings: np.ndarray [N, D]，与 chunks 一一对应
        id_prefix: chunk id 前缀，便于按文档清理；默认 doc_name + 自增
        """
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"chunks ({len(chunks)}) 与 embeddings ({embeddings.shape[0]}) 数量不匹配"
            )
        if len(chunks) == 0:
            return []

        ids = []
        for i, c in enumerate(chunks):
            meta = c.get("metadata", {}) or {}
            doc = meta.get("doc_name", "doc")
            page = meta.get("page", "0")
            kind = meta.get("kind", "x")
            base = f"{id_prefix or doc}::p{page}::{kind}::{i}"
            ids.append(base)

        # ChromaDB metadata 仅支持标量；过滤 None 与非标量
        flat_metadatas = []
        for c in chunks:
            m = {}
            for k, v in (c.get("metadata") or {}).items():
                if v is None:
                    continue
                if isinstance(v, (str, int, float, bool)):
                    m[k] = v
                else:
                    m[k] = str(v)
            flat_metadatas.append(m)

        docs = [c["text"] for c in chunks]
        embs = embeddings.tolist()

        for i in range(0, len(ids), batch_size):
            j = i + batch_size
            self.collection.add(
                ids=ids[i:j],
                documents=docs[i:j],
                embeddings=embs[i:j],
                metadatas=flat_metadatas[i:j],
            )
        logger.info(f"add_chunks: {len(ids)} 条入库 (total={self.collection.count()})")
        return ids

    # ── 检索 ──────────────────────────────────────────────────────────────

    def query(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int = 5,
        where: Optional[dict] = None,
    ) -> QueryResult:
        """
        query_embedding: [D] 或 [1, D]
        where: 元数据过滤，例如 {"doc_name": "spec_with_tables"}
        """
        emb = query_embedding
        if emb.ndim == 1:
            emb = emb[None, :]

        res = self.collection.query(
            query_embeddings=emb.tolist(),
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        hits: List[RetrievedChunk] = []
        if not res.get("ids") or not res["ids"][0]:
            return QueryResult(query=query, hits=hits)

        for i in range(len(res["ids"][0])):
            dist = res["distances"][0][i]
            # cosine_distance ∈ [0, 2]，相似度 = 1 - dist
            score = 1.0 - float(dist)
            hits.append(
                RetrievedChunk(
                    text=res["documents"][0][i],
                    metadata=res["metadatas"][0][i] or {},
                    score=score,
                    chunk_id=res["ids"][0][i],
                )
            )
        return QueryResult(query=query, hits=hits)

    # ── 维护 ──────────────────────────────────────────────────────────────

    def count(self) -> int:
        return self.collection.count()

    def reset_collection(self) -> None:
        """删掉当前集合并重建（用于重新灌入）"""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception as e:
            logger.debug(f"delete_collection 失败可能因不存在: {e}")
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"集合已重置: {self.collection_name}")

    def delete_by_doc(self, doc_name: str) -> int:
        """按 doc_name 删除该文档的所有 chunks，返回删除数量"""
        before = self.collection.count()
        self.collection.delete(where={"doc_name": doc_name})
        after = self.collection.count()
        return before - after


# ── 工具：从 jsonl 加载 chunks ───────────────────────────────────────────────


def load_chunks_jsonl(path: str | Path) -> List[dict]:
    import json

    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            out.append({"text": obj["text"], "metadata": obj.get("metadata", {})})
    return out


def iter_chunks_jsonl(
    paths: Iterable[str | Path],
    min_chars: int = 0,
) -> List[dict]:
    """
    合并多份 jsonl，强行打上 doc_name=文件名（不含 .chunks）兜底。

    min_chars > 0 时，丢掉文本短于阈值的 chunk——主要用于过滤页眉/页脚噪声。
    对很多 NVIDIA / 国标 PDF，pdfplumber 会把页脚 "Product Name | <page>" 拆成
    独立段落 → 切片器产 N 个几乎一样的短 chunk → 检索时排在头部把真正的事实
    挤出 Top-K。设 min_chars=60 实测能消掉这种 boilerplate。
    """
    merged = []
    for p in paths:
        p = Path(p)
        items = load_chunks_jsonl(p)
        fallback_doc = p.name.replace(".chunks.jsonl", "")
        for it in items:
            it.setdefault("metadata", {})
            it["metadata"].setdefault("doc_name", fallback_doc)
            if min_chars > 0 and len(it.get("text", "").strip()) < min_chars:
                continue
            merged.append(it)
    return merged
