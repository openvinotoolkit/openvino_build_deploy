"""Custom OpenVINO Sentence Transformer embedding for LlamaIndex."""

from llama_index.core.embeddings import BaseEmbedding
from optimum.intel import OVSentenceTransformer
from typing import List, Any
import asyncio

class OpenVINOSentenceTransformerEmbedding(BaseEmbedding):
    """Custom embedding class for OpenVINO Sentence Transformers."""
    
    def __init__(self, model_path: str, device: str = "CPU", **kwargs):
        super().__init__(**kwargs)
        # Use object.__setattr__ to bypass pydantic validation
        object.__setattr__(self, '_model', OVSentenceTransformer.from_pretrained(model_path, device=device))
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for a single query."""
        return self._model.encode([query])[0].tolist()
    
    def _get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text."""
        return self._model.encode([text])[0].tolist()
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts."""
        embeddings = self._model.encode(texts)
        return [emb.tolist() for emb in embeddings]
    
    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Get embedding for a single query (async version)."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._get_query_embedding, query
        )
    
    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text (async version)."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._get_text_embedding, text
        )
    
    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts (async version)."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._get_text_embeddings, texts
        )
