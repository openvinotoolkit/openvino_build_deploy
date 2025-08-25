from typing import Iterable, List, Optional, Any, Type, Callable
import PIL
import uuid
from langchain_community.vectorstores.lancedb import LanceDB
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel
from lancedb.pydantic import LanceModel, Vector

# class Metadata(BaseModel):
#     """Metadata for LanceDB."""
#     transcript: str
#     source: str
#     time_of_frame_ms: float
#     transcript_for_inference: str

# class LanceSchema(LanceModel):
#     """Schema for LanceDB table."""
#     id: str
#     vector: Vector(512)
#     text: str
#     metadata: Metadata

class MultimodalLanceDB(LanceDB):
    """`MultimodalLanceDB` vector store.
    This class extends the `LanceDB` vector store to support multimodal data,
    specifically text and image pairs. It allows you to add text-image pairs
    to the LanceDB database and perform similarity searches on them.

    To use, you should have ``lancedb`` python package installed.
    You can install it with ``pip install lancedb``.

    Args:
        connection: LanceDB connection to use. If not provided, a new connection
                    will be created.
        embedding: Embedding to use for the vectorstore.
        vector_key: Key to use for the vector in the database. Defaults to ``vector``.
        id_key: Key to use for the id in the database. Defaults to ``id``.
        text_key: Key to use for the text in the database. Defaults to ``text``.
        table_name: Name of the table to use. Defaults to ``vectorstore``.
        api_key: API key to use for LanceDB cloud database.
        region: Region to use for LanceDB cloud database.
        mode: Mode to use for adding data to the table. Valid values are
              ``append`` and ``overwrite``. Defaults to ``overwrite``.



    Example:
        .. code-block:: python
            vectorstore = LanceDB(uri='/lancedb', embedding_function)
            vectorstore.add_texts(['text1', 'text2'])
            vectorstore.add_text_image_pairs(['text1', 'text2'], ['path/to/image1.jpg', 'path/to/image2.jpg'])
            result = vectorstore.similarity_search('text1')
    """

    def __init__(
        self, *args, **kwargs: Any
    ):
        super().__init__(*args, **kwargs)
        
    def add_text_image_pairs(self, texts: Iterable[str], 
                             images: Iterable[str | PIL.Image.Image],
                             metadatas: Optional[List[dict]] = None,
                             ids: Optional[List[str]] = None,
                             **kwargs: Any) -> List[str]:
        # Implementation for adding text-image pairs
        assert len(texts) == len(images), "Texts and images must have the same length"
        if metadatas and len(metadatas) > 0:
            assert len(metadatas) == len(texts), "Metadatas and Texts must have the same length"
        docs = []
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        embeddings = self._embedding.embed_image_text_pairs(texts, images)
        for idx, text in enumerate(texts):
            embedding = embeddings[idx]
            metadata = metadatas[idx] if metadatas else {"id" : ids[idx], "image": images[idx]}
            doc = {
                self._vector_key: embedding,
                self._id_key: ids[idx],
                self._text_key: text,
                "metadata": metadata,
            }
            for key, value in metadata.items():
                doc[key] = value
            docs.append(doc)

        tbl = self.get_table()
        if tbl is None:
            tbl = self._connection.create_table(self._table_name, data=docs)
            self._table = tbl
        else:
            if self.api_key is None:
                tbl.add(docs, mode=self.mode)
            else:
                tbl.add(docs)
        
        self._fts_index = None
        return ids
    
    @classmethod
    def from_text_image_pairs(
        cls: Type["MultimodalLanceDB"],
        texts: List[str],
        images: List[str | PIL.Image.Image],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        connection: Optional[Any] = None,
        vector_key: Optional[str] = "vector",
        id_key: Optional[str] = "id",
        text_key: Optional[str] = "text",
        table_name: Optional[str] = "vectorstore",
        api_key: Optional[str] = None,
        region: Optional[str] = None,
        mode: Optional[str] = "overwrite",
        distance: Optional[str] = "l2",
        reranker: Optional[Any] = None,
        relevance_score_fn: Optional[Callable[[float], float]] = None,
        **kwargs: Any,
    ) -> "MultimodalLanceDB":
        
        instance = MultimodalLanceDB(
            connection=connection,
            embedding=embedding,
            vector_key=vector_key,
            id_key=id_key,
            text_key=text_key,
            table_name=table_name,
            api_key=api_key,
            region=region,
            mode=mode,
            distance=distance,
            reranker=reranker,
            relevance_score_fn=relevance_score_fn,
            **kwargs,
        )
        instance.add_text_image_pairs(
            texts=texts,
            images=images,
            metadatas=metadatas,
        )
        return instance