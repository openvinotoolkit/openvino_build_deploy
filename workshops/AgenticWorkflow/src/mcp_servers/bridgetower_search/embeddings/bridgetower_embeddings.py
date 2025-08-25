from typing import List, Any
from pydantic.v1 import BaseModel
from langchain_core.embeddings import Embeddings
from tqdm import tqdm
from mcp_servers.bridgetower_search.openvino_bridgetower.openvino_bridgetower import (
    OpenVINOBridgeTower
)
import PIL


class BridgeTowerEmbeddings(BaseModel, Embeddings):
    """ BridgeTower embedding model """
    openVINOBridgeTower: Any

    def __init__(self, text_vision_model_path: str, text_model_path: str, vision_model_path: str):
        """Initialize the BridgeTower embeddings with the given model paths."""
        super().__init__()
        self.openVINOBridgeTower = OpenVINOBridgeTower(
            text_vision_model_path=text_vision_model_path,
            text_model_path=text_model_path,
            vision_model_path=vision_model_path
        )
        

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using BridgeTower.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        embeddings = self.openVINOBridgeTower.embed_texts(texts)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a query using BridgeTower.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self.embed_documents([text])[0]

    def embed_image_text_pairs(self, texts: List[str], images: List[str | PIL.Image.Image]) -> List[List[float]]:
        """Embed a list of image-text pairs using BridgeTower.

        Args:
            texts: The list of texts to embed.
            images: The list of path-to-images or PIL.Image.Image objects to embed
            
        Returns:
            List of embeddings, one for each image-text pairs.
        """

        # the length of texts must be equal to the length of images
        assert len(texts)==len(images), "the len of texts should be equal to the len of images"

        embeddings = self.openVINOBridgeTower.embed_text_image_pairs(texts, images)
        return embeddings