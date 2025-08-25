from typing import List
import os
import openvino as ov
from mcp_servers.bridgetower_search.openvino_bridgetower.bridgetower_custom import (
    BridgeTowerForITC, 
)
from transformers import BridgeTowerProcessor
import torch
import torch.nn.functional as F
from torchvision.io import ImageReadMode, read_image
import torchvision.transforms.functional as transform
import PIL
from PIL import Image

class OpenVINOBridgeTower:
    """
    OpenVINO BridgeTower class for handling OpenVINO model inference.
    This class is a placeholder for the actual implementation.
    """

    def __init__(self, 
                 text_vision_model_path: str, 
                 text_model_path: str, 
                 vision_model_path: str, 
                 max_length: int = 100, 
                 batch_size: int = 10):
        """
        Initialize the OpenVINO BridgeTower with the given model path.

        Args:
            model_path (str): Path to the OpenVINO BridgeTower model.
            max_length (int): Maximum length of text input sequences.
            batch_size (int): Batch size for processing inputs.
        """
        self.batch_size = batch_size
        self.static_shapes = True
        self.padding = "max_length"
        self.truncation = True
        self.use_graphs = True
        self.max_length = max_length
        self.clear_embedding_cache = False

        self.core = ov.Core()
        self.model_name = "BridgeTower/bridgetower-large-itm-mlm-itc"
        self.text_vision_model_path = text_vision_model_path
        self.text_model_path = text_model_path
        self.vision_model_path = vision_model_path

        # Use environment variable for device configuration, fallback to GPU
        self.device = os.getenv("BRIDGETOWER_MODEL_DEVICE", "GPU")

        print(f"🔧 BridgeTower Model Configuration:")
        print(f"   📱 Device: {self.device}")
        print(f"   📄 Text-Vision Model: {os.path.basename(self.text_vision_model_path)}")
        print(f"   📄 Text Model: {os.path.basename(self.text_model_path)}")

        self.text_vision_model = self.core.compile_model(model=self.text_vision_model_path, device_name=self.device)
        print(f"   ✅ Text-Vision Model compiled on {self.device}")

        self.text_model = self.core.compile_model(model=self.text_model_path, device_name=self.device)
        print(f"   ✅ Text Model compiled on {self.device}")

        # self.vision_model = self.core.compile_model(model=self.vision_model_path, device_name=self.device)

        self.model_cfg = BridgeTowerForITC.from_pretrained(self.model_name)
        self.processor = BridgeTowerProcessor.from_pretrained(self.model_name)
        

    def embed_texts(self, texts: List[str]) -> list:
        """
        Embed list of text using the OpenVINO BridgeTower model.

        Args:
            text: Input text for embedding.

        Returns:
            Embedding for the input text.
        """
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
        
            encoding = self.processor.tokenizer(
                batch_texts,
                return_tensors="pt",
                max_length=self.max_length,
                padding=self.padding,
                truncation=self.truncation,
            ) 
            encoding = dict(encoding)
        
            with torch.no_grad():                
                outputs = self.text_model(encoding)
            embeddings += outputs[0].tolist()
        return embeddings

    def embed_images(self, image: List[str | PIL.Image.Image]) -> list:
        raise NotImplementedError("Embedding images is not implemented in this OpenVINO BridgeTower class.")
    

    def embed_text_image_pairs(self, texts: List[str], images: List[str|PIL.Image.Image]) -> list:
        """
        Embed list of text-image pairs using the OpenVINO BridgeTower model.

        Args:
            texts: List of input texts for embedding.
            images: List of input image paths for embedding.

        Returns:
            Embedding for the text-image pair.
        """
        assert len(texts) == len(images), "The length of texts must be equal to the length of images."
        image_list = []
        text_list = []
        embeddings = []

        for image_path, text in zip(images, texts):
            # Read and preprocess the image
            if isinstance(image_path, PIL.Image.Image):
                img = image_path
            elif isinstance(image_path, str) and os.path.exists(image_path):
                img = read_image(image_path, mode=ImageReadMode.RGB)
                img = transform.to_pil_image(img)
            image_list.append(img)
            text_list.append(text)

        # Process the images and texts in batches
        for i in range(0, len(image_list), self.batch_size):
            batch_images = image_list[i:i + self.batch_size]
            batch_texts = text_list[i:i + self.batch_size]
            batch = self.processor(batch_images, batch_texts, return_tensors="pt",
                                   max_length=self.max_length,
                                   padding=self.padding,
                                   truncation=self.truncation)
            batch = dict(batch)
            with torch.no_grad():
                outputs = self.text_vision_model(batch)

            for k in range(len(batch_texts)):
                embeddings.append(outputs[0][k, 2, :].tolist())
        return embeddings

    def embed(self, text: str = None, image: str | PIL.Image.Image = None) -> list:
        """
        Perform embedding using the OpenVINO BridgeTower model.

        Args:
            text: Input text for embedding (optional).
            image: Input image for embedding (optional).

        Returns:
            Embedding for the text, image, or image-text pair.
        """
        assert text is not None or image is not None, "Either text or image must be provided for embedding."
        if text is not None and image is None:
            return self.embed_texts([text])
        elif text is None and image is not None:
            return self.embed_images(image)
        else:
            return self.embed_text_image_pairs([text], [image])
