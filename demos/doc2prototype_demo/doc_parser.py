"""
Doc2Prototype - Document Parser Module
Uses PaddleOCR-VL with OpenVINO for document understanding.
"""

import time
import re
from pathlib import Path
from typing import Optional

import openvino as ov
from PIL import Image


# Prompt templates for different document understanding tasks
PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
    "layout": "Layout Recognition:",
    "flowchart": "Please analyze this flowchart and describe all nodes, connections, and the flow logic in detail.",
    "api_doc": "Please extract all API endpoints, parameters, request/response formats from this document.",
    "ui_mockup": "Please analyze this UI mockup and describe all components, layout structure, and interactive elements.",
    "technical_doc": "Please analyze this technical document and extract all key information including sections, tables, code snippets, and relationships.",
}


def clean_model_text(text: str) -> str:
    """Remove PaddleOCR-VL location markers from generated text."""
    text = re.sub(r"<\|LOC_\d+\|>", "", text)
    return re.sub(r"[ \t]+\n", "\n", text).strip()


class DocParser:
    """PaddleOCR-VL OpenVINO document parser."""

    def __init__(
        self,
        ov_model_path: str,
        device: str = "CPU",
        llm_int4_compress: bool = False,
        llm_int8_compress: bool = True,
        vision_int8_quant: bool = False,
        max_new_tokens: int = 2048,
    ):
        self.ov_model_path = ov_model_path
        self.device = device
        self.max_new_tokens = max_new_tokens

        # Lazy-load the model
        from ov_paddleocr_vl import OVPaddleOCRVLForCausalLM

        self._core = ov.Core()
        self._llm_infer_list = []
        self._vision_infer = []

        print(f"[DocParser] Loading PaddleOCR-VL OpenVINO model from {ov_model_path}...")
        start = time.time()

        self._model = OVPaddleOCRVLForCausalLM(
            core=self._core,
            ov_model_path=ov_model_path,
            device=device,
            llm_int4_compress=llm_int4_compress,
            llm_int8_compress=llm_int8_compress,
            vision_int8_quant=vision_int8_quant,
            llm_infer_list=self._llm_infer_list,
            vision_infer=self._vision_infer,
        )

        elapsed = time.time() - start
        print(f"[DocParser] Model loaded in {elapsed:.2f}s on {device}")

    def parse(
        self,
        image_path: str,
        task: str = "ocr",
        custom_prompt: Optional[str] = None,
    ) -> dict:
        """
        Parse a document image and return structured results.

        Args:
            image_path: Path to the image file
            task: One of 'ocr', 'table', 'formula', 'chart', 'layout',
                  'flowchart', 'api_doc', 'ui_mockup', 'technical_doc'
            custom_prompt: Custom prompt override

        Returns:
            dict with keys: 'raw_text', 'task', 'inference_time', 'image_path'
        """
        prompt_text = custom_prompt or PROMPTS.get(task, PROMPTS["ocr"])
        image = Image.open(image_path).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        start = time.time()
        result = self._model.chat(messages=messages)
        elapsed = time.time() - start

        # Handle tuple return (text, metadata) or plain string
        if isinstance(result, tuple):
            raw_text = result[0] if result[0] else ""
        else:
            raw_text = result if result else ""
        raw_text = clean_model_text(str(raw_text))

        return {
            "raw_text": raw_text,
            "task": task,
            "prompt": prompt_text,
            "inference_time": elapsed,
            "image_path": str(image_path),
        }

    def parse_batch(
        self,
        image_paths: list[str],
        task: str = "ocr",
        custom_prompt: Optional[str] = None,
    ) -> list[dict]:
        """Parse multiple document images."""
        results = []
        for path in image_paths:
            results.append(self.parse(path, task, custom_prompt))
        return results
