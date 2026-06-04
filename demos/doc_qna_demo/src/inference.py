"""
PaddleOCR-VL OpenVINO 推理封装

支持两种 IR 布局：
  - 单文件夹 VLMPipeline 布局：用 openvino_genai.VLMPipeline 加载（官方 notebook 输出）
  - 拆分布局：分别有 llm_embd / llm_stateful_int4 / vision_int8 / vision_mlp
    （ModelScope `zhaohb/PaddleOCR-VL-1.5-ov` 的格式），用 paddleocr_vl_openvino 包加载

同时暴露 Tesseract 基线和可选的 PyTorch 对照组。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = (
    "请完整识别这个文档图片中的所有内容（包括文字、表格、公式、图表标题），"
    "保留原始排版与阅读顺序，输出为结构化 Markdown 格式。"
    "表格请输出为 Markdown 表格语法，公式请输出为 LaTeX。"
)


@dataclass
class InferenceResult:
    """单次推理结果与计时"""

    text: str = ""
    backend: str = ""
    image_path: str = ""
    elapsed_ms: float = 0.0
    extras: dict = field(default_factory=dict)


# ── OpenVINO GenAI 后端 ──────────────────────────────────────────────────────


def _detect_ir_layout(ir_dir: Path) -> str:
    """
    返回 'split' / 'vlmpipeline' / 'unknown'

    split 特征：包含 PaddleOCR-VL-1.5-ov/llm_stateful_int4.xml 或类似拆分文件
    vlmpipeline 特征：单层目录中含 openvino_*.xml + tokenizer.xml
    """
    if (ir_dir / "PaddleOCR-VL-1.5-ov").is_dir() or any(
        ir_dir.glob("**/llm_stateful*.xml")
    ):
        return "split"
    if (ir_dir / "openvino_language_model.xml").exists() or list(
        ir_dir.glob("openvino_*.xml")
    ):
        return "vlmpipeline"
    return "unknown"


class OpenVINOPaddleOCRVL:
    """
    自动选择后端的 PaddleOCR-VL OpenVINO 封装

    传入 ir_dir 后：
      1. split 布局 → 使用 paddleocr_vl_openvino 包加载，layout-detect → per-region VL
      2. vlmpipeline 布局 → 使用 openvino_genai.VLMPipeline 全图推理

    对外都暴露相同 `.infer(image, prompt) -> InferenceResult` 接口。
    """

    def __init__(
        self,
        ir_dir: str | Path,
        device: str = "CPU",
        max_new_tokens: int = 2048,
        layout: Optional[str] = None,
    ):
        self.ir_dir = Path(ir_dir)
        self.device = device
        self.max_new_tokens = max_new_tokens

        if not self.ir_dir.exists():
            raise FileNotFoundError(
                f"未找到 OpenVINO IR 目录: {self.ir_dir}\n"
                f"请先放置 IR（参考 openvino/README.md）"
            )

        self.layout = layout or _detect_ir_layout(self.ir_dir)
        logger.info(f"IR 布局: {self.layout}  路径: {self.ir_dir}")

        if self.layout == "split":
            self._init_split()
        elif self.layout == "vlmpipeline":
            self._init_vlmpipeline()
        else:
            raise RuntimeError(
                f"无法识别 IR 布局: {self.ir_dir}\n"
                f"支持 split (zhaohb/PaddleOCR-VL-1.5-ov) 或 vlmpipeline (openvino_*.xml)"
            )

    # ── 布局 1: VLMPipeline 单文件夹 ──────────────────────────────────────

    def _init_vlmpipeline(self):
        try:
            import openvino_genai as ov_genai
        except ImportError as e:
            raise ImportError("openvino_genai 未安装，请 `pip install openvino-genai`") from e
        self.pipe = ov_genai.VLMPipeline(str(self.ir_dir), self.device)
        self.gen_config = ov_genai.GenerationConfig()
        self.gen_config.max_new_tokens = self.max_new_tokens

    def _vlmpipeline_infer(self, pil: Image.Image, prompt: str) -> str:
        import openvino as ov

        arr = np.array(pil.convert("RGB"))
        if arr.ndim == 3:
            arr = arr[None, ...]
        tensor = ov.Tensor(arr.astype(np.uint8))
        out = self.pipe.generate(prompt, image=tensor, generation_config=self.gen_config)
        return out if isinstance(out, str) else getattr(out, "texts", [str(out)])[0]

    # ── 布局 2: split (zhaohb 格式) ────────────────────────────────────────

    def _init_split(self):
        try:
            from paddleocr_vl_openvino.paddleocr_vl_pipeline.ov_paddleocr_vl_pipeline import (
                PaddleOCRVL,
            )
        except ImportError as e:
            raise ImportError(
                "split 布局需要 paddleocr_vl_openvino 包：\n"
                "  pip install --no-deps "
                "https://github.com/zhaohb/paddleocr_vl_ov/releases/download/v0.3.0/paddleocr_vl_openvino-0.3.0-py3-none-any.whl\n"
                "  pip install \"transformers==4.54.0\" \"openvino>=2025.4.1,<2026\" "
                "\"torch>=2.8.0\" torchvision opencv-python sentencepiece shapely"
            ) from e

        # 拆分布局期望两个子目录（含 FP16 + INT4 + INT8 完整集合）
        vlm_dir = self._find_subdir(["PaddleOCR-VL-1.5-ov", "PaddleOCR-VL"])
        layout_dir = self._find_subdir(["PP-DoclayoutV3-ov", "PP-DoclayoutV3"])

        # wheel 校验完整性：需要 vision.xml / llm_stateful.xml 等 FP16 文件
        # 缺失则自动从 ModelScope 下载到 ~/.cache/modelscope（首次约 2.7 GB）
        logger.info(f"  VL: {vlm_dir.name}  layout: {layout_dir.name}")
        # llm_int4_compress=True 选择 llm_stateful_int4.xml；vision_int8_quant=True 选 vision_int8.xml
        self.pipe = PaddleOCRVL(
            layout_model_path=str(layout_dir),
            vlm_model_path=str(vlm_dir),
            vlm_device=self.device,
            layout_device=self.device,        # 默认是 NPU，这里强制对齐
            llm_int4_compress=True,
            vision_int8_quant=True,
        )

    def _find_subdir(self, candidates: List[str]) -> Path:
        for name in candidates:
            p = self.ir_dir / name
            if p.is_dir():
                return p
        for sub in self.ir_dir.iterdir():
            if sub.is_dir() and any(sub.glob("*.xml")):
                return sub
        raise FileNotFoundError(f"未在 {self.ir_dir} 找到子目录: {candidates}")

    def _split_infer(self, pil: Image.Image, prompt: str) -> str:
        # PaddleOCRVL.predict 是生成器，每次 yield 一个 PaddleOCRVLResult
        bgr = np.array(pil.convert("RGB"))[:, :, ::-1].copy()
        result = next(iter(self.pipe.predict(bgr, max_new_tokens=self.max_new_tokens)), None)
        if result is None:
            return ""
        md = getattr(result, "markdown", None)
        if md is None and hasattr(result, "get"):
            md = result.get("markdown", "")
        # markdown 字段在不同版本里可能是 str 或 dict（含 markdown_texts/markdown_images）
        if isinstance(md, dict):
            md = md.get("markdown_texts", "") or "\n".join(
                str(v) for v in md.values() if v
            )
        return md or ""

    # ── 统一对外接口 ──────────────────────────────────────────────────────

    def infer(
        self,
        image: str | Path | Image.Image | np.ndarray,
        prompt: str = DEFAULT_PROMPT,
    ) -> InferenceResult:
        if isinstance(image, (str, Path)):
            image_path = str(image)
            pil = Image.open(image_path).convert("RGB")
        elif isinstance(image, np.ndarray):
            image_path = ""
            pil = Image.fromarray(image)
        else:
            image_path = ""
            pil = image

        t0 = time.perf_counter()
        if self.layout == "split":
            text = self._split_infer(pil, prompt)
        else:
            text = self._vlmpipeline_infer(pil, prompt)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        return InferenceResult(
            text=text,
            backend=f"openvino_{self.layout}",
            image_path=image_path,
            elapsed_ms=elapsed_ms,
            extras={"layout": self.layout},
        )


# ── PyTorch 后端（Benchmark 对照） ─────────────────────────────────────────────


class PyTorchPaddleOCRVL:
    """HuggingFace transformers 加载 PaddleOCR-VL，用作速度对照组"""

    def __init__(
        self,
        hf_repo: str = "PaddlePaddle/PaddleOCR-VL",
        device: str = "cpu",
        max_new_tokens: int = 2048,
        torch_dtype: Optional[str] = None,
    ):
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError as e:
            raise ImportError(
                "需要安装 torch + transformers: `pip install torch transformers`"
            ) from e
        import torch

        self.device = device
        self.max_new_tokens = max_new_tokens

        logger.info(f"加载 PyTorch PaddleOCR-VL: {hf_repo} on {device}")
        kwargs = {"trust_remote_code": True}
        if torch_dtype:
            kwargs["torch_dtype"] = getattr(torch, torch_dtype)
        self.processor = AutoProcessor.from_pretrained(hf_repo, **kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(hf_repo, **kwargs).to(device)
        self.model.eval()

    def infer(
        self,
        image: str | Path | Image.Image | np.ndarray,
        prompt: str = DEFAULT_PROMPT,
    ) -> InferenceResult:
        import torch

        if isinstance(image, (str, Path)):
            image_path = str(image)
            pil = Image.open(image_path).convert("RGB")
        elif isinstance(image, np.ndarray):
            image_path = ""
            pil = Image.fromarray(image)
        else:
            image_path = ""
            pil = image

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text_in = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(images=pil, text=text_in, return_tensors="pt").to(self.device)

        t0 = time.perf_counter()
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        text = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
        return InferenceResult(
            text=text,
            backend="pytorch",
            image_path=image_path,
            elapsed_ms=elapsed_ms,
        )


# ── Tesseract 基线（解析质量对比） ────────────────────────────────────────────


class TesseractEngine:
    """pytesseract 封装，用作解析质量基线"""

    def __init__(self, lang: str = "chi_sim+eng", psm: int = 3):
        try:
            import pytesseract  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "未安装 pytesseract，请 `pip install pytesseract` 并安装 tesseract 二进制"
            ) from e
        self.lang = lang
        self.psm = psm

    def infer(
        self,
        image: str | Path | Image.Image | np.ndarray,
        prompt: str = "",  # 兼容统一接口
    ) -> InferenceResult:
        import pytesseract

        if isinstance(image, (str, Path)):
            image_path = str(image)
            pil = Image.open(image_path).convert("RGB")
        elif isinstance(image, np.ndarray):
            image_path = ""
            pil = Image.fromarray(image)
        else:
            image_path = ""
            pil = image

        config = f"--psm {self.psm}"
        t0 = time.perf_counter()
        text = pytesseract.image_to_string(pil, lang=self.lang, config=config)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        return InferenceResult(
            text=text,
            backend="tesseract",
            image_path=image_path,
            elapsed_ms=elapsed_ms,
        )


# ── 工具：批量推理 ───────────────────────────────────────────────────────────


def iter_images(image_dir: str | Path, exts={".png", ".jpg", ".jpeg", ".bmp", ".tiff"}) -> List[Path]:
    return sorted(p for p in Path(image_dir).iterdir() if p.suffix.lower() in exts)
