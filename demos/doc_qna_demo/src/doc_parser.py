"""
PaddleOCR-VL OpenVINO 文档解析器

输入：单页图片或 text-layer 文本
输出：结构化 Markdown（保留页码 / 段落标题元信息）

注意：与 qnn/src/doc_parser.py 不同，本模块不再做布局检测 + 区域裁剪，
而是直接调用 PaddleOCR-VL 全图 prompt（这也是 OpenVINO Notebook 的官方用法）。
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .inference import DEFAULT_PROMPT, OpenVINOPaddleOCRVL
from .pdf_preprocessor import PageContent, PreprocessResult

logger = logging.getLogger(__name__)


@dataclass
class ParsedPage:
    page_no: int
    markdown: str
    source: str                    # "text_layer" | "paddleocr_vl"
    elapsed_ms: float = 0.0


@dataclass
class ParsedDocument:
    doc_name: str
    pages: List[ParsedPage] = field(default_factory=list)
    timings: dict = field(default_factory=dict)

    def to_markdown(self) -> str:
        out = [f"# {self.doc_name}\n"]
        for p in self.pages:
            out.append(f"\n<!-- page {p.page_no} · source={p.source} -->\n")
            out.append(p.markdown.strip())
            out.append("")
        return "\n".join(out)


def _normalize_markdown(md: str) -> str:
    """轻量规整 PaddleOCR-VL 输出："""
    md = md.replace("\r\n", "\n").replace("\r", "\n")
    # 折叠 3+ 连续空行
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


def parse_document(
    preprocessed: PreprocessResult,
    engine: Optional[OpenVINOPaddleOCRVL] = None,
    prompt: str = DEFAULT_PROMPT,
    ir_dir: Optional[str | Path] = None,
    device: str = "CPU",
) -> ParsedDocument:
    """
    根据 preprocess 结果解析整份 PDF。

    - text_layer 页面：直接复用文字层
    - ocr 页面：调用 PaddleOCR-VL OpenVINO 推理
    """
    doc = ParsedDocument(doc_name=preprocessed.pdf_path.stem)

    needs_ocr = any(p.source == "ocr" for p in preprocessed.pages)
    if needs_ocr and engine is None:
        if ir_dir is None:
            raise ValueError("含 OCR 页面但未提供 engine / ir_dir")
        engine = OpenVINOPaddleOCRVL(ir_dir, device=device)

    total_t0 = time.perf_counter()
    ocr_total = 0.0
    for page in preprocessed.pages:
        parsed = _parse_one_page(page, engine=engine, prompt=prompt)
        doc.pages.append(parsed)
        if parsed.source == "paddleocr_vl":
            ocr_total += parsed.elapsed_ms
        logger.info(
            f"  page {page.page_no:>3} [{parsed.source}] "
            f"{parsed.elapsed_ms:>6.0f} ms  · {len(parsed.markdown)} chars"
        )

    doc.timings = {
        "total_ms": (time.perf_counter() - total_t0) * 1000,
        "ocr_ms": ocr_total,
        "num_pages": len(doc.pages),
        "num_ocr_pages": sum(1 for p in doc.pages if p.source == "paddleocr_vl"),
    }
    return doc


def _parse_one_page(
    page: PageContent,
    engine: Optional[OpenVINOPaddleOCRVL],
    prompt: str,
) -> ParsedPage:
    if page.source == "text_layer":
        return ParsedPage(
            page_no=page.page_no,
            markdown=_normalize_markdown(page.text or ""),
            source="text_layer",
            elapsed_ms=0.0,
        )

    if engine is None or page.image_path is None:
        return ParsedPage(
            page_no=page.page_no,
            markdown="",
            source="paddleocr_vl",
            elapsed_ms=0.0,
        )

    res = engine.infer(page.image_path, prompt=prompt)
    return ParsedPage(
        page_no=page.page_no,
        markdown=_normalize_markdown(res.text),
        source="paddleocr_vl",
        elapsed_ms=res.elapsed_ms,
    )
