"""
Phase 2 端到端管线：PDF → 预处理 → PaddleOCR-VL 解析 → 表格感知切片
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .chunker import Chunk, chunk_document
from .doc_parser import ParsedDocument, parse_document
from .inference import OpenVINOPaddleOCRVL
from .pdf_preprocessor import preprocess_pdf

logger = logging.getLogger(__name__)


@dataclass
class PipelineOutput:
    document: ParsedDocument
    chunks: List[Chunk]


def run_pipeline(
    pdf_path: str | Path,
    ir_dir: str | Path = "./models/paddleocr_vl_ov",
    device: str = "CPU",
    cache_dir: Optional[str | Path] = None,
    dpi: int = 200,
    force_ocr: bool = False,
    engine: Optional[OpenVINOPaddleOCRVL] = None,
) -> PipelineOutput:
    pdf_path = Path(pdf_path)
    logger.info(f"== Phase 2 pipeline: {pdf_path.name} ==")

    pre = preprocess_pdf(pdf_path, cache_dir=cache_dir, dpi=dpi, force_ocr=force_ocr)
    logger.info(f"  preprocess.mode = {pre.mode} ({pre.num_pages} 页)")

    needs_ocr = any(p.source == "ocr" for p in pre.pages)
    if needs_ocr and engine is None:
        engine = OpenVINOPaddleOCRVL(ir_dir, device=device)

    doc = parse_document(pre, engine=engine)
    chunks = chunk_document(doc)
    logger.info(f"  生成 {len(chunks)} 个 chunk")
    return PipelineOutput(document=doc, chunks=chunks)
