#!/usr/bin/env python3
"""Prepare PaddleOCR-VL OpenVINO IR files for Doc2Prototype."""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path


def download_model(model_id: str, cache_dir: str) -> Path:
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    try:
        from modelscope import snapshot_download

        print(f"[prepare] downloading {model_id} from ModelScope")
        return Path(snapshot_download(model_id, cache_dir=str(cache_path)))
    except Exception as exc:
        print(f"[prepare] ModelScope download failed: {exc}")

    try:
        from huggingface_hub import snapshot_download

        print(f"[prepare] downloading {model_id} from Hugging Face")
        return Path(
            snapshot_download(
                repo_id=model_id,
                cache_dir=str(cache_path),
                local_dir=str(cache_path / "hf" / model_id.replace("/", "__")),
            )
        )
    except Exception as exc:
        raise RuntimeError(f"failed to download model from ModelScope or Hugging Face: {exc}") from exc


def patch_model(pretrained_dir: Path) -> None:
    src = Path(__file__).parent / "modeling_paddleocr_vl.py"
    dst = pretrained_dir / "modeling_paddleocr_vl.py"
    backup = pretrained_dir / "modeling_paddleocr_vl.py.bak"

    if not src.exists():
        raise FileNotFoundError(f"missing local patch file: {src}")
    if dst.exists() and not backup.exists():
        shutil.copy2(dst, backup)
    shutil.copy2(src, dst)
    print(f"[prepare] patched {dst}")


def convert_to_openvino(
    pretrained_dir: Path,
    output_dir: Path,
    device: str,
    llm_int4: bool,
    llm_int8: bool,
    vision_int8: bool,
) -> Path:
    if output_dir.exists() and any(output_dir.glob("*.xml")):
        print(f"[prepare] OpenVINO model already exists at {output_dir}")
        return output_dir

    from ov_paddleocr_vl import PaddleOCR_VL_OV

    print("[prepare] converting PaddleOCR-VL to OpenVINO IR")
    start = time.perf_counter()
    model = PaddleOCR_VL_OV(
        pretrained_model_path=str(pretrained_dir),
        ov_model_path=str(output_dir),
        device=device,
        llm_int4_compress=llm_int4,
        llm_int8_compress=llm_int8,
        vision_int8_quant=vision_int8,
    )
    model.export_paddleocr_vl_to_ov()
    print(f"[prepare] conversion completed in {time.perf_counter() - start:.2f}s")
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and convert PaddleOCR-VL for Doc2Prototype.")
    parser.add_argument("--model-id", default="PaddlePaddle/PaddleOCR-VL-1.5")
    parser.add_argument("--cache-dir", default="_cache")
    parser.add_argument("--output-dir", default="ov_paddleocr_vl_model")
    parser.add_argument("--device", default="CPU", choices=("CPU", "GPU", "NPU", "AUTO"))
    parser.add_argument("--int4", action="store_true", help="Use INT4 LLM compression instead of INT8.")
    parser.add_argument("--vision-int8", action="store_true", help="Enable vision model INT8 quantization.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        import openvino as ov

        print(f"[prepare] OpenVINO {ov.__version__}")
    except Exception as exc:
        print(f"[prepare] OpenVINO import failed: {exc}", file=sys.stderr)
        raise

    pretrained_dir = download_model(args.model_id, args.cache_dir)
    patch_model(pretrained_dir)
    convert_to_openvino(
        pretrained_dir=pretrained_dir,
        output_dir=Path(args.output_dir),
        device=args.device,
        llm_int4=args.int4,
        llm_int8=not args.int4,
        vision_int8=args.vision_int8,
    )


if __name__ == "__main__":
    main()
