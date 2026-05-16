#!/usr/bin/env python3
"""Prepare an official/reference PaddleOCR-VL image for local demo runs.

The script first tries to reuse the OpenVINO Notebooks PaddleOCR-VL sample
image when that checkout is available beside this demo's parent workspace. If
not found, it downloads the public PaddleOCR-VL demo image.
"""

from __future__ import annotations

import argparse
import os
import shutil
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "examples" / "official_paddleocr_vl_sample.png"
DOWNLOAD_URL = "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/paddleocr_vl_demo.png"


def notebook_candidates() -> list[Path]:
    candidates = []
    env_path = os.environ.get("OPENVINO_PADDLEOCR_VL_SAMPLE")
    if env_path:
        candidates.append(Path(env_path).expanduser())

    try:
        workspace_root = ROOT.parents[2]
        candidates.append(workspace_root / "openvino_notebooks" / "notebooks" / "paddleocr_vl" / "test.png")
    except IndexError:
        pass

    candidates.append(Path.home() / "openvino_notebooks" / "notebooks" / "paddleocr_vl" / "test.png")
    return candidates


def prepare(output: Path, force_download: bool = False) -> tuple[Path, str]:
    output.parent.mkdir(parents=True, exist_ok=True)

    if not force_download:
        for candidate in notebook_candidates():
            if candidate.exists():
                shutil.copyfile(candidate, output)
                return output, f"copied from OpenVINO Notebooks sample: {candidate}"

    with urllib.request.urlopen(DOWNLOAD_URL, timeout=60) as response:
        output.write_bytes(response.read())
    return output, f"downloaded from PaddleOCR-VL official demo: {DOWNLOAD_URL}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare an official/reference PaddleOCR-VL sample image.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Where to write the sample image.")
    parser.add_argument("--force-download", action="store_true", help="Skip local OpenVINO Notebooks lookup.")
    args = parser.parse_args()

    output, source = prepare(Path(args.output), force_download=args.force_download)
    print(f"[sample] {source}")
    print(f"[sample] wrote: {output}")


if __name__ == "__main__":
    main()
