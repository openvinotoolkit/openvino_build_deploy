#!/usr/bin/env python3
"""Command-line MVP demo for Doc2Prototype.

The MVP path is intentionally narrow:

    input image/text -> OpenVINO PaddleOCR-VL parse -> structured JSON
    -> downstream generator -> saved Markdown/code artifacts

Use image input for the full OpenVINO path, or a text/Markdown file for fast
local validation of the structure and generation stages.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from code_generator import CodeGenerator
from structure_extractor import extract_structure, to_json_string, to_mermaid


SUPPORTED_TASKS = ("api_doc", "flowchart", "technical_doc")
TEXT_SUFFIXES = {".txt", ".md", ".markdown"}


def _now_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _openvino_version() -> str:
    try:
        import openvino as ov

        return getattr(ov, "__version__", "unknown")
    except Exception as exc:
        return f"unavailable: {exc}"


def _parse_input(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, float]]:
    input_path = Path(args.input)
    timings: dict[str, float] = {"model_load": 0.0, "openvino_inference": 0.0}

    if args.raw_text:
        return {
            "raw_text": args.raw_text,
            "task": args.task,
            "prompt": "raw_text_argument",
            "inference_time": 0.0,
            "image_path": "",
            "source_path": "",
            "parser": "raw_text_argument",
            "uses_openvino": False,
        }, timings

    if input_path.suffix.lower() in TEXT_SUFFIXES:
        return {
            "raw_text": _read_text(input_path),
            "task": args.task,
            "prompt": "raw_text_file",
            "inference_time": 0.0,
            "image_path": "",
            "source_path": str(input_path),
            "parser": "raw_text_file",
            "uses_openvino": False,
        }, timings

    model_path = Path(args.ov_model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"OpenVINO model path not found: {model_path}. "
            "Run run_demo.py once to prepare ov_paddleocr_vl_model, or use a .md/.txt input for a structure-only smoke test."
        )

    from doc_parser import DocParser

    load_start = time.perf_counter()
    parser = DocParser(
        ov_model_path=str(model_path),
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )
    timings["model_load"] = time.perf_counter() - load_start

    parse_result = parser.parse(str(input_path), task=args.task, custom_prompt=args.prompt)
    parse_result["source_path"] = str(input_path)
    parse_result["parser"] = "PaddleOCR-VL OpenVINO"
    parse_result["uses_openvino"] = True
    timings["openvino_inference"] = float(parse_result.get("inference_time", 0.0))
    return parse_result, timings


def _default_code_type(task: str) -> str:
    if task == "flowchart":
        return "mermaid_diagram"
    if task == "technical_doc":
        return "markdown_summary"
    return "api_skeleton"


def _artifact_name(task: str, code_type: str) -> str:
    if code_type == "api_skeleton":
        return "generated_api.py"
    if code_type == "mermaid_diagram":
        return "generated_flowchart.mmd"
    if code_type == "markdown_summary":
        return "generated_summary.md"
    if code_type == "test_cases":
        return "generated_tests.py"
    if code_type == "frontend_page":
        return "generated_page.tsx"
    return f"generated_{task}.txt"


def _build_markdown_report(run: dict[str, Any], structured_json: str, generated_code: str) -> str:
    metrics = run["metrics"]
    parse = run["parse"]
    structured = run["structured"]

    lines = [
        "# Doc2Prototype MVP Run",
        "",
        "## Command",
        "",
        f"`{run['command']}`",
        "",
        "## Input",
        "",
        f"- Task: `{run['input']['task']}`",
        f"- Source: `{run['input']['source'] or 'raw text argument'}`",
        f"- Parser: `{parse['parser']}`",
        f"- OpenVINO device: `{run['openvino']['device']}`",
        f"- OpenVINO version: `{run['openvino']['version']}`",
        f"- OpenVINO model: `{run['openvino']['model_path']}`",
        "",
        "## Timings",
        "",
        "| Stage | Seconds |",
        "| --- | ---: |",
    ]
    for key in ("model_load", "openvino_inference", "structure_extraction", "generation", "total"):
        lines.append(f"| {key} | {metrics.get(key, 0.0):.3f} |")

    lines.extend(
        [
            "",
            "## Structured Output",
            "",
            f"- Schema version: `{structured.get('schema_version', 'n/a')}`",
            f"- Document type: `{structured.get('document_type', run['input']['task'])}`",
        ]
    )
    if run["input"]["task"] == "api_doc":
        lines.append(f"- Endpoints: `{len(structured.get('endpoints', []))}`")
    if run["input"]["task"] == "flowchart":
        lines.append(f"- Nodes: `{len(structured.get('nodes', []))}`")
        lines.append(f"- Edges: `{len(structured.get('edges', []))}`")

    lines.extend(
        [
            "",
            "```json",
            structured_json,
            "```",
            "",
            "## Generated Artifact",
            "",
            f"- Type: `{run['generated']['code_type']}`",
            f"- Generation backend: `{run['generated']['backend']}`",
            "",
            "```",
            generated_code,
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_outputs(
    output_dir: Path,
    run: dict[str, Any],
    structured_json: str,
    generated_code: str,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = output_dir / "raw_parse.md"
    structured_path = output_dir / "structured.json"
    generated_path = output_dir / _artifact_name(run["input"]["task"], run["generated"]["code_type"])
    run_path = output_dir / "run.json"
    report_path = output_dir / "result.md"

    raw_path.write_text(run["parse"]["raw_text"], encoding="utf-8")
    structured_path.write_text(structured_json + "\n", encoding="utf-8")
    generated_path.write_text(generated_code + "\n", encoding="utf-8")
    run_path.write_text(json.dumps(run, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report_path.write_text(_build_markdown_report(run, structured_json, generated_code), encoding="utf-8")

    artifacts = {
        "raw_parse": str(raw_path),
        "structured_json": str(structured_path),
        "generated": str(generated_path),
        "run_json": str(run_path),
        "markdown_report": str(report_path),
    }

    if run["input"]["task"] == "flowchart" and run["structured"].get("nodes"):
        mermaid_path = output_dir / "structured_flowchart.mmd"
        mermaid_path.write_text(to_mermaid(run["structured"], "flowchart") + "\n", encoding="utf-8")
        artifacts["structured_mermaid"] = str(mermaid_path)

    return artifacts


def run_mvp(args: argparse.Namespace) -> dict[str, Any]:
    total_start = time.perf_counter()

    parse_result, timings = _parse_input(args)

    structure_start = time.perf_counter()
    structured = extract_structure(parse_result["raw_text"], args.task)
    timings["structure_extraction"] = time.perf_counter() - structure_start

    code_type = args.code_type or _default_code_type(args.task)
    generator = CodeGenerator(model_path=args.code_model_path, device=args.device)
    gen_start = time.perf_counter()
    generated = generator.generate(structured, code_type=code_type)
    timings["generation"] = time.perf_counter() - gen_start
    timings["total"] = time.perf_counter() - total_start

    backend = "OpenVINO/HF model" if args.code_model_path else "deterministic template"
    generated_code = generated["code"]

    run = {
        "schema_version": "doc2prototype.mvp_run.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "input": {
            "source": str(Path(args.input)) if args.input else "",
            "task": args.task,
        },
        "openvino": {
            "uses_openvino": bool(parse_result.get("uses_openvino")),
            "device": args.device,
            "model_path": args.ov_model_path,
            "version": _openvino_version(),
        },
        "parse": parse_result,
        "structured": structured,
        "generated": {
            "code_type": generated["code_type"],
            "generation_time": generated["generation_time"],
            "backend": backend,
        },
        "metrics": timings,
    }

    structured_json = to_json_string(structured)
    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs") / f"mvp_{_now_slug()}"
    artifacts = _write_outputs(output_dir, run, structured_json, generated_code)
    run["artifacts"] = artifacts

    try:
        from mvp_visualizer import create_visualizations

        visual_artifacts = create_visualizations(run, output_dir, project_root=Path.cwd())
        artifacts.update(visual_artifacts)
        run["artifacts"] = artifacts
    except Exception as exc:
        warning_path = output_dir / "visualization_warning.txt"
        warning_path.write_text(
            f"Visualization skipped: {type(exc).__name__}: {exc}\n",
            encoding="utf-8",
        )
        artifacts["visualization_warning"] = str(warning_path)
        run["artifacts"] = artifacts

    (output_dir / "run.json").write_text(json.dumps(run, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Doc2Prototype command-line MVP: parse -> JSON -> generated artifact."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="examples/api_doc_sample.md",
        help="Input image path for OpenVINO parsing, or .md/.txt for fast structure-only validation.",
    )
    parser.add_argument(
        "--task",
        choices=SUPPORTED_TASKS,
        default="api_doc",
        help="MVP document type. Keep this narrow for the next report.",
    )
    parser.add_argument(
        "--device",
        choices=("CPU", "GPU", "NPU", "AUTO"),
        default="CPU",
        help="OpenVINO device for image parsing and optional code model.",
    )
    parser.add_argument("--ov-model-path", default="ov_paddleocr_vl_model")
    parser.add_argument("--code-model-path", default=None, help="Optional local OpenVINO/HF coder model path.")
    parser.add_argument("--code-type", default=None, help="Override generated artifact type.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults to outputs/mvp_<timestamp>.")
    parser.add_argument("--prompt", default=None, help="Optional custom PaddleOCR-VL prompt for image input.")
    parser.add_argument("--raw-text", default=None, help="Inline raw parser text. Skips image/OpenVINO parsing.")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        result = run_mvp(args)
    except Exception as exc:
        print(f"[mvp] failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise

    artifacts = result["artifacts"]
    print("[mvp] completed")
    print(f"[mvp] task: {result['input']['task']}")
    print(f"[mvp] openvino: {result['openvino']['uses_openvino']} device={result['openvino']['device']}")
    print(f"[mvp] total_time: {result['metrics']['total']:.3f}s")
    print(f"[mvp] report: {artifacts['markdown_report']}")
    print(f"[mvp] structured_json: {artifacts['structured_json']}")
    print(f"[mvp] generated: {artifacts['generated']}")
    if "visual_report" in artifacts:
        print(f"[mvp] visual_report: {artifacts['visual_report']}")


if __name__ == "__main__":
    main()
