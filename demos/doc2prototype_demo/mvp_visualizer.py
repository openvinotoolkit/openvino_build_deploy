#!/usr/bin/env python3
"""Static visualizations for Doc2Prototype MVP outputs."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
from typing import Any


METHOD_COLORS = {
    "GET": "#2563eb",
    "POST": "#059669",
    "PUT": "#d97706",
    "PATCH": "#7c3aed",
    "DELETE": "#dc2626",
}


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _wrap(value: str, width: int = 44) -> list[str]:
    words = str(value).split()
    if not words:
        return [""]
    lines: list[str] = []
    current: list[str] = []
    current_len = 0
    for word in words:
        next_len = current_len + len(word) + (1 if current else 0)
        if current and next_len > width:
            lines.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len = next_len
    if current:
        lines.append(" ".join(current))
    return lines


def _relative_link(from_dir: Path, target: Path) -> str:
    try:
        return Path(os.path.relpath(target.resolve(), from_dir.resolve())).as_posix()
    except ValueError:
        return target.as_posix()


def render_metrics_svg(run: dict[str, Any]) -> str:
    metrics = run.get("metrics", {})
    stages = [
        ("Model load", float(metrics.get("model_load", 0.0))),
        ("OpenVINO inference", float(metrics.get("openvino_inference", 0.0))),
        ("Structure extraction", float(metrics.get("structure_extraction", 0.0))),
        ("Generation", float(metrics.get("generation", 0.0))),
    ]

    width = 920
    row_h = 54
    top = 72
    height = top + row_h * len(stages) + 56
    label_w = 220
    bar_w = 560
    max_value = max([value for _, value in stages] + [1.0])

    rows = []
    for i, (label, value) in enumerate(stages):
        y = top + i * row_h
        w = max(3, int(bar_w * value / max_value))
        rows.append(
            f"""
  <text x="36" y="{y + 25}" class="label">{_esc(label)}</text>
  <rect x="{label_w}" y="{y}" width="{bar_w}" height="30" rx="4" class="bar-bg"/>
  <rect x="{label_w}" y="{y}" width="{w}" height="30" rx="4" class="bar"/>
  <text x="{label_w + bar_w + 22}" y="{y + 22}" class="value">{value:.3f}s</text>
"""
        )

    total = float(metrics.get("total", 0.0))
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <style>
    .title {{ font: 700 24px Arial, sans-serif; fill: #172033; }}
    .subtitle {{ font: 400 14px Arial, sans-serif; fill: #64748b; }}
    .label {{ font: 600 15px Arial, sans-serif; fill: #334155; }}
    .value {{ font: 600 14px Arial, sans-serif; fill: #0f172a; }}
    .bar-bg {{ fill: #e2e8f0; }}
    .bar {{ fill: #0f766e; }}
    .panel {{ fill: #ffffff; stroke: #d7dee8; }}
  </style>
  <rect x="1" y="1" width="{width - 2}" height="{height - 2}" rx="8" class="panel"/>
  <text x="36" y="38" class="title">MVP Pipeline Timing</text>
  <text x="36" y="59" class="subtitle">Total: {total:.3f}s, device: {_esc(run.get("openvino", {}).get("device", "CPU"))}</text>
  {"".join(rows)}
</svg>
"""


def render_api_svg(structured: dict[str, Any]) -> str:
    endpoints = structured.get("endpoints", [])
    width = 1040
    card_h = 74
    gap = 14
    top = 82
    height = top + max(1, len(endpoints)) * (card_h + gap) + 34

    rows = []
    if not endpoints:
        rows.append('<text x="40" y="110" class="empty">No endpoints extracted.</text>')

    for i, endpoint in enumerate(endpoints):
        y = top + i * (card_h + gap)
        method = str(endpoint.get("method", "GET")).upper()
        color = METHOD_COLORS.get(method, "#475569")
        path = endpoint.get("path", "/")
        description = endpoint.get("description", "")
        params = endpoint.get("parameters", [])
        param_text = ", ".join(param.get("name", "") for param in params) or "none"
        rows.append(
            f"""
  <rect x="34" y="{y}" width="{width - 68}" height="{card_h}" rx="8" class="card"/>
  <rect x="54" y="{y + 21}" width="82" height="32" rx="5" fill="{color}"/>
  <text x="95" y="{y + 42}" class="method" text-anchor="middle">{_esc(method)}</text>
  <text x="158" y="{y + 31}" class="path">{_esc(path)}</text>
  <text x="158" y="{y + 55}" class="desc">{_esc(description or "No description")}</text>
  <text x="{width - 260}" y="{y + 45}" class="params">params: {_esc(param_text)}</text>
"""
        )

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <style>
    .title {{ font: 700 25px Arial, sans-serif; fill: #172033; }}
    .subtitle {{ font: 400 14px Arial, sans-serif; fill: #64748b; }}
    .card {{ fill: #ffffff; stroke: #d7dee8; }}
    .method {{ font: 700 14px Arial, sans-serif; fill: white; }}
    .path {{ font: 700 18px Arial, sans-serif; fill: #0f172a; }}
    .desc {{ font: 400 14px Arial, sans-serif; fill: #475569; }}
    .params {{ font: 600 13px Arial, sans-serif; fill: #64748b; }}
    .empty {{ font: 500 16px Arial, sans-serif; fill: #64748b; }}
    .panel {{ fill: #f8fafc; stroke: #d7dee8; }}
  </style>
  <rect x="1" y="1" width="{width - 2}" height="{height - 2}" rx="8" class="panel"/>
  <text x="36" y="40" class="title">Extracted API Surface</text>
  <text x="36" y="62" class="subtitle">{len(endpoints)} endpoint(s) generated from structured JSON</text>
  {"".join(rows)}
</svg>
"""


def _node_shape(node: dict[str, Any], x: int, y: int, width: int, height: int) -> str:
    label = _esc(node.get("label", "Step"))
    node_type = node.get("type", "process")
    lines = _wrap(label, width=32)[:3]
    text_y = y + height / 2 - (len(lines) - 1) * 9
    text = "".join(
        f'<text x="{x + width / 2:.0f}" y="{text_y + idx * 18:.0f}" class="node-text" text-anchor="middle">{line}</text>'
        for idx, line in enumerate(lines)
    )

    if node_type == "decision":
        points = f"{x + width / 2},{y} {x + width},{y + height / 2} {x + width / 2},{y + height} {x},{y + height / 2}"
        shape = f'<polygon points="{points}" class="node decision"/>'
    elif node_type in {"start", "end"}:
        shape = f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="{height / 2:.0f}" class="node terminal"/>'
    else:
        shape = f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="8" class="node process"/>'
    return shape + text


def render_flowchart_svg(structured: dict[str, Any]) -> str:
    nodes = structured.get("nodes", [])
    edges = structured.get("edges", [])
    width = 860
    node_w = 300
    node_h = 72
    gap = 56
    x = (width - node_w) // 2
    top = 88
    height = top + max(1, len(nodes)) * (node_h + gap) + 30

    node_pos = {}
    parts = []
    for i, node in enumerate(nodes):
        y = top + i * (node_h + gap)
        node_pos[node.get("id", f"node_{i}")] = (x, y)
        parts.append(_node_shape(node, x, y, node_w, node_h))

    edge_parts = []
    for edge in edges:
        src = edge.get("from")
        dst = edge.get("to")
        if src not in node_pos or dst not in node_pos:
            continue
        sx, sy = node_pos[src]
        dx, dy = node_pos[dst]
        x1 = sx + node_w / 2
        y1 = sy + node_h
        x2 = dx + node_w / 2
        y2 = dy
        edge_parts.append(
            f'<path d="M {x1:.0f} {y1:.0f} C {x1:.0f} {y1 + 28:.0f}, {x2:.0f} {y2 - 28:.0f}, {x2:.0f} {y2:.0f}" class="edge" marker-end="url(#arrow)"/>'
        )
        if edge.get("label"):
            edge_parts.append(
                f'<text x="{(x1 + x2) / 2:.0f}" y="{(y1 + y2) / 2:.0f}" class="edge-label">{_esc(edge["label"])}</text>'
            )

    if not nodes:
        parts.append('<text x="40" y="116" class="empty">No flowchart nodes extracted.</text>')

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#64748b"/>
    </marker>
  </defs>
  <style>
    .title {{ font: 700 25px Arial, sans-serif; fill: #172033; }}
    .subtitle {{ font: 400 14px Arial, sans-serif; fill: #64748b; }}
    .node {{ stroke-width: 2; }}
    .terminal {{ fill: #ecfdf5; stroke: #059669; }}
    .process {{ fill: #eff6ff; stroke: #2563eb; }}
    .decision {{ fill: #fff7ed; stroke: #ea580c; }}
    .node-text {{ font: 600 13px Arial, sans-serif; fill: #172033; }}
    .edge {{ fill: none; stroke: #64748b; stroke-width: 2; }}
    .edge-label {{ font: 600 12px Arial, sans-serif; fill: #475569; }}
    .empty {{ font: 500 16px Arial, sans-serif; fill: #64748b; }}
    .panel {{ fill: #f8fafc; stroke: #d7dee8; }}
  </style>
  <rect x="1" y="1" width="{width - 2}" height="{height - 2}" rx="8" class="panel"/>
  <text x="36" y="40" class="title">Extracted Flowchart</text>
  <text x="36" y="62" class="subtitle">{len(nodes)} node(s), {len(edges)} edge(s)</text>
  {"".join(edge_parts)}
  {"".join(parts)}
</svg>
"""


def _render_html(run: dict[str, Any], output_dir: Path, artifacts: dict[str, str]) -> str:
    structured = run.get("structured", {})
    task = run.get("input", {}).get("task", "")
    source = run.get("input", {}).get("source") or ""
    source_path = Path(source)
    if source and not source_path.is_absolute():
        source_path = Path.cwd() / source_path
    source_link = _relative_link(output_dir, source_path) if source and source_path.exists() else ""

    visual = "api_endpoints.svg" if task == "api_doc" else "flowchart.svg"
    generated_link = Path(artifacts.get("generated", "")).name if artifacts.get("generated") else ""

    layout_links = []
    for key, label in (("layout_overlay", "Layout Overlay"), ("text_heatmap", "Text Density Heatmap")):
        if key in artifacts:
            layout_links.append(f'<figure><img src="{_esc(Path(artifacts[key]).name)}" alt="{label}"><figcaption>{label}</figcaption></figure>')

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Doc2Prototype MVP Visual Report</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5f7fb;
      --panel: #ffffff;
      --text: #172033;
      --muted: #64748b;
      --line: #d7dee8;
      --accent: #0f766e;
    }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Arial, sans-serif;
    }}
    main {{
      max-width: 1160px;
      margin: 0 auto;
      padding: 28px;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      gap: 20px;
      align-items: flex-start;
      margin-bottom: 22px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 28px;
      letter-spacing: 0;
    }}
    .muted {{ color: var(--muted); }}
    .pill {{
      display: inline-block;
      padding: 6px 10px;
      border-radius: 999px;
      background: #e6f4f1;
      color: #0f766e;
      font-weight: 700;
      font-size: 13px;
      white-space: nowrap;
    }}
    section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      margin: 16px 0;
    }}
    h2 {{
      margin: 0 0 14px;
      font-size: 18px;
      letter-spacing: 0;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 16px;
    }}
    .kv {{
      display: grid;
      grid-template-columns: 140px 1fr;
      gap: 8px 14px;
      font-size: 14px;
    }}
    .kv b {{ color: #334155; }}
    img, object {{
      max-width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: white;
    }}
    figure {{ margin: 0; }}
    figcaption {{
      margin-top: 8px;
      color: var(--muted);
      font-size: 13px;
    }}
    pre {{
      overflow: auto;
      max-height: 360px;
      padding: 14px;
      background: #0f172a;
      color: #e2e8f0;
      border-radius: 6px;
      font-size: 13px;
      line-height: 1.45;
    }}
    a {{ color: var(--accent); font-weight: 700; }}
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>Doc2Prototype MVP Visual Report</h1>
      <div class="muted">Static visualization generated from <code>run.json</code>.</div>
    </div>
    <span class="pill">{_esc(task)}</span>
  </header>

  <section>
    <h2>Run Summary</h2>
    <div class="kv">
      <b>Source</b><span>{f'<a href="{_esc(source_link)}">{_esc(source)}</a>' if source_link else _esc(source or "raw text")}</span>
      <b>OpenVINO</b><span>{_esc(run.get("openvino", {}).get("uses_openvino", False))}</span>
      <b>Device</b><span>{_esc(run.get("openvino", {}).get("device", ""))}</span>
      <b>Schema</b><span>{_esc(structured.get("schema_version", ""))}</span>
      <b>Generated</b><span>{f'<a href="{_esc(generated_link)}">{_esc(generated_link)}</a>' if generated_link else ""}</span>
    </div>
  </section>

  <section>
    <h2>Pipeline Timing</h2>
    <object data="metrics.svg" type="image/svg+xml"></object>
  </section>

  <section>
    <h2>Structured Visualization</h2>
    <object data="{_esc(visual)}" type="image/svg+xml"></object>
  </section>

  {f'<section><h2>Document Layout</h2><div class="grid">{"".join(layout_links)}</div></section>' if layout_links else ''}

  <section>
    <h2>Structured JSON Preview</h2>
    <pre>{_esc(json.dumps(structured, indent=2, ensure_ascii=False))}</pre>
  </section>
</main>
</body>
</html>
"""


def _maybe_generate_layout(run: dict[str, Any], output_dir: Path, project_root: Path) -> dict[str, str]:
    source = run.get("input", {}).get("source") or ""
    if not source:
        return {}
    source_path = Path(source)
    if not source_path.is_absolute():
        source_path = project_root / source_path
    if source_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".webp"} or not source_path.exists():
        return {}

    raw_text = run.get("parse", {}).get("raw_text", "")
    artifacts: dict[str, str] = {}
    try:
        from layout_visualizer import generate_heatmap, visualize_layout

        layout_path = output_dir / "layout_overlay.png"
        heatmap_path = output_dir / "text_heatmap.png"
        visualize_layout(str(source_path), raw_text, output_path=str(layout_path))
        generate_heatmap(str(source_path), raw_text, output_path=str(heatmap_path))
        artifacts["layout_overlay"] = str(layout_path)
        artifacts["text_heatmap"] = str(heatmap_path)
    except Exception as exc:
        (output_dir / "visualization_warning.txt").write_text(
            f"Layout visualization skipped: {type(exc).__name__}: {exc}\n",
            encoding="utf-8",
        )
    return artifacts


def create_visualizations(run: dict[str, Any], output_dir: Path, project_root: Path | None = None) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    project_root = Path.cwd() if project_root is None else Path(project_root)

    structured = run.get("structured", {})
    task = run.get("input", {}).get("task", structured.get("document_type", "api_doc"))

    artifacts: dict[str, str] = {}

    metrics_path = output_dir / "metrics.svg"
    metrics_path.write_text(render_metrics_svg(run), encoding="utf-8")
    artifacts["metrics_svg"] = str(metrics_path)

    if task == "flowchart":
        visual_path = output_dir / "flowchart.svg"
        visual_path.write_text(render_flowchart_svg(structured), encoding="utf-8")
        artifacts["flowchart_svg"] = str(visual_path)
    else:
        visual_path = output_dir / "api_endpoints.svg"
        visual_path.write_text(render_api_svg(structured), encoding="utf-8")
        artifacts["api_endpoints_svg"] = str(visual_path)

    artifacts.update(_maybe_generate_layout(run, output_dir, project_root))

    merged_artifacts = dict(run.get("artifacts", {}))
    merged_artifacts.update(artifacts)
    html_path = output_dir / "visual_report.html"
    html_path.write_text(_render_html(run, output_dir, merged_artifacts), encoding="utf-8")
    artifacts["visual_report"] = str(html_path)
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate static visualizations for a Doc2Prototype MVP run.")
    parser.add_argument("run_json", help="Path to outputs/<run>/run.json")
    parser.add_argument("--project-root", default=None, help="Project root used to resolve relative input paths.")
    args = parser.parse_args()

    run_path = Path(args.run_json)
    run = json.loads(run_path.read_text(encoding="utf-8"))
    output_dir = run_path.parent
    project_root = Path(args.project_root) if args.project_root else output_dir.parents[1]
    artifacts = create_visualizations(run, output_dir, project_root=project_root)

    run["artifacts"] = {**run.get("artifacts", {}), **artifacts}
    run_path.write_text(json.dumps(run, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("[visualizer] completed")
    for key, path in artifacts.items():
        print(f"[visualizer] {key}: {path}")


if __name__ == "__main__":
    main()
