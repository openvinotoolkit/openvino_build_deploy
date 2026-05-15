#!/usr/bin/env python3
"""Generate simple raster sample inputs for the MVP demo."""

from __future__ import annotations

import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def render_text(markdown_path: Path, output_path: Path) -> None:
    text = markdown_path.read_text(encoding="utf-8")
    title_font = load_font(34)
    body_font = load_font(24)
    mono_font = load_font(23)

    lines: list[tuple[str, ImageFont.ImageFont]] = []
    for raw in text.splitlines():
        if raw.startswith("# "):
            lines.append((raw[2:], title_font))
            lines.append(("", body_font))
            continue
        wrapped = textwrap.wrap(raw, width=72) or [""]
        for line in wrapped:
            font = mono_font if line.startswith(("GET ", "POST ", "PATCH ", "DELETE ", "PUT ")) else body_font
            lines.append((line, font))

    margin = 54
    line_height = 34
    width = 1280
    height = max(720, margin * 2 + line_height * len(lines))
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    y = margin
    for line, font in lines:
        fill = (20, 20, 20)
        if line.startswith(("GET ", "POST ", "PATCH ", "DELETE ", "PUT ")):
            fill = (24, 78, 119)
        draw.text((margin, y), line, font=font, fill=fill)
        y += line_height

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(output_path)


def main() -> None:
    render_text(EXAMPLES / "api_doc_sample.md", EXAMPLES / "api_doc_sample.png")
    render_text(EXAMPLES / "flowchart_sample.md", EXAMPLES / "flowchart_sample.png")


if __name__ == "__main__":
    main()
