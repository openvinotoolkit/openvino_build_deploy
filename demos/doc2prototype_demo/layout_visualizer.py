"""
Layout Visualizer for Doc2Prototype
====================================
Visual analysis of document parsing results from PaddleOCR-VL.
Generates annotated images with bounding boxes, region classification,
and text density heatmaps.
"""

import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    _DEMO_UTILS = Path(__file__).resolve().parents[1] / "utils"
    if _DEMO_UTILS.exists():
        sys.path.insert(0, str(_DEMO_UTILS))
    from demo_utils import draw_ov_watermark
except Exception:
    draw_ov_watermark = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Color palette (BGR for OpenCV, used everywhere for consistency)
COLORS: Dict[str, Tuple[int, int, int]] = {
    "header":    (255, 140, 0),    # Blue (BGR)
    "body":      (0, 180, 0),      # Green
    "table":     (0, 140, 255),    # Orange
    "figure":    (180, 0, 180),    # Purple
    "code":      (0, 0, 220),      # Red
    "unknown":   (128, 128, 128),  # Gray
}

# RGBA versions for PIL semi-transparent overlays
COLORS_RGBA: Dict[str, Tuple[int, int, int, int]] = {
    "header":  (0, 140, 255, 80),
    "body":    (0, 180, 0, 50),
    "table":   (255, 140, 0, 70),
    "figure":  (180, 0, 180, 60),
    "code":    (220, 0, 0, 70),
    "unknown": (128, 128, 128, 40),
}

# Grid configuration for layout detection
DEFAULT_GRID_ROWS = 6
DEFAULT_GRID_COLS = 4

OUTPUT_DIR = Path(__file__).parent / "outputs"


def apply_openvino_watermark(image: np.ndarray) -> np.ndarray:
    """Add an OpenVINO watermark to visual outputs."""
    if draw_ov_watermark is not None:
        try:
            draw_ov_watermark(image, alpha=0.45, size=0.16)
            return image
        except Exception:
            pass

    h, w = image.shape[:2]
    label = "OpenVINO"
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = max(0.6, w / 1700)
    thickness = 2
    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
    x = max(12, w - tw - 24)
    y = max(th + 12, h - 20)
    cv2.rectangle(image, (x - 10, y - th - 10), (x + tw + 10, y + 8), (255, 255, 255), cv2.FILLED)
    cv2.rectangle(image, (x - 10, y - th - 10), (x + tw + 10, y + 8), (180, 180, 180), 1)
    cv2.putText(image, label, (x, y), font, scale, (0, 104, 181), thickness, cv2.LINE_AA)
    return image


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TextSegment:
    """A single text segment extracted from OCR output."""
    text: str
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float = 1.0
    element_type: str = "unknown"

    @property
    def center(self) -> Tuple[int, int]:
        x = (self.bbox[0] + self.bbox[2]) // 2
        y = (self.bbox[1] + self.bbox[3]) // 2
        return x, y

    @property
    def area(self) -> int:
        return max(0, self.bbox[2] - self.bbox[0]) * max(0, self.bbox[3] - self.bbox[1])


@dataclass
class GridRegion:
    """A grid cell in the document layout."""
    row: int
    col: int
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    segments: List[TextSegment] = field(default_factory=list)
    region_type: str = "unknown"
    confidence: float = 0.0

    @property
    def text_density(self) -> float:
        """Characters per pixel-area unit."""
        area = max(1, (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1]))
        total_chars = sum(len(s.text) for s in self.segments)
        return total_chars / area * 10000  # per 10k pixels


@dataclass
class LayoutSummary:
    """JSON-serializable summary of the detected layout."""
    image_size: Tuple[int, int]
    grid: Tuple[int, int]
    total_segments: int
    regions: List[Dict[str, Any]]
    element_counts: Dict[str, int]
    dominant_type: str


# ---------------------------------------------------------------------------
# OCR text parsing
# ---------------------------------------------------------------------------

def parse_ocr_text(ocr_text: str, image_shape: Optional[Tuple[int, int]] = None) -> List[TextSegment]:
    """
    Parse raw PaddleOCR-VL text output into TextSegment objects.

    Handles several common PaddleOCR output formats:
      1.  json-style:  [{"text": "...", "bbox": [x1,y1,x2,y2], "confidence": 0.99}, ...]
      2.  line-based:  "text\t[x1, y1, x2, y2]\tconfidence"
      3.  plain lines: each non-empty line is a segment (bboxes auto-generated)

    Parameters
    ----------
    ocr_text : str
        Raw OCR output string.
    image_shape : tuple (H, W), optional
        If provided, used to assign synthetic bounding boxes to plain text.

    Returns
    -------
    list of TextSegment
    """
    ocr_text = ocr_text.strip()
    if not ocr_text:
        return []

    # --- Attempt JSON parse ---
    if ocr_text.startswith("["):
        try:
            data = json.loads(ocr_text)
            segments = []
            for item in data:
                if isinstance(item, dict):
                    text = item.get("text", "")
                    bbox = item.get("bbox", item.get("bounding_box", [0, 0, 0, 0]))
                    conf = float(item.get("confidence", item.get("score", 1.0)))
                    if text:
                        segments.append(TextSegment(text=text, bbox=tuple(int(v) for v in bbox), confidence=conf))
            if segments:
                return segments
        except json.JSONDecodeError:
            pass

    # --- Attempt tab-separated format ---
    segments = []
    tab_pattern = re.compile(r'^(.*?)\t\[?([\d,\s]+)\]?\t?([\d.]*)$')
    for line in ocr_text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = tab_pattern.match(line)
        if m:
            text = m.group(1).strip()
            coords = [int(c.strip()) for c in m.group(2).split(",") if c.strip()]
            conf = float(m.group(3)) if m.group(3) else 1.0
            if len(coords) >= 4 and text:
                segments.append(TextSegment(text=text, bbox=tuple(coords[:4]), confidence=conf))
                continue

    if segments:
        return segments

    # --- Fallback: plain lines with synthetic bounding boxes ---
    lines = [l.strip() for l in ocr_text.splitlines() if l.strip()]
    if not lines:
        return []

    h, w = (image_shape[0], image_shape[1]) if image_shape else (1100, 850)
    line_height = max(20, h // max(len(lines), 1))
    margin = int(w * 0.08)

    for i, text in enumerate(lines):
        y1 = i * line_height
        y2 = min((i + 1) * line_height, h)
        # Estimate width from character count
        est_char_w = max(8, w // 80)
        x2 = min(margin + len(text) * est_char_w, w - margin)
        segments.append(TextSegment(text=text, bbox=(margin, y1, x2, y2), confidence=0.85))

    return segments


# ---------------------------------------------------------------------------
# Element type classification
# ---------------------------------------------------------------------------

def classify_element(segment: TextSegment, image_height: int) -> str:
    """
    Heuristic classification of a text segment into an element type.

    Rules (ordered by priority):
      - code:     contains code-like tokens or indentation patterns
      - table:    text contains pipe chars, tabs, or grid-like alignment
      - header:   short text near top of image, ALL CAPS, or starts with #
      - figure:   text matches common figure-caption patterns
      - body:     everything else
    """
    text = segment.text.strip()
    lower = text.lower()

    # Code detection
    code_indicators = ["def ", "class ", "import ", "function ", "return ", "const ",
                       "var ", "{", "}", "();", "=>", "//", "#include", "```",
                       "print(", "console.", "SELECT ", "FROM ", "WHERE "]
    if any(tok in text for tok in code_indicators):
        return "code"
    # Heavily indented
    if text.startswith("    ") or text.startswith("\t"):
        return "code"

    # Table detection
    if "|" in text and text.count("|") >= 2:
        return "table"
    if "\t" in segment.text and segment.text.count("\t") >= 2:
        return "table"

    # Figure caption
    fig_patterns = [r'^fig(ure)?\.?\s*\d', r'^image\s*\d', r'^diagram\s*\d',
                    r'^chart\s*\d', r'^photo\s*\d', r'^illustration']
    if any(re.match(p, lower) for p in fig_patterns):
        return "figure"

    # Header detection
    top_third = image_height / 3
    _, y1, _, y2 = segment.bbox
    is_short = len(text) < 80
    is_near_top = y1 < top_third
    is_all_caps = text.isupper() and len(text) > 2
    starts_with_hash = text.startswith("#")
    is_title_case = text.istitle() and is_short

    if is_all_caps and is_short:
        return "header"
    if starts_with_hash:
        return "header"
    if is_near_top and is_short and (is_title_case or len(text.split()) <= 8):
        return "header"

    return "body"


def classify_segments(segments: List[TextSegment], image_height: int) -> List[TextSegment]:
    """Classify all segments in-place and return them."""
    for seg in segments:
        seg.element_type = classify_element(seg, image_height)
    return segments


# ---------------------------------------------------------------------------
# Grid-based layout detection
# ---------------------------------------------------------------------------

def build_grid(
    image_shape: Tuple[int, int],
    segments: List[TextSegment],
    rows: int = DEFAULT_GRID_ROWS,
    cols: int = DEFAULT_GRID_COLS,
) -> List[GridRegion]:
    """
    Divide the image into a grid and map text segments to regions.

    Returns a list of GridRegion objects covering the entire image.
    """
    h, w = image_shape
    cell_h = h // rows
    cell_w = w // cols
    regions: List[GridRegion] = []

    for r in range(rows):
        for c in range(cols):
            x1 = c * cell_w
            y1 = r * cell_h
            x2 = (c + 1) * cell_w if c < cols - 1 else w
            y2 = (r + 1) * cell_h if r < rows - 1 else h
            regions.append(GridRegion(row=r, col=c, bbox=(x1, y1, x2, y2)))

    # Assign segments to regions based on center point
    for seg in segments:
        cx, cy = seg.center
        col_idx = min(cx // cell_w, cols - 1)
        row_idx = min(cy // cell_h, rows - 1)
        idx = row_idx * cols + col_idx
        if 0 <= idx < len(regions):
            regions[idx].segments.append(seg)

    # Determine dominant type per region
    for region in regions:
        if not region.segments:
            region.region_type = "unknown"
            region.confidence = 0.0
            continue
        type_counts: Dict[str, int] = defaultdict(int)
        for s in region.segments:
            type_counts[s.element_type] += 1
        dominant = max(type_counts, key=type_counts.get)
        region.region_type = dominant
        region.confidence = type_counts[dominant] / len(region.segments)

    return regions


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _get_font(size: int = 16) -> ImageFont.FreeTypeFont:
    """Try to load a clean TTF font, fall back to default."""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            return ImageFont.truetype(fp, size)
    return ImageFont.load_default()


def draw_segment_boxes(
    image: np.ndarray,
    segments: List[TextSegment],
    alpha: float = 0.35,
) -> np.ndarray:
    """
    Draw semi-transparent colored bounding boxes and labels for each segment.

    Uses PIL for clean text rendering on top of an OpenCV image.
    """
    overlay = image.copy()
    h, w = image.shape[:2]

    for seg in segments:
        x1, y1, x2, y2 = seg.bbox
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        color = COLORS.get(seg.element_type, COLORS["unknown"])
        # Fill with semi-transparent color
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, cv2.FILLED)
        # Border
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    # Blend overlay
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    # Draw labels using PIL for clean typography
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = _get_font(14)

    for seg in segments:
        x1, y1, x2, y2 = seg.bbox
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        color_rgb = COLORS.get(seg.element_type, COLORS["unknown"])
        color_rgb_pil = (color_rgb[2], color_rgb[1], color_rgb[0])  # BGR -> RGB

        label = f"{seg.element_type} ({seg.confidence:.0%})"
        # Background strip for label
        try:
            bbox_text = draw.textbbox((0, 0), label, font=font)
            tw, th = bbox_text[2] - bbox_text[0], bbox_text[3] - bbox_text[1]
        except AttributeError:
            tw, th = draw.textsize(label, font=font)

        label_y = max(0, y1 - th - 4)
        draw.rectangle([x1, label_y, x1 + tw + 8, label_y + th + 4], fill=color_rgb_pil)
        draw.text((x1 + 4, label_y + 2), label, fill=(255, 255, 255), font=font)

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def draw_grid_overlay(
    image: np.ndarray,
    regions: List[GridRegion],
    rows: int,
    cols: int,
    alpha: float = 0.15,
) -> np.ndarray:
    """Draw subtle grid lines and region-type color washes."""
    h, w = image.shape[:2]
    overlay = image.copy()

    for region in regions:
        if region.region_type == "unknown" or not region.segments:
            continue
        x1, y1, x2, y2 = region.bbox
        color = COLORS.get(region.region_type, COLORS["unknown"])
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, cv2.FILLED)

    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    # Grid lines
    cell_h = h // rows
    cell_w = w // cols
    for r in range(1, rows):
        cv2.line(image, (0, r * cell_h), (w, r * cell_h), (200, 200, 200), 1, cv2.LINE_AA)
    for c in range(1, cols):
        cv2.line(image, (c * cell_w, 0), (c * cell_w, h), (200, 200, 200), 1, cv2.LINE_AA)

    return image


def draw_legend(image: np.ndarray) -> np.ndarray:
    """Draw a compact legend in the top-right corner."""
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = _get_font(13)
    w = image.shape[1]

    items = [("header", "Header"), ("body", "Body Text"),
             ("table", "Table"), ("figure", "Figure"), ("code", "Code")]

    box_w, box_h = 18, 14
    padding = 8
    line_h = box_h + 6
    legend_w = 130
    legend_h = len(items) * line_h + padding * 2
    x0 = w - legend_w - 15
    y0 = 15

    # Background
    draw.rectangle([x0, y0, x0 + legend_w, y0 + legend_h],
                   fill=(30, 30, 30), outline=(100, 100, 100))

    for i, (key, label) in enumerate(items):
        y = y0 + padding + i * line_h
        color_rgb = COLORS.get(key, COLORS["unknown"])
        color_pil = (color_rgb[2], color_rgb[1], color_rgb[0])
        draw.rectangle([x0 + padding, y, x0 + padding + box_w, y + box_h], fill=color_pil)
        draw.text((x0 + padding + box_w + 6, y - 1), label, fill=(230, 230, 230), font=font)

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Heatmap generation
# ---------------------------------------------------------------------------

def generate_heatmap(
    image_path: str,
    ocr_text: str,
    output_path: Optional[str] = None,
    grid_rows: int = 20,
    grid_cols: int = 15,
    colormap: int = cv2.COLORMAP_JET,
    blur_ksize: int = 51,
) -> Tuple[np.ndarray, str]:
    """
    Generate a text-density heatmap overlaid on the source image.

    Parameters
    ----------
    image_path : str
        Path to the source document image.
    ocr_text : str
        Raw OCR text output.
    output_path : str, optional
        Where to save the heatmap. Defaults to outputs/<stem>_heatmap.png.
    grid_rows, grid_cols : int
        Resolution of the density grid.
    colormap : int
        OpenCV colormap constant.
    blur_ksize : int
        Kernel size for Gaussian blur (must be odd).

    Returns
    -------
    (heatmap_image, saved_path)
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    h, w = image.shape[:2]
    segments = parse_ocr_text(ocr_text, (h, w))

    # Build density map
    density = np.zeros((grid_rows, grid_cols), dtype=np.float64)
    cell_h = h / grid_rows
    cell_w = w / grid_cols

    for seg in segments:
        cx, cy = seg.center
        r = min(int(cy / cell_h), grid_rows - 1)
        c = min(int(cx / cell_w), grid_cols - 1)
        density[r, c] += len(seg.text)

    # Normalize 0-255
    if density.max() > 0:
        density = (density / density.max() * 255).astype(np.uint8)
    else:
        density = density.astype(np.uint8)

    # Resize to image size and apply colormap
    density_resized = cv2.resize(density, (w, h), interpolation=cv2.INTER_CUBIC)
    if blur_ksize > 0:
        blur_ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        density_resized = cv2.GaussianBlur(density_resized, (blur_ksize, blur_ksize), 0)

    heatmap_color = cv2.applyColorMap(density_resized, colormap)

    # Blend
    blended = cv2.addWeighted(image, 0.55, heatmap_color, 0.45, 0)

    # Add color bar legend
    bar_w = 30
    bar_x = w - bar_w - 20
    bar_y1, bar_y2 = 60, h - 60
    gradient = np.linspace(255, 0, bar_y2 - bar_y1, dtype=np.uint8).reshape(-1, 1)
    gradient = np.repeat(gradient, bar_w, axis=1)
    gradient_color = cv2.applyColorMap(gradient, colormap)
    blended[bar_y1:bar_y2, bar_x:bar_x + bar_w] = gradient_color

    pil_img = Image.fromarray(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = _get_font(12)
    draw.text((bar_x - 5, bar_y1 - 20), "High", fill=(255, 255, 255), font=font)
    draw.text((bar_x - 3, bar_y2 + 5), "Low", fill=(255, 255, 255), font=font)
    draw.text((w // 2 - 80, 15), "Text Density Heatmap", fill=(255, 255, 255), font=_get_font(16))
    blended = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    blended = apply_openvino_watermark(blended)

    if output_path is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stem = Path(image_path).stem
        output_path = str(OUTPUT_DIR / f"{stem}_heatmap.png")

    cv2.imwrite(output_path, blended)
    print(f"[layout_visualizer] Heatmap saved to {output_path}")
    return blended, output_path


# ---------------------------------------------------------------------------
# Main visualization function
# ---------------------------------------------------------------------------

def visualize_layout(
    image_path: str,
    ocr_text: str,
    output_path: Optional[str] = None,
    grid_rows: int = DEFAULT_GRID_ROWS,
    grid_cols: int = DEFAULT_GRID_COLS,
) -> Tuple[np.ndarray, LayoutSummary, str]:
    """
    Main entry point: produce an annotated image showing detected layout.

    Parameters
    ----------
    image_path : str
        Path to the source document image.
    ocr_text : str
        Raw PaddleOCR-VL text output.
    output_path : str, optional
        Destination for the annotated image. Defaults to outputs/<stem>_layout.png.
    grid_rows, grid_cols : int
        Grid resolution for layout region detection.

    Returns
    -------
    (annotated_image, layout_summary, saved_path)
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    h, w = image.shape[:2]
    segments = parse_ocr_text(ocr_text, (h, w))
    segments = classify_segments(segments, h)
    regions = build_grid((h, w), segments, grid_rows, grid_cols)

    # Draw layers
    annotated = draw_grid_overlay(image.copy(), regions, grid_rows, grid_cols)
    annotated = draw_segment_boxes(annotated, segments)
    annotated = draw_legend(annotated)
    annotated = apply_openvino_watermark(annotated)

    # Build summary
    element_counts: Dict[str, int] = defaultdict(int)
    region_dicts: List[Dict[str, Any]] = []
    for region in regions:
        element_counts[region.region_type] += len(region.segments)
        region_dicts.append({
            "row": region.row,
            "col": region.col,
            "bbox": list(region.bbox),
            "type": region.region_type,
            "confidence": round(region.confidence, 3),
            "segment_count": len(region.segments),
            "text_preview": region.segments[0].text[:80] if region.segments else "",
        })

    dominant = max(element_counts, key=element_counts.get) if element_counts else "unknown"
    summary = LayoutSummary(
        image_size=(w, h),
        grid=(grid_rows, grid_cols),
        total_segments=len(segments),
        regions=region_dicts,
        element_counts=dict(element_counts),
        dominant_type=dominant,
    )

    # Save
    if output_path is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stem = Path(image_path).stem
        output_path = str(OUTPUT_DIR / f"{stem}_layout.png")

    cv2.imwrite(output_path, annotated)

    # Save JSON summary alongside
    json_path = Path(output_path).with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(asdict(summary), f, indent=2, default=str)

    print(f"[layout_visualizer] Annotated image saved to {output_path}")
    print(f"[layout_visualizer] Layout summary  saved to {json_path}")

    return annotated, summary, output_path


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

def _create_test_image(path: str, w: int = 850, h: int = 1100) -> str:
    """Create a synthetic document-like test image."""
    img = np.ones((h, w, 3), dtype=np.uint8) * 245  # light gray bg

    # Title area
    cv2.rectangle(img, (60, 40), (w - 60, 110), (255, 255, 255), cv2.FILLED)
    cv2.putText(img, "Document Title Area", (80, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 30, 30), 2)

    # Simulated paragraphs
    y = 150
    for i in range(6):
        line_w = np.random.randint(300, 700)
        cv2.rectangle(img, (70, y), (70 + line_w, y + 12), (200, 200, 200), cv2.FILLED)
        y += 22
    y += 20

    # Simulated code block
    cv2.rectangle(img, (70, y), (w - 70, y + 160), (240, 240, 240), cv2.FILLED)
    cv2.rectangle(img, (70, y), (w - 70, y + 160), (180, 180, 180), 1)
    for j in range(6):
        line_w = np.random.randint(150, 500)
        cv2.rectangle(img, (85, y + 18 + j * 22), (85 + line_w, y + 28 + j * 22), (190, 190, 190), cv2.FILLED)
    y += 200

    # Simulated table
    cv2.rectangle(img, (70, y), (w - 70, y + 140), (255, 255, 255), cv2.FILLED)
    for r in range(4):
        cv2.line(img, (70, y + r * 35), (w - 70, y + r * 35), (200, 200, 200), 1)
    for c in range(4):
        cx = 70 + c * (w - 140) // 4
        cv2.line(img, (cx, y), (cx, y + 140), (200, 200, 200), 1)
    y += 180

    # More body text
    for i in range(4):
        line_w = np.random.randint(250, 650)
        cv2.rectangle(img, (70, y), (70 + line_w, y + 12), (200, 200, 200), cv2.FILLED)
        y += 22

    cv2.imwrite(path, img)
    return path


def _test():
    """Run standalone test with a synthetic image and sample OCR text."""
    print("=" * 60)
    print("  layout_visualizer.py  --  Standalone Test")
    print("=" * 60)

    test_dir = Path(__file__).parent / "outputs"
    test_dir.mkdir(parents=True, exist_ok=True)
    img_path = str(test_dir / "test_document.png")

    print("\n[1] Creating synthetic test image ...")
    _create_test_image(img_path)
    print(f"    Image: {img_path}")

    sample_ocr = """
Document Title: Understanding Neural Networks
# Introduction
Neural networks are a class of machine learning models.
They are inspired by biological neural networks in the brain.
This document provides an overview of key concepts.
# Architecture
A typical neural network consists of layers of neurons.
def forward(self, x):
    h = self.hidden(x)
    return self.output(h)
import torch.nn as nn
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
| Layer   | Units | Activation |
|---------|-------|------------|
| Input   | 784   | None       |
| Hidden  | 128   | ReLU       |
| Output  | 10    | Softmax    |
Fig. 1 Architecture diagram of the proposed model.
The results show significant improvement over baselines.
Training was performed on a single GPU for 50 epochs.
"""

    print("\n[2] Running visualize_layout() ...")
    annotated, summary, out_path = visualize_layout(img_path, sample_ocr)
    print(f"    Annotated image: {out_path}")
    print(f"    Summary JSON:    {Path(out_path).with_suffix('.json')}")
    print(f"    Total segments:  {summary.total_segments}")
    print(f"    Dominant type:   {summary.dominant_type}")
    print(f"    Element counts:  {summary.element_counts}")

    print("\n[3] Running generate_heatmap() ...")
    _, heat_path = generate_heatmap(img_path, sample_ocr)
    print(f"    Heatmap: {heat_path}")

    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    _test()
