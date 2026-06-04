"""
表格感知切片器

策略：
  1. 解析 ParsedDocument 的每页 Markdown
  2. 预处理：把 HTML <table> 块转成 Markdown 管道表格（PaddleOCR-VL 默认输出 HTML）
  3. 识别 Markdown 表格块（连续 `| ... |` 行 + 至少一行分隔符 `|---|---|`）
  4. 表格切片：表头 + 单行 → 一个 chunk（保留列名上下文）
  5. 非表格切片：按二级标题 / 段落分组，单个 chunk 控制在 [MIN_CHUNK, MAX_CHUNK] 字符范围
     超长段落做二次切分，保留 OVERLAP 字符重叠
  6. 元数据：{doc_name, page, section_title, kind: "text"|"table"|"table_header"}
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from html.parser import HTMLParser
from typing import Iterable, List, Optional

from .doc_parser import ParsedDocument, ParsedPage

logger = logging.getLogger(__name__)

MIN_CHUNK = 200
MAX_CHUNK = 500
OVERLAP = 50

TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")
TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
HTML_TABLE_RE = re.compile(r"<table\b[^>]*>.*?</table>", re.DOTALL | re.IGNORECASE)


# ── HTML 表格 → Markdown 表格预处理 ──────────────────────────────────────────


class _HTMLTableToRows(HTMLParser):
    """把单个 <table>...</table> 解析为 rows: List[List[(text, is_header)]]"""

    def __init__(self):
        super().__init__()
        self.rows: List[List[tuple[str, bool]]] = []
        self._row: Optional[List[tuple[str, bool]]] = None
        self._cell: Optional[List[str]] = None
        self._cell_is_header = False

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []
            self._cell_is_header = tag == "th"
        elif tag == "br" and self._cell is not None:
            self._cell.append(" ")

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None
        elif tag in ("td", "th") and self._cell is not None and self._row is not None:
            text = "".join(self._cell).strip()
            text = text.replace("|", "\\|").replace("\n", " ").replace("\r", " ")
            text = re.sub(r"\s+", " ", text)
            self._row.append((text, self._cell_is_header))
            self._cell = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def _html_table_to_markdown(html: str) -> str:
    parser = _HTMLTableToRows()
    try:
        parser.feed(html)
        parser.close()
    except Exception as e:
        logger.warning("HTML 表格解析失败，保留原文: %s", e)
        return html

    rows = parser.rows
    if not rows:
        return html

    # 确定列数：取最大 cell 数
    n_cols = max(len(r) for r in rows)
    if n_cols == 0:
        return html

    def pad(cells: List[tuple[str, bool]]) -> List[str]:
        out = [c[0] for c in cells]
        while len(out) < n_cols:
            out.append("")
        return out[:n_cols]

    # 找表头：首个全 <th> 行；否则把首行当表头
    header_idx = 0
    for i, r in enumerate(rows):
        if r and all(c[1] for c in r):
            header_idx = i
            break

    header = pad(rows[header_idx])
    body_rows = [pad(r) for j, r in enumerate(rows) if j != header_idx]

    md_lines = ["| " + " | ".join(header) + " |"]
    md_lines.append("| " + " | ".join(["---"] * n_cols) + " |")
    for r in body_rows:
        md_lines.append("| " + " | ".join(r) + " |")
    return "\n".join(md_lines)


def _normalize_html_tables(md: str) -> str:
    """把 markdown 文本里的 <table> 块就地替换为 Markdown 管道表格，前后留空行"""

    def _sub(match: re.Match) -> str:
        return "\n\n" + _html_table_to_markdown(match.group(0)) + "\n\n"

    return HTML_TABLE_RE.sub(_sub, md)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"text": self.text, "metadata": self.metadata}


# ── 表格块识别 ────────────────────────────────────────────────────────────────


@dataclass
class TableBlock:
    header: str
    separator: str
    rows: List[str]


def _detect_table_blocks(lines: List[str]) -> List[tuple[int, int, TableBlock]]:
    """
    返回 [(start_idx, end_idx_exclusive, TableBlock)]，索引基于 lines。
    """
    out = []
    i = 0
    n = len(lines)
    while i < n - 1:
        if TABLE_LINE_RE.match(lines[i]) and TABLE_SEP_RE.match(lines[i + 1] or ""):
            header = lines[i]
            sep = lines[i + 1]
            j = i + 2
            rows = []
            while j < n and TABLE_LINE_RE.match(lines[j]):
                rows.append(lines[j])
                j += 1
            if rows:
                out.append((i, j, TableBlock(header=header, separator=sep, rows=rows)))
                i = j
                continue
        i += 1
    return out


# ── 切片：表格 ────────────────────────────────────────────────────────────────


def _table_to_chunks(
    block: TableBlock,
    base_meta: dict,
    caption: Optional[str] = None,
) -> List[Chunk]:
    """
    每行 + 表头 → 一个 chunk；同时附加一个表头总览 chunk 便于召回。

    caption（可选）：紧贴表格的上文标题/说明行（如 "Table 2. Memory Specifications"），
    会拼到每个 chunk 的开头，给检索一个稳定的语义锚——很多 PDF 的具体表格用通用列名
    （Specification / Description / Value），单看表头与查询匹配度低，加上 caption 后
    "Memory size | 80 GB" 这种行才更容易被 "HBM 显存容量" 召回。
    """
    chunks: List[Chunk] = []

    def _wrap(body: str) -> str:
        return f"{caption}\n\n{body}" if caption else body

    overview = "\n".join([block.header, block.separator])
    overview_meta = {**base_meta, "kind": "table_header", "row_count": len(block.rows)}
    if caption:
        overview_meta["table_caption"] = caption
    chunks.append(Chunk(text=_wrap(overview), metadata=overview_meta))

    for idx, row in enumerate(block.rows, start=1):
        text = "\n".join([block.header, block.separator, row])
        row_meta = {**base_meta, "kind": "table", "row_index": idx}
        if caption:
            row_meta["table_caption"] = caption
        chunks.append(Chunk(text=_wrap(text), metadata=row_meta))
    return chunks


# ── 切片：文字段 ──────────────────────────────────────────────────────────────


def _split_long_paragraph(text: str, max_chars: int = MAX_CHUNK, overlap: int = OVERLAP) -> List[str]:
    """超长段落按字符滑窗切分，保留 overlap 字符重叠"""
    if len(text) <= max_chars:
        return [text]
    pieces = []
    step = max_chars - overlap
    if step <= 0:
        step = max_chars
    i = 0
    while i < len(text):
        pieces.append(text[i : i + max_chars])
        if i + max_chars >= len(text):
            break
        i += step
    return pieces


def _flush_buffer(buf: List[str], base_meta: dict) -> List[Chunk]:
    """把累积的段落 buffer 输出成 chunks（合并到 ≥ MIN_CHUNK 后再切）"""
    if not buf:
        return []
    text = "\n\n".join(s for s in buf if s.strip())
    if not text.strip():
        return []
    chunks: List[Chunk] = []
    if len(text) <= MAX_CHUNK:
        chunks.append(Chunk(text=text, metadata={**base_meta, "kind": "text"}))
        return chunks
    for piece in _split_long_paragraph(text):
        chunks.append(Chunk(text=piece, metadata={**base_meta, "kind": "text"}))
    return chunks


# ── 主流程 ────────────────────────────────────────────────────────────────────


def chunk_page(page: ParsedPage, doc_name: str) -> List[Chunk]:
    md = page.markdown or ""
    if not md.strip():
        return []

    md = _normalize_html_tables(md)
    lines = md.split("\n")
    table_spans = _detect_table_blocks(lines)
    table_idx_set = set()
    for s, e, _ in table_spans:
        for k in range(s, e):
            table_idx_set.add(k)

    chunks: List[Chunk] = []
    section_title: Optional[str] = None
    buf: List[str] = []
    cur_paragraph: List[str] = []

    base_meta = {
        "doc_name": doc_name,
        "page": page.page_no,
    }

    def base_with_section() -> dict:
        m = dict(base_meta)
        if section_title:
            m["section_title"] = section_title
        return m

    i = 0
    n = len(lines)
    table_iter = iter(table_spans)
    next_table = next(table_iter, None)

    while i < n:
        # 命中表格起点 → flush 当前 buffer，输出表格 chunk
        if next_table and i == next_table[0]:
            if cur_paragraph:
                buf.append(" ".join(cur_paragraph).strip())
                cur_paragraph = []
            chunks.extend(_flush_buffer(buf, base_with_section()))
            buf = []

            # 找最近一条非空非表格行作 table caption（如 "Table 2. Memory Specifications"）
            caption: Optional[str] = None
            for j in range(next_table[0] - 1, -1, -1):
                if j in table_idx_set:
                    break
                s = lines[j].strip()
                if not s:
                    continue
                # 去掉 markdown 标题前缀
                m_h = HEADING_RE.match(s)
                caption = m_h.group(2).strip() if m_h else s
                break

            _, end, block = next_table
            chunks.extend(_table_to_chunks(block, base_with_section(), caption=caption))
            i = end
            next_table = next(table_iter, None)
            continue

        line = lines[i]
        stripped = line.strip()

        m = HEADING_RE.match(stripped)
        if m:
            # 标题：先 flush 之前的内容，再切换 section_title
            if cur_paragraph:
                buf.append(" ".join(cur_paragraph).strip())
                cur_paragraph = []
            chunks.extend(_flush_buffer(buf, base_with_section()))
            buf = []
            section_title = m.group(2).strip()
            # 标题本身也作为 chunk 的一部分（拼到下一个段落）
            buf.append(stripped)
            i += 1
            continue

        if stripped == "":
            if cur_paragraph:
                buf.append(" ".join(cur_paragraph).strip())
                cur_paragraph = []
                # 检查 buffer 是否已经够大
                cur_text_len = sum(len(s) for s in buf)
                if cur_text_len >= MIN_CHUNK:
                    chunks.extend(_flush_buffer(buf, base_with_section()))
                    buf = []
            i += 1
            continue

        cur_paragraph.append(stripped)
        i += 1

    if cur_paragraph:
        buf.append(" ".join(cur_paragraph).strip())
    chunks.extend(_flush_buffer(buf, base_with_section()))

    return chunks


def chunk_document(doc: ParsedDocument) -> List[Chunk]:
    out: List[Chunk] = []
    for page in doc.pages:
        out.extend(chunk_page(page, doc_name=doc.doc_name))
    return out


# ── 序列化 ────────────────────────────────────────────────────────────────────


def chunks_to_jsonl(chunks: Iterable[Chunk]) -> str:
    import json

    return "\n".join(json.dumps(c.to_dict(), ensure_ascii=False) for c in chunks)
