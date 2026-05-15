"""
Doc2Prototype - Structure Extractor Module
Converts PaddleOCR-VL raw output into structured JSON representations.
"""

import json
import re
from typing import Any, Optional


SCHEMA_VERSION = "doc2prototype.mvp.v1"


# JSON Schema definitions for different document types
SCHEMAS = {
    "flowchart": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "nodes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "label": {"type": "string"},
                        "type": {"type": "string", "enum": ["start", "end", "process", "decision", "io", "subprocess"]},
                        "description": {"type": "string"},
                    },
                },
            },
            "edges": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "from": {"type": "string"},
                        "to": {"type": "string"},
                        "label": {"type": "string"},
                    },
                },
            },
        },
    },
    "api_doc": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "base_url": {"type": "string"},
            "endpoints": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "method": {"type": "string"},
                        "description": {"type": "string"},
                        "parameters": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "type": {"type": "string"},
                                    "required": {"type": "boolean"},
                                    "description": {"type": "string"},
                                },
                            },
                        },
                        "response": {"type": "object"},
                    },
                },
            },
        },
    },
    "ui_mockup": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "layout": {"type": "string"},
            "components": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "type": {"type": "string"},
                        "label": {"type": "string"},
                        "position": {"type": "object"},
                        "children": {"type": "array"},
                    },
                },
            },
        },
    },
    "table": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "headers": {"type": "array", "items": {"type": "string"}},
            "rows": {"type": "array", "items": {"type": "array"}},
            "metadata": {"type": "object"},
        },
    },
    "technical_doc": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "sections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "heading": {"type": "string"},
                        "level": {"type": "integer"},
                        "content": {"type": "string"},
                        "subsections": {"type": "array"},
                    },
                },
            },
            "tables": {"type": "array"},
            "code_blocks": {"type": "array"},
            "key_points": {"type": "array", "items": {"type": "string"}},
        },
    },
}


def extract_structure(
    raw_text: str,
    task: str,
    use_llm: bool = False,
    llm_model=None,
) -> dict[str, Any]:
    """
    Extract structured information from PaddleOCR-VL raw output.

    Args:
        raw_text: Raw text output from PaddleOCR-VL
        task: Document type for schema selection
        use_llm: Whether to use LLM for extraction (slower but more accurate)
        llm_model: Optional LLM model for structured extraction

    Returns:
        Structured dict following the appropriate schema
    """
    raw_text = re.sub(r"<\|LOC_\d+\|>", "", raw_text or "").strip()

    if use_llm and llm_model:
        return _extract_with_llm(raw_text, task, llm_model)

    # Rule-based extraction fallback
    return _extract_with_rules(raw_text, task)


def _extract_with_rules(raw_text: str, task: str) -> dict[str, Any]:
    """Rule-based structured extraction."""
    if task == "flowchart":
        return _extract_flowchart(raw_text)
    elif task == "api_doc":
        return _extract_api_doc(raw_text)
    elif task == "ui_mockup":
        return _extract_ui_mockup(raw_text)
    elif task == "table":
        return _extract_table(raw_text)
    else:
        return _extract_generic(raw_text, task)


def _extract_flowchart(text: str) -> dict:
    """Extract flowchart structure from text description."""
    nodes = []
    edges = []
    seen_labels = set()

    def has_word(value: str, word: str) -> bool:
        return re.search(rf'\b{re.escape(word)}\b', value, re.IGNORECASE) is not None

    def classify_node(label: str) -> str:
        low = label.lower()
        if has_word(low, "start") or "开始" in low:
            return "start"
        if has_word(low, "end") or "结束" in low:
            return "end"
        if "?" in label or has_word(low, "decide") or "判断" in low or has_word(low, "decision"):
            return "decision"
        if has_word(low, "input") or has_word(low, "output") or has_word(low, "io") or "输入" in low or "输出" in low:
            return "io"
        return "process"

    def normalize_ref(value: str) -> str:
        value = re.sub(r'^(?:Step|Node|阶段|步骤)\s*\d+\s*[:\-：]\s*', '', value.strip(), flags=re.IGNORECASE)
        value = re.sub(r'\s+', ' ', value)
        return value.strip().lower()

    # Pattern 1: "Step N: label" or "Node N: label"
    step_pattern = r'(?:Step|Node|阶段|步骤)\s*(\d+)\s*[:\-：]\s*(.+?)(?:\n|$)'
    for match in re.finditer(step_pattern, text, re.IGNORECASE):
        label = match.group(2).strip().rstrip('.')
        if label in seen_labels:
            continue
        seen_labels.add(label)
        node_id = f"node_{match.group(1)}"
        nodes.append({"id": node_id, "label": label, "type": classify_node(label)})

    # Pattern 2: Bullet/list items if no steps found
    if not nodes:
        bullet_pattern = r'[\-\*•]\s+(.+?)(?:\n|$)'
        for i, match in enumerate(re.finditer(bullet_pattern, text)):
            label = match.group(1).strip()
            if label in seen_labels:
                continue
            seen_labels.add(label)
            node_id = f"node_{i}"
            nodes.append({"id": node_id, "label": label, "type": classify_node(label)})

    def resolve_ref(value: str) -> str:
        ref = normalize_ref(value)
        for node in nodes:
            label_ref = normalize_ref(node["label"])
            if ref == label_ref or label_ref.startswith(ref) or ref.startswith(label_ref):
                return node["id"]
        return value.strip()

    # Edge patterns: "A -> B", "A ->|label| B", "A leads to B"
    for line in text.splitlines():
        line = line.strip().strip("-*•").strip()
        if not line:
            continue

        src = dst = label = ""
        arrow_match = re.match(r'(.+?)\s*(?:->|→|=>)\s*(?:\|([^|]+)\|\s*)?(.+)$', line)
        phrase_match = re.match(r'(.+?)\s+\b(?:leads to|goes to|connects to|then)\b\s+(.+)$', line, re.IGNORECASE)
        if arrow_match:
            src = arrow_match.group(1).strip()
            label = (arrow_match.group(2) or "").strip()
            dst = arrow_match.group(3).strip()
        elif phrase_match:
            src = phrase_match.group(1).strip()
            dst = phrase_match.group(2).strip()

        if src and dst and len(src) < 100 and len(dst) < 100:
            edge = {"from": resolve_ref(src), "to": resolve_ref(dst)}
            if label:
                edge["label"] = label
            edges.append(edge)

    if nodes and not edges:
        for src, dst in zip(nodes, nodes[1:]):
            edges.append({"from": src["id"], "to": dst["id"]})

    return {
        "schema_version": SCHEMA_VERSION,
        "document_type": "flowchart",
        "title": "Extracted Flowchart",
        "nodes": nodes,
        "edges": edges,
        "raw_analysis": text,
    }


def _extract_api_doc(text: str) -> dict:
    """Extract API documentation structure."""
    endpoints = []

    method_pattern = r'\b(GET|POST|PUT|DELETE|PATCH)\s+([/\w.\-{}:]+)(?:\s*(?:-|–|—|:)\s*([^\n]+))?'
    for match in re.finditer(method_pattern, text, re.IGNORECASE):
        method = match.group(1).upper()
        path = match.group(2).strip()
        description = (match.group(3) or "").strip()
        parameters = []
        for param in re.findall(r'\{([^}]+)\}', path):
            parameters.append({
                "name": param,
                "type": "string",
                "required": True,
                "location": "path",
                "description": f"Path parameter `{param}`.",
            })
        for param in re.findall(r':([A-Za-z_][A-Za-z0-9_]*)', path):
            if not any(existing["name"] == param for existing in parameters):
                parameters.append({
                    "name": param,
                    "type": "string",
                    "required": True,
                    "location": "path",
                    "description": f"Path parameter `{param}`.",
                })
        endpoints.append({
            "method": method,
            "path": path,
            "description": description,
            "parameters": parameters,
            "response": {
                "status": "success",
                "data": "object",
            },
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "document_type": "api_doc",
        "title": "Extracted API Documentation",
        "base_url": "",
        "endpoints": endpoints,
        "raw_analysis": text,
    }


def _extract_ui_mockup(text: str) -> dict:
    """Extract UI mockup structure."""
    components = []
    component_types = ["button", "input", "form", "table", "list", "card", "modal", "nav", "header", "footer", "sidebar"]

    for ctype in component_types:
        pattern = rf'{ctype}\s*[:\-]?\s*(.+?)(?:\n|$)'
        for match in re.finditer(pattern, text, re.IGNORECASE):
            components.append({
                "id": f"{ctype}_{len(components)}",
                "type": ctype,
                "label": match.group(1).strip(),
            })

    return {
        "schema_version": SCHEMA_VERSION,
        "document_type": "ui_mockup",
        "title": "Extracted UI Layout",
        "layout": "flex",
        "components": components,
        "raw_analysis": text,
    }


def _extract_table(text: str) -> dict:
    """Extract table structure."""
    lines = text.strip().split('\n')
    headers = []
    rows = []

    for line in lines:
        cells = [c.strip() for c in re.split(r'\s{2,}|\t|\|', line) if c.strip()]
        if cells:
            if not headers:
                headers = cells
            else:
                rows.append(cells)

    return {
        "schema_version": SCHEMA_VERSION,
        "document_type": "table",
        "title": "Extracted Table",
        "headers": headers,
        "rows": rows,
        "raw_analysis": text,
    }


def _extract_generic(text: str, task: str) -> dict:
    """Generic extraction for technical documents."""
    sections = []
    current_section = {"heading": "Main Content", "level": 1, "content": ""}

    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue

        # Detect headings
        if re.match(r'^#+\s', line) or re.match(r'^\d+\.\s', line):
            if current_section["content"]:
                sections.append(current_section)
            current_section = {
                "heading": line.lstrip('#').strip(),
                "level": 1,
                "content": "",
            }
        else:
            current_section["content"] += line + "\n"

    if current_section["content"]:
        sections.append(current_section)

    return {
        "schema_version": SCHEMA_VERSION,
        "document_type": task,
        "title": f"Extracted {task.replace('_', ' ').title()}",
        "sections": sections,
        "key_points": [],
        "raw_analysis": text,
    }


def _extract_with_llm(raw_text: str, task: str, llm_model) -> dict:
    """Use LLM for more accurate structured extraction."""
    schema = SCHEMAS.get(task, SCHEMAS["technical_doc"])

    prompt = f"""You are a document structure extraction expert.
Given the following raw text from a document parser, extract structured information.

Raw text:
{raw_text[:3000]}

Target schema:
{json.dumps(schema, indent=2)}

Return ONLY valid JSON matching the schema. No explanation."""

    response = llm_model.generate(prompt)

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # Fallback to rule-based
        return _extract_with_rules(raw_text, task)


def to_mermaid(structured: dict, doc_type: str = "flowchart") -> str:
    """Convert structured data to Mermaid diagram syntax."""
    if doc_type == "flowchart":
        lines = ["flowchart TD"]
        for node in structured.get("nodes", []):
            nid = node["id"]
            label = node["label"]
            ntype = node.get("type", "process")
            if ntype == "start":
                lines.append(f"    {nid}([{label}])")
            elif ntype == "end":
                lines.append(f"    {nid}([{label}])")
            elif ntype == "decision":
                lines.append(f"    {nid}{{{label}}}")
            else:
                lines.append(f"    {nid}[{label}]")

        for edge in structured.get("edges", []):
            label = edge.get("label", "")
            if label:
                lines.append(f"    {edge['from']} -->|{label}| {edge['to']}")
            else:
                lines.append(f"    {edge['from']} --> {edge['to']}")

        return "\n".join(lines)

    return f"%% No Mermaid conversion for {doc_type}"


def to_json_string(structured: dict, indent: int = 2) -> str:
    """Convert structured data to formatted JSON string."""
    return json.dumps(structured, indent=indent, ensure_ascii=False)
