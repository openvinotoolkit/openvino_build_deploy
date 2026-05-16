"""
Doc2Prototype - Code Generator Module
Uses downstream LLM to generate code prototypes from structured document data.
"""

import json
import re
import time
from typing import Optional


# Prompt templates for code generation
CODE_GEN_PROMPTS = {
    "api_skeleton": """You are a senior backend developer. Based on the following API documentation structure, generate a complete FastAPI backend skeleton.

API Documentation:
{structured_data}

Requirements:
1. Create all endpoint functions with proper type hints
2. Include Pydantic models for request/response
3. Add docstrings
4. Include proper error handling
5. Generate realistic mock data

Output ONLY the Python code, no explanations.""",

    "frontend_page": """You are a senior frontend developer. Based on the following UI mockup structure, generate a React component.

UI Mockup:
{structured_data}

Requirements:
1. Use functional components with hooks
2. Include TypeScript types
3. Use Tailwind CSS for styling
4. Make it responsive
5. Include realistic placeholder data

Output ONLY the React/TypeScript code, no explanations.""",

    "mermaid_diagram": """Based on the following flowchart structure, generate a clean Mermaid.js diagram definition.

Flowchart:
{structured_data}

Output ONLY the Mermaid syntax, starting with 'flowchart TD' or similar.""",

    "test_cases": """You are a QA engineer. Based on the following document structure, generate comprehensive test cases.

Document:
{structured_data}

Requirements:
1. Use pytest format
2. Include positive and negative test cases
3. Add parameterized tests where appropriate
4. Include setup/teardown
5. Add docstrings explaining each test

Output ONLY the Python test code, no explanations.""",

    "full_scaffold": """You are a senior full-stack architect. Based on the following document, generate a complete project scaffold.

Document:
{structured_data}

Generate a project structure with:
1. Backend (FastAPI): models, routes, services
2. Frontend (React): components, pages, types
3. Tests: unit and integration tests
4. Configuration: .env, docker-compose
5. README.md

Output the code organized by file paths (e.g., 'backend/main.py'), no explanations.""",

    "markdown_summary": """You are a technical writer. Based on the following extracted document structure, generate a concise Markdown technical summary.

Document:
{structured_data}

Requirements:
1. Summarize the document purpose
2. List the detected sections
3. Preserve important OCR details that may need manual review
4. Keep the output concise

Output ONLY Markdown.""",
}


class CodeGenerator:
    """Generate code prototypes from structured document data."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "CPU",
        use_openvino: bool = True,
        max_new_tokens: int = 4096,
    ):
        self.model_path = model_path
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._tokenizer = None

        if model_path and use_openvino:
            self._load_openvino_model(model_path, device)
        elif model_path:
            self._load_hf_model(model_path)

    def _load_openvino_model(self, model_path: str, device: str):
        """Load model using OpenVINO."""
        try:
            from optimum.intel import OVModelForCausalLM
            from transformers import AutoTokenizer

            print(f"[CodeGenerator] Loading OpenVINO model from {model_path}...")
            start = time.time()

            self._model = OVModelForCausalLM.from_pretrained(
                model_path,
                device=device,
            )
            self._tokenizer = AutoTokenizer.from_pretrained(model_path)

            elapsed = time.time() - start
            print(f"[CodeGenerator] Model loaded in {elapsed:.2f}s")
        except Exception as e:
            print(f"[CodeGenerator] Failed to load OpenVINO model: {e}")
            print("[CodeGenerator] Falling back to template-based generation")

    def _load_hf_model(self, model_path: str):
        """Load model using HuggingFace transformers."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            print(f"[CodeGenerator] Loading HuggingFace model from {model_path}...")
            self._model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                trust_remote_code=True,
            )
            self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        except Exception as e:
            print(f"[CodeGenerator] Failed to load HF model: {e}")

    @staticmethod
    def _safe_identifier(value: str, fallback: str = "endpoint") -> str:
        """Convert paths and labels into valid Python identifiers."""
        value = re.sub(r"\{([^}]+)\}", r"_\1", value)
        value = value.replace(":", "_")
        value = re.sub(r"[^0-9A-Za-z_]+", "_", value)
        value = re.sub(r"_+", "_", value).strip("_").lower()
        if not value:
            value = fallback
        if value[0].isdigit():
            value = f"{fallback}_{value}"
        return value

    @staticmethod
    def _safe_class_name(value: str, fallback: str = "GeneratedModel") -> str:
        parts = re.split(r"[^0-9A-Za-z]+", value)
        name = "".join(part[:1].upper() + part[1:] for part in parts if part)
        if not name:
            return fallback
        if name[0].isdigit():
            name = f"{fallback}{name}"
        return name

    @staticmethod
    def _path_parameters(path: str) -> list[str]:
        params = re.findall(r"\{([^}]+)\}", path)
        params.extend(re.findall(r":([A-Za-z_][A-Za-z0-9_]*)", path))
        ordered = []
        for param in params:
            if param not in ordered:
                ordered.append(param)
        return ordered

    def generate(
        self,
        structured_data: dict,
        code_type: str = "api_skeleton",
        custom_prompt: Optional[str] = None,
    ) -> dict:
        """
        Generate code from structured document data.

        Args:
            structured_data: Structured JSON from document parser
            code_type: One of 'api_skeleton', 'frontend_page', 'mermaid_diagram',
                       'test_cases', 'full_scaffold', 'markdown_summary'
            custom_prompt: Custom prompt override

        Returns:
            dict with keys: 'code', 'code_type', 'generation_time'
        """
        structured_str = json.dumps(structured_data, indent=2, ensure_ascii=False)

        if custom_prompt:
            prompt = custom_prompt.format(structured_data=structured_str)
        else:
            template = CODE_GEN_PROMPTS.get(code_type, CODE_GEN_PROMPTS["api_skeleton"])
            prompt = template.format(structured_data=structured_str)

        start = time.time()

        if self._model and self._tokenizer:
            code = self._generate_with_model(prompt)
        else:
            code = self._generate_template(structured_data, code_type)

        elapsed = time.time() - start

        return {
            "code": code,
            "code_type": code_type,
            "generation_time": elapsed,
            "structured_input": structured_data,
        }

    def _generate_with_model(self, prompt: str) -> str:
        """Generate code using loaded LLM."""
        try:
            inputs = self._tokenizer(prompt, return_tensors="pt")
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )
            response = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract only the generated part
            response = response[len(prompt):]
            return response.strip()
        except Exception as e:
            return f"// Generation error: {e}\n// Falling back to template"

    def _generate_template(self, structured_data: dict, code_type: str) -> str:
        """Template-based code generation when no LLM is available."""
        if code_type == "api_skeleton":
            return self._template_api_skeleton(structured_data)
        elif code_type == "frontend_page":
            return self._template_frontend(structured_data)
        elif code_type == "mermaid_diagram":
            return self._template_mermaid(structured_data)
        elif code_type == "test_cases":
            return self._template_test_cases(structured_data)
        elif code_type == "full_scaffold":
            return self._template_full_scaffold(structured_data)
        elif code_type == "markdown_summary":
            return self._template_markdown_summary(structured_data)
        else:
            return self._template_api_skeleton(structured_data)

    def _template_api_skeleton(self, data: dict) -> str:
        """Generate FastAPI skeleton from structured data."""
        endpoints = data.get("endpoints", [])

        code = '''"""Auto-generated FastAPI skeleton from Doc2Prototype."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn

app = FastAPI(title="{title}", version="1.0.0")


# ---- Models ----

'''.format(title=data.get("title", "API"))

        if not endpoints:
            endpoints = [
                {
                    "method": "GET",
                    "path": "/health",
                    "description": "Health check endpoint",
                    "parameters": [],
                }
            ]

        # Generate models from endpoints
        for ep in endpoints:
            method = ep.get("method", "GET").lower()
            path = ep.get("path", "/")
            func_name = self._safe_identifier(f"{method}_{path}")
            class_name = self._safe_class_name(f"{func_name}_response")

            code += f'''
class {class_name}(BaseModel):
    """Response model for {path}."""
    status: str = "success"
    data: Optional[dict] = None


'''

        code += "# ---- Endpoints ----\n\n"

        for ep in endpoints:
            method = ep.get("method", "GET").lower()
            path = ep.get("path", "/")
            func_name = self._safe_identifier(f"{method}_{path}")
            desc = ep.get("description", "")
            signature_parts = [f"{name}: str" for name in self._path_parameters(path)]
            if method in {"post", "put", "patch"}:
                signature_parts.append("payload: Optional[dict] = None")
            signature = ", ".join(signature_parts)

            code += f'''
@app.{method}("{path}")
async def {func_name}({signature}):
    """
    {desc or f"Endpoint: {method.upper()} {path}"}
    """
    # TODO: Implement business logic
    return {{"status": "success", "message": "Not implemented yet"}}

'''

        code += '''
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
        return code

    def _template_frontend(self, data: dict) -> str:
        """Generate React component from UI mockup data."""
        components = data.get("components", [])
        title = data.get("title", "Page")

        code = f'''// Auto-generated React component from Doc2Prototype
// UI: {title}

import React, {{ useState }} from 'react';

interface ComponentProps {{
  children?: React.ReactNode;
}}

'''

        for comp in components:
            comp_type = comp.get("type", "div")
            comp_label = comp.get("label", "Component")

            if comp_type == "button":
                code += f'''const {comp_label.replace(" ", "")}Button: React.FC = () => {{
  const [clicked, setClicked] = useState(false);

  return (
    <button
      className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
      onClick={{() => setClicked(!clicked)}}
    >
      {comp_label} {{clicked ? '✓' : ''}}
    </button>
  );
}};

'''
            elif comp_type in ("input", "form"):
                code += f'''const {comp_label.replace(" ", "")}Input: React.FC = () => {{
  const [value, setValue] = useState('');

  return (
    <div className="mb-4">
      <label className="block text-sm font-medium mb-1">{comp_label}</label>
      <input
        type="text"
        value={{value}}
        onChange={{(e) => setValue(e.target.value)}}
        className="w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
        placeholder="Enter {comp_label.lower()}..."
      />
    </div>
  );
}};

'''

        code += 'const ' + title.replace(" ", "") + '''Page: React.FC = () => {
  return (
    <div className="max-w-4xl mx-auto p-6">
      <h1 className="text-2xl font-bold mb-6">''' + title + '''</h1>
      <div className="space-y-4">
        {/* Components */}
      </div>
    </div>
  );
};

export default ''' + title.replace(" ", "") + '''Page;
'''
        return code

    def _template_mermaid(self, data: dict) -> str:
        """Generate Mermaid diagram."""
        nodes = data.get("nodes", [])
        edges = data.get("edges", [])

        lines = ["flowchart TD"]

        for node in nodes:
            nid = node.get("id", f"n{len(lines)}")
            label = node.get("label", "Step")
            ntype = node.get("type", "process")

            if ntype == "decision":
                lines.append(f"    {nid}{{{label}}}")
            elif ntype in ("start", "end"):
                lines.append(f"    {nid}([{label}])")
            else:
                lines.append(f"    {nid}[{label}]")

        for edge in edges:
            label = edge.get("label", "")
            if label:
                lines.append(f"    {edge['from']} -->|{label}| {edge['to']}")
            else:
                lines.append(f"    {edge['from']} --> {edge['to']}")

        return "\n".join(lines)

    def _template_test_cases(self, data: dict) -> str:
        """Generate pytest test cases."""
        endpoints = data.get("endpoints", [])

        code = '''"""Auto-generated test cases from Doc2Prototype."""

import pytest
from fastapi.testclient import TestClient

# Import your app
# from main import app
# client = TestClient(app)


class TestDocumentStructure:
    """Tests for document structure validation."""

    def test_document_has_title(self, structured_data):
        """Verify document has a title."""
        assert "title" in structured_data
        assert len(structured_data["title"]) > 0

    def test_document_sections_not_empty(self, structured_data):
        """Verify document has content."""
        sections = structured_data.get("sections", [])
        assert len(sections) > 0

'''

        for ep in endpoints:
            method = ep.get("method", "GET").lower()
            path = ep.get("path", "/")
            func_name = self._safe_identifier(f"{method}_{path}")
            class_name = self._safe_class_name(f"{func_name}_endpoint", fallback="GeneratedEndpoint")

            code += f'''
class Test{class_name}:
    """Tests for {method.upper()} {path}."""

    def test_{func_name}_success(self):
        """Test successful response."""
        # response = client.{method}("{path}")
        # assert response.status_code == 200
        pass

    def test_{func_name}_not_found(self):
        """Test 404 handling."""
        pass

'''

        return code

    def _template_full_scaffold(self, data: dict) -> str:
        """Generate full project scaffold."""
        parts = []

        parts.append("# Project Scaffold")
        parts.append(f"# Generated from: {data.get('title', 'Document')}")
        parts.append("#\n# Structure:")
        parts.append("# backend/")
        parts.append("#   main.py")
        parts.append("#   models.py")
        parts.append("#   routes/")
        parts.append("# frontend/")
        parts.append("#   src/")
        parts.append("#     App.tsx")
        parts.append("#     components/")
        parts.append("# tests/")
        parts.append("# docker-compose.yml")
        parts.append("# README.md")
        parts.append("")

        parts.append("=" * 60)
        parts.append("BACKEND - main.py")
        parts.append("=" * 60)
        parts.append(self._template_api_skeleton(data))

        parts.append("\n" + "=" * 60)
        parts.append("FRONTEND - App.tsx")
        parts.append("=" * 60)
        parts.append(self._template_frontend(data))

        parts.append("\n" + "=" * 60)
        parts.append("TESTS")
        parts.append("=" * 60)
        parts.append(self._template_test_cases(data))

        return "\n".join(parts)

    def _template_markdown_summary(self, data: dict) -> str:
        """Generate a concise Markdown summary from generic document structure."""
        title = data.get("title", "Extracted Technical Document")
        sections = data.get("sections", [])
        raw_analysis = data.get("raw_analysis", "")

        lines = [
            f"# {title}",
            "",
            "## Summary",
        ]

        if sections:
            first_content = sections[0].get("content", "").strip()
            summary = " ".join(first_content.split())[:420]
            lines.append(summary or "The document was parsed into structured sections for downstream review.")
        else:
            summary = " ".join(str(raw_analysis).split())[:420]
            lines.append(summary or "No readable content was extracted from the document.")

        lines.extend(["", "## Detected Sections"])
        if sections:
            for index, section in enumerate(sections, start=1):
                heading = section.get("heading") or f"Section {index}"
                content = " ".join(section.get("content", "").split())
                preview = content[:180] + ("..." if len(content) > 180 else "")
                lines.append(f"- **{heading}**: {preview or 'No body text detected.'}")
        else:
            lines.append("- No explicit sections detected.")

        raw_lines = [line.strip() for line in str(raw_analysis).splitlines() if line.strip()]
        if raw_lines:
            lines.extend(["", "## OCR Review Notes"])
            for line in raw_lines[:8]:
                lines.append(f"- {line[:220]}")

        return "\n".join(lines)


def download_code_model(
    model_id: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    cache_dir: str = "./_models",
) -> str:
    """Download a code generation model for local inference."""
    from pathlib import Path

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    try:
        from modelscope import snapshot_download
        print(f"Downloading {model_id} from ModelScope...")
        local_dir = snapshot_download(model_id, cache_dir=str(cache_path))
        return local_dir
    except Exception:
        pass

    try:
        from huggingface_hub import snapshot_download as hf_download
        print(f"Downloading {model_id} from HuggingFace...")
        local_dir = hf_download(
            repo_id=model_id,
            cache_dir=str(cache_path),
            local_dir=str(cache_path / model_id.replace("/", "__")),
        )
        return local_dir
    except Exception as e:
        raise RuntimeError(f"Failed to download model: {e}")
