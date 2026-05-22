#!/usr/bin/env python3
"""Downstream agent workflow for Doc2Prototype.

The agent pipeline consumes structured document JSON, plans the downstream
artifact, invokes the code/summary generator, and reviews whether the generated
artifact covers the extracted structure.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Optional

from code_generator import CodeGenerator


@dataclass
class AgentStep:
    agent: str
    action: str
    status: str
    summary: str


def _normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def _contains(haystack: str, needle: Any) -> bool:
    normalized_needle = _normalize(needle)
    return bool(normalized_needle) and normalized_needle in _normalize(haystack)


def _flowchart_node_covered(generated_code: str, node: dict[str, Any]) -> bool:
    if _contains(generated_code, node.get("id")) or _contains(generated_code, node.get("label")):
        return True

    label = str(node.get("label", ""))
    prefix = re.split(r"\s+-\s+|\s+--\s+|\s+:\s+", label, maxsplit=1)[0]
    if _contains(generated_code, prefix):
        return True

    label_tokens = [token for token in re.findall(r"[A-Za-z0-9_]+", label.lower()) if len(token) > 2]
    if not label_tokens:
        return False
    generated_tokens = set(re.findall(r"[A-Za-z0-9_]+", generated_code.lower()))
    overlap = sum(1 for token in label_tokens if token in generated_tokens)
    return overlap >= min(2, len(label_tokens))


def _task_goal(task: str, code_type: str) -> str:
    if task == "api_doc":
        return "Generate a FastAPI skeleton from extracted API endpoints."
    if task == "flowchart":
        return "Generate a Mermaid diagram from extracted flowchart nodes and edges."
    if task == "technical_doc":
        return "Generate a concise Markdown summary from extracted document sections."
    return f"Generate `{code_type}` from the structured document."


def _input_summary(structured_data: dict[str, Any], task: str) -> dict[str, Any]:
    if task == "api_doc":
        return {
            "document_type": structured_data.get("document_type", task),
            "endpoint_count": len(structured_data.get("endpoints", [])),
        }
    if task == "flowchart":
        return {
            "document_type": structured_data.get("document_type", task),
            "node_count": len(structured_data.get("nodes", [])),
            "edge_count": len(structured_data.get("edges", [])),
        }
    if task == "technical_doc":
        return {
            "document_type": structured_data.get("document_type", task),
            "section_count": len(structured_data.get("sections", [])),
            "key_point_count": len(structured_data.get("key_points", [])),
        }
    return {"document_type": structured_data.get("document_type", task)}


def _review_api_doc(structured_data: dict[str, Any], generated_code: str) -> tuple[str, list[str], list[str]]:
    findings: list[str] = []
    passed: list[str] = []
    endpoints = structured_data.get("endpoints", [])
    if not endpoints:
        findings.append("No API endpoints were extracted, so the generated API skeleton has no endpoint coverage to verify.")
        return "needs_attention", passed, findings

    missing: list[str] = []
    for endpoint in endpoints:
        method = endpoint.get("method", "")
        path = endpoint.get("path", "")
        if not _contains(generated_code, path):
            missing.append(f"{method} {path}".strip())
    if missing:
        findings.append("Generated artifact is missing extracted endpoints: " + ", ".join(missing))
    else:
        passed.append(f"All {len(endpoints)} extracted endpoints are represented in the generated artifact.")

    status = "pass" if not findings else "needs_attention"
    return status, passed, findings


def _review_flowchart(structured_data: dict[str, Any], generated_code: str) -> tuple[str, list[str], list[str]]:
    findings: list[str] = []
    passed: list[str] = []
    nodes = structured_data.get("nodes", [])
    edges = structured_data.get("edges", [])
    if not nodes:
        findings.append("No flowchart nodes were extracted, so the generated diagram has no node coverage to verify.")
        return "needs_attention", passed, findings

    missing_nodes = [
        node.get("label", node.get("id", ""))
        for node in nodes
        if not _flowchart_node_covered(generated_code, node)
    ]
    if missing_nodes:
        findings.append("Generated diagram is missing extracted nodes: " + ", ".join(map(str, missing_nodes)))
    else:
        passed.append(f"All {len(nodes)} extracted nodes are represented in the generated diagram.")

    if edges:
        passed.append(f"The structured input contains {len(edges)} directed edges for downstream diagram generation.")

    status = "pass" if not findings else "needs_attention"
    return status, passed, findings


def _review_technical_doc(structured_data: dict[str, Any], generated_code: str) -> tuple[str, list[str], list[str]]:
    findings: list[str] = []
    passed: list[str] = []
    sections = structured_data.get("sections", [])
    if not generated_code.strip():
        findings.append("Generated summary is empty.")
    else:
        passed.append("Generated summary is non-empty.")

    if sections:
        passed.append(f"The structured input contains {len(sections)} section(s) for summary generation.")
    else:
        findings.append("No document sections were extracted.")

    status = "pass" if not findings else "needs_attention"
    return status, passed, findings


def review_generated_artifact(
    structured_data: dict[str, Any],
    generated_code: str,
    task: str,
    code_type: str,
) -> dict[str, Any]:
    if task == "api_doc":
        status, passed, findings = _review_api_doc(structured_data, generated_code)
    elif task == "flowchart":
        status, passed, findings = _review_flowchart(structured_data, generated_code)
    elif task == "technical_doc":
        status, passed, findings = _review_technical_doc(structured_data, generated_code)
    else:
        findings = []
        passed = ["Generated artifact is non-empty."] if generated_code.strip() else []
        status = "pass" if passed else "needs_attention"
        if not passed:
            findings.append("Generated artifact is empty.")

    return {
        "agent": "ReviewAgent",
        "status": status,
        "code_type": code_type,
        "passed_checks": passed,
        "findings": findings,
    }


class DownstreamAgentPipeline:
    """Plan, generate, and review downstream artifacts from structured JSON."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "CPU",
        model_backend: str = "auto",
        max_new_tokens: int = 4096,
    ):
        self.generator = CodeGenerator(
            model_path=model_path,
            device=device,
            backend=model_backend,
            max_new_tokens=max_new_tokens,
        )
        self.model_path = model_path
        self.device = device
        self.model_backend = model_backend

    @property
    def backend(self) -> str:
        loaded_backend = getattr(self.generator, "loaded_backend", "template")
        if loaded_backend == "openvino":
            return "OpenVINO Coder model inside agent workflow"
        if loaded_backend == "hf":
            return "local HuggingFace Coder model inside agent workflow"
        return "deterministic agent workflow with template generator"

    def _build_plan(self, structured_data: dict[str, Any], task: str, code_type: str) -> dict[str, Any]:
        return {
            "agent": "PlannerAgent",
            "goal": _task_goal(task, code_type),
            "input_summary": _input_summary(structured_data, task),
            "output_type": code_type,
            "handoff": "Pass the structured JSON to the generator, then review generated artifact coverage.",
        }

    def run(
        self,
        structured_data: dict[str, Any],
        task: str,
        code_type: str,
    ) -> dict[str, Any]:
        start = time.perf_counter()
        steps: list[AgentStep] = []

        plan = self._build_plan(structured_data, task, code_type)
        steps.append(
            AgentStep(
                agent="PlannerAgent",
                action="derive_downstream_plan",
                status="completed",
                summary=f"Prepared `{code_type}` generation plan for `{task}`.",
            )
        )

        generated = self.generator.generate(structured_data, code_type=code_type)
        generation_backend = generated.get("backend", getattr(self.generator, "loaded_backend", "template"))
        backend_summary = {
            "openvino": "OpenVINO Coder model inside agent workflow",
            "hf": "local HuggingFace Coder model inside agent workflow",
            "template": "deterministic agent workflow with template generator",
            "template_fallback": "deterministic template fallback after model generation failure",
        }.get(generation_backend, self.backend)
        steps.append(
            AgentStep(
                agent="GeneratorAgent",
                action="generate_artifact",
                status="completed",
                summary=f"Generated `{code_type}` using {backend_summary}.",
            )
        )

        review = review_generated_artifact(structured_data, generated["code"], task, code_type)
        steps.append(
            AgentStep(
                agent="ReviewAgent",
                action="review_artifact_coverage",
                status=review["status"],
                summary="Reviewed generated artifact against structured extraction results.",
            )
        )

        trace = {
            "schema_version": "doc2prototype.agent_trace.v1",
            "workflow": "structured_output_to_downstream_agent",
            "backend": backend_summary,
            "requested_model_backend": self.model_backend,
            "loaded_model_backend": getattr(self.generator, "loaded_backend", "template"),
            "generation_backend": generation_backend,
            "device": self.device,
            "model_path": self.model_path or "",
            "plan": plan,
            "steps": [asdict(step) for step in steps],
            "review": review,
            "elapsed": time.perf_counter() - start,
        }

        return {
            "code": generated["code"],
            "code_type": generated["code_type"],
            "generation_time": generated["generation_time"],
            "agent_time": trace["elapsed"],
            "backend": backend_summary,
            "trace": trace,
        }


def build_agent_review_markdown(trace: dict[str, Any]) -> str:
    plan = trace.get("plan", {})
    review = trace.get("review", {})
    lines = [
        "# Downstream Agent Review",
        "",
        "## Workflow",
        "",
        f"- Workflow: `{trace.get('workflow', '')}`",
        f"- Backend: `{trace.get('backend', '')}`",
        f"- Device: `{trace.get('device', '')}`",
        f"- Goal: {plan.get('goal', '')}",
        "",
        "## Input Summary",
        "",
        "```json",
        json.dumps(plan.get("input_summary", {}), indent=2, ensure_ascii=False),
        "```",
        "",
        "## Agent Steps",
        "",
    ]
    for step in trace.get("steps", []):
        lines.append(f"- `{step.get('agent')}` / `{step.get('action')}` / `{step.get('status')}`: {step.get('summary')}")

    lines.extend(
        [
            "",
            "## Review Result",
            "",
            f"- Status: `{review.get('status', '')}`",
        ]
    )

    passed_checks = review.get("passed_checks", [])
    findings = review.get("findings", [])
    if passed_checks:
        lines.append("- Passed checks:")
        lines.extend(f"  - {item}" for item in passed_checks)
    if findings:
        lines.append("- Findings:")
        lines.extend(f"  - {item}" for item in findings)
    if not passed_checks and not findings:
        lines.append("- No checks were recorded.")

    return "\n".join(lines) + "\n"
