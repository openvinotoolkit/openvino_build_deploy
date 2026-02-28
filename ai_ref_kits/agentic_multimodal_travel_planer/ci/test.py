#!/usr/bin/env python3
"""Sanity checks for agentic multimodal travel planner.

Checks covered:
1) Live LLM sanity: OVMS endpoints and a real completion response.
2) MCP sanity: startup manager can launch mock MCP servers on configured ports.
3) Agent sanity: agent manager can launch mock agent runners on configured ports.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import yaml
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGENTS_CONFIG_PATH = PROJECT_ROOT / "config" / "agents_config.yaml"
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from beeai_framework.adapters.a2a.agents.agent import A2AAgent
    from beeai_framework.adapters.a2a.agents.events import A2AAgentUpdateEvent
    from beeai_framework.agents.requirement.events import (
        RequirementAgentFinalAnswerEvent,
    )
    from beeai_framework.memory import UnconstrainedMemory
except ImportError:
    A2AAgent = None  # type: ignore[assignment]
    A2AAgentUpdateEvent = None  # type: ignore[assignment]
    RequirementAgentFinalAnswerEvent = None  # type: ignore[assignment]
    UnconstrainedMemory = None  # type: ignore[assignment]

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore[assignment]

try:
    from utils.streaming_citation_parser import StreamingCitationParser
    from utils.util import extract_response_text
except ImportError:
    StreamingCitationParser = None  # type: ignore[assignment]
    extract_response_text = None  # type: ignore[assignment]


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
        if value > 0:
            return value
    except ValueError:
        pass
    return default


def _is_port_open(port: int, host: str = "localhost") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.4)
        return sock.connect_ex((host, port)) == 0


def _http_get_json(url: str, timeout: int = 20) -> dict:
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
            return json.loads(payload) if payload else {}
    except urllib.error.URLError as exc:
        raise RuntimeError(f"GET failed for {url}: {exc}") from exc


def _load_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


def _strip_model_provider_prefix(model_name: str) -> str:
    # BeeAI config can use "openai:<model_id>".
    if model_name.startswith("openai:"):
        return model_name.split(":", 1)[1]
    return model_name


def _force_localhost(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    _assert(
        parsed.scheme in {"http", "https"},
        f"Invalid URL scheme in config: {url}",
    )
    _assert(
        parsed.port is not None,
        f"URL must include explicit port in config: {url}",
    )
    netloc = f"localhost:{parsed.port}"
    return urllib.parse.urlunparse(
        (
            parsed.scheme,
            netloc,
            parsed.path,
            parsed.params,
            parsed.query,
            parsed.fragment,
        )
    )


def _resolve_llm_vlm_targets_from_config() -> tuple[str, str, str]:
    agents_cfg = _load_yaml(AGENTS_CONFIG_PATH)
    mcp_cfg = _load_yaml(PROJECT_ROOT / "config" / "mcp_config.yaml")

    travel_router = agents_cfg.get("travel_router", {})
    llm_cfg = travel_router.get("llm", {}) if isinstance(travel_router, dict) else {}
    image_mcp = mcp_cfg.get("image_mcp", {})

    _assert(
        isinstance(llm_cfg, dict) and llm_cfg.get("api_base"),
        "Missing required config: travel_router.llm.api_base in agents_config.yaml",
    )
    _assert(
        isinstance(llm_cfg, dict) and llm_cfg.get("model"),
        "Missing required config: travel_router.llm.model in agents_config.yaml",
    )
    _assert(
        isinstance(image_mcp, dict) and image_mcp.get("ovms_base_url"),
        "Missing required config: image_mcp.ovms_base_url in mcp_config.yaml",
    )

    llm_base = _force_localhost(str(llm_cfg["api_base"]).rstrip("/"))
    llm_model = _strip_model_provider_prefix(str(llm_cfg["model"]))
    vlm_base = _force_localhost(str(image_mcp["ovms_base_url"]).rstrip("/"))
    return llm_base, vlm_base, llm_model


def _pick_model_from_models_endpoint(models_payload: dict) -> str:
    data = models_payload.get("data")
    if isinstance(data, list) and data and isinstance(data[0], dict):
        model_id = data[0].get("id")
        if model_id:
            return str(model_id)
    raise RuntimeError(
        "No model id found in /v3/models response. "
        f"Payload={json.dumps(models_payload, ensure_ascii=True)}"
    )


def _ensure_v3_base(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v3"):
        return base
    return f"{base}/v3"


def _wait_for_models_payload(
    base_url: str, label: str, timeout_s: int = 180, interval_s: float = 2.0
) -> dict:
    deadline = time.time() + timeout_s
    last_payload: dict = {}
    while time.time() < deadline:
        payload = _http_get_json(f"{base_url}/models")
        last_payload = payload if isinstance(payload, dict) else {}
        data = last_payload.get("data")
        if isinstance(data, list) and data:
            return last_payload
        time.sleep(interval_s)

    raise RuntimeError(
        f"{label} models did not become ready within {timeout_s}s. "
        f"Last payload={json.dumps(last_payload, ensure_ascii=True)}"
    )


def _serialize_completion_for_logs(completion: object) -> str:
    try:
        if hasattr(completion, "model_dump"):
            payload = completion.model_dump()
        else:
            payload = str(completion)
        return json.dumps(payload, ensure_ascii=True)
    except Exception:
        return str(completion)


def check_live_llm_sanity() -> None:
    llm_base, vlm_base, _configured_llm_model = (
        _resolve_llm_vlm_targets_from_config()
    )
    llm_base = _ensure_v3_base(llm_base)
    vlm_base = _ensure_v3_base(vlm_base)

    llm_models = _wait_for_models_payload(llm_base, "LLM")
    vlm_models = _wait_for_models_payload(vlm_base, "VLM")
    llm_model = _pick_model_from_models_endpoint(llm_models)
    vlm_model = _pick_model_from_models_endpoint(vlm_models)

    # Test LLM chat completion
    print(f"Testing LLM chat completion at {llm_base}...")
    try:
        llm_client = OpenAI(base_url=llm_base, api_key="unused")
        llm_completion = llm_client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "hello"},
            ],
            max_tokens=100,
            extra_body={"top_k": 1},
            stream=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"OpenAI SDK chat completion failed for LLM {llm_base}: {exc}"
        ) from exc

    print(f"LLM response: {_serialize_completion_for_logs(llm_completion)}")
    llm_choices = getattr(llm_completion, "choices", [])
    _assert(llm_choices, "No LLM choices returned.")

    # Test VLM chat completion
    print(f"Testing VLM chat completion at {vlm_base}...")
    try:
        vlm_client = OpenAI(base_url=vlm_base, api_key="unused")
        vlm_completion = vlm_client.chat.completions.create(
            model=vlm_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "hello"},
            ],
            max_tokens=100,
            extra_body={"top_k": 1},
            stream=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"OpenAI SDK chat completion failed for VLM {vlm_base}: {exc}"
        ) from exc

    print(f"VLM response: {_serialize_completion_for_logs(vlm_completion)}")
    vlm_choices = getattr(vlm_completion, "choices", [])
    _assert(vlm_choices, "No VLM choices returned.")
    print("Live LLM and VLM sanity checks passed.")


def _wait_for_ports(ports: list[int], timeout_s: int = 120) -> None:
    deadline = time.time() + timeout_s
    pending = set(ports)
    while time.time() < deadline and pending:
        for port in list(pending):
            if _is_port_open(port):
                pending.remove(port)
        if pending:
            time.sleep(0.5)
    _assert(not pending, f"Timed out waiting for ports: {sorted(pending)}")


def _mcp_ports_from_config() -> list[int]:
    mcp_cfg_path = PROJECT_ROOT / "config" / "mcp_config.yaml"
    mcp_cfg = _load_yaml(mcp_cfg_path)
    return [
        int(section.get("mcp_port"))
        for section in mcp_cfg.values()
        if isinstance(section, dict) and section.get("mcp_port")
    ]


def _agent_ports_from_config() -> list[int]:
    agents_cfg = _load_yaml(AGENTS_CONFIG_PATH)
    return [
        int(section.get("port"))
        for section in agents_cfg.values()
        if (
            isinstance(section, dict)
            and section.get("enabled", True)
            and section.get("port")
        )
    ]


def _enabled_agents_from_config() -> list[tuple[str, dict]]:
    agents_cfg = _load_yaml(AGENTS_CONFIG_PATH)
    return [
        (name, cfg)
        for name, cfg in agents_cfg.items()
        if (
            isinstance(cfg, dict)
            and cfg.get("enabled", True)
            and cfg.get("port")
        )
    ]


def _agent_url_from_card(card_payload: dict) -> str:
    if isinstance(card_payload, dict):
        raw_url = card_payload.get("url")
        if isinstance(raw_url, str) and raw_url.strip():
            parsed = urllib.parse.urlparse(raw_url.strip())
            if parsed.scheme in {"http", "https"} and parsed.port is not None:
                return f"{parsed.scheme}://127.0.0.1:{parsed.port}"
    raise RuntimeError(
        "Agent card missing valid 'url' with explicit port. "
        f"Payload={json.dumps(card_payload, ensure_ascii=True)}"
    )


def check_mcp_services_up() -> None:
    ports = _mcp_ports_from_config()
    _assert(ports, "No MCP ports found in mcp_config.yaml")
    _wait_for_ports(ports, timeout_s=120)
    print(f"MCP ports are up: {ports}")


def _verify_ovms_reachable(timeout_s: int = 15) -> None:
    """Verify OVMS LLM is still responding (fail fast if container crashed)."""
    try:
        llm_base, _, _ = _resolve_llm_vlm_targets_from_config()
        llm_base = _ensure_v3_base(llm_base)
        _http_get_json(f"{llm_base}/models", timeout=timeout_s)
    except Exception as exc:
        raise RuntimeError(
            "OVMS (LLM) not responding; container may have crashed. "
            f"Check 'docker logs ovms-llm'. Error: {exc}"
        ) from exc


def check_agent_services_up() -> None:
    ports = _agent_ports_from_config()
    _assert(ports, "No enabled agent ports found in agents_config.yaml")
    _wait_for_ports(ports, timeout_s=120)
    print(f"Agent ports are up: {ports}")

    # Fail fast if OVMS has crashed (agents depend on it)
    if not os.environ.get("SKIP_OVMS_HEALTH_CHECK"):
        ovms_timeout = _int_env("OVMS_HEALTH_CHECK_TIMEOUT_SECONDS", 15)
        print("Checking OVMS (LLM) is reachable...", flush=True)
        _verify_ovms_reachable(timeout_s=ovms_timeout)
        print("OVMS reachable.", flush=True)

    # Brief warm-up so agents are fully ready (CI can be slow to serve first request)
    warmup_s = _int_env("AGENT_WARMUP_SECONDS", 10)
    if warmup_s > 0:
        print(f"Warming up agents for {warmup_s}s...", flush=True)
        time.sleep(warmup_s)

    query_timeout_s = _int_env("AGENT_QUERY_TIMEOUT_SECONDS", 120)
    query_retries = _int_env("AGENT_QUERY_RETRIES", 3)
    retry_sleep_s = _int_env("AGENT_QUERY_RETRY_SLEEP_SECONDS", 5)

    for agent_name, cfg in _enabled_agents_from_config():
        agent_port = int(cfg["port"])
        card_base_url = f"http://127.0.0.1:{agent_port}"
        card_url = f"{card_base_url}/.well-known/agent-card.json"
        card_payload = _http_get_json(card_url)
        print(f"{agent_name} agent-card: {json.dumps(card_payload, ensure_ascii=True)}")
        agent_url = _agent_url_from_card(card_payload)

        # Use "hello" to match the working agent test script
        query = "hello"

        print(f"Querying {agent_name} at {agent_url} with: '{query}'...", flush=True)
        last_error: RuntimeError | None = None
        response_text = ""
        t0 = time.monotonic()
        for attempt in range(1, query_retries + 1):
            try:
                response_text = _query_agent(
                    query=query,
                    agent_url=agent_url,
                    timeout_s=query_timeout_s,
                )
                break
            except RuntimeError as exc:
                last_error = exc
                elapsed = time.monotonic() - t0
                print(
                    f"{agent_name} query attempt {attempt}/{query_retries} failed after {elapsed:.1f}s: {exc}",
                    flush=True,
                )
                if attempt < query_retries:
                    time.sleep(retry_sleep_s)
        if not response_text or response_text == "[No response]":
            reason = (
                str(last_error) if last_error
                else "no response from agent (timeout or no events)"
            )
            raise RuntimeError(
                f"{agent_name} query failed after {query_retries} attempts: {reason}"
            ) from last_error
        elapsed = time.monotonic() - t0
        print(f"{agent_name} response ({elapsed:.1f}s): {response_text}")
    print("Agent endpoint sanity passed.")


def _timeout_msg(agent_url: str, timeout_s: int) -> str:
    return f"Timed out after {timeout_s}s waiting for agent response from {agent_url}"


def _append_chunk(text_chunks: list[str], parser: object, delta: object) -> None:
    """Append parsed or raw delta to text_chunks."""
    if parser:
        clean_text, _ = parser.process_chunk(delta)
        if clean_text:
            text_chunks.append(clean_text)
            return
    text_chunks.append(delta if isinstance(delta, str) else str(delta))


def _query_agent(query: str, agent_url: str, timeout_s: int = 60) -> str:
    """Send a message to an agent and return response text."""
    _assert(
        A2AAgent is not None and UnconstrainedMemory is not None,
        "Missing beeai_framework dependency for agent queries.",
    )
    text_chunks: list[str] = []
    last_text = ""
    last_processed_length = 0

    async def _run_query() -> str:
        nonlocal last_processed_length, last_text
        if load_dotenv:
            load_dotenv()
        response_ready = asyncio.Event()
        client = A2AAgent(url=agent_url, memory=UnconstrainedMemory())
        parser = StreamingCitationParser() if StreamingCitationParser else None

        async def capture_events(data: object, event: object) -> None:
            nonlocal last_processed_length, last_text
            event_name = getattr(event, "name", "unknown")

            if (
                event_name == "final_answer"
                and RequirementAgentFinalAnswerEvent
                and isinstance(data, RequirementAgentFinalAnswerEvent)
                and getattr(data, "delta", None)
            ):
                _append_chunk(text_chunks, parser, data.delta)
                response_ready.set()

            if A2AAgentUpdateEvent and isinstance(data, A2AAgentUpdateEvent):
                value = getattr(data, "value", None)
                if isinstance(value, tuple) and len(value) >= 2:
                    task = value[0]
                    current_text = ""
                    if getattr(task, "history", None):
                        last_msg = task.history[-1]
                        if getattr(last_msg, "parts", None):
                            for part in last_msg.parts:
                                root = getattr(part, "root", None)
                                if root is not None and hasattr(root, "text"):
                                    current_text += root.text or ""
                    if len(current_text) > last_processed_length:
                        delta = current_text[last_processed_length:]
                        last_processed_length = len(current_text)
                        last_text = current_text
                        _append_chunk(text_chunks, parser, delta)
                        response_ready.set()

        run_handle = (
            client.run(query)
            .on("update", capture_events)
            .on("final_answer", capture_events)
        )

        async def _await_response():
            return await run_handle

        response_task = asyncio.create_task(_await_response())
        ready_task = asyncio.create_task(response_ready.wait())

        done, pending = await asyncio.wait(
            {response_task, ready_task},
            return_when=asyncio.FIRST_COMPLETED,
            timeout=timeout_s,
        )
        timed_out = len(done) == 0

        if ready_task in done:
            text = "".join(text_chunks).strip()
            if text:
                response_task.cancel()
                return text

        if response_task in done:
            response = response_task.result()
            if parser:
                final_chunk = parser.finalize()
                if final_chunk:
                    text_chunks.append(final_chunk)

            text = ""
            if extract_response_text:
                text = extract_response_text(response)

            if not text and text_chunks:
                text = "".join(text_chunks).strip()

            if not text and last_text:
                text = last_text

            if text:
                return text

        for task in pending:
            task.cancel()

        if timed_out:
            raise RuntimeError(_timeout_msg(agent_url, timeout_s))
        return "[No response]"

    try:
        return asyncio.run(_run_query())
    except asyncio.TimeoutError as exc:
        raise RuntimeError(_timeout_msg(agent_url, timeout_s)) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Agent query failed for {agent_url}: {exc}"
        ) from exc


def check_overall_placeholder() -> None:
    # Placeholder step kept to preserve workflow stage order.
    print(
        "Overall check placeholder: travel_router validation handled in "
        "--check-agents."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sanity checks for travel planner kit"
    )
    parser.add_argument(
        "--check-ovms",
        action="store_true",
        help="Check live OVMS LLM/VLM endpoints",
    )
    parser.add_argument(
        "--check-mcp",
        action="store_true",
        help="Check MCP services are up from config ports",
    )
    parser.add_argument(
        "--check-agents",
        action="store_true",
        help="Check enabled agents are up and return a simple query response",
    )
    parser.add_argument(
        "--check-overall",
        action="store_true",
        help="Placeholder step for overall validation",
    )
    args = parser.parse_args()

    if args.check_ovms:
        check_live_llm_sanity()
        return
    if args.check_mcp:
        check_mcp_services_up()
        return
    if args.check_agents:
        check_agent_services_up()
        return
    if args.check_overall:
        check_overall_placeholder()
        return

    # Default behavior for local/manual usage: run the same staged checks.
    check_live_llm_sanity()
    check_mcp_services_up()
    check_agent_services_up()
    check_overall_placeholder()
    print("All sanity checks passed.")


if __name__ == "__main__":
    main()
