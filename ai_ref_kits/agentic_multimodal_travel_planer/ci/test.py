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


def _try_http_get_json(url: str, timeout: int = 20) -> dict | None:
    """GET JSON or return None on transient errors (connection refused, timeout)."""
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
            if not payload:
                return {}
            return json.loads(payload)
    except (
        urllib.error.URLError,
        TimeoutError,
        OSError,
        json.JSONDecodeError,
    ):
        return None


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
        isinstance(llm_cfg, dict),
        "Missing required config: travel_router.llm in agents_config.yaml",
    )
    _assert(
        isinstance(image_mcp, dict) and image_mcp.get("ovms_base_url"),
        "Missing required config: image_mcp.ovms_base_url in mcp_config.yaml",
    )

    # Env overrides for tests/CI: use small model / port without changing agents_config.yaml
    api_base_raw = os.environ.get("AGENT_LLM_API_BASE_OVERRIDE")
    if not api_base_raw and os.environ.get("AGENT_LLM_PORT_OVERRIDE"):
        api_base_raw = f"http://127.0.0.1:{os.environ.get('AGENT_LLM_PORT_OVERRIDE').strip()}/v3"
    if not api_base_raw and isinstance(llm_cfg, dict):
        api_base_raw = llm_cfg.get("api_base")
    model_raw = os.environ.get("AGENT_LLM_MODEL_OVERRIDE") or (llm_cfg.get("model") if isinstance(llm_cfg, dict) else None)
    _assert(api_base_raw, "Missing travel_router.llm.api_base or AGENT_LLM_API_BASE_OVERRIDE")
    _assert(model_raw, "Missing travel_router.llm.model or AGENT_LLM_MODEL_OVERRIDE")

    llm_base = _force_localhost(str(api_base_raw).rstrip("/"))
    llm_model = _strip_model_provider_prefix(str(model_raw))
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
    models_url = f"{base_url}/models"
    get_timeout = min(30, max(5, int(timeout_s // 6)))
    while time.time() < deadline:
        payload = _try_http_get_json(models_url, timeout=get_timeout)
        if isinstance(payload, dict):
            last_payload = payload
            data = last_payload.get("data")
            if isinstance(data, list) and data:
                return last_payload
        time.sleep(interval_s)

    raise RuntimeError(
        f"{label} models did not become ready within {timeout_s}s "
        f"(retried while connection refused or empty /v3/models). "
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
        if "chat model error" in response_text.lower():
            raise RuntimeError(
                f"{agent_name} returned 'Chat Model error' (LLM/OVMS request failed)"
            )
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


def _query_supervisor_multi_turn(
    agent_url: str,
    messages: list[str],
    timeout_s: int = 300,
) -> str:
    """Send a sequence of messages to the supervisor (same session); return last response."""
    _assert(
        A2AAgent is not None and UnconstrainedMemory is not None,
        "Missing beeai_framework dependency for agent queries.",
    )
    if load_dotenv:
        load_dotenv()
    memory = UnconstrainedMemory()
    client = A2AAgent(url=agent_url, memory=memory)
    parser = StreamingCitationParser() if StreamingCitationParser else None
    last_response = ""

    async def _run_one(query: str) -> str:
        nonlocal last_response
        text_chunks = []
        last_text = ""
        last_processed_length = 0
        response_ready = asyncio.Event()

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

        async def _await_run():
            return await run_handle

        response_task = asyncio.create_task(_await_run())
        ready_task = asyncio.create_task(response_ready.wait())
        wait_timeout = timeout_s if timeout_s > 0 else None
        done, pending = await asyncio.wait(
            {response_task, ready_task},
            return_when=asyncio.FIRST_COMPLETED,
            timeout=wait_timeout,
        )
        if response_task in done and not response_task.cancelled():
            try:
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
                if not text:
                    text = last_text
                if text:
                    return text
            except Exception:
                pass
        if ready_task in done:
            text = "".join(text_chunks).strip() or last_text
            if text:
                # Give the run more time to complete (streaming may never "finish")
                extra_s = min(120, timeout_s // 2) if timeout_s > 0 else None
                done2, pending2 = await asyncio.wait(
                    {response_task},
                    timeout=extra_s,
                )
                if response_task in done2 and not response_task.cancelled():
                    try:
                        response = response_task.result()
                        if parser:
                            final_chunk = parser.finalize()
                            if final_chunk:
                                text_chunks.append(final_chunk)
                        if extract_response_text:
                            t = extract_response_text(response)
                        else:
                            t = "".join(text_chunks).strip() or last_text
                        if t:
                            return t
                    except Exception:
                        pass
                for t in pending2:
                    t.cancel()
                return text
        for t in pending:
            t.cancel()
        raise RuntimeError(_timeout_msg(agent_url, timeout_s))

    async def _run_all() -> str:
        nonlocal last_response
        for msg in messages:
            last_response = await _run_one(msg)
        return last_response

    try:
        return asyncio.run(_run_all())
    except Exception as exc:
        raise RuntimeError(
            f"Supervisor multi-turn failed for {agent_url}: {exc}"
        ) from exc


def _print_response_preview(label: str, response: str, max_chars: int = 1200) -> None:
    """Print a truncated preview of the supervisor response for debugging."""
    text = (response or "").strip()
    if not text:
        print(f"  [{label}] (empty response)", flush=True)
        return
    if len(text) <= max_chars:
        print(f"  [{label}] response: {text}", flush=True)
    else:
        print(f"  [{label}] response ({len(text)} chars): {text[:max_chars]}...", flush=True)


def _contains_failure_fallback(text_lower: str) -> bool:
    """Detect common fallback/error responses that should fail E2E checks."""
    failure_markers = [
        "unable to",
        "cannot",
        "can't",
        "not available",
        "try again later",
        "contact support",
        "check your account balance",
        "service unavailable",
        "temporarily unavailable",
        "i'm sorry",
        "sorry",
    ]
    return any(marker in text_lower for marker in failure_markers)


def _requests_missing_information(text_lower: str) -> bool:
    """Detect when the supervisor asks for additional required fields."""
    markers = [
        "missing information",
        "please provide the missing",
        "i need",
        "provide the missing",
        "need all",
    ]
    return any(marker in text_lower for marker in markers)


def check_overall() -> None:
    """Run end-to-end flows through supervisor with explicit confirmation.

    This test sends each user prompt, then sends "yes" as a second turn to
    trigger the actual downstream agent + MCP execution path.
    """
    _assert(
        A2AAgent is not None and UnconstrainedMemory is not None,
        "Missing beeai_framework dependency for --check-overall.",
    )
    agents_cfg = _load_yaml(AGENTS_CONFIG_PATH)
    router_cfg = agents_cfg.get("travel_router", {})
    _assert(
        isinstance(router_cfg, dict) and router_cfg.get("port"),
        "travel_router not found in agents_config.yaml",
    )
    port = int(router_cfg["port"])
    agent_url = f"http://127.0.0.1:{port}"
    # Allow disabling timeout for slow CI environments by setting
    # AGENT_QUERY_TIMEOUT_SECONDS=0 (or a negative value).
    # Use a sensible positive default when unset or invalid to avoid waiting forever.
    DEFAULT_TIMEOUT_SECONDS = 300
    timeout_raw = os.getenv("AGENT_QUERY_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS)).strip()
    try:
        timeout_s = int(timeout_raw)
    except ValueError:
        timeout_s = DEFAULT_TIMEOUT_SECONDS

    # Flight Finder: first prompt, then either confirm ("yes") or provide
    # explicit details if the first turn asks for missing information.
    flight_prompt = "Give me flights from Milan to Berlin for March 1st to March 10th"
    print(f"Check overall (Flight Finder): {flight_prompt!r} -> second turn", flush=True)
    flight_first_response = _query_supervisor_multi_turn(
        agent_url, [flight_prompt], timeout_s=timeout_s
    )
    _print_response_preview("Flight Finder first response", flight_first_response)
    flight_first_lower = flight_first_response.lower()
    if _requests_missing_information(flight_first_lower):
        flight_second_message = (
            "from Milan to Berlin, departure date 2026-03-01, "
            "return date 2026-03-10, class economy"
        )
    else:
        flight_second_message = "yes"
    print(
        f"  [Flight Finder second message] {flight_second_message}",
        flush=True,
    )
    flight_response = _query_supervisor_multi_turn(
        agent_url, [flight_prompt, flight_second_message], timeout_s=timeout_s
    )
    flight_lower = flight_response.lower()
    _assert(
        "chat model error" not in flight_lower,
        f"Flight Finder returned 'Chat Model error' (LLM/OVMS request failed). Response: {flight_response[:500]!r}",
    )
    _assert(
        not _contains_failure_fallback(flight_lower),
        f"Flight Finder returned fallback/error response instead of offers. Response: {flight_response[:500]!r}",
    )
    _assert(
        ("missing information" not in flight_lower and "please provide the missing" not in flight_lower)
        and (
            "here are" in flight_lower
            or "option" in flight_lower
            or "$" in flight_response
            or "€" in flight_response
            or ("flight" in flight_lower and len(flight_response) > 80)
        ),
        f"Flight Finder did not return flight options. Response: {flight_response[:500]!r}",
    )
    print("Flight Finder flow OK.", flush=True)
    _print_response_preview("Flight Finder", flight_response)

    # Hotel Finder: first prompt, then either confirm ("yes") or provide
    # explicit details if the first turn asks for missing information.
    hotel_prompt = "Give me hotels in Milan for March 1st to March 10th for 2 guests"
    print(f"Check overall (Hotel Finder): {hotel_prompt!r} -> second turn", flush=True)
    hotel_first_response = _query_supervisor_multi_turn(
        agent_url, [hotel_prompt], timeout_s=timeout_s
    )
    _print_response_preview("Hotel Finder first response", hotel_first_response)
    hotel_first_lower = hotel_first_response.lower()
    if _requests_missing_information(hotel_first_lower):
        hotel_second_message = (
            "destination Milan, check-in date 2026-03-01, "
            "check-out date 2026-03-10, guests 2"
        )
    else:
        hotel_second_message = "yes"
    print(
        f"  [Hotel Finder second message] {hotel_second_message}",
        flush=True,
    )
    hotel_response = _query_supervisor_multi_turn(
        agent_url, [hotel_prompt, hotel_second_message], timeout_s=timeout_s
    )
    hotel_lower = hotel_response.lower()
    _assert(
        "chat model error" not in hotel_lower,
        f"Hotel Finder returned 'Chat Model error' (LLM/OVMS request failed). Response: {hotel_response[:500]!r}",
    )
    _assert(
        not _contains_failure_fallback(hotel_lower),
        f"Hotel Finder returned fallback/error response instead of offers. Response: {hotel_response[:500]!r}",
    )
    _assert(
        ("missing information" not in hotel_lower and "please provide the missing" not in hotel_lower)
        and (
            "here are" in hotel_lower
            or "option" in hotel_lower
            or "$" in hotel_response
            or "€" in hotel_response
            or ("hotel" in hotel_lower and len(hotel_response) > 80)
        ),
        f"Hotel Finder did not return hotel information. Response: {hotel_response[:500]!r}",
    )
    print("Hotel Finder flow OK.", flush=True)
    _print_response_preview("Hotel Finder", hotel_response)
    print("Check overall passed.", flush=True)


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
        help="E2E flows: Flight Finder and Hotel Finder via supervisor (prompt -> confirm yes -> expect results)",
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
        check_overall()
        return

    # Default behavior for local/manual usage: run the same staged checks.
    check_live_llm_sanity()
    check_mcp_services_up()
    check_agent_services_up()
    check_overall()
    print("All sanity checks passed.")


if __name__ == "__main__":
    main()
