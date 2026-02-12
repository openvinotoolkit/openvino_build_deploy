#!/usr/bin/env python3
"""Sanity checks for agentic multimodal travel planner.

Checks covered:
1) Live LLM sanity: OVMS endpoints and a real completion response.
2) MCP sanity: startup manager can launch mock MCP servers on configured ports.
3) Agent sanity: agent manager can launch mock agent runners on configured ports.
"""

from __future__ import annotations

import asyncio
import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        check=False,
    )


def _is_port_open(port: int, host: str = "localhost") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.4)
        return sock.connect_ex((host, port)) == 0


def _env_truthy(name: str, default: str = "false") -> bool:
    value = os.getenv(name, default).strip().lower()
    return value in {"1", "true", "yes", "on"}


def _http_get_json(url: str, timeout: int = 20) -> dict:
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
            return json.loads(payload) if payload else {}
    except urllib.error.URLError as exc:
        raise RuntimeError(f"GET failed for {url}: {exc}") from exc


def _http_post_json(url: str, body: dict, timeout: int = 60) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
            return json.loads(payload) if payload else {}
    except urllib.error.URLError as exc:
        raise RuntimeError(f"POST failed for {url}: {exc}") from exc


def _http_post_json_with_404_fallback(
    urls: list[str], body: dict, timeout: int = 60
) -> dict:
    last_404: RuntimeError | None = None
    for url in urls:
        try:
            return _http_post_json(url, body, timeout=timeout)
        except RuntimeError as exc:
            if "HTTP Error 404" in str(exc):
                last_404 = exc
                continue
            raise
    if last_404:
        raise last_404
    raise RuntimeError("No completion endpoint candidates were provided.")


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
    _assert(parsed.scheme in {"http", "https"}, f"Invalid URL scheme in config: {url}")
    _assert(parsed.port is not None, f"URL must include explicit port in config: {url}")
    netloc = f"localhost:{parsed.port}"
    return urllib.parse.urlunparse(
        (parsed.scheme, netloc, parsed.path, parsed.params, parsed.query, parsed.fragment)
    )


def _resolve_llm_vlm_targets_from_config() -> tuple[str, str, str]:
    agents_cfg = _load_yaml(PROJECT_ROOT / "config" / "agents_config.yaml")
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


def check_live_llm_sanity() -> None:
    llm_base, vlm_base, llm_model = _resolve_llm_vlm_targets_from_config()

    # Basic health endpoints.
    llm_models = _http_get_json(f"{llm_base}/models")
    vlm_models = _http_get_json(f"{vlm_base}/models")
    _assert(
        isinstance(llm_models.get("data"), list),
        "LLM /models endpoint did not return expected payload.",
    )
    _assert(
        isinstance(vlm_models.get("data"), list),
        "VLM /models endpoint did not return expected payload.",
    )

    # Real minimal completion call against LLM endpoint.
    completion_body = {
        "model": llm_model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "hello"},
        ],
        "stream": False,
    }
    completion_candidates = [f"{llm_base}/chat/completions"]
    if llm_base.endswith("/v3"):
        completion_candidates.append(f"{llm_base[:-3]}/v1/chat/completions")
    elif llm_base.endswith("/v1"):
        completion_candidates.append(f"{llm_base[:-3]}/v3/chat/completions")
    else:
        completion_candidates.append(f"{llm_base}/v1/chat/completions")
        completion_candidates.append(f"{llm_base}/v3/chat/completions")

    completion = _http_post_json_with_404_fallback(
        completion_candidates, completion_body
    )
    choices = completion.get("choices", [])
    _assert(isinstance(choices, list) and len(choices) > 0, "No LLM choices returned.")
    print("Live LLM sanity checks passed.")


def _write_mock_mcp_script(path: Path) -> None:
    content = textwrap.dedent(
        """
        #!/usr/bin/env python3
        import argparse
        from http.server import BaseHTTPRequestHandler, HTTPServer

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"ok")

            def log_message(self, format, *args):
                return

        parser = argparse.ArgumentParser()
        parser.add_argument("command", nargs="?")
        parser.add_argument("--protocol")
        parser.add_argument("--port", type=int, default=3000)
        args = parser.parse_args()

        server = HTTPServer(("localhost", args.port), Handler)
        print("MCP server started", flush=True)
        server.serve_forever()
        """
    ).strip()
    path.write_text(content + "\n", encoding="utf-8")
    path.chmod(0o755)


def _write_mock_agent_runner(path: Path) -> None:
    content = textwrap.dedent(
        """
        #!/usr/bin/env python3
        import argparse
        from http.server import BaseHTTPRequestHandler, HTTPServer
        from pathlib import Path
        import yaml

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/.well-known/agent-card.json":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(b'{"name":"mock-agent"}')
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"ok")

            def log_message(self, format, *args):
                return


        parser = argparse.ArgumentParser()
        parser.add_argument("--agent", required=True)
        args = parser.parse_args()

        cfg_file = Path(__file__).resolve().parent.parent / "config" / "agents_config.yaml"
        config = yaml.safe_load(cfg_file.read_text(encoding="utf-8")) or {}
        port = int(config[args.agent]["port"])

        server = HTTPServer(("localhost", port), Handler)
        print("Uvicorn running on mock server", flush=True)
        server.serve_forever()
        """
    ).strip()
    path.write_text(content + "\n", encoding="utf-8")
    path.chmod(0o755)


def check_mcp_and_agents_sanity() -> None:
    sys.path.insert(0, str(PROJECT_ROOT))

    # Imports are intentionally local so tests remain self-contained.
    import start_agents  # pylint: disable=import-error
    import start_mcp_servers  # pylint: disable=import-error

    with tempfile.TemporaryDirectory(prefix="agentic_ci_") as temp_dir:
        temp_root = Path(temp_dir)
        os.chdir(temp_root)

        (temp_root / "config").mkdir(parents=True, exist_ok=True)
        (temp_root / "mcp_tools").mkdir(parents=True, exist_ok=True)
        (temp_root / "agents").mkdir(parents=True, exist_ok=True)

        mcp_cfg = {
            "image_mcp": {
                "script": "mcp_tools/ai_builder_mcp_image.py",
                "mcp_port": 3330,
            },
            "hotel_finder": {
                "script": "mcp_tools/ai_builder_mcp_hotel_finder.py",
                "mcp_port": 3331,
            },
            "flight_finder": {
                "script": "mcp_tools/ai_builder_mcp_flights.py",
                "mcp_port": 3332,
            },
        }
        with open(temp_root / "config" / "mcp_config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(mcp_cfg, f, sort_keys=False)

        for section in mcp_cfg.values():
            _write_mock_mcp_script(temp_root / section["script"])

        agent_ports = {
            "travel_router": 9446,
            "flight_finder": 9448,
            "hotel_finder": 9449,
            "image_captioning": 9447,
        }
        agent_cfg = {
            name: {"port": port, "enabled": True}
            for name, port in agent_ports.items()
        }
        with open(
            temp_root / "config" / "agents_config.yaml", "w", encoding="utf-8"
        ) as f:
            yaml.safe_dump(agent_cfg, f, sort_keys=False)

        mock_runner = temp_root / "agents" / "agent_runner.py"
        _write_mock_agent_runner(mock_runner)

        # Patch module-level paths to temporary fixtures.
        start_mcp_servers.CONFIG_PATH = Path("config/mcp_config.yaml")
        start_mcp_servers.LOG_DIR = Path("logs")
        start_mcp_servers.LOG_DIR.mkdir(exist_ok=True)

        start_agents.CONFIG_PATH = Path("config/agents_config.yaml")
        start_agents.AGENT_RUNNER = Path("agents/agent_runner.py")
        start_agents.LOG_DIR = Path("logs")
        start_agents.LOG_DIR.mkdir(exist_ok=True)

        # Start MCP servers and verify ports.
        targets = start_mcp_servers.select_targets(start_mcp_servers.load_config())
        started_mcp = []
        for name, section in targets:
            ok = start_mcp_servers.start_mcp_server(name, section)
            _assert(ok, f"Failed to start mock MCP server: {name}")
            started_mcp.append(name)
        for port in [section["mcp_port"] for section in mcp_cfg.values()]:
            _assert(_is_port_open(port), f"Mock MCP port not open: {port}")
        print(f"MCP sanity checks passed for: {', '.join(started_mcp)}")

        # Start agents and verify ports.
        started_agents = []
        for name, section in agent_cfg.items():
            ok = start_agents.start_agent(name, section)
            _assert(ok, f"Failed to start mock agent: {name}")
            started_agents.append(name)
        for port in agent_ports.values():
            _assert(_is_port_open(port), f"Mock agent port not open: {port}")
        print(f"Agent sanity checks passed for: {', '.join(started_agents)}")

        # Cleanup started processes.
        start_mcp_servers.stop_mcp_servers(targets, kill_all=False)
        start_agents.stop_agents()


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
    agents_cfg_path = PROJECT_ROOT / "config" / "agents_config.yaml"
    agents_cfg = _load_yaml(agents_cfg_path)
    return [
        int(section.get("port"))
        for section in agents_cfg.values()
        if isinstance(section, dict) and section.get("enabled", True) and section.get("port")
    ]


def check_mcp_services_up() -> None:
    ports = _mcp_ports_from_config()
    _assert(ports, "No MCP ports found in mcp_config.yaml")
    _wait_for_ports(ports, timeout_s=120)
    print(f"MCP ports are up: {ports}")


def check_agent_services_up() -> None:
    ports = _agent_ports_from_config()
    _assert(ports, "No enabled agent ports found in agents_config.yaml")
    _wait_for_ports(ports, timeout_s=120)
    print(f"Agent ports are up: {ports}")


def _query_travel_router(query: str, router_url: str) -> str:
    from beeai_framework.adapters.a2a.agents.agent import A2AAgent
    from beeai_framework.memory import UnconstrainedMemory
    from utils.util import extract_response_text

    async def _run_query() -> str:
        client = A2AAgent(url=router_url, memory=UnconstrainedMemory())
        response = await client.run(query)
        return extract_response_text(response)

    return asyncio.run(_run_query())


def check_live_router_query_on_running_stack() -> None:
    agents_cfg = _load_yaml(PROJECT_ROOT / "config" / "agents_config.yaml")
    query = "Reply with exactly OK."
    expected_token = "ok"
    router_cfg = agents_cfg.get("travel_router", {})
    _assert(
        isinstance(router_cfg, dict) and router_cfg.get("port"),
        "Missing required config: travel_router.port in agents_config.yaml",
    )
    router_port = int(router_cfg["port"])
    router_url = f"http://localhost:{router_port}"
    check_mcp_services_up()
    check_agent_services_up()
    response_text = _query_travel_router(query=query, router_url=router_url)
    print(f"Travel router response: {response_text}")
    _assert(
        expected_token in response_text.lower(),
        (
            f"Travel router response did not include expected token "
            f"'{expected_token}'."
        ),
    )
    print("Live travel router query sanity passed.")


def check_live_router_query() -> None:
    mcp_ports = _mcp_ports_from_config()
    agent_ports = _agent_ports_from_config()

    try:
        start_mcp = _run([sys.executable, "start_mcp_servers.py"], cwd=PROJECT_ROOT)
        _assert(start_mcp.returncode == 0, f"Failed to start MCP servers: {start_mcp.stderr.strip()}")
        if mcp_ports:
            _wait_for_ports(mcp_ports, timeout_s=120)

        start_agents = _run([sys.executable, "start_agents.py"], cwd=PROJECT_ROOT)
        _assert(start_agents.returncode == 0, f"Failed to start agents: {start_agents.stderr.strip()}")
        if agent_ports:
            _wait_for_ports(agent_ports, timeout_s=120)

        check_live_router_query_on_running_stack()
    finally:
        _run([sys.executable, "start_agents.py", "--stop"], cwd=PROJECT_ROOT)
        _run(
            [sys.executable, "start_mcp_servers.py", "--stop", "--kill"],
            cwd=PROJECT_ROOT,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity checks for travel planner kit")
    parser.add_argument("--check-ovms", action="store_true", help="Check live OVMS LLM/VLM endpoints")
    parser.add_argument("--check-mcp", action="store_true", help="Check MCP services are up from config ports")
    parser.add_argument("--check-agents", action="store_true", help="Check enabled agents are up from config ports")
    parser.add_argument("--check-overall", action="store_true", help="Send a query to travel_router and validate response")
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
        check_live_router_query_on_running_stack()
        return

    check_live_llm_sanity()
    if _env_truthy("LIVE_ROUTER_QUERY", "false"):
        check_live_router_query()
    else:
        check_mcp_and_agents_sanity()
    print("All sanity checks passed.")


if __name__ == "__main__":
    main()
