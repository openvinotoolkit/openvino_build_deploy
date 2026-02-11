#!/usr/bin/env python3
"""Sanity checks for agentic multimodal travel planner.

Checks covered:
1) LLM/docker sanity: launcher script wiring and --help validation.
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


def _is_port_open(port: int, host: str = "127.0.0.1") -> bool:
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


def check_llm_docker_sanity() -> None:
    script_path = PROJECT_ROOT / "download_and_run_models_linux.sh"
    _assert(script_path.exists(), f"Missing file: {script_path}")

    script_text = script_path.read_text(encoding="utf-8")
    _assert(
        'OVMS_IMAGE="openvino/model_server:latest"' in script_text,
        "Docker launcher does not define OVMS image.",
    )
    _assert(
        "docker run -d" in script_text and "docker pull" in script_text,
        "Docker launcher does not include expected docker commands.",
    )

    help_result = _run(["bash", str(script_path), "--help"], cwd=PROJECT_ROOT)
    _assert(
        help_result.returncode == 0,
        f"Docker launcher --help failed: {help_result.stderr.strip()}",
    )
    _assert(
        "--llm-model" in help_result.stdout and "--vlm-model" in help_result.stdout,
        "Docker launcher help text is missing expected options.",
    )

    print("LLM/docker sanity checks passed.")


def check_live_llm_sanity() -> None:
    llm_base = os.getenv("LLM_BASE_URL", "http://127.0.0.1:8001/v3").rstrip("/")
    vlm_base = os.getenv("VLM_BASE_URL", "http://127.0.0.1:8002/v3").rstrip("/")
    llm_model = os.getenv("LLM_MODEL_ID", "OpenVINO/Qwen3-8B-int4-ov")

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
    completion = _http_post_json(
        f"{llm_base}/chat/completions",
        {
            "model": llm_model,
            "messages": [{"role": "user", "content": "Reply with OK"}],
            "max_tokens": 8,
            "temperature": 0,
        },
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

        server = HTTPServer(("127.0.0.1", args.port), Handler)
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

        server = HTTPServer(("127.0.0.1", port), Handler)
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


def _query_travel_router(query: str, router_url: str) -> str:
    from beeai_framework.adapters.a2a.agents.agent import A2AAgent
    from beeai_framework.memory import UnconstrainedMemory
    from utils.util import extract_response_text

    async def _run_query() -> str:
        client = A2AAgent(url=router_url, memory=UnconstrainedMemory())
        response = await client.run(query)
        return extract_response_text(response)

    return asyncio.run(_run_query())


def check_live_router_query() -> None:
    mcp_cfg_path = PROJECT_ROOT / "config" / "mcp_config.yaml"
    agents_cfg_path = PROJECT_ROOT / "config" / "agents_config.yaml"
    mcp_cfg = yaml.safe_load(mcp_cfg_path.read_text(encoding="utf-8")) or {}
    agents_cfg = yaml.safe_load(agents_cfg_path.read_text(encoding="utf-8")) or {}

    mcp_ports = [
        int(section.get("mcp_port"))
        for section in mcp_cfg.values()
        if isinstance(section, dict) and section.get("mcp_port")
    ]
    agent_ports = [
        int(section.get("port"))
        for section in agents_cfg.values()
        if isinstance(section, dict) and section.get("enabled", True) and section.get("port")
    ]

    query = os.getenv("TRAVEL_ROUTER_TEST_QUERY", "Reply with exactly OK.")
    expected_token = os.getenv("TRAVEL_ROUTER_EXPECTED_TOKEN", "ok").lower()
    router_port = int(os.getenv("TRAVEL_ROUTER_PORT", "9996"))
    router_url = f"http://127.0.0.1:{router_port}"

    try:
        start_mcp = _run([sys.executable, "start_mcp_servers.py"], cwd=PROJECT_ROOT)
        _assert(start_mcp.returncode == 0, f"Failed to start MCP servers: {start_mcp.stderr.strip()}")
        if mcp_ports:
            _wait_for_ports(mcp_ports, timeout_s=120)

        start_agents = _run([sys.executable, "start_agents.py"], cwd=PROJECT_ROOT)
        _assert(start_agents.returncode == 0, f"Failed to start agents: {start_agents.stderr.strip()}")
        if agent_ports:
            _wait_for_ports(agent_ports, timeout_s=120)

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
    finally:
        _run([sys.executable, "start_agents.py", "--stop"], cwd=PROJECT_ROOT)
        _run(
            [sys.executable, "start_mcp_servers.py", "--stop", "--kill"],
            cwd=PROJECT_ROOT,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity checks for travel planner kit")
    parser.add_argument(
        "--skip-docker",
        action="store_true",
        help="Skip LLM/docker launcher checks",
    )
    args = parser.parse_args()

    if not args.skip_docker:
        check_llm_docker_sanity()
    if _env_truthy("LIVE_LLM_CHECK", "false"):
        check_live_llm_sanity()
    if _env_truthy("LIVE_ROUTER_QUERY", "false"):
        check_live_router_query()
    else:
        check_mcp_and_agents_sanity()
    print("All sanity checks passed.")


if __name__ == "__main__":
    main()
