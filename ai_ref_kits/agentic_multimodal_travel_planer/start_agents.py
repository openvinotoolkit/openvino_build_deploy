"""Agent startup script to create a travel planner multimodal agentic system.

This module manages the lifecycle of multiple agents, starting worker agents
before the supervisor agent to ensure proper dependency initialization.
"""
import socket
import subprocess
import sys
import time
from pathlib import Path

import yaml

CONFIG_PATH = Path("config/agents_config.yaml")
AGENT_RUNNER = Path("agents/agent_runner.py")
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


def is_port_in_use(port: int) -> bool:
    """Check if a port is currently in use.

    Args:
        port: The port number to check.

    Returns:
        True if the port is in use, False otherwise.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            return sock.connect_ex(("127.0.0.1", port)) == 0
    except OSError:
        return False


def kill_processes_on_port(port: int) -> None:
    """Kill any processes using the specified port.

    Args:
        port: The port number to clear.
    """
    try:
        result = subprocess.run(
            ["lsof", "-t", f"-i:{port}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.stdout.strip():
            pids = result.stdout.strip().split("\n")
            for pid in pids:
                if pid:
                    subprocess.run(["kill", "-9", pid], check=False)
                    print(f"Killed process {pid} on port {port}")
    except FileNotFoundError:
        # lsof not available; try pkill as fallback
        subprocess.run(["pkill", "-f", str(AGENT_RUNNER)], check=False)


def start_agent(name, config):
    """Start an individual agent process.

    Args:
        name: The name of the agent to start.
        config: Configuration dictionary for the agent.

    Returns:
        True if agent started successfully, False otherwise.
    """
    port = config.get("port")
    if not port:
        print(f"Agent '{name}' missing port, skipping.")
        return False

    # Kill any process using the port
    if is_port_in_use(port):
        kill_processes_on_port(port)
        time.sleep(0.5)

    log_file = LOG_DIR / f"{name}.log"

    try:
        # Start process with output redirected to log file
        with open(log_file, "w") as log:
            proc = subprocess.Popen(
                [sys.executable, str(AGENT_RUNNER), "--agent", name],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

        # Wait for agent to be ready by monitoring log output
        ready = False
        start_time = time.time()
        timeout_s = 30
        log_position = 0

        while time.time() - start_time < timeout_s:
            # Check if process died early
            if proc.poll() is not None:
                print(
                    f"Agent '{name}' exited early (code: {proc.returncode})"
                )
                return False

            # Read new log content
            if log_file.exists():
                with open(log_file, "r") as f:
                    f.seek(log_position)
                    new_content = f.read()
                    log_position = f.tell()

                    if new_content:
                        # Check for readiness indicators
                        if "uvicorn running on" in new_content.lower():
                            ready = True
                            break

            # Also verify port is in use
            if is_port_in_use(port):
                ready = True
                break

            time.sleep(0.3)

        if ready:
            time.sleep(0.5)
            print(f"Agent '{name}' started on port {port}")
            return True

        print(f"Warning: Agent '{name}' timed out waiting for readiness.")
        return False

    except Exception as e:
        print(f"Failed to start agent '{name}': {e}")
        return False


def stop_agents():
    subprocess.run(["pkill", "-f", str(AGENT_RUNNER)], check=False)
    print("All agents stopped.")


def main():
    """Main entry point for starting agents.

    Starts all worker agents first, then starts the supervisor agent to
    ensure proper dependency initialization.
    """
    if len(sys.argv) > 1 and sys.argv[1] == "--stop":
        stop_agents()
        return

    if not CONFIG_PATH.exists():
        print(f"Config file not found: {CONFIG_PATH}")
        sys.exit(1)

    with open(CONFIG_PATH, "r") as file:
        config = yaml.safe_load(file) or {}

    # Separate supervisor from other agents
    supervisor_name = "travel_router"
    supervisor_config = None
    other_agents = {}

    for name, agent_conf in config.items():
        if name == supervisor_name:
            supervisor_config = agent_conf
        else:
            other_agents[name] = agent_conf

    started = []
    failed = []

    # Start all non-supervisor agents first
    print("Starting worker agents...")
    for name, agent_conf in other_agents.items():
        if start_agent(name, agent_conf):
            started.append(name)
        else:
            failed.append(name)

    # Start supervisor last if it exists
    if supervisor_config:
        print(f"\nStarting supervisor agent ({supervisor_name})...")
        if start_agent(supervisor_name, supervisor_config):
            started.append(supervisor_name)
        else:
            failed.append(supervisor_name)

    if started:
        print(f"\nSuccessfully started agents: {', '.join(started)}")
    if failed:
        print(f"Failed to start agents: {', '.join(failed)}")
    print(f"\nLogs are in {LOG_DIR}/\n")


if __name__ == "__main__":
    main()
