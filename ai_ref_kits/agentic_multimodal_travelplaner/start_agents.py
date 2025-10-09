import subprocess
import yaml
import socket
from pathlib import Path
import sys

CONFIG_PATH = Path("config/agents_config.yaml")
AGENT_RUNNER = Path("agents/agent_runner.py")
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


def start_agent(name, config):
    port = config.get("port")
    if not port:
        print(f"Agent '{name}' missing port, skipping.")
        return

    # Kill any process using the port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        if sock.connect_ex(("127.0.0.1", port)) == 0:
            subprocess.run(
                ["lsof", "-t", f"-i:{port}"], capture_output=True, text=True
            ).stdout.splitlines()
            subprocess.run(["pkill", "-f", str(AGENT_RUNNER)], check=False)
            print(f"Port {port} in use, previous process killed.")

    log_file = LOG_DIR / f"{name}.log"

    proc = subprocess.Popen(
        ["python", str(AGENT_RUNNER), "--agent", name],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )

    for line in proc.stdout:
        with open(log_file, "a") as log:
            log.write(line)
            log.flush()
        if "Uvicorn running on" in line:
            print(f"Agent '{name}' started on port {port}")
            break


def stop_agents():
    subprocess.run(["pkill", "-f", str(AGENT_RUNNER)], check=False)
    print("All agents stopped.")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--stop":
        stop_agents()
        return

    if not CONFIG_PATH.exists():
        print(f"Config file not found: {CONFIG_PATH}")
        return

    with open(CONFIG_PATH, "r") as file:
        config = yaml.safe_load(file) or {}

    for name, agent_conf in config.items():
        start_agent(name, agent_conf)

    print("\nAll agents started.Logs are in /logs/.\n")


if __name__ == "__main__":
    main()



