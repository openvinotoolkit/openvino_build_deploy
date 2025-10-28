import subprocess
import yaml
import socket
from pathlib import Path
import sys
import time
import urllib.request
from typing import Dict, List, Tuple


CONFIG_PATH = Path("config/mcp_config.yaml")
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# GitHub URLs for known MCP servers
GITHUB_MCP_URLS = {
    "ai_builder_mcp_flights.py": (
        "https://raw.githubusercontent.com/intel/intel-ai-assistant-builder/"
        "main/mcp/mcp_servers/mcp_google_flight/server.py"
    ),
    "ai_builder_mcp_hotel_finder.py": (
        "https://raw.githubusercontent.com/intel/intel-ai-assistant-builder/"
        "main/mcp/mcp_servers/mcp_google_hotel/server.py"
    ),
}


def is_port_in_use(port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            return sock.connect_ex(("127.0.0.1", port)) == 0
    except OSError:
        return False


def kill_processes_on_port(port: int) -> None:
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
        # lsof not available; best effort skip
        pass


def download_script_if_missing(name: str, script_path: Path) -> bool:
    """Download script from GitHub if it doesn't exist locally."""
    script_name = script_path.name

    if script_name not in GITHUB_MCP_URLS:
        print(f"✗ Script '{script_name}' not found and no download URL available")
        return False

    url = GITHUB_MCP_URLS[script_name]

    try:
        print(f"Downloading '{name}' from GitHub...")
        print(f"  URL: {url}")
        print(f"  Destination: {script_path}")

        # Create parent directory if needed
        script_path.parent.mkdir(parents=True, exist_ok=True)

        with urllib.request.urlopen(url, timeout=30) as response:
            content = response.read()

        with open(script_path, "wb") as f:
            f.write(content)

        # Make executable
        script_path.chmod(0o755)

        print(f"✓ Downloaded '{name}' successfully")
        return True

    except urllib.error.URLError as e:
        print(f"✗ Failed to download '{name}': {e}")
        return False
    except Exception as e:
        print(f"✗ Error downloading '{name}': {e}")
        return False


def start_mcp_server(name: str, conf: Dict) -> bool:
    script = conf.get("script")
    if not script:
        print(f"MCP '{name}' missing script, skipping.")
        return False

    port = conf.get("mcp_port")

    # If a port is specified, ensure it is free
    if port:
        if is_port_in_use(port):
            kill_processes_on_port(port)
            time.sleep(0.5)

    # Resolve script path
    script_path = Path(script)
    if not script_path.exists():
        print(f"Script for '{name}' not found: {script_path}")
        print(f"Attempting to download from GitHub...")
        if not download_script_if_missing(name, script_path):
            return False
        print()

    log_file = LOG_DIR / f"mcp_{name}.log"

    # Build command
    cmd: List[str] = [sys.executable, str(script_path)]
    script_lower = script_path.name.lower()
    if "ai_builder_mcp" in script_lower:
        cmd += ["start"]
        if port:
            cmd += ["--port", str(port)]

    try:
        # Start process with output redirected to log file
        with open(log_file, "w") as log:
            proc = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,  # Detach from parent
            )

        # Wait for server to be ready by monitoring log output
        ready = False
        start_time = time.time()
        timeout_s = 30
        log_position = 0

        while time.time() - start_time < timeout_s:
            # Check if process died early
            if proc.poll() is not None:
                print(f"MCP '{name}' exited early (code: {proc.returncode})")
                return False
            
            # Read new log content
            if log_file.exists():
                with open(log_file, "r") as f:
                    f.seek(log_position)
                    new_content = f.read()
                    log_position = f.tell()
                    
                    if new_content:
                        # Check for readiness indicators in new content
                        content_lower = new_content.lower()
                        if any(s in content_lower for s in [
                            "uvicorn running on",
                            "server started successfully",
                            "mcp server started",
                            "starting simple video mcp server",
                        ]):
                            ready = True
                            break
            
            # Also verify port is in use if specified
            if port and is_port_in_use(port):
                ready = True
                break
            
            time.sleep(0.3)

        if ready:
            time.sleep(0.5)  # Small settle time
            status = f"MCP '{name}' started" + (f" on port {port}" if port else "")
            print(status)
            return True

        print(f"Warning: MCP '{name}' timed out waiting for readiness.")
        return False

    except Exception as e:
        print(f"Failed to start MCP '{name}': {e}")
        return False


def load_config() -> Dict:
    if not CONFIG_PATH.exists():
        print(f"Config file not found: {CONFIG_PATH}")
        return {}
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f) or {}


def select_targets(
    cfg: Dict, only: List[str] = None
) -> List[Tuple[str, Dict]]:
    items: List[Tuple[str, Dict]] = []
    for name, section in (cfg or {}).items():
        if not isinstance(section, dict):
            continue
        if "script" not in section:
            continue
        if only and name not in only:
            continue
        items.append((name, section))
    return items


def stop_mcp_servers(
    targets: List[Tuple[str, Dict]],
    kill_all: bool = False,
) -> None:
    killed = 0
    for name, section in targets:
        port = section.get("mcp_port")
        if port:
            try:
                result = subprocess.run(
                    ["lsof", "-t", f"-i:{port}"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                for pid in result.stdout.strip().split("\n"):
                    if pid:
                        subprocess.run(["kill", "-9", pid], check=False)
                        print(f"Killed process {pid} on port {port}")
                        killed += 1
            except FileNotFoundError:
                # Best effort if lsof missing
                pass

        # Also try to kill by script name as a fallback
        script = section.get("script")
        if script:
            try:
                subprocess.run(
                    ["pkill", "-f", str(Path(script).name)],
                    check=False,
                )
            except Exception:
                pass

    # Aggressive kill mode: try to kill by common MCP script patterns
    if kill_all:
        patterns = [
            "ai_builder_mcp_hotel_finder.py",
            "ai_builder_mcp_flights.py",
            "simple_video_mcp_server.py",
            "image_mcp_new.py",
            "image_mcp.py",
        ]
        for pat in patterns:
            try:
                subprocess.run(["pkill", "-f", pat], check=False)
            except Exception:
                pass

    print(
        f"Stopped {killed} process(es). All selected MCP servers stopped."
    )


def download_mcp_servers(targets: List[Tuple[str, Dict]]) -> None:
    """Download MCP server scripts from GitHub to their configured paths."""
    downloaded = 0
    skipped = 0

    for name, section in targets:
        script = section.get("script")
        if not script:
            continue

        script_path = Path(script)

        # Try to download
        if download_script_if_missing(name, script_path):
            downloaded += 1
        else:
            skipped += 1

    print(
        f"\nDownload complete: {downloaded} downloaded, {skipped} skipped"
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Start or stop MCP servers defined in config/mcp_config.yaml"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Stop the selected servers instead of starting them",
    )
    parser.add_argument(
        "--kill",
        action="store_true",
        help=(
            "Aggressive mode when used with --stop: kill by ports and known"
            " MCP process patterns"
        ),
    )
    parser.add_argument(
        "--only",
        nargs="*",
        help=(
            "Names of MCP servers to operate on (default: all with a"
            " script entry)"
        ),
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help=(
            "Download MCP server scripts from GitHub to paths specified"
            " in config"
        ),
    )

    args = parser.parse_args()

    cfg = load_config()
    if not cfg:
        sys.exit(1)

    targets = select_targets(cfg, args.only)
    if not targets:
        print("No MCP servers matched the selection.")
        sys.exit(1)

    if args.download:
        download_mcp_servers(targets)
        return

    if args.stop:
        stop_mcp_servers(targets, kill_all=args.kill)
        return

    started: List[str] = []
    failed: List[str] = []
    for name, section in targets:
        if start_mcp_server(name, section):
            started.append(name)
        else:
            failed.append(name)

    if started:
        print(f"\nSuccessfully started MCP servers: {', '.join(started)}")
    if failed:
        print(f"Failed to start MCP servers: {', '.join(failed)}")
    print(f"\nLogs are in {LOG_DIR}/\n")


if __name__ == "__main__":
    main()
