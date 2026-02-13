"""MCP server startup script for the agentic multimodal travel planner.

This module manages MCP (Model Context Protocol) servers, providing
functionality to start, stop, and download server scripts from GitHub.
"""

import os
import subprocess  # nosec B404 - controlled argv lists, no shell=True usage
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import urlparse

import yaml
from dotenv import find_dotenv, load_dotenv, set_key

from utils.util import is_port_in_use, kill_processes_on_port

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
ALLOWED_DOWNLOAD_SCHEMES = {"https"}
ALLOWED_DOWNLOAD_HOSTS = {"raw.githubusercontent.com"}


def _validate_download_url(url: str) -> None:
    """Validate URL scheme and host before network download.

    Args:
        url: URL string to validate.

    Raises:
        ValueError: If URL scheme/host is not explicitly allowed.
    """
    parsed = urlparse(url)
    scheme = (parsed.scheme or "").lower()
    host = (parsed.hostname or "").lower()

    if scheme not in ALLOWED_DOWNLOAD_SCHEMES:
        raise ValueError(
            f"Unsupported URL scheme '{scheme}'. Allowed: "
            f"{sorted(ALLOWED_DOWNLOAD_SCHEMES)}"
        )

    if host not in ALLOWED_DOWNLOAD_HOSTS:
        raise ValueError(
            f"Unsupported download host '{host}'. Allowed: "
            f"{sorted(ALLOWED_DOWNLOAD_HOSTS)}"
        )


def check_and_set_api_key() -> bool:
    """Check if SERP_API_KEY is set, prompt user if missing.
    
    Returns:
        True if API key is available, False otherwise.
    """
    # Load existing .env file
    load_dotenv()
    
    api_key = os.getenv("SERP_API_KEY")
    
    if api_key:
        print("✓ SERP_API_KEY found in environment")
        return True
    
    # Check if we need API key for any servers that will be started
    # (This check happens later, but we can check here too)
    print("\n⚠️  SERP_API_KEY not found in environment")
    print("   The SERP API key is required for hotel and flight search functionality.")
    print("   You can get a free API key from: https://serpapi.com/users/sign_up")
    print()
    
    # Prompt user for API key
    try:
        user_input = input("Enter your SERP_API_KEY (or press Enter to skip): ").strip()
        
        if not user_input:
            print("⚠️  Skipping API key setup. Hotel and flight search will not work.")
            print("   You can set SERP_API_KEY later by:")
            print("   1. Adding it to a .env file: SERP_API_KEY=your_key_here")
            print("   2. Exporting it: export SERP_API_KEY=your_key_here")
            return False
        
        # Save to .env file
        env_path = find_dotenv()
        if not env_path:
            # Create .env file in current directory
            env_path = Path.cwd() / ".env"
            env_path.touch()
        
        set_key(env_path, "SERP_API_KEY", user_input)
        os.environ["SERP_API_KEY"] = user_input
        
        print(f"✓ API key saved to {env_path}")
        return True
        
    except (KeyboardInterrupt, EOFError):
        print("\n⚠️  API key setup cancelled")
        return False
    except Exception as e:
        print(f"✗ Error saving API key: {e}")
        print("   You can manually set SERP_API_KEY in your environment")
        return False


def download_script_if_missing(name: str, script_path: Path) -> bool:
    """Download script from GitHub if it doesn't exist locally.

    Args:
        name: The friendly name of the MCP server.
        script_path: Path where the script should be saved.

    Returns:
        True if download was successful, False otherwise.
    """
    script_name = script_path.name

    if script_name not in GITHUB_MCP_URLS:
        print(
            f"✗ Script '{script_name}' not found and no download URL available"
        )
        return False

    url = GITHUB_MCP_URLS[script_name]

    try:
        print(f"Downloading '{name}' from GitHub...")
        print(f"  URL: {url}")
        print(f"  Destination: {script_path}")
        _validate_download_url(url)

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
    """Start an individual MCP server process.

    Args:
        name: The name of the MCP server to start.
        conf: Configuration dictionary for the MCP server.

    Returns:
        True if server started successfully, False otherwise.
    """
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
        cmd += ["start", "--protocol", "sse"]
        if port:
            cmd += ["--port", str(port)]

    try:
        # Start process with output redirected to log file
        with open(log_file, "w") as log:
            proc = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
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
            time.sleep(0.5)
            status = (
                f"MCP '{name}' started" +
                (f" on port {port}" if port else "")
            )
            print(status)
            return True

        print(f"Warning: MCP '{name}' timed out waiting for readiness.")
        return False

    except Exception as e:
        print(f"Failed to start MCP '{name}': {e}")
        return False


def load_config() -> Dict:
    """Load MCP server configuration from YAML file.

    Returns:
        Configuration dictionary, or empty dict if file not found.
    """
    if not CONFIG_PATH.exists():
        print(f"Config file not found: {CONFIG_PATH}")
        return {}
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f) or {}


def select_targets(
    cfg: Dict, only: List[str] = None
) -> List[Tuple[str, Dict]]:
    """Select MCP server targets from configuration.

    Args:
        cfg: Configuration dictionary.
        only: Optional list of server names to filter by.

    Returns:
        List of tuples containing (server_name, config_dict).
    """
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
    """Stop running MCP server processes.

    Args:
        targets: List of (server_name, config_dict) tuples to stop.
        kill_all: If True, aggressively kill by known process patterns.
    """
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
    """Download MCP server scripts from AI Assistant Builder GitHub to their configured paths.

    Args:
        targets: List of (server_name, config_dict) tuples to download.
    """
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
    """Main entry point for managing MCP servers.

    Provides command-line interface for starting, stopping, and downloading
    MCP server scripts.
    """
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

    # Check for API key before starting servers (only if starting servers that need it)
    # Check if any target servers require SERP_API_KEY
    servers_needing_key = ["hotel_finder", "flight_finder", "ai_builder_mcp_hotel_finder", "ai_builder_mcp_flights"]
    needs_api_key = any(
        any(needle in name.lower() or needle in str(section.get("script", "")).lower() 
            for needle in servers_needing_key)
        for name, section in targets
    )
    
    if needs_api_key:
        check_and_set_api_key()
        print()  # Add blank line for readability

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
