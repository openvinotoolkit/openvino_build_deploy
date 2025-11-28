"""
Simple configuration loader for agents.
"""

import asyncio
import os
import shutil
import socket
import subprocess
import time
import yaml
from datetime import datetime
from pathlib import Path
from typing import Awaitable, Callable, Dict, Iterable
from urllib.parse import urlparse

import requests
from PIL import Image

def validate_llm_endpoint(api_base, timeout=5):
    """Validate if the LLM API endpoint is accessible"""
    try:
        # For OpenVINO Model Server, try the models endpoint
        models_url = f"{api_base.rstrip('/')}/models"
        response = requests.get(models_url, timeout=timeout)
        if response.status_code == 200:
            return True, "OpenVINO Model Server is accessible"
    except requests.exceptions.RequestException:
        pass
    
    try:
        # Try the base URL
        response = requests.get(api_base, timeout=timeout)
        if response.status_code in [200, 404, 405]:  # 405 Method Not Allowed is also OK
            return True, "LLM endpoint is accessible"
    except requests.exceptions.RequestException:
        pass
    
    try:
        # For OpenAI-compatible APIs, try the health endpoint
        health_url = f"{api_base.rstrip('/')}/health"
        response = requests.get(health_url, timeout=timeout)
        if response.status_code == 200:
            return True, "LLM endpoint is healthy"
    except requests.exceptions.RequestException:
        pass
    
    # Check if the server is at least responding on the port
    try:
        # Extract host and port from api_base using urlparse
        parsed = urlparse(
            api_base if '://' in api_base else f'http://{api_base}'
        )
        host = parsed.hostname or 'localhost'
        port = parsed.port or (443 if parsed.scheme == 'https' else 80)
        
        # Convert 0.0.0.0 to localhost for socket check
        if host == '0.0.0.0':
            host = 'localhost'
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        
        if result == 0:
            return True, f"Server is running on {host}:{port}"
    except Exception:
        pass
    
    return False, "LLM endpoint not accessible"


def _create_tool(tool_config, agent_config=None):
    """Create a tool instance based on configuration"""
    from beeai_framework.tools.think import ThinkTool
    from beeai_framework.tools.weather import OpenMeteoTool
    
    tool_map = {
        'ThinkTool': ThinkTool,
        'OpenMeteoTool': OpenMeteoTool,
        'FinalAnswerTool': None,  # FinalAnswerTool is built into RequirementAgent
    }
    
    tool_name = tool_config['name']
    
    # Handle Bridge Tower tools
    if tool_name in ['MultimodalSearchTool', 'ImageTextRetrievalTool', 'VideoContentSearchTool']:
        if tool_config.get('enabled', True) and agent_config:
            try:
                from utils.bridgetower_tools import create_bridgetower_tools
                bridgetower_tools = create_bridgetower_tools(agent_config)
                # Return the specific tool requested
                for tool in bridgetower_tools:
                    if tool.name == tool_name:
                        return tool
            except ImportError as e:
                print(f"⚠️ Bridge Tower tools not available: {e}")
        return None
    
    # Handle standard tools
    if tool_name in tool_map and tool_config.get('enabled', True):
        if tool_map[tool_name] is None:
            # FinalAnswerTool is built into RequirementAgent, return a marker
            return 'FinalAnswerTool'
        return tool_map[tool_name]()
    elif tool_name == 'HandoffTool' and tool_config.get('auto_discovered', False):
        # HandoffTools will be created dynamically in the supervisor agent
        return None
    return None


def _create_handoff_tools(supervised_agents_config):
    """Create HandoffTools for supervised agents"""
    import requests
    from beeai_framework.tools.handoff import HandoffTool
    from beeai_framework.adapters.a2a.agents.agent import A2AAgent
    from beeai_framework.memory import UnconstrainedMemory
    
    handoff_tools = []
    
    for agent_config in supervised_agents_config:
        agent_name = agent_config['name']
        agent_url = agent_config['url']
        
        # Override URL with environment variable if available
        port_env_var = agent_config.get('port_env_var')
        if port_env_var and port_env_var in os.environ:
            port = os.environ[port_env_var]
            base_url = agent_url.rsplit(':', 1)[0]  # Remove port
            agent_url = f"{base_url}:{port}"
        
        try:
            print(f"🔗 Testing connection to {agent_name} at {agent_url}")
            
            # Test connectivity first
            agent_card_url = f"{agent_url}/.well-known/agent-card.json"
            response = requests.get(agent_card_url, timeout=3)
            
            if response.status_code != 200:
                print(f"⚠️ Agent {agent_name} not available (HTTP {response.status_code}), skipping HandoffTool creation")
                continue
                
            # Get agent card for description
            base_description = f"Consult {agent_name} for specialized tasks via A2A protocol."
            try:
                agent_card = response.json()
                base_description = agent_card.get('description', base_description)
                print(f"✅ Found agent card for {agent_name}")
            except Exception:
                print(f"⚠️ Invalid agent card for {agent_name}, using default description")
            
            # Add format requirements to description based on agent type
            if agent_name == "flight_finder":
                description = (
                    f"{base_description} REQUIRES structured input with keys: "
                    "from, to, departure_date, class. Example: "
                    "{'from': 'Toronto', 'to': 'Rome', 'departure_date': "
                    "'2025-12-15', 'class': 'economy'}. "
                    "DO NOT use {'task': '...'} format."
                )
            elif agent_name == "hotel_finder":
                description = (
                    f"{base_description} REQUIRES structured input with keys: "
                    "destination, check_in_date, check_out_date, guests. "
                    "Example: {'destination': 'Paris', 'check_in_date': "
                    "'2025-12-20', 'check_out_date': '2025-12-25', 'guests': 2}. "
                    "DO NOT use {'task': '...'} format."
                )
            else:
                description = base_description
            
            # Create A2A agent connection
            a2a_agent = A2AAgent(url=agent_url, memory=UnconstrainedMemory())
            
            # Create HandoffTool
            handoff_tool = HandoffTool(
                a2a_agent,
                name=agent_name,
                description=description
            )
            
            handoff_tools.append(handoff_tool)
            print(f"✅ Created HandoffTool for {agent_name}")
            
        except requests.RequestException as e:
            print(
                f"⚠️ Agent {agent_name} not reachable ({str(e)}), "
                "skipping HandoffTool creation"
            )
        except Exception as e:
            print(f"❌ Failed to create HandoffTool for {agent_name}: {e}")
    
    if not handoff_tools:
        print("⚠️ No specialized agents available - router will use fallback knowledge")
        print("⚠️ Returning empty handoff tools list to force direct responses")
    
    return handoff_tools


def load_config(agent_name: str):
    """
    Load configuration for a specific agent from YAML files and set environment variables.
    
    Args:
        agent_name: Name of the agent (e.g., 'hotel_finder', 'flight_finder', 'video_search')
    
    Returns:
        dict: Configuration dictionary with all needed values
    """
    config_dir = Path(__file__).parent.parent / "config"
    
    # Load agent config
    with open(config_dir / "agents_config.yaml", 'r') as f:
        agents_config = yaml.safe_load(f)
    
    # Load prompts config
    with open(config_dir / "agents_prompts.yaml", 'r') as f:
        prompts_config = yaml.safe_load(f)
    
    # Get agent specific config
    agent_config = agents_config[agent_name]
    agent_llm_config = agent_config['llm']
    
    # Use config values directly (no environment variable overrides)
    port = agent_config['port']
    llm_model = agent_llm_config['model']
    llm_temperature = agent_llm_config['temperature']
    api_base = agent_llm_config['api_base']
    api_key = agent_llm_config['api_key']
    
    # Validate LLM endpoint before proceeding
    print(f"Validating LLM endpoint: {api_base}")
    is_valid, message = validate_llm_endpoint(api_base)
    
    if not is_valid:
        print(f"{message}")
        print(f"Please check if your LLM server is running on {api_base}")
        raise ConnectionError(f"LLM endpoint validation failed: {message}")
    else:
        print(f"{message}")
    
    # Set environment variables for the agent
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_API_BASE"] = api_base
    
    # Create tools from configuration
    tools = []
    tools_config = agent_config.get('tools', [])
    
    # Handle tools as list (standard format)
    if isinstance(tools_config, list):
        for tool_config in tools_config:
            tool = _create_tool(tool_config, agent_config)
            if tool:
                tools.append(tool)
    # Handle tools as dict (legacy MCP format) - skip for now, will be handled externally
    elif isinstance(tools_config, dict):
        print(f"Detected MCP tools configuration - tools will be provided externally")
        pass
    
    # If this is a supervisor agent, create HandoffTools for supervised agents
    if 'supervised_agents' in agent_config:
        handoff_tools = _create_handoff_tools(agent_config['supervised_agents'])
        tools.extend(handoff_tools)
    
    # Get prompt and inject today's date for flight_finder
    prompt = prompts_config[f'{agent_name}_prompt'].strip()
    if agent_name == 'flight_finder':
        today_date = datetime.now().strftime('%Y-%m-%d')
        prompt = prompt.replace('[TODAY\'S DATE]', today_date)
    
    # Build the return config
    config = {
        'port': port,
        'llm_model': llm_model,
        'llm_temperature': llm_temperature,
        'api_base': api_base,
        'api_key': api_key,
        'name': agent_config['name'],
        'description': agent_config['description'],
        'prompt': prompt,
        'middleware': agent_config['middleware'],
        'memory_size': agent_config['memory_size'],
        'tools': tools,
        'is_supervisor': 'supervised_agents' in agent_config,
        'mcp_config': agent_config.get('mcp_config'),  # Add MCP configuration
        'requirements': agent_config.get('requirements', [])
    }
    
    # Only add supervised_agents if it exists in the original config
    if 'supervised_agents' in agent_config:
        config['supervised_agents'] = agent_config['supervised_agents']
    
    return config


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
        # lsof not available; best effort skip
        pass


def run_async_in_thread(
    coro_factory: Callable[..., Awaitable], *args, **kwargs
):
    """Run an async callable inside a fresh event loop on a worker thread."""
    new_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(new_loop)
    try:
        coroutine = coro_factory(*args, **kwargs)
        return new_loop.run_until_complete(coroutine)
    finally:
        new_loop.close()


def _path_within_directory(path: Path, directory: Path) -> bool:
    """Safely check that path resides within directory."""
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        common = os.path.commonpath([str(path), str(directory)])
        return common == str(directory)


def save_uploaded_image(image_input, destination_dir, prefix="caption_image"):
    """Persist uploaded/cached image input into destination_dir."""
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)

    timestamp = int(time.time())
    saved_image_path = destination_dir / f"{prefix}_{timestamp}.jpg"

    if isinstance(image_input, str):
        source_path = Path(image_input)
        try:
            source_resolved = source_path.resolve()
            dest_resolved = destination_dir.resolve()
        except Exception as exc:
            raise ValueError(
                f"Error: Could not resolve path: {image_input} ({exc})"
            ) from exc

        if not _path_within_directory(source_resolved, dest_resolved):
            raise ValueError(f"Error: Unsafe file path: {image_input}")

        if not source_path.exists():
            raise FileNotFoundError(
                f"Error: Image file not found at {image_input}"
            )

        shutil.copy2(source_path, saved_image_path)
        return saved_image_path

    if hasattr(image_input, "shape"):
        try:
            pil_image = Image.fromarray(image_input)
            pil_image.save(saved_image_path)
            return saved_image_path
        except Exception as exc:
            raise RuntimeError(
                f"Error processing uploaded image: {exc}. "
                "Please ensure PIL (Pillow) is installed."
            ) from exc

    raise TypeError(
        f"Error: Unsupported image input type: {type(image_input)}"
    )


def extract_agent_handoffs_from_log(
    log_path, cache: Dict[str, Dict[str, Iterable]]
):
    """Parse agent handoff events from a log file using a cache."""
    log_path = Path(log_path)
    if not log_path.exists():
        return []

    cache_key = str(log_path)
    cache.setdefault(cache_key, {"position": 0, "seen_handoffs": set()})

    new_steps = []
    entry = cache[cache_key]

    try:
        file_size = log_path.stat().st_size
        last_position = entry["position"]
        seen_handoffs = entry["seen_handoffs"]

        if file_size > last_position:
            with open(
                log_path, "r", encoding="utf-8", errors="ignore"
            ) as handle:
                handle.seek(last_position)
                for line in handle:
                    line = line.strip()
                    if "--> 🔍 HandoffTool[" in line:
                        parts = line.split("HandoffTool[")[1].split("]")
                        agent_name = parts[0]
                        handoff_id = f"{agent_name}_start"
                        if handoff_id not in seen_handoffs:
                            new_steps.append(
                                f"🔄 Delegating to {agent_name}..."
                            )
                            seen_handoffs.add(handoff_id)
                    elif "<-- 🔍 HandoffTool[" in line:
                        parts = line.split("HandoffTool[")[1].split("]")
                        agent_name = parts[0]
                        handoff_id = f"{agent_name}_complete"
                        if handoff_id not in seen_handoffs:
                            new_steps.append(f"✅ {agent_name} completed")
                            seen_handoffs.add(handoff_id)

                entry["position"] = handle.tell()
    except Exception:
        pass

    return new_steps

