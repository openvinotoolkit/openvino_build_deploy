"""
Simple configuration loader for agents.
"""

import os
import yaml
import requests
from pathlib import Path


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
    import socket
    try:
        # Extract host and port from api_base
        if '://' in api_base:
            host_port = api_base.split('://', 1)[1]
        else:
            host_port = api_base
        
        if ':' in host_port:
            host, port = host_port.split(':', 1)
            port = int(port.split('/')[0])  # Remove any path after port
        else:
            host = host_port.split('/')[0]
            port = 80
        
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
            description = f"Consult {agent_name} for specialized tasks via A2A protocol."
            try:
                agent_card = response.json()
                description = agent_card.get('description', description)
                print(f"✅ Found agent card for {agent_name}")
            except Exception:
                print(f"⚠️ Invalid agent card for {agent_name}, using default description")
            
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
            print(f"⚠️ Agent {agent_name} not reachable ({str(e)}), skipping HandoffTool creation")
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
    
    # Build the return config
    config = {
        'port': port,
        'llm_model': llm_model,
        'llm_temperature': llm_temperature,
        'api_base': api_base,
        'api_key': api_key,
        'name': agent_config['name'],
        'description': agent_config['description'],
        'prompt': prompts_config[f'{agent_name}_prompt'].strip(),
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
