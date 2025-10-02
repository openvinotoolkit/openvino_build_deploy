#!/usr/bin/env python3
"""
Generic Agent Runner - Modular script to run any A2A agent based on YAML config
It reads configuration from config/agents_config.yaml and prompts from config/agent_prompts.yaml

This script can run:
- Regular agents with MCP tools
- Supervisor agents that coordinate other agents via HandoffTools
- All agents can be run in daemon mode for background operation
"""
# Standard library imports
import asyncio
import os
import sys
import yaml
from pathlib import Path
import argparse
from contextlib import AsyncExitStack
import threading
import requests

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

#Local imports
from utils.config import load_config

# BeeAI Framework imports
from beeai_framework.agents.experimental import RequirementAgent
from beeai_framework.agents.requirement.requirements.conditional import ConditionalRequirement
from beeai_framework.adapters.a2a.serve.server import A2AServer, A2AServerConfig
from beeai_framework.adapters.a2a.agents.agent import A2AAgent
from beeai_framework.backend import ChatModel, ChatModelParameters
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.serve.utils import LRUMemoryManager
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware
from beeai_framework.tools import Tool
from beeai_framework.tools.mcp import MCPTool
from beeai_framework.tools.handoff import HandoffTool

# MCP client imports
from mcp.client.sse import sse_client
from mcp.client.session import ClientSession

# =============================================================================
# MCP TOOLS MANAGEMENT
# =============================================================================

async def setup_mcp_tools(config):
    """Setup MCP tools from multiple MCP servers with persistent connections"""
    mcp_config = config.get('mcp_config')

    if not mcp_config:
        return None
    
    all_mcp_tools = []
    
    if 'url' in mcp_config:
        # Legacy single server format
        servers = [{'name': 'default', 'url': mcp_config['url'], 'enabled': True}]
    elif 'servers' in mcp_config:
        # New multi-server format
        servers = mcp_config['servers']
    else:
        print("No MCP servers configured - skipping MCP connections")
        return None
    
    # Connect to each enabled MCP server with persistent connections
    for server_config in servers:
        if not server_config.get('enabled', True):
            print(f"Skipping disabled MCP server: {server_config['name']}")
            continue
            
        server_name = server_config['name']
        server_url = server_config['url'] + "/sse"
        server_desc = server_config.get('description', '')
        
        print(f"Attempting to connect to MCP server '{server_name}' at {server_url}...")
        print(f"Server description: {server_desc}")
        
        try:
            print(f"Establishing SSE connection to {server_url}...")
            # Use correct MCP client approach
            async with sse_client(server_url) as (read_stream, write_stream):
                print(f"SSE connection established to '{server_name}'")
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    print(f"MCP session initialized successfully")
                    server_tools = await MCPTool.from_client(session)
            
            print(f"Successfully connected to '{server_name}' ({server_desc})")
            print(f"Found {len(server_tools)} tools:")
            for tool in server_tools:
                print(f"  - {tool.name}: {tool.description}")
            
            all_mcp_tools.extend(server_tools)
            print(f"Total tools now: {len(all_mcp_tools)}")
            
        except Exception as e:
            print(f"Failed to connect to MCP server '{server_name}': {e}")
            print(f"Error details: {type(e).__name__}: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            continue
    
    if all_mcp_tools:
        print(f"\nTotal MCP tools loaded: {len(all_mcp_tools)}")
        return all_mcp_tools
    else:
        print("No MCP tools were successfully loaded")
        return None

# =============================================================================
# SUPERVISOR AGENT MANAGEMENT
# =============================================================================

def create_supervisor_agent(config, available_agents=None, agent_cards=None):
    """
    Create a SUPERVISOR agent that coordinates other agents via HandoffTools
    
    This function:
    1. Discovers supervised agents (hotel_finder, flight_finder, etc.)
    2. Creates HandoffTools for each discovered agent
    3. Creates a supervisor agent with these coordination tools
    """
    
    print(f"Creating SUPERVISOR agent: {config.get('name', 'Unnamed')}")
    
    # Discover supervised agents if not provided
    if available_agents is None or agent_cards is None:
        available_agents, agent_cards = discover_supervised_agents(config)
    
    # Create supervisor tools (HandoffTools)
    supervisor_tools = []
    for agent_name in available_agents:
        agent_description = agent_cards.get(agent_name, {}).get("description", agent_name)
        supervisor_tools.append(
            HandoffTool(
                name=agent_name,
                description=agent_description,
                target=available_agents[agent_name].run
            )
        )
        print(f"   HandoffTool created for: {agent_name}")
    
    print(f" Created {len(supervisor_tools)} supervisor coordination tools")
    
    # =============================================================================
    # CREATE SUPERVISOR AGENT
    # =============================================================================
    print("Creating supervisor agent with coordination tools...")
    return create_agent(config, supervisor_tools)

def discover_supervised_agents(config):
    """Discover A2A agents that the supervisor can delegate to"""
    available_agents = {}
    agent_cards = {}
    
    if 'supervised_agents' not in config:
        return available_agents, agent_cards
    
    for agent_config in config['supervised_agents']:
        agent_name = agent_config['name']
        agent_url = agent_config['url']
        
        # Override URL with environment variable if available
        port_env_var = agent_config.get('port_env_var')
        if port_env_var and port_env_var in os.environ:
            port = os.environ[port_env_var]
            base_url = agent_url.rsplit(':', 1)[0]  # Remove port
            agent_url = f"{base_url}:{port}"
        
        try:
            print(f"Discovering A2A agent {agent_name} at {agent_url}...")
            
            # Try to get the Agent Card
            agent_card_url = f"{agent_url}/.well-known/agent-card.json"
            response = requests.get(agent_card_url, timeout=5)
            
            if response.status_code == 200:
                agent_card = response.json()
                agent_cards[agent_name] = agent_card
                
                # Create A2A client for this agent
                available_agents[agent_name] = A2AAgent(url=agent_url, memory=UnconstrainedMemory())
                print(f"✅ {agent_name}: Connected successfully")
            else:
                print(f"⚠️ {agent_name}: Agent card not found (HTTP {response.status_code})")
                
        except requests.RequestException as e:
            print(f"[ERROR] {agent_name}: Not accessible ({e})")
    
    print(f"Discovered {len(available_agents)} A2A agents")
    return available_agents, agent_cards

# =============================================================================
# AGENT CREATION 
# =============================================================================

def create_agent(config, tools=None):
    """
    Create a REGULAR A2A agent from configuration
    
    This function creates standard agents (not supervisor agents).
    For supervisor agents, SEE create_supervisor_agent() instead.
    """
    
    # AGENT TYPE DETECTION
    is_supervisor = 'supervised_agents' in config
    agent_type = "SUPERVISOR" if is_supervisor else "REGULAR"
    print(f" Creating {agent_type} agent: {config.get('name', 'Unnamed')}")
    
    if is_supervisor:
        print(" WARNING: This is a supervisor agent config but using create_agent()")
        print("   Consider using create_supervisor_agent() instead")
    
    print(f"Setting up LLM: {config['llm_model']}")
    llm = ChatModel.from_name(
        config['llm_model'],
        ChatModelParameters(temperature=config['llm_temperature'])
    )
    llm.tool_choice_support = {"auto", "none"}

    # LOGGING: Create middleware if enabled
    middlewares = []
    middleware_config = config['middleware']['trajectory']
    if middleware_config.get('enabled', True):
        middlewares.append(GlobalTrajectoryMiddleware(
            included=[Tool, ChatModel] if "ChatModel" in middleware_config['included_types'] else [Tool],
            pretty=middleware_config['pretty'],
            prefix_by_type={Tool: middleware_config['tool_prefix']}
        ))

    # Handle mixed tool configuration (MCP tools + additional traditional tools)
    agent_tools = tools or []
    
    if not tools:
        # Use traditional tools configuration only if no MCP tools
        tools_config = config.get('tools', [])
        agent_tools = tools_config if isinstance(tools_config, list) else []
    else:
        print(f"Agent could use {len(tools)} MCP tools")
    
    # Print tool names for debugging
    if agent_tools:
        print("Agent tools:")
        for t in agent_tools:
            print(f"  - {t.name}")
    else:
        print("No tools configured for this agent")

    requirements = []
    if agent_tools:  # Only process requirements if tools exist
        for r in config.get('requirements', []):
            tool_instance = next((t for t in agent_tools if t.name == r["tool_name"]), None)
            if not tool_instance:
                print(f"Warning: Tool {r['tool_name']} not found — skipping requirement")
                continue
            kwargs = {k: v for k, v in r.items() if k != "tool_name"}
            requirements.append(ConditionalRequirement(tool_instance, **kwargs))
    
    print(f"Requirements: {len(requirements)} tool usage rules")
    

    print(f"Creating RequirementAgent: {config.get('name', 'Unnamed')}")
    return RequirementAgent(
        llm=llm,
        tools=agent_tools,
        memory=UnconstrainedMemory(),
        instructions=config['prompt'],
        middlewares=middlewares,
        name=config.get('name'),
        description=config.get('description'),
        requirements=requirements
    )

# =============================================================================
# SERVER MANAGEMENT
# =============================================================================

def create_server(agent, config):
    """Create and configure the A2A server"""
    return A2AServer(
        config=A2AServerConfig(port=config['port']),
        memory_manager=LRUMemoryManager(maxsize=config['memory_size'])
    ).register(agent, name=config['name'], description=config['description'])

def run_server_background(server, config):
    """Run server in background thread"""
    def server_thread():
        try:
            server.serve()
        except Exception as e:
            print(f"Error in {config['name']} server thread: {e}")
    
    thread = threading.Thread(target=server_thread, daemon=True)
    thread.start()
    return thread

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_available_agents():
    """Dynamically get available agents from config directory"""
    config_dir = Path(__file__).parent.parent / "config"
    config_file = config_dir / "agents_config.yaml"
    
    if not config_file.exists():
        return []
    
    with open(config_file, 'r') as f:
        agents_config = yaml.safe_load(f)
    
    # Return all agent names except travel_router (which runs in the UI script)
    return list(agents_config.keys())

# =============================================================================
# MAIN AGENT RUNNER
# =============================================================================

async def run_agent(agent_name, background=False):
    """Run a single agent with optional MCP support"""
    
    # Load Agent configuration
    config = load_config(agent_name)
    
    print(f"LLM for {config['name']}: {config['llm_model']}")
    
    # Check if the agent will use any MCP tools
    mcp_config = config.get('mcp_config')
    has_valid_mcp = False
    
    if mcp_config:
        # Check if MCP config has actual servers
        if 'url' in mcp_config or ('servers' in mcp_config and mcp_config['servers']):
            has_valid_mcp = True
    
    if has_valid_mcp:
        print(f"MCP configuration detected - starting agent with MCP connections...")
        
        # Handle both old single-server format and new multi-server format  
        if 'url' in mcp_config:
            servers = [{'name': 'default', 'url': mcp_config['url'], 'enabled': True}]
        elif 'servers' in mcp_config and mcp_config['servers']:
            servers = mcp_config['servers']
        else:
            print("No MCP servers configured, check config/agents_config.yaml")
            return
        
        if not servers:
            print("No servers found in MCP configuration")
            return
        
        # Filter enabled servers
        enabled_servers = [s for s in servers if s and s.get('enabled', True)]
        
        if not enabled_servers:
            print("No enabled MCP servers found")
            return
        
        print(f"Connecting to {len(enabled_servers)} MCP server(s)...")
        
        # Use AsyncExitStack to manage multiple async context managers
        async with AsyncExitStack() as stack:
            all_mcp_tools = []
            
            # Connect to all enabled servers
            for server_config in enabled_servers:
                server_url = server_config['url'] + "/sse"
                server_name = server_config['name']
                
                print(f"Connecting to MCP server '{server_name}' at {server_url}...")
                
                try:
                    # Properly enter context managers and register them with the stack
                    read_stream, write_stream = await stack.enter_async_context(
                        sse_client(server_url)
                    )
                    
                    session = await stack.enter_async_context(
                        ClientSession(read_stream, write_stream)
                    )
                    
                    await session.initialize()
                    print(f"MCP session '{server_name}' initialized")
                    
                    # Get tools from this server
                    mcp_tools = await MCPTool.from_client(session)
                    print(f"Found {len(mcp_tools)} tools from '{server_name}':")
                    for tool in mcp_tools:
                        print(f"  - {tool.name}")
                    
                    all_mcp_tools.extend(mcp_tools)
                    
                except Exception as e:
                    print(f"Failed to connect to MCP server '{server_name}': {e}")
                    continue
            
            if not all_mcp_tools:
                print("No MCP tools available from any server")
                return
            
            print(f"\nTotal MCP tools available: {len(all_mcp_tools)}")
            
            # Create agent with all tools from all servers
            if 'supervised_agents' in config:
                print("Detected SUPERVISOR agent - using supervisor creation")
                agent = create_supervisor_agent(config)
            else:
                print("Detected REGULAR A2Aagent - using standard creation")
                agent = create_agent(config, all_mcp_tools)
            server = create_server(agent, config)
            
            print(f"Server ready at http://127.0.0.1:{config['port']}")
            print("Press Ctrl+C to stop")
            
            # Run server while keeping all MCP connections alive
            try:
                await asyncio.to_thread(server.serve)
            except (KeyboardInterrupt, asyncio.CancelledError):
                print(f"\n{config['name']} stopped")
    
    else:
        # No MCP tools or empty MCP config
        print(f"No valid MCP configuration found for {config['name']} - starting without MCP tools")
        if 'supervised_agents' in config:
            print(" Detected SUPERVISOR agent - using supervisor creation")
            agent = create_supervisor_agent(config)
        else:
            print(" Detected REGULAR agent - using standard creation")
            agent = create_agent(config, None)
        server = create_server(agent, config)
        
        print(f"Server ready at http://127.0.0.1:{config['port']}")
        print("Press Ctrl+C to stop")
        
        try:
            await asyncio.to_thread(server.serve)
        except (KeyboardInterrupt, asyncio.CancelledError):
            print(f"\n{config['name']} stopped")

def main():
    """Main entry point"""
    
    # Get available agents declared on the config file
    available_agents = get_available_agents()

    parser = argparse.ArgumentParser(description='Generic A2A Agent Runner')
    
    parser.add_argument('--agent', choices=available_agents,
                       help=f'Agent to run. Available: {", ".join(available_agents)}')
    parser.add_argument('--daemon', action='store_true', help='Run silently in background')
    parser.add_argument('--stop', action='store_true', help='Stop running agent')
    parser.add_argument('--logs', action='store_true', help='Show agent logs')
    args = parser.parse_args()
    
    if args.stop:
        os.system(f"pkill -f 'python.*agent_runner_copy.py.*--agent {args.agent}'")
        print(f"Agent '{args.agent}' stopped")
    elif args.logs:
        os.system(f"tail -f /tmp/agent_{args.agent}.log 2>/dev/null || echo 'No logs found for {args.agent}'")
    elif args.daemon:
        os.system(f"python3 {__file__} --agent {args.agent} > /tmp/agent_{args.agent}.log 2>&1 &")
        print(f"Agent '{args.agent}' started successfully in background")
    else:
        try:
            asyncio.run(run_agent(args.agent))
        except Exception as e:
            print(f"Error starting {args.agent}: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
