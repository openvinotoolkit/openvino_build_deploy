#!/usr/bin/env python3
"""Generic Agent Runner - Modular script to run any A2A agent based on YAML config"""
"""It reads configuration from config/agents_config.yaml and prompts from config/agent_prompts.yaml"""

import asyncio
import os
import sys
import yaml
from pathlib import Path
import argparse
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import load_config
from beeai_framework.agents.experimental import RequirementAgent
from beeai_framework.adapters.a2a.serve.server import A2AServer, A2AServerConfig
from beeai_framework.backend import ChatModel, ChatModelParameters
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.serve.utils import LRUMemoryManager
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware
from beeai_framework.tools import Tool
from beeai_framework.tools.mcp import MCPTool
from beeai_framework.agents.requirement.requirements.conditional import ConditionalRequirement
from mcp.client.sse import sse_client
from mcp.client.session import ClientSession



async def setup_mcp_tools(config):
    """Setup MCP tools from multiple MCP servers with persistent connections"""
    mcp_config = config.get('mcp_config')
    if not mcp_config:
        return None
    
    all_mcp_tools = []
    
    # Handle both old single-server format and new multi-server format
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
        server_url = server_config['url']
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


def create_agent(config, tools=None):
    """Create the agent from config"""
    llm = ChatModel.from_name(
        config['llm_model'],
        ChatModelParameters(temperature=config['llm_temperature'])
    )
    # Fix for Qwen3-8B model - remove "single" from tool_choice_support
    llm.tool_choice_support.discard("single")

    # Create middleware if enabled
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
    for t in agent_tools:
        print(t.name)

    requirements = []
    for r in config.get('requirements', []):
        tool_instance = next((t for t in agent_tools if t.name == r["tool_name"]), None)
        if not tool_instance:
            print(f"Warning: Tool {r['tool_name']} not found — skipping requirement")
            continue
        kwargs = {k: v for k, v in r.items() if k != "tool_name"}
        requirements.append(ConditionalRequirement(tool_instance, **kwargs))
    print(requirements)
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

def create_server(agent, config):
    """Create and configure the A2A server"""
    return A2AServer(
        config=A2AServerConfig(port=config['port']),
        memory_manager=LRUMemoryManager(maxsize=config['memory_size'])
    ).register(agent, name=config['name'], description=config['description'])


def get_available_agents():
    """Dynamically get available agents from config directory"""
    config_dir = Path(__file__).parent.parent / "config"
    config_file = config_dir / "agents_config.yaml"
    
    if not config_file.exists():
        return []
    
    with open(config_file, 'r') as f:
        agents_config = yaml.safe_load(f)
    
    # Return all agent names except travel_router (which runs in the UI script)
    return [name for name in agents_config.keys() if name != 'travel_router']


def run_agent(agent_name):
    """Run a single agent"""

    # Load configuration
    config = load_config(agent_name)
    print(f"Starting {config['name']} on port {config['port']}")
    print(f"Model: {config['llm_model']}")
    
    # Check if MCP tools are configured - run with persistent connection
    if 'mcp_config' in config:
        print(f"MCP configuration detected - starting with persistent MCP connections...")
        asyncio.run(run_agent_with_mcp(agent_name, config))
    else:
        # No MCP - run normally
        print(f"No MCP configuration found for {config['name']}")
        agent = create_agent(config, None)
        server = create_server(agent, config)
        
        print(f"Server ready at http://127.0.0.1:{config['port']}")
        print("Press Ctrl+C to stop")
        
        try:
            server.serve()
        except KeyboardInterrupt:
            print(f"\n{config['name']} stopped")


async def run_agent_with_mcp(agent_name, config):
    """Run agent with persistent MCP connections"""
    mcp_config = config.get('mcp_config')
    
    # Handle both old single-server format and new multi-server format  
    if 'url' in mcp_config:
        servers = [{'name': 'default', 'url': mcp_config['url'], 'enabled': True}]
    elif 'servers' in mcp_config:
        servers = mcp_config['servers']
    else:
        print("No MCP servers configured")
        return
    
    # Connect to first enabled MCP server with persistent connection
    for server_config in servers:
        if not server_config.get('enabled', True):
            continue
            
        server_url = server_config['url']
        server_name = server_config['name']
        
        print(f"Connecting to MCP server '{server_name}' at {server_url}...")
        
        try:
            # Keep connection alive for the lifetime of the agent
            async with sse_client(server_url) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    print(f"MCP session initialized")
                    
                    mcp_tools = await MCPTool.from_client(session)
                    print(f"Found {len(mcp_tools)} MCP tools")
                    for tool in mcp_tools:
                        print(f"  - {tool.name}")
                    
                    # Create agent inside the MCP connection context
                    agent = create_agent(config, mcp_tools)
                    server = create_server(agent, config)
                    
                    print(f"Server ready at http://127.0.0.1:{config['port']}")
                    print("Press Ctrl+C to stop")
                    
                    # Run server in a separate task to keep MCP connection alive
                    try:
                        await asyncio.to_thread(server.serve)
                    except KeyboardInterrupt:
                        print(f"\n{config['name']} stopped")
                    
                    break  # Only connect to first enabled server
                    
        except Exception as e:
            print(f"Failed to connect to MCP server '{server_name}': {e}")
            continue


def main():
    """Main entry point"""
    
    available_agents = get_available_agents()
    parser = argparse.ArgumentParser(description='Generic A2A Agent Runner')
    parser.add_argument('--agent', choices=available_agents,
                       help=f'Agent to run. Available: {", ".join(available_agents)}')
    args = parser.parse_args()
    
    try:
        run_agent(args.agent)
    except Exception as e:
        print(f"Error starting {args.agent}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
