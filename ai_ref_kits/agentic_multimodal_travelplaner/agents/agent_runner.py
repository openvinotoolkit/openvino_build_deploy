#!/usr/bin/env python3
"""
Generic Agent Runner - Modular script to run any A2A agent based on YAML config.
It reads configuration from config/agents_config.yaml and prompts from
config/agent_prompts.yaml.

This script can run:
- Regular agents with MCP tools
- Supervisor agents that coordinate other agents via HandoffTools
- All agents can be run in daemon mode for background operation
"""

import sys
from pathlib import Path
import argparse
import asyncio
import multiprocessing
import os
from contextlib import AsyncExitStack
import requests
import yaml

from mcp.client.sse import sse_client
from mcp.client.session import ClientSession

from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.agents.requirement.requirements.conditional import (
    ConditionalRequirement,
)
from beeai_framework.adapters.a2a.agents.agent import A2AAgent
from beeai_framework.adapters.a2a.serve.server import (
    A2AServer,
    A2AServerConfig,
)
from beeai_framework.backend import ChatModel, ChatModelParameters
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware
from beeai_framework.serve.utils import LRUMemoryManager
from beeai_framework.tools import Tool
from beeai_framework.tools.handoff import HandoffTool
from beeai_framework.tools.mcp import MCPTool

# Add parent directory to path for imports (before local imports!)
sys.path.append(str(Path(__file__).parent.parent))   # noqa: E402
from utils.config import load_config


# =============================================================================
# MCP TOOLS MANAGEMENT
# =============================================================================


class MCPToolsManager:
    """Manages MCP server connections and tool discovery"""

    async def setup_mcp_tools_with_stack(self, config, stack):
        """Setup MCP tools using
        AsyncExitStack for proper connection management.
        If the connection fails, the error is caught and
        the loop continues to the next server.
        """
        mcp_config = config.get('mcp_config')
        if not mcp_config:
            return []
        # Handle both old single-server format and new multi-server format
        servers = (
            [{'name': 'default', 'url': mcp_config['url'], 'enabled': True}]
            if 'url' in mcp_config
            else mcp_config.get('servers', [])
        )

        if not servers:
            return []

        enabled_servers = [s for s in servers if s and s.get('enabled', True)]
        if not enabled_servers:
            return []

        all_mcp_tools = []

        for server_config in enabled_servers:
            server_url = server_config['url'] + "/sse"

            try:
                read_stream, write_stream = await stack.enter_async_context(
                    sse_client(server_url)
                )
                session = await stack.enter_async_context(
                    ClientSession(read_stream, write_stream)
                )
                await session.initialize()

                mcp_tools = await MCPTool.from_client(session)
                all_mcp_tools.extend(mcp_tools)

            except (ConnectionError, OSError, TimeoutError):
                continue

        return all_mcp_tools

# =============================================================================
# AGENT CREATION
# =============================================================================


class AgentFactory:
    """Factory class for creating different types of agents"""

    def create_agent(self, config, tools=None):
        """Create an agent from configuration (regular or supervisor)"""
        if 'supervised_agents' in config:
            return self._create_supervisor_agent(config, tools)
        else:
            return self._create_regular_agent(config, tools)

    def _create_regular_agent(self, config, tools=None):
        """Create a regular A2A agent from configuration"""
        llm = ChatModel.from_name(
            config['llm_model'],
            ChatModelParameters(temperature=config['llm_temperature'])
        )
        llm.tool_choice_support = {"auto", "none"}

        # Create middleware if enabled
        middlewares = []
        middleware_config = config['middleware']['trajectory']
        if middleware_config.get('enabled', True):
            middlewares.append(GlobalTrajectoryMiddleware(
                included=(
                    [Tool, ChatModel]
                    if "ChatModel" in middleware_config['included_types']
                    else [Tool]
                ),
                pretty=middleware_config['pretty'],
                prefix_by_type={Tool: middleware_config['tool_prefix']}
            ))

        # Handle tool configuration
        # If tools are provided externally (MCP tools), use them
        # Otherwise, check if tools is a list in config
        if tools:
            agent_tools = tools
        else:
            tools_config = config.get('tools', [])
            # If tools is a dict with mcp_tools: true, it means
            # tools are loaded externally
            if (isinstance(tools_config, dict) and
                    tools_config.get('mcp_tools', False)):
                # Empty for now, will be loaded by run_agent
                agent_tools = []
            else:
                agent_tools = (
                    tools_config if isinstance(tools_config, list) else []
                )

        print(f"Agent using {len(agent_tools)} tools")

        # Process requirements
        requirements = []
        if agent_tools:
            for r in config.get('requirements', []):
                tool_instance = next(
                    (t for t in agent_tools if t.name == r["tool_name"]),
                    None
                )
                if tool_instance:
                    kwargs = {k: v for k, v in r.items() if k != "tool_name"}
                    requirements.append(ConditionalRequirement(
                        tool_instance, **kwargs
                    ))

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

    def _create_supervisor_agent(self, config, mcp_tools=None):
        """Create a SUPERVISOR agent
        that coordinates other agents
        via HandoffTools"""

        available_agents, agent_cards = self._discover_supervised_agents(
            config
        )
        print("***********CREATING SUPERVISOR AGENT***********")
        # Create supervisor tools (HandoffTools)
        supervisor_tools = []
        for agent_name, agent_instance in available_agents.items(
        ):
            agent_description = agent_cards.get(agent_name, {}).get(
                "description", agent_name
            )
            supervisor_tools.append(HandoffTool(
                name=agent_name,
                description=agent_description,
                target=agent_instance
            ))

        # Combine MCP tools with HandoffTools if provided
        if mcp_tools:
            all_tools = mcp_tools + supervisor_tools
            print(f"Supervisor using {len(mcp_tools)} MCP tools + "
                  f"{len(supervisor_tools)} HandoffTools = {len(all_tools)} total tools")
        else:
            all_tools = supervisor_tools
            print(f"Supervisor using {len(supervisor_tools)} "
                  f"HandoffTools (no MCP tools)")

        return self._create_regular_agent(config, all_tools)

    def _discover_supervised_agents(self, config):
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
                base_url = agent_url.rsplit(':', 1)[0]
                agent_url = f"{base_url}:{port}"

            try:
                agent_card_url = f"{agent_url}/.well-known/agent-card.json"
                response = requests.get(agent_card_url, timeout=5)

                if response.status_code == 200:
                    agent_card = response.json()
                    agent_cards[agent_name] = agent_card
                    available_agents[agent_name] = A2AAgent(
                        url=agent_url, memory=UnconstrainedMemory()
                    )
                    print(f"{agent_name}: Connected successfully")
                else:
                    print(
                        f"{agent_name}: Agent card not found "
                        f"(HTTP {response.status_code})"
                    )

            except requests.RequestException as e:
                print(f" {agent_name}: Not accessible ({e})")

        print(f"Discovered {len(available_agents)} A2A agents for supervision")
        return available_agents, agent_cards

# =============================================================================
# SERVER MANAGEMENT
# =============================================================================


class ServerManager:
    """Manages A2A server creation and lifecycle"""

    def create_server(self, agent, config):
        """Create and configure the A2A server"""
        return A2AServer(
            config=A2AServerConfig(port=config['port']),
            memory_manager=LRUMemoryManager(
                maxsize=config['memory_size']
            )
        ).register(
            agent, name=config['name'], description=config['description']
        )

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


class ConfigManager:
    """Manages configuration loading and agent discovery"""

    @staticmethod
    def get_available_agents():
        """Dynamically get available agents from config directory"""
        config_dir = Path(__file__).parent.parent / "config"
        config_file = config_dir / "agents_config.yaml"

        if not config_file.exists():
            return []

        with open(config_file, 'r', encoding='utf-8') as f:
            agents_config = yaml.safe_load(f)

        return list(agents_config.keys())

# =============================================================================
# MAIN AGENT RUNNER
# =============================================================================


class AgentRunner:
    """Main orchestrator class for running agents"""

    def __init__(self):
        self.mcp_manager = MCPToolsManager()
        self.agent_factory = AgentFactory()
        self.server_manager = ServerManager()
        self.config_manager = ConfigManager()

    async def run_agent(self, agent_name):
        """Run an agent based on its configuration.

        Args:
            agent_name: Name of the agent to run from config
        """
        config = load_config(agent_name)
        mcp_config = config.get('mcp_config')
        has_valid_mcp = mcp_config and (
            'url' in mcp_config or
            ('servers' in mcp_config and mcp_config['servers'])
        )

        if has_valid_mcp:
            # Keep AsyncExitStack alive for the entire agent lifecycle
            async with AsyncExitStack() as stack:
                all_mcp_tools = (
                    await self.mcp_manager.setup_mcp_tools_with_stack(
                        config, stack
                    )
                )
                if not all_mcp_tools:
                    return
                agent = self.agent_factory.create_agent(config, all_mcp_tools)
                server = self.server_manager.create_server(agent, config)

                # Run server in the same process to keep MCP connections alive
                print(f"Starting {agent_name} with MCP tools...")
                try:
                    await asyncio.to_thread(server.serve)
                except KeyboardInterrupt:
                    print(f"\n{agent_name} stopped")
        else:
            agent = self.agent_factory.create_agent(config, None)
            server = self.server_manager.create_server(agent, config)

            process = multiprocessing.Process(target=server.serve)
            process.start()

            try:
                while process.is_alive():
                    await asyncio.sleep(1)
            except (KeyboardInterrupt, asyncio.CancelledError):
                process.terminate()
                process.join()

    def get_available_agents(self):
        """Get list of available agents"""
        return self.config_manager.get_available_agents()


def main():
    """Main entry point"""
    runner = AgentRunner()
    available_agents = runner.get_available_agents()

    parser = argparse.ArgumentParser(description='Generic A2A Agent Runner')
    parser.add_argument(
        '--agent',
        choices=available_agents,
        help=f'Agent to run. Available: {", ".join(available_agents)}'
    )
    args = parser.parse_args()

    try:
        asyncio.run(runner.run_agent(args.agent))
    except KeyboardInterrupt:
        pass
    except (RuntimeError, ValueError) as e:
        print(f"Error starting {args.agent}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
