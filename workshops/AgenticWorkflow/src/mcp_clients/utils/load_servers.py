from llama_index.tools.mcp import BasicMCPClient, McpToolSpec
from .utils import (
    load_env, llm, 
    # custom_handle_reasoning_failure, 
    streaming_agent_workflow
)
# from llama_index.core.agent import ReActChatFormatter
from llama_index.core.llms import MessageRole
from mcp import types
from llama_index.core.agent.workflow import ReActAgent, AgentWorkflow
from llama_index.core.workflow import Context
import os
from llama_index.core.memory import ChatMemoryBuffer

async def load_mcp_search_servers():
    load_env()
    URL_VIDEO_PROCESSING_MCP_SERVER = os.getenv("URL_VIDEO_PROCESSING_MCP_SERVER")
    print("inside main")
    video_search_client = BasicMCPClient(URL_VIDEO_PROCESSING_MCP_SERVER)
    video_search_tool_spec = McpToolSpec(client=video_search_client)
    video_search_tools = await video_search_tool_spec.to_tool_list_async()