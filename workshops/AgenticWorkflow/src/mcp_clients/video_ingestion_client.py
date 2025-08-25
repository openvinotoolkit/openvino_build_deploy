import asyncio
from typing import Optional, Any
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import sys
import os
import pathlib
import base64
from dotenv import load_dotenv
from mcp_clients.utils.utils import NotificationCollector
from mcp.client.sse import sse_client
import gradio as gr

load_dotenv()

    
class VideoIngestionClient:
    def __init__(self, progress: Any = None):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.collector = NotificationCollector()
        self.progress_updates = []
        self.device = 'GPU'  # BridgeTower runs on CPU by default, can be changed to 'AUTO' or other devices if needed.
        
    async def progress_callback(self,
        progress: float, total: float | None, message: str | None
    ) -> None:
        """Collect progress updates for testing (async version)."""
        self.progress_updates.append((progress, total, message))
        text = f"Current Progress: {progress}/{total} - {message}"
        print(text)
        gr_progress = gr.Progress()
        gr_progress(progress=progress/total, desc=message)
        # yield text

    async def connect_to_sse_server(self, server_url: str):
        """
        Connect to an MCP server using SSE.
        
        Args:
            server_url (str): URL of the MCP server
        """
        print(f"Connecting to MCP server at {server_url}...")
        # Increase timeout for long-running embedding operations
        self.stdio, self.write = await self.exit_stack.enter_async_context(
            sse_client(server_url, timeout=600.0)  # 10 minute timeout
        )
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write),
        )
        print("Connecting to MCP server...")
        await self.session.initialize()
        print("MCP server initialized successfully.")
        response = await self.session.list_tools()
        tools = response.tools
        print(f"Connected to MCP server with tools: {[tool.name for tool in tools]}")
        
    async def connect(self, server_script_path: str):
        """
        Connect to an MCP server
        
        Args:
            server_script_path (str): Path to the MCP server script (.py or .js)
        """
        print(f"Connecting to MCP server at {server_script_path}...")
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a Python (.py) or JavaScript (.js) file.")
        
        command = "python" if is_python else "node"
        env = os.environ.copy()
        env['PYTHONPATH'] = os.getenv("HOME_DIR")
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=env,
        )

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write),
        )
        print("Connecting to MCP server...")
        await self.session.initialize()
        print("MCP server initialized successfully.")
        response = await self.session.list_tools()
        tools = response.tools
        print(f"Connected to MCP server with tools: {[tool.name for tool in tools]}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

    async def ingest_video(self, path: str) -> str:
        
        # result_tmp = await self.session.call_tool('long_running_task', {}, progress_callback=progress_callback, )
        # print(f"Result from long_running_task: {result_tmp}")
        # for msg in self.collector.log_messages:
        #     print(f"Log message: {msg}")

        # Assume query is a path to an mp4 file
        video_path = pathlib.Path(path)
        if not video_path.is_file() or video_path.suffix.lower() != ".mp4":
            return "Invalid video file. Currently, the system only supports .mp4 files."

        with open(video_path, "rb") as f:
            video_bytes = f.read()
        
        # Encode video bytes to base64
        video_base64 = base64.b64encode(video_bytes).decode('utf-8')

        tool_args = {"b64_file": video_base64, "filename": f"tmp_vid.mp4", "mode" : "overwrite"}
        # print(f"Calling save_videos tool with args: {tool_args}")
        result = await self.session.call_tool('ingest_videos', tool_args, progress_callback=self.progress_callback)
        # for res in result.content:
        #     print(f"=== {res}")
        text_response = result.content[0].text
        print(f"Text response is {text_response}")
        return text_response

async def ingest_video(path):
    gr_progress = gr.Progress()
    gr_progress(progress=0, desc="Connecting to MCP server...")
    client = VideoIngestionClient()
    try:
        print(f"video path is {path}")
        # await client.connect(sys.argv[1])
        URL_VIDEO_PROCESSING_MCP_SERVER = os.getenv("URL_VIDEO_PROCESSING_MCP_SERVER")
        print(f"Connecting to MCP server at {URL_VIDEO_PROCESSING_MCP_SERVER}...")
        await client.connect_to_sse_server(URL_VIDEO_PROCESSING_MCP_SERVER)
        await client.ingest_video(path)
        # await client.chat_loop()
        return "Video ingestion completed successfully."
    finally:
        await client.cleanup()
