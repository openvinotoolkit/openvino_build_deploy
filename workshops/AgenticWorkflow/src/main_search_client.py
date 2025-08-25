import asyncio
from typing import Optional
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

load_dotenv()
progress_updates = []

async def progress_callback(
    progress: float, total: float | None, message: str | None
) -> None:
    """Collect progress updates for testing (async version)."""
    progress_updates.append((progress, total, message))
    print(f"Progress: {progress}/{total} - {message}")
    
class MCPSearchClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.collector = NotificationCollector()

    async def connect_to_sse_server(self, server_url: str):
        """
        Connect to an MCP server using SSE.
        
        Args:
            server_url (str): URL of the MCP server
        """
        print(f"Connecting to MCP server at {server_url}...")
        self.stdio, self.write = await self.exit_stack.enter_async_context(
            sse_client(server_url)
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

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def process_query(self, query: str) -> str:
        
        # result_tmp = await self.session.call_tool('long_running_task', {}, progress_callback=progress_callback, )
        # print(f"Result from long_running_task: {result_tmp}")
        # for msg in self.collector.log_messages:
        #     print(f"Log message: {msg}")

        # Assume query is a path to an mp4 file
        video_path = pathlib.Path(query)
        if not video_path.is_file() or video_path.suffix.lower() != ".mp4":
            return await self.search(query)

        with open(video_path, "rb") as f:
            video_bytes = f.read()
        
        # Encode video bytes to base64
        video_base64 = base64.b64encode(video_bytes).decode('utf-8')

        tool_args = {"b64_file": video_base64, "filename": f"tmp_vid.mp4", "mode" : "overwrite"}
        # print(f"Calling save_videos tool with args: {tool_args}")
        result = await self.session.call_tool('ingest_videos', tool_args, progress_callback=progress_callback)
        # for res in result.content:
        #     print(f"=== {res}")
        text_response = result.content[0].text
        print(f"Text response is {text_response}")
        return text_response
    
    async def search(self, query: str) -> str:
        """Search for a query in the vector store"""
        tool_args = {"query": query}
        # print("calling tool...")
        
        result = await self.session.call_tool('search_from_video', tool_args, progress_callback=progress_callback)
        if result.isError:
            return "Error during search: " + result.content[0].text
        else:
            return f"{result.content[0].text}"
    
async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    client = MCPSearchClient()
    try:
        await client.connect(sys.argv[1])
        # URL_VIDEO_PROCESSING_MCP_SERVER = os.getenv("URL_VIDEO_PROCESSING_MCP_SERVER")
        # print(f"Connecting to MCP server at {URL_VIDEO_PROCESSING_MCP_SERVER}...")
        # await client.connect_to_sse_server(URL_VIDEO_PROCESSING_MCP_SERVER)
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())