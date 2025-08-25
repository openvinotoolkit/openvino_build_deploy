from mcp_servers.bridgetower_search.video_processing_server import mcp
# import asyncio

if __name__ == "__main__":
    # asyncio.run(main())
    # mcp.run(transport="stdio")
    mcp.run(transport="sse")