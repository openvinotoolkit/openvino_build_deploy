from mcp_servers.shopping_cart.shopping_cart_server import mcp
# import asyncio

if __name__ == "__main__":
    # asyncio.run(main())
    # mcp.run(transport="stdio")
    mcp.run(transport="sse")