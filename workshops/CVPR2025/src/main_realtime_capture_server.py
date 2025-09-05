"""
Main entry point for the Real-time Capture MCP Server

This script starts the real-time capture server that processes live webcam video
and microphone audio using BridgeTower embeddings for multimodal search.

Usage:
    python main_realtime_capture_server.py

The server will run on localhost:3002 with SSE transport.
"""

from mcp_servers.realtime_capture.realtime_capture_server import mcp

if __name__ == "__main__":
    print("🎥 Starting Real-time Capture MCP Server...")
    print("📹 Webcam and 🎤 Microphone → BridgeTower → Vector DB")
    print("🌐 Server URL: http://localhost:3002/sse")
    print("Press Ctrl+C to stop the server")
    
    mcp.run(transport="sse")
