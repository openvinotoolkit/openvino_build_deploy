"""
Real-time Capture Client

This client demonstrates how to interact with the real-time capture MCP server
for live webcam and microphone processing using BridgeTower embeddings.

Features:
- Start/stop real-time capture
- Process and store recent audio-visual data
- Search through captured content
- Monitor capture status

Usage:
    python realtime_capture_client.py
"""

import asyncio
import os
import sys
from pathlib import Path
from mcp_clients.utils.utils import load_env
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

async def main():
    """Main function to demonstrate real-time capture client"""
    
    print("🎥 Real-time Capture Client")
    print("=" * 50)
    
    # Load environment
    load_env()
    
    # Connect to real-time capture MCP server
    SERVER_URL = "http://localhost:3002/sse"
    print(f"Connecting to real-time capture server: {SERVER_URL}")
    
    try:
        client = BasicMCPClient(SERVER_URL)
        tool_spec = McpToolSpec(client=client)
        tools = await tool_spec.to_tool_list_async()
        
        print(f"✅ Connected! Available tools: {[tool.metadata.name for tool in tools]}")
        print()
        
        # Interactive loop
        while True:
            print("\n📋 Available Commands:")
            print("1. start - Start real-time capture")
            print("2. stop - Stop real-time capture")
            print("3. process - Process and store recent data")
            print("4. search - Search captured data")
            print("5. status - Get capture status")
            print("6. quit - Exit")
            
            choice = input("\n🎯 Enter command (1-6): ").strip().lower()
            
            if choice in ['1', 'start']:
                print("\n🚀 Starting real-time capture...")
                
                # Get capture parameters
                try:
                    frame_rate = input("Frame rate (frames/sec, default=2): ").strip()
                    frame_rate = int(frame_rate) if frame_rate else 2
                    
                    audio_duration = input("Audio segment duration (seconds, default=5): ").strip()
                    audio_duration = int(audio_duration) if audio_duration else 5
                    
                    # Find the start tool
                    start_tool = next((tool for tool in tools if tool.metadata.name == "start_realtime_capture"), None)
                    if start_tool:
                        result = await start_tool.call(frame_rate=frame_rate, audio_duration=audio_duration)
                        print(f"✅ {result}")
                    else:
                        print("❌ Start tool not found")
                        
                except ValueError:
                    print("❌ Invalid input. Please enter numeric values.")
                except Exception as e:
                    print(f"❌ Error starting capture: {e}")
            
            elif choice in ['2', 'stop']:
                print("\n⏹️ Stopping real-time capture...")
                
                try:
                    stop_tool = next((tool for tool in tools if tool.metadata.name == "stop_realtime_capture"), None)
                    if stop_tool:
                        result = await stop_tool.call()
                        print(f"✅ {result}")
                    else:
                        print("❌ Stop tool not found")
                        
                except Exception as e:
                    print(f"❌ Error stopping capture: {e}")
            
            elif choice in ['3', 'process']:
                print("\n⚙️ Processing and storing recent data...")
                
                try:
                    frame_count = input("Number of recent frames to process (default=3): ").strip()
                    frame_count = int(frame_count) if frame_count else 3
                    
                    audio_count = input("Number of recent audio segments to process (default=1): ").strip()
                    audio_count = int(audio_count) if audio_count else 1
                    
                    process_tool = next((tool for tool in tools if tool.metadata.name == "process_and_store_recent_data"), None)
                    if process_tool:
                        result = await process_tool.call(frame_count=frame_count, audio_count=audio_count)
                        print(f"✅ {result}")
                    else:
                        print("❌ Process tool not found")
                        
                except ValueError:
                    print("❌ Invalid input. Please enter numeric values.")
                except Exception as e:
                    print(f"❌ Error processing data: {e}")
            
            elif choice in ['4', 'search']:
                print("\n🔍 Searching captured data...")
                
                try:
                    query = input("Enter search query: ").strip()
                    if not query:
                        print("❌ Please enter a search query")
                        continue
                    
                    top_k = input("Number of results (default=3): ").strip()
                    top_k = int(top_k) if top_k else 3
                    
                    search_tool = next((tool for tool in tools if tool.metadata.name == "search_realtime_data"), None)
                    if search_tool:
                        result = await search_tool.call(query=query, top_k=top_k)
                        
                        # Parse and display results
                        import json
                        try:
                            result_data = json.loads(result)
                            print(f"\n📊 Search Results for: '{query}'")
                            print(f"Found {result_data['_meta']['results_count']} results")
                            
                            for content in result_data['content']:
                                if content['type'] == 'text':
                                    print(f"\n💬 Analysis: {content['text']}")
                                elif content['type'] == 'image':
                                    print(f"🖼️ Image found - Frame ID: {content.get('frame_id', 'N/A')}")
                                    print(f"   Timestamp: {content.get('timestamp', 'N/A')}")
                                    print(f"   Transcript: {content.get('transcript', 'N/A')[:100]}...")
                                    
                        except json.JSONDecodeError:
                            print(f"✅ Raw result: {result}")
                    else:
                        print("❌ Search tool not found")
                        
                except ValueError:
                    print("❌ Invalid input for number of results.")
                except Exception as e:
                    print(f"❌ Error searching: {e}")
            
            elif choice in ['5', 'status']:
                print("\n📊 Getting capture status...")
                
                try:
                    status_tool = next((tool for tool in tools if tool.metadata.name == "get_capture_status"), None)
                    if status_tool:
                        result = await status_tool.call()
                        
                        # Parse and display status
                        import json
                        try:
                            status_data = json.loads(result)
                            print(f"\n📋 Capture Status:")
                            print(f"   Active: {'✅ Yes' if status_data.get('capture_active') else '❌ No'}")
                            print(f"   Frame Buffer: {status_data.get('frame_buffer_size', 0)} frames")
                            print(f"   Audio Buffer: {status_data.get('audio_buffer_size', 0)} segments")
                            print(f"   Frame Queue: {status_data.get('frame_queue_size', 0)} pending")
                            print(f"   Audio Queue: {status_data.get('audio_queue_size', 0)} pending")
                            
                            if 'camera_available' in status_data:
                                print(f"   Camera: {'✅ Available' if status_data['camera_available'] else '❌ Unavailable'}")
                                
                            print(f"   Last Update: {status_data.get('timestamp', 'N/A')}")
                            
                        except json.JSONDecodeError:
                            print(f"Raw status: {result}")
                    else:
                        print("❌ Status tool not found")
                        
                except Exception as e:
                    print(f"❌ Error getting status: {e}")
            
            elif choice in ['6', 'quit']:
                print("\n👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid choice. Please enter 1-6.")
                
    except Exception as e:
        print(f"❌ Failed to connect to server: {e}")
        print("Make sure the real-time capture server is running on localhost:3002")

if __name__ == "__main__":
    asyncio.run(main())
