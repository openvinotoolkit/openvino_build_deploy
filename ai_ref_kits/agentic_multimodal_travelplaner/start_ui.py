#!/usr/bin/env python3
"""
Agentic Tourism UI - Gradio interface for Travel Router
Connects to travel_router agent running via agent_runner_copy.py
"""

import os
import sys
import io
import asyncio
from pathlib import Path
from typing import Tuple

import gradio as gr
import uvicorn
import yaml
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

# BeeAI Framework imports
from beeai_framework.adapters.a2a.agents.agent import A2AAgent
from beeai_framework.memory import UnconstrainedMemory

print("✅ Basic imports successful")

# Set OpenAI environment variables from agent config
config_dir = Path(__file__).parent / "config"
with open(config_dir / "agents_config.yaml", 'r') as f:
    agents_config = yaml.safe_load(f)

llm_config = agents_config['travel_router']['llm']
os.environ["OPENAI_API_KEY"] = llm_config['api_key']
os.environ["OPENAI_API_BASE"] = llm_config['api_base']

# Import video ingestion with proper error handling
try:
    print("🔄 Importing video_ingestion...")
    from mcp_tools.video_ingestion import ingest_video
    print("✅ Video ingestion client imported")
except ImportError as e:
    print(f"❌ Failed to import video_ingestion: {e}")
    def ingest_video(video_file):
        return "❌ Video ingestion not available - MCP server import failed"

class TravelRouterClient:
    """UI client that connects to the travel_router agent"""
    
    def __init__(self):
        """Initialize the TravelRouterClient to connect to travel_router"""
        self.client = None
        self.initialized = False
        self.memory = UnconstrainedMemory()  # Persistent memory across queries
    
    async def initialize(self):
        """Initialize connection to the travel_router agent"""
        print("Initializing connection to Travel Router...")
        load_dotenv()
        
        try:
            # Connect to the running travel_router agent
            travel_router_port = os.getenv("TRAVEL_ROUTER_PORT", "9996")
            travel_router_url = f"http://127.0.0.1:{travel_router_port}"
            
            print(f"🔗 Connecting to Travel Router at {travel_router_url}")
            
            # Create A2A client to connect to the travel_router
            self.client = A2AAgent(url=travel_router_url, memory=self.memory)
            print("✅ Connected to Travel Router successfully")
            
            self.initialized = True
            print("MultiAgentWorkflow initialization completed successfully")
            return True
            
        except Exception as e:
            print(f"Failed to initialize Travel Router Agent: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return False
    
    async def chat(self, query: str) -> str:
        """Process a query through the travel router"""
        if not self.initialized:
            await self.initialize()
        
        if not self.initialized:
            return "Error: Travel Router not available. Please check the agent configuration."
        
        try:
            print(f"Router processing query: {query}")
            
            # Call the travel router client
            response = await self.client.run(query)
            
            print("Router agent completed processing")
            
            # Extract response text
            if hasattr(response, 'result') and hasattr(response.result, 'final_answer'):
                result = response.result.final_answer
            elif hasattr(response, 'state') and hasattr(response.state, 'final_answer'):
                result = response.state.final_answer
            elif hasattr(response, 'final_answer'):
                result = response.final_answer
            elif hasattr(response, 'last_message') and hasattr(response.last_message, 'text'):
                result = response.last_message.text
            elif hasattr(response, 'text'):
                result = response.text
            else:
                result = str(response)
            
            return result
                
        except Exception as e:
            print(f"Error with router agent: {e}")
            import traceback
            print(f"Error traceback: {traceback.format_exc()}")
            return f"Error processing request: {e}"


print("✅ Travel Router Client defined inline")

import asyncio
from llama_index.core.memory import ChatMemoryBuffer
import io
import sys
from contextlib import redirect_stdout
print("✅ LlamaIndex imports successful")

print("✅ All imports completed successfully")

app = FastAPI()

# Use the actual demo video from the workspace data directory
project_root = Path(__file__).parent.parent  # Go up to CVPR2025 directory
example_path = project_root / "data" / "input_vid.mp4"
# examples = [
#     ["What dessert is included in this video?"],
#     ["Tell me how to make a trifle. I want to make one myself"],
#     ["I think I need to buy some ingredients first. I need product information on online shopping platforms."],
#     ["Search for custard for me in these online shopping platforms"],
# ]

examples = [
    "Clear my cart",
    "What dessert is included in the video?",
    "What are the ingredients of the trifle?", 
    "Add those ingredients to my cart", 
    "Show my cart", 
    "what paint is the best for kitchens?",
    "what is the price of it?", 
    "how many gallons of paint do I need to cover 600 sq ft?",
    "add them to my cart",
    "what else do I need to complete my projects?", 
    "add 2 brushes to my cart",
]

# Use data directory in the project root instead of hardcoded Linux path
project_root = Path(__file__).parent.parent  # Go up to CVPR2025 directory
static_dir = project_root / "data"
static_dir.mkdir(exist_ok=True)  # Create the directory if it doesn't exist

# mount FastAPI StaticFiles server
app.mount("/static", StaticFiles(directory=static_dir), name="static")

css = """
body {
    padding: 15px;
    box-sizing: border-box;
    overflow-x: hidden;
}

#agent-steps {
    border: 2px solid #ddd;
    border-radius: 8px;
    padding: 12px;
    background-color: #f9f9f9;
    margin-top: 0; /* Remove top margin to align with other components */
    height: 100%; /* Ensure the same height as other components */
    box-sizing: border-box; /* Include padding in height calculation */
}

#shopping-cart {
    border: 2px solid #4CAF50;
    border-radius: 8px;
    padding: 12px;
    background-color: #f0f8f0;
    margin-top: 0; /* Remove top margin to align with other components */
    height: 100%; /* Ensure the same height as other components */
    box-sizing: border-box; /* Include padding in height calculation */
}

/* Fix row alignment issues */
.gradio-row {
    align-items: flex-start !important; /* Align all items to the top of the row */
}

/* Make all components in the main row the same height */
.gradio-row > .gradio-column {
    height: 100%;
    display: flex;
    flex-direction: column;
}

/* Ensure the chatbot and other components align properly */
.gradio-chatbot {
    margin-top: 0 !important;
}

/* Improve shopping cart table styling */
#shopping-cart table {
    width: 100%;
    border-collapse: collapse;
    table-layout: auto; /* Let the browser calculate column widths based on content */
}

#shopping-cart th,
#shopping-cart td {
    padding: 8px;
    text-align: left;
    min-width: 50px; /* Ensure minimum width for all columns */
}

#shopping-cart th:nth-child(2), /* Qty column */
#shopping-cart td:nth-child(2) {
    text-align: center;
    width: 50px;
}

#shopping-cart th:nth-child(3), /* Price column */
#shopping-cart td:nth-child(3),
#shopping-cart th:nth-child(4), /* Total column */
#shopping-cart td:nth-child(4) {
    text-align: right;
    min-width: 80px;
}

#shopping-cart th:first-child, /* Product column */
#shopping-cart td:first-child {
    width: auto; /* Let product name take remaining space */
}

.sample-prompt-btn {
    min-height: 35px !important;
    font-size: 0.85em !important;
    margin: 2px !important;
    padding: 4px 8px !important;
}

.intel-header {
    margin: 0px;
    padding: 0 15px;
    background: #0054ae;
    height: 60px;
    width: 100%;
    display: flex;
    align-items: center;
    position: relative;
    box-sizing: border-box;
    margin-bottom: 15px;
}

.intel-logo {
    margin-left: 20px;
    margin-right: 20px;
    width: 60px;
    height: 60px;
}
  
.intel-title {
    height: 60px;
    line-height: 60px;
    color: white;
    font-size: 24px;    
}

.gradio-container {
    max-width: 100% !important;
    padding: 0 !important; 
    box-sizing: border-box;
    overflow-x: hidden;
}

/* Override Gradio's generated padding classes */
.padding.svelte-phx28p,
[class*="padding svelte-"],
.gradio-container [class*="padding"] {
    padding: 0 !important;
}

.intel-header-wrapper {
    width: 100%;
    max-width: 100%;
    margin-left: 0;
    position: relative;
    padding: 0;
    box-sizing: border-box;
}

.gradio-container > .main {
    padding: 20px !important;
    max-width: 1800px;
    margin: 0 auto;
    box-sizing: border-box;
}

/* Fix label alignment issues */
.gradio-column > .label-wrap {
    margin-top: 0;
}

/* Ensure consistent spacing for all components */
.gradio-box, .gradio-chatbot, .gradio-markdown {
    margin-top: 0 !important;
}

/* Responsive adjustments */
@media (max-width: 768px) {
    #agent-steps, #shopping-cart {
        padding: 8px;
    }
    
    .intel-logo {
        margin-left: 10px;
        margin-right: 10px;
        width: 50px;
        height: 50px;
    }
    
    .intel-title {
        font-size: 20px;
    }
    
    /* Adjust table for mobile */
    #shopping-cart th,
    #shopping-cart td {
        padding: 4px;
        font-size: 0.9em;
    }
}
"""

# prepare Travel Router Client (URL-based connection)
print("🔄 Initializing Travel Router Client...")
travel_router_client = None
try:
    # Create client that connects to travel router A2A server via URL
    travel_router_port = os.getenv("TRAVEL_ROUTER_PORT", "9996")
    print(f"🔗 Connecting to Travel Router at http://127.0.0.1:{travel_router_port}")
    
    travel_router_client = TravelRouterClient()  # This now uses URL-based client
    
    # Initialize the client asynchronously
    import asyncio
    try:
        # Try to get the current event loop
        loop = asyncio.get_running_loop()
        # If we're in a running loop, run in a new thread
        import concurrent.futures
        
        def run_init_in_thread():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(travel_router_client.initialize())
            finally:
                new_loop.close()
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_init_in_thread)
            future.result()  # Wait for completion
            
    except RuntimeError:
        # No running event loop, we can use asyncio.run directly
        asyncio.run(travel_router_client.initialize())
        
    print("✅ Travel Router Client initialized")
except Exception as e:
    print(f"❌ Failed to initialize Travel Router Client: {e}")
    import traceback
    traceback.print_exc()
    travel_router_client = None

memory = ChatMemoryBuffer.from_defaults(token_limit=40000)
chatbox_msg = []
stop_requested = False  # Global flag for stopping queries
print("✅ Memory and chat initialized")

def markdown_adoption(st: str):
    st = st.replace("\n", "\n\n")
    return st
def safe_ingest_video(video_file):
    """Wrapper for ingest_video with error handling"""
    if ingest_video is None:
        return "❌ Video ingestion not available - MCP server connection failed during startup"
    
    try:
        # Handle async function in Gradio thread context
        import asyncio
        
        # Check if there's a running event loop
        try:
            loop = asyncio.get_running_loop()
            # If we're in a running loop (like AnyIO worker thread), 
            # we need to run the async function in a new thread with its own loop
            import concurrent.futures
            import threading
            
            def run_async_in_thread():
                # Create a new event loop for this thread
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(ingest_video(video_file))
                finally:
                    new_loop.close()
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_async_in_thread)
                return future.result()
                
        except RuntimeError:
            # No running event loop, we can use asyncio.run directly
            return asyncio.run(ingest_video(video_file))
            
    except Exception as e:
        return f"❌ Video ingestion failed: {str(e)}"

def stop_query():
    """Stop the current query"""
    global stop_requested
    stop_requested = True
    return "🛑 Stopping query...", chatbox_msg

def clear_all():
    global chatbox_msg, memory, travel_router_client, stop_requested
    # Clear chatbox messages and memory
    chatbox_msg = []
    memory = ChatMemoryBuffer.from_defaults(token_limit=40000)
    stop_requested = False  # Reset stop flag
    
    # Clear workflow memory if available
    if travel_router_client and travel_router_client.memory:
        travel_router_client.memory = UnconstrainedMemory()
    return "", chatbox_msg

async def run_agent_workflow(query: str):
    global chatbox_msg, travel_router_client, memory, stop_requested

    print(f"Received query: {query}", flush=True)
    chatbox_msg.append({"role": "user", "content": query})
    stop_requested = False  # Reset stop flag for new query

    if travel_router_client is None or not travel_router_client.initialized:
        print("Travel Router not available or not initialized", flush=True)
        yield "Travel Router not available or not initialized.", chatbox_msg
        return

    try:
        print(f"Sending query to Travel Router: {query}", flush=True)
        
        # Show initial processing message
        initial_log = f"""
🤖 **Processing Query:** {query}

**Router Status:** ✅ Active
**Memory:** ✅ Maintained
**Agent Tools:** ✅ Available

*Processing through Travel Router...*
"""
        yield initial_log, chatbox_msg
        
        # Use the URL-based client to chat with travel router
        # Capture tool call logs during processing
        tool_call_log = ""
        
        # Show tool call status
        tool_status_log = f"""
🤖 **Processing Query:** {query}

**Router Status:** ✅ Active
**Memory:** ✅ Maintained
**Agent Tools:** ✅ Available

*Analyzing query and routing to specialized agents...*
"""
        yield tool_status_log, chatbox_msg
        
        # Capture console output to detect tool calls
        captured_output = io.StringIO()
        
        # Show connection status
        connection_status = "✅ Connected to Travel Router" if travel_router_client and travel_router_client.initialized else "❌ Not connected"
        
        tool_prediction_log = f"""
🔍 **Travel Router Status**

{connection_status}

*Processing query through Travel Router...*
"""
        yield tool_prediction_log, chatbox_msg
        
        response = await travel_router_client.chat(query)
        
        print(f"Received response from Travel Router: {response[:100]}...", flush=True)
        
        # Just show that processing is complete
        tools_completed_log = f"""
🔧 **Processing Complete**

✅ **Travel Router** - Query processed successfully

*Generating final response...*
"""
        yield tools_completed_log, chatbox_msg
        
        # Simulate streaming by showing the response progressively
        words = response.split()
        partial_response = ""
        
        for i, word in enumerate(words):
            # Check if stop was requested
            if stop_requested:
                yield "🛑 Query stopped by user.", chatbox_msg
                return
                
            partial_response += word + " "
            
            # Update chat history with partial response
            temp_chatbox = chatbox_msg.copy()
            temp_chatbox.append({"role": "assistant", "content": partial_response.strip()})
            
            # Create log with progress
            progress = f"""
🤖 **Generating Response** ({i+1}/{len(words)} words)

**Streaming:** {partial_response.strip()}...

**Status:** ✅ Active | **Memory:** ✅ Maintained
"""
            
            yield progress, temp_chatbox
            
            # Small delay to make streaming visible
            import asyncio
            await asyncio.sleep(0.05)  # 50ms delay between words
        
        # Final response
        chatbox_msg.append({"role": "assistant", "content": response})
        
        # Show detailed logs instead of generic completion message
        final_log = f"""
🤖 **Agent Processing Complete**

**Final Response:** {response}

**Processing Details:**
- Query processed by Travel Router
- Memory maintained across conversation
- Tool calls executed as needed
- Response generated successfully

✅ **Ready for next query**
"""
        
        yield final_log, chatbox_msg
        
    except Exception as e:
        error_msg = f"Error communicating with Travel Router: {e}"
        print(f"Error: {error_msg}", flush=True)
        chatbox_msg.append({"role": "assistant", "content": error_msg})
        yield error_msg, chatbox_msg





# def please(str):
#     loop = asyncio.get_event_loop()
#     loop.run_until_complete(run_agent_workflow(str))
# async def run_agent_workflow(query: str):
#     print("HELPING1111")
#     handler = travel_router_client.query(
#         query,
#         memory=memory,
#     )
#     print("HELPING2222")
#     fn_response = ""
#     img_list = []
#     async for res in handler:
#         log, text_res = res
#         yield log

# def chat(query:str):
#     """
#     Function to handle chat queries.
#     """
#     return asyncio.create_task(run_agent_workflow(query))
    # history = [
    #     {"role": "assistant", "content": "I am happy to provide you that report and plot."},
    #     {"role": "assistant", "content": "Here is the report and plot you requested."},
    # ]
    # return "abc", history

# chat("What dessert is included in this video?")


# print(f"=====> RESPONSE: {resp}")

# Set Gradio temporary directory to avoid Windows path issues
import tempfile
gradio_temp_dir = Path(__file__).parent.parent / "temp"
gradio_temp_dir.mkdir(exist_ok=True)
# Set environment variable for Gradio temp directory
os.environ["GRADIO_TEMP_DIR"] = str(gradio_temp_dir)
print("✅ Gradio temp directory set")

print("🔄 Creating Gradio interface...")
with gr.Blocks(theme=gr.themes.Soft(), css=".disclaimer{font-variant-caps:all-small-caps;}") as demo:
# with gr.Blocks(theme=gr.themes.Soft(), css=css) as demo: 
    # Title
    gr.Markdown("<h1><center>Agentic Multimodal RAG 🤖</center></h1>")
    gr.Markdown("<center>Powered by OpenVINO + MCP + A2A Tools</center>") 

    with gr.Row():
        # === Left Column: Video + Log/Cart Below ===
        with gr.Column(scale=2):
            video_file = gr.Video(value=str(example_path), label="1) Upload or choose a video", interactive=True)
            build_btn  = gr.Button("2) Build Vector Store", variant="primary")
            status     = gr.Textbox("Vector store is already pre-built", interactive=False, show_label=False)

            # with gr.Row():                
                # cart_md = gr.Markdown("### 🛒 Your Actions / Cart", label="Cart", height=300)

                # === Right Column: Chat UI ===
        with gr.Column(scale=2):
            chatbot   = gr.Chatbot(label="Conversation", height=500, type="messages")
            with gr.Row():
                msg      = gr.Textbox(placeholder="Type your message…", show_label=False, container=False, interactive=True)
            with gr.Row():
                send_btn = gr.Button("Send", variant="primary")
                stop_btn = gr.Button("Stop")
                clr_btn  = gr.Button("Clear")
            gr.Examples(examples, inputs=[msg], label="Click example, then Send")
        with gr.Column(scale=2):
            log_window = gr.Markdown("### 🤖 Agent’s Reasoning Log", label="Logs", height=300)
    
    # Register listeners    
    build_btn.click(fn=safe_ingest_video, inputs=[video_file], outputs=[status])
    send_btn.click(fn=run_agent_workflow, inputs=[msg], outputs=[log_window, chatbot])
    
    # Register stop button
    stop_btn.click(fn=stop_query, inputs=[], outputs=[log_window, chatbot])
    
    clr_btn.click(fn=clear_all, inputs=[], outputs=[log_window, chatbot])

print("✅ Gradio interface created successfully")

if __name__ == "__main__":
    try:
        print("🚀 Starting Agentic Multimodal RAG Application...")
        print("📹 Make sure your MCP servers are running:")
        print("   - Video Search Server: python main_search_server.py")
        print("   - Shopping Cart Server: python main_shopping_cart_server.py")
        print("🌐 Opening web interface...")
        
        # Get port from environment variable or use default
        port = int(os.getenv("GRADIO_SERVER_PORT", 7860))
        print(f"🌐 Starting on port {port}")
        
        demo.launch(
            server_name="0.0.0.0",  # Allow external access
            server_port=port,       # Use environment variable or default
            inbrowser=True,         # Automatically open browser
            share=False,            # Don't create public link
            debug=False,            # Set to True for debugging
            show_error=True,        # Show errors in interface
            quiet=False             # Show startup messages
        )
    except Exception as e:
        print(f"❌ Error launching Gradio interface: {e}")
        print("💡 Try running with: python gradio_helper.py")
        print("🔧 If issues persist, check if ports 7860 is available")
        import traceback
        traceback.print_exc()
# app = gr.mount_gradio_app(app, demo, path="/demo")
# share = False
# enable_queue = False
