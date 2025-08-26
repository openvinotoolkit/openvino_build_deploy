from typing import Tuple
import gradio as gr
from fastapi import FastAPI
import uvicorn
print("✅ Basic imports successful")

# Import video ingestion with proper error handling
ingest_video = None
try:
    print("🔄 Importing video_ingestion_client...")
    from mcp_clients.video_ingestion_client import ingest_video
    print("✅ Video ingestion client imported")
except ImportError as e:
    print(f"❌ Failed to import video_ingestion_client: {e}")
    def ingest_video(video_file):
        return "❌ Video ingestion not available - MCP server import failed"

from pathlib import Path
from fastapi.staticfiles import StaticFiles

# Import multiagent workflow with error handling
MultiAgentWorkflow = None
shorten_memory = None
try:
    print("🔄 Importing MultiAgentWorkflow...")
    from mcp_clients.multiagents_workflow import MultiAgentWorkflow, shorten_memory
    print("✅ MultiAgentWorkflow imported")
except ImportError as e:
    print(f"❌ Failed to import MultiAgentWorkflow: {e}")

import asyncio
from llama_index.core.memory import ChatMemoryBuffer
print("✅ LlamaIndex imports successful")

from llama_index.core.agent.workflow import (
    AgentOutput,
    ToolCall,
    ToolCallResult,
)
from llama_index.core.workflow.events import StopEvent

print("✅ All imports completed successfully")

app = FastAPI()

# Use the actual demo video from the workspace data directory
project_root = Path(__file__).parent.parent  # Go up to CVPR2025 directory
example_path = project_root / "data" / "friends.mp4"

# Check if the example video exists, otherwise use None
default_video = str(example_path) if example_path.exists() else None
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

# prepare MultiAgentWorkflow
print("🔄 Initializing MultiAgentWorkflow...")
multiagent_workflow = None
try:
    if MultiAgentWorkflow is not None:
        multiagent_workflow = MultiAgentWorkflow()
        # Initialize the workflow asynchronously
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
                    return new_loop.run_until_complete(multiagent_workflow.initialize())
                finally:
                    new_loop.close()
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_init_in_thread)
                future.result()  # Wait for completion
                
        except RuntimeError:
            # No running event loop, we can use asyncio.run directly
            asyncio.run(multiagent_workflow.initialize())
            
        print("✅ MultiAgentWorkflow initialized")
    else:
        print("⚠️ MultiAgentWorkflow not available - using demo mode")
except Exception as e:
    print(f"❌ Failed to initialize MultiAgentWorkflow: {e}")
    import traceback
    traceback.print_exc()
    multiagent_workflow = None

memory = ChatMemoryBuffer.from_defaults(token_limit=40000)
chatbox_msg = []
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

def clear_all():
    global chatbox_msg, memory
    # Clear chatbox messages and memory
    chatbox_msg = []
    memory = ChatMemoryBuffer.from_defaults(token_limit=40000)
    return "", chatbox_msg

async def run_agent_workflow(query: str):
    # for i in range(3):
    #     yield str(i)
    global chatbox_msg, multiagent_workflow, memory
    
    # Check if multiagent workflow is available and initialized
    if multiagent_workflow is None:
        yield "❌ MultiAgentWorkflow not available - check MCP server connections", [{"role": "user", "content": query}, {"role": "assistant", "content": "❌ MultiAgentWorkflow not available. Please ensure MCP servers are running and restart the application."}]
        return
    
    if not hasattr(multiagent_workflow, 'agent_workflow') or multiagent_workflow.agent_workflow is None:
        yield "❌ MultiAgentWorkflow not properly initialized - check MCP server connections", [{"role": "user", "content": query}, {"role": "assistant", "content": "❌ MultiAgentWorkflow not properly initialized. Please ensure MCP servers are running and restart the application."}]
        return
    
    chatbox_msg.append({"role": "user", "content": query})
    multiagent_workflow.video_search_agent_tool_memory = []
    handler = multiagent_workflow.agent_workflow.run(
        query,
        memory=memory,
    )
    log = ""
    current_agent = None
    current_tool_calls = ""
    fn_response = ""
    async for event in handler.stream_events():
        if (
            hasattr(event, "current_agent_name")
            and event.current_agent_name != current_agent
        ):
            current_agent = event.current_agent_name
            log += f"\n{'='*10}"
            log += f"🤖 Agent: {current_agent}"
            log += f"{'='*10}\n\n"
            yield log, chatbox_msg

        # if isinstance(event, AgentStream):
        #     if event.delta:
        #         print(event.delta, end="", flush=True)
        # elif isinstance(event, AgentInput):
        #     print("📥 Input:", event.input)
        elif isinstance(event, AgentOutput):
            if event.response.content:
                log += f"📤 Output:\n{markdown_adoption(event.response.content)}\n\n"
                yield log, chatbox_msg
            if event.tool_calls:
                log += f"🛠️ Planning to use tools: {[call.tool_name for call in event.tool_calls]} \n\n"
                yield log, chatbox_msg
        elif isinstance(event, ToolCallResult):
            log += f"🔧 Tool Result ({event.tool_name}):\n\n"
            yield log, chatbox_msg
            log += f"  Arguments: {event.tool_kwargs} \n\n"
            yield log, chatbox_msg
            log += f"  Output: {event.tool_output} \n\n"
            yield log, chatbox_msg
        elif isinstance(event, ToolCall):
            log += f"🔨 Calling Tool: {event.tool_name} \n\n"
            yield log, chatbox_msg
            log += f"  With arguments: {event.tool_kwargs} \n\n"
            yield log, chatbox_msg

        if isinstance(event, StopEvent):
            fn_response = event.result
            log += f"\n{'='*50}\n\n"
            log += f"🤖 Final Response: {markdown_adoption(str(fn_response))}\n\n"
            # chatbox_msg.append({"role": "assistant", "content": str(fn_response)})
            # yield log, chatbox_msg
            partial_text = ""
            if multiagent_workflow.video_search_agent_tool_memory:
                partial_text = "According to audio and following frames from the video: \n"                    
                for imgs in multiagent_workflow.video_search_agent_tool_memory:
                    for base64_img in imgs:
                        # Use JPEG MIME type since extracted frames are JPG files
                        partial_text += f'<img src="data:image/jpeg;base64,{base64_img}">\n'
            partial_text += f"\n{str(fn_response)}"
            chatbox_msg.append({"role": "assistant", "content": partial_text})
            multiagent_workflow.video_search_agent_tool_memory = []
            if shorten_memory is not None:
                memory = shorten_memory(memory)
            yield log, chatbox_msg
    

# def please(str):
#     loop = asyncio.get_event_loop()
#     loop.run_until_complete(run_agent_workflow(str))
# async def run_agent_workflow(query: str):
#     print("HELPING1111")
#     handler = multiagent_workflow.query(
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
import os
import stat

# Use a dedicated gradio_tmp directory in the project root
gradio_temp_dir = Path(__file__).parent.parent / "gradio_tmp"
gradio_temp_dir.mkdir(exist_ok=True)

# Set full permissions for the temp directory
try:
    os.chmod(str(gradio_temp_dir), stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
except:
    pass  # Ignore permission errors on Windows

# Set environment variable for Gradio temp directory
os.environ["GRADIO_TEMP_DIR"] = str(gradio_temp_dir)
print("✅ Gradio temp directory set")

print("🔄 Creating Gradio interface...")
with gr.Blocks(theme=gr.themes.Soft(), css=".disclaimer{font-variant-caps:all-small-caps;}") as demo:
# with gr.Blocks(theme=gr.themes.Soft(), css=css) as demo: 
    # Title
    gr.Markdown("<h1><center>Agentic Multimodal RAG 🤖</center></h1>")
    gr.Markdown("<center>Powered by OpenVINO + MCP Tools</center>") 

    with gr.Row():
        # === Left Column: Video + Log/Cart Below ===
        with gr.Column(scale=2):
            video_file = gr.Video(value=default_video, label="1) Upload or choose a video", interactive=True)
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
    clr_btn.click(fn=clear_all, inputs=[], outputs=[log_window, chatbot])

print("✅ Gradio interface created successfully")

if __name__ == "__main__":
    try:
        print("🚀 Starting Agentic Multimodal RAG Application...")
        print("📹 Make sure your MCP servers are running:")
        print("   - Video Search Server: python main_search_server.py")
        print("   - Shopping Cart Server: python main_shopping_cart_server.py")
        print("🌐 Opening web interface...")
        
        demo.launch(
            server_name="0.0.0.0",  # Allow external access
            server_port=7860,       # Default Gradio port
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