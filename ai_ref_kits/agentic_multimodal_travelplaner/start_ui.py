from typing import Tuple
import gradio as gr
from fastapi import FastAPI
import uvicorn
from beeai_framework.workflows.agent.agent import AgentWorkflowInput
print("✅ Basic imports successful")

# Set OpenAI environment variables from agent config
import os
import yaml
from pathlib import Path

config_dir = Path(__file__).parent / "config"
with open(config_dir / "agents_config.yaml", 'r') as f:
    agents_config = yaml.safe_load(f)

# Set environment variables from travel_router config
llm_config = agents_config['travel_router']['llm']
os.environ["OPENAI_API_KEY"] = llm_config['api_key']
os.environ["OPENAI_API_BASE"] = llm_config['api_base']

# Import video ingestion with proper error handling
ingest_video = None
try:
    print("🔄 Importing video_ingestion...")
    from mcp_tools.video_ingestion import ingest_video
    print("✅ Video ingestion client imported")
except ImportError as e:
    print(f"❌ Failed to import video_ingestion: {e}")
    def ingest_video(video_file):
        return "❌ Video ingestion not available - MCP server import failed"

from pathlib import Path
from fastapi.staticfiles import StaticFiles

# Travel Router Client - built into start_ui.py using direct agent workflow
import requests
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))
from beeai_framework.agents.experimental import RequirementAgent
from beeai_framework.backend import ChatModel, ChatModelParameters
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.workflows.agent.agent import AgentWorkflow, AgentWorkflowInput
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware
from beeai_framework.tools import Tool
from beeai_framework.adapters.a2a.agents.agent import A2AAgent
from dotenv import load_dotenv

class MultiAgentWorkflow:
    """Travel Router Agent that coordinates with A2A agents directly in the UI"""
    
    def __init__(self, agent_urls=None):
        """Initialize the MultiAgentWorkflow with direct agent integration"""
        self.router_agent = None
        self.agent_workflow = None
        self.initialized = False
        self.config = None
        self.available_agents = {}
        self.agent_cards = {}
        
        # Default agent URLs
        self.agent_urls = agent_urls or [
            f"http://127.0.0.1:{os.getenv('HOTEL_SEARCHER_PORT', '9999')}",
            f"http://127.0.0.1:{os.getenv('FLIGHT_SEARCHER_PORT', '9998')}",
            f"http://127.0.0.1:{os.getenv('VIDEO_SEARCHER_PORT', '9997')}"
        ]
    
    async def initialize(self):
        """Initialize the router agent with YAML configuration and A2A agents"""
        print("Initializing Travel Router Agent from YAML config...")
        load_dotenv()
        
        try:
            # Load configuration from YAML
            from utils.config import load_config
            self.config = load_config('travel_router')
            print(f"Loaded YAML config for {self.config['name']}")
            print(f"Model: {self.config['llm_model']}")
            print(f"Supervised agents: {len(self.config['supervised_agents'])}")
            
            # Discover A2A agents
            for agent_config in self.config['supervised_agents']:
                agent_name = agent_config['name']
                agent_url = agent_config['url']
                
                # Override URL with environment variable if available
                port_env_var = agent_config.get('port_env_var')
                if port_env_var and port_env_var in os.environ:
                    port = os.environ[port_env_var]
                    base_url = agent_url.rsplit(':', 1)[0]  # Remove port
                    agent_url = f"{base_url}:{port}"
                
                try:
                    print(f"Discovering A2A agent {agent_name} at {agent_url}...")
                    
                    # Try to get the Agent Card
                    agent_card_url = f"{agent_url}/.well-known/agent-card.json"
                    response = requests.get(agent_card_url, timeout=5)
                    
                    if response.status_code == 200:
                        agent_card = response.json()
                        print(f"[OK] {agent_name}: {agent_card.get('description', 'Available')}")
                        
                        # Create A2A agent connection
                        agent = A2AAgent(url=agent_url, memory=UnconstrainedMemory())
                        self.available_agents[agent_name] = agent
                        self.agent_cards[agent_name] = agent_card
                    else:
                        print(f"[WARN] {agent_name}: Server responded with status {response.status_code}")
                        
                except requests.RequestException as e:
                    print(f"[ERROR] {agent_name}: Not accessible ({e})")
            
            print(f"Discovered {len(self.available_agents)} A2A agents")
            
            # Create LLM
            llm = ChatModel.from_name(
                self.config['llm_model'],
                ChatModelParameters(temperature=self.config['llm_temperature'])
            )
            # Fix for Phi-4 model - remove "single" from tool_choice_support
            llm.tool_choice_support.discard("single")
            
            # Create middleware
            middleware_config = self.config['middleware']['trajectory']
            middlewares = [GlobalTrajectoryMiddleware(
                included=[Tool, ChatModel] if "ChatModel" in middleware_config['included_types'] else [Tool],
                pretty=middleware_config['pretty'],
                prefix_by_type={Tool: middleware_config['tool_prefix']}
            )]
            
            # Create and return agent
            self.router_agent = RequirementAgent(
                llm=llm,
                tools=self.config['tools'],  # Includes ThinkTool and HandoffTools
                memory=UnconstrainedMemory(),
                instructions=self.config['prompt'],
                middlewares=middlewares
            )
            
            print("Router agent created successfully")
            
            # Create workflow
            agent_workflow = AgentWorkflow(name="TravelRouterWorkflow")
            agent_workflow.add_agent(self.router_agent)
            self.agent_workflow = agent_workflow
            
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
            
            response = await self.agent_workflow.run(
                [AgentWorkflowInput(prompt=query)]
            )
            
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
multiagent_workflow = None
try:
    # Create client that connects to travel router A2A server via URL
    travel_router_port = os.getenv("TRAVEL_ROUTER_PORT", "9996")
    print(f"🔗 Connecting to Travel Router at http://127.0.0.1:{travel_router_port}")
    
    multiagent_workflow = MultiAgentWorkflow()  # This now uses URL-based client
    
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
                return new_loop.run_until_complete(multiagent_workflow.initialize())
            finally:
                new_loop.close()
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_init_in_thread)
            future.result()  # Wait for completion
            
    except RuntimeError:
        # No running event loop, we can use asyncio.run directly
        asyncio.run(multiagent_workflow.initialize())
        
    print("✅ Travel Router Client initialized")
except Exception as e:
    print(f"❌ Failed to initialize Travel Router Client: {e}")
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
    global chatbox_msg, multiagent_workflow, memory

    print(f"Received query: {query}", flush=True)
    chatbox_msg.append({"role": "user", "content": query})

    if multiagent_workflow is None or not multiagent_workflow.initialized:
        print("Travel Router not available or not initialized", flush=True)
        yield "Travel Router not available or not initialized.", chatbox_msg
        return

    try:
        print(f"Sending query to Travel Router: {query}", flush=True)
        
        # Show initial processing message
        initial_log = "🤖 Processing your request..."
        yield initial_log, chatbox_msg
        
        # Use the URL-based client to chat with travel router
        response = await multiagent_workflow.chat(query)
        
        print(f"Received response from Travel Router: {response[:100]}...", flush=True)
        
        # Simulate streaming by showing the response progressively
        words = response.split()
        partial_response = ""
        
        for i, word in enumerate(words):
            partial_response += word + " "
            
            # Update chat history with partial response
            temp_chatbox = chatbox_msg.copy()
            temp_chatbox.append({"role": "assistant", "content": partial_response.strip()})
            
            # Create log with progress
            progress = f"🤖 Generating response... ({i+1}/{len(words)} words)"
            
            yield progress, temp_chatbox
            
            # Small delay to make streaming visible
            import asyncio
            await asyncio.sleep(0.05)  # 50ms delay between words
        
        # Final response
        chatbox_msg.append({"role": "assistant", "content": response})
        final_log = f"✅ Response complete!"
        
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
