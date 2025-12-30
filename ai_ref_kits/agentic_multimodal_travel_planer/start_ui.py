#!/usr/bin/env python3
"""Gradio UI for the agentic multimodal travel planner.

This module provides a web-based user interface for interacting with the
travel router agent, supporting both text queries and image analysis through
a chat interface.
"""

import asyncio
import concurrent.futures
import os
import traceback
from pathlib import Path

import gradio as gr
from beeai_framework.adapters.a2a.agents.agent import A2AAgent
from beeai_framework.memory import UnconstrainedMemory
from dotenv import load_dotenv
from utils.util import (
    extract_agent_handoffs_from_log,
    run_async_in_thread,
    save_uploaded_image,
)


class TravelRouterClient:
    """UI client that connects to the travel_router agent.

    This client manages the connection to the travel router agent and handles
    query processing through the A2A (Agent-to-Agent) protocol.

    Attributes:
        client: A2AAgent instance for communication.
        initialized: Boolean flag indicating connection status.
        memory: Memory instance for maintaining conversation history.
    """

    def __init__(self):
        """Initialize the TravelRouterClient."""
        self.client = None
        self.initialized = False
        self.memory = UnconstrainedMemory()

    async def initialize(self):
        """Initialize connection to the travel_router agent.

        Returns:
            True if connection successful, False otherwise.
        """
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
            return True

        except Exception as e:
            print(f"Failed to initialize Travel Router Agent: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            return False

    async def chat(self, query: str) -> str:
        """Process a query through the travel router.

        Args:
            query: User query text to process.

        Returns:
            Response text from the agent.
        """
        if not self.initialized:
            await self.initialize()

        if not self.initialized:
            return (
                "Error: Travel Router not available. Please check the agent "
                "configuration."
            )

        try:
            # Call the travel router client
            response = await self.client.run(query)

            # Debug: print response structure
            print(f"Response type: {type(response)}")
            print(
                f"Response attrs: "
                f"{[a for a in dir(response) if not a.startswith('_')]}"
            )

            return self._extract_response_text(response)

        except Exception as e:
            print(f"Error with router agent: {e}")
            print(f"Error traceback: {traceback.format_exc()}")
            return f"Error processing request: {e}"

    def _extract_response_text(self, response) -> str:
        """Return the first text-like value available on the response."""

        def _as_text(value):
            if value is None:
                return None
            if isinstance(value, str):
                return value
            if isinstance(value, (list, tuple)):
                for item in reversed(value):
                    text = _as_text(item)
                    if text:
                        return text
                return None
            for attr in ("text", "content", "final_answer", "output"):
                if hasattr(value, attr):
                    text = _as_text(getattr(value, attr))
                    if text:
                        return text
            return None

        sources = (
            (
                "result.final_answer",
                getattr(
                    getattr(response, "result", None),
                    "final_answer",
                    None,
                ),
            ),
            (
                "state.final_answer",
                getattr(
                    getattr(response, "state", None),
                    "final_answer",
                    None,
                ),
            ),
            ("final_answer", getattr(response, "final_answer", None)),
            ("output", getattr(response, "output", None)),
            ("last_message", getattr(response, "last_message", None)),
            ("text", getattr(response, "text", None)),
            ("messages[-1]", getattr(response, "messages", None)),
        )

        for label, candidate in sources:
            text = _as_text(candidate)
            if text:
                print(f"Found in {label}: {text[:100]}...")
                return text

        fallback = str(response)
        print(f"Fallback to str: {fallback[:100]}...")
        return fallback if fallback else "No response received"


# Example prompts for the UI
EXAMPLES = [
    "Which city is shown in this image?",
    "Describe what's in this image",
    "How many people are in this photo?",
    "What is this building?",
    "Describe the scene in detail",
    "Give me flights from New York to Paris for March 1st to March 10th",
    "Give me hotels in Paris for March 1st to March 10th for 2 guests",
    (
        "Give me flights from New York to Paris for March 1st to March 10th "
        "in business class"
    ),
]

# Global state variables
chatbox_msg = []
stop_requested = False
current_image_path = None
log_cache = {}  # Cache for log file positions
workflow_steps = []  # Track workflow steps


def initialize_travel_router_client():
    """Initialize the travel router client connection.

    Returns:
        Initialized TravelRouterClient instance or None on failure.
    """
    print("🔄 Initializing Travel Router Client...")
    try:
        travel_router_port = os.getenv("TRAVEL_ROUTER_PORT", "9996")
        print(
            f"🔗 Connecting to Travel Router at "
            f"http://127.0.0.1:{travel_router_port}"
        )

        client = TravelRouterClient()

        # Initialize the client asynchronously
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    run_async_in_thread, client.initialize
                )
                future.result()
        except RuntimeError:
            # No running event loop, we can use asyncio.run directly
            asyncio.run(client.initialize())

        print("✅ Travel Router Client initialized")
        return client
    except Exception as e:
        print(f"❌ Failed to initialize Travel Router Client: {e}")
        traceback.print_exc()
        return None


# Initialize travel router client
travel_router_client = initialize_travel_router_client()


async def image_captioning(image_input):
    """Process and save uploaded image for analysis.

    Args:
        image_input: Image input from Gradio (file path or numpy array).

    Returns:
        Status message string.
    """
    project_root = Path(__file__).parent.parent
    tmp_dir = project_root / "tmp_files"

    try:
        saved_image_path = save_uploaded_image(
            image_input=image_input,
            destination_dir=tmp_dir,
            prefix="caption_image",
        )
        print(f"✅ Image saved to {saved_image_path}", flush=True)
    except Exception as exc:
        return str(exc)

    global current_image_path
    current_image_path = str(saved_image_path)
    return "✅ Image uploaded successfully"


def stop_query():
    """Stop the current query execution.

    Returns:
        Tuple of (status message, chatbox messages, send_btn state,
                  stop_btn state, clear_btn state).
    """
    global stop_requested
    stop_requested = True
    return (
        "🛑 Stopping query...",
        chatbox_msg,
        gr.update(interactive=True),   # send_btn enabled
        gr.update(interactive=False),  # stop_btn disabled
        gr.update(interactive=True),   # clear_btn enabled
    )


def add_workflow_step(step_text):
    """Add a step to the workflow display.

    Args:
        step_text: Text describing the workflow step.
    """
    global workflow_steps
    import datetime
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    workflow_steps.append(f"{timestamp} | {step_text}")
    # Keep only last 20 steps
    if len(workflow_steps) > 20:
        workflow_steps.pop(0)


def extract_agent_handoffs():
    """Extract agent handoff events from travel_router log."""
    logs_dir = Path(__file__).parent / "logs"
    travel_router_log = logs_dir / "travel_router.log"
    return extract_agent_handoffs_from_log(travel_router_log, log_cache)


def read_latest_logs():
    """Display workflow steps.

    Returns:
        Formatted markdown string with workflow steps.
    """
    global workflow_steps

    if not workflow_steps:
        return "### 🤖 Agent Workflow\n\n_Waiting for activity..._"

    # Format output
    log_content = "### 🤖 Agent Workflow\n\n"
    log_content += '\n\n'.join(workflow_steps)

    return log_content


def reset_all_agent_memories():
    """Reset memory for travel router and all supervised agents."""
    global travel_router_client
    # Reinitialize the travel router client to create a fresh session
    # This creates a new A2AAgent with fresh memory, effectively starting
    # a new conversation session with the server-side agent
    if travel_router_client:
        try:
            # Create new memory instance
            travel_router_client.memory = UnconstrainedMemory()
            # Reinitialize the A2AAgent client with fresh memory
            travel_router_port = os.getenv("TRAVEL_ROUTER_PORT", "9996")
            travel_router_url = f"http://127.0.0.1:{travel_router_port}"
            travel_router_client.client = A2AAgent(
                url=travel_router_url,
                memory=travel_router_client.memory
            )
            print("✅ Travel Router client reinitialized with fresh memory")
        except Exception as e:
            print(f"⚠️  Error reinitializing client: {e}")
    print("✅ All agent memories cleared")


def clear_all():
    """Clear all chat history and reset state.

    Returns:
        Tuple of (empty log window, empty chatbox).
    """
    global chatbox_msg, travel_router_client, stop_requested
    global current_image_path, workflow_steps
    # Clear chatbox messages and memory
    chatbox_msg = []
    stop_requested = False
    current_image_path = None
    workflow_steps = []
    # Reset agent memories
    try:
        reset_all_agent_memories()
    except Exception as e:
        print(f"⚠️  Error resetting memories: {e}")

    return read_latest_logs(), chatbox_msg


async def run_agent_workflow(query: str):
    """Execute agent workflow for user query.

    Args:
        query: User query text.

    Yields:
        Tuple of (log window text, chatbox messages, input field update,
                  send_btn state, stop_btn state, clear_btn state).
    """
    global chatbox_msg, travel_router_client, stop_requested
    global current_image_path, workflow_steps, log_cache

    # Clear workflow steps for new query
    workflow_steps = []

    # Mark current log position to only capture new handoffs
    logs_dir = Path(__file__).parent / "logs"
    travel_router_log = logs_dir / "travel_router.log"
    if travel_router_log.exists():
        cache_key = str(travel_router_log)
        try:
            current_size = travel_router_log.stat().st_size
            log_cache[cache_key] = {
                'position': current_size,
                'seen_handoffs': set()
            }
        except Exception:
            pass

    # Create enhanced query that includes image path if available
    enhanced_query = query
    if current_image_path:
        # Include a clear text hint so router can parse image_path
        enhanced_query = f"{query} : <image_path> = <{current_image_path}> "
        add_workflow_step("📸 Image included in query")

    chatbox_msg.append({"role": "user", "content": enhanced_query})
    stop_requested = False

    if travel_router_client is None or not travel_router_client.initialized:
        add_workflow_step("❌ Travel Router not available")
        yield (
            read_latest_logs(),
            chatbox_msg,
            gr.update(value="", interactive=True),   # msg cleared and enabled
            gr.update(interactive=True),   # send_btn enabled
            gr.update(interactive=False),  # stop_btn disabled
            gr.update(interactive=True),   # clear_btn enabled
        )
        return

    try:
        add_workflow_step("📤 Sending query to Travel Router")
        # Disable msg, send and clear, enable stop
        yield (
            read_latest_logs(),
            chatbox_msg,
            gr.update(value="", interactive=False),  # msg cleared and disabled
            gr.update(interactive=False),  # send_btn disabled
            gr.update(interactive=True),   # stop_btn enabled
            gr.update(interactive=False),  # clear_btn disabled
        )

        msg_text = f"Sending query to Travel Router: {enhanced_query}"
        print(msg_text, flush=True)

        add_workflow_step("🤔 Travel Router is processing...")
        yield (
            read_latest_logs(),
            chatbox_msg,
            gr.update(value="", interactive=False),  # msg cleared and disabled
            gr.update(interactive=False),  # send_btn disabled
            gr.update(interactive=True),   # stop_btn enabled
            gr.update(interactive=False),  # clear_btn disabled
        )

        # Create task for chat and monitor for handoffs
        chat_task = asyncio.create_task(
            travel_router_client.chat(enhanced_query)
        )

        # Poll for handoffs while waiting for response
        while not chat_task.done():
            await asyncio.sleep(0.2)  # Poll more frequently
            new_handoffs = extract_agent_handoffs()
            if new_handoffs:
                for handoff in new_handoffs:
                    add_workflow_step(handoff)
                yield (
                    read_latest_logs(),
                    chatbox_msg,
                    gr.update(value="", interactive=False),  # msg disabled
                    gr.update(interactive=False),  # send_btn disabled
                    gr.update(interactive=True),   # stop_btn enabled
                    gr.update(interactive=False),  # clear_btn disabled
                )

        # Get the final response
        response = await chat_task
        # Remove trailing colon if present (UI formatting issue)
        if isinstance(response, str) and response.rstrip().endswith(':'):
            response = response.rstrip()[:-1].rstrip()

        # Give it a moment for logs to flush
        await asyncio.sleep(0.3)

        # Check for any final handoffs multiple times
        for _ in range(3):
            final_handoffs = extract_agent_handoffs()
            if final_handoffs:
                for handoff in final_handoffs:
                    add_workflow_step(handoff)
                yield (
                    read_latest_logs(),
                    chatbox_msg,
                    gr.update(value="", interactive=False),  # msg disabled
                    gr.update(interactive=False),  # send_btn disabled
                    gr.update(interactive=True),   # stop_btn enabled
                    gr.update(interactive=False),  # clear_btn disabled
                )
            await asyncio.sleep(0.1)

        add_workflow_step("✅ Received response from Travel Router")

        # Add response to chat history
        chatbox_msg.append({"role": "assistant", "content": response})

        # Return final result - re-enable msg, send and clear, disable stop
        yield (
            read_latest_logs(),
            chatbox_msg,
            gr.update(value="", interactive=True),   # msg cleared and enabled
            gr.update(interactive=True),   # send_btn enabled
            gr.update(interactive=False),  # stop_btn disabled
            gr.update(interactive=True),   # clear_btn enabled
        )

    except Exception as e:
        error_msg = f"Error communicating with Travel Router: {e}"
        add_workflow_step(f"❌ Error: {str(e)[:50]}")
        chatbox_msg.append({"role": "assistant", "content": error_msg})
        # Re-enable msg, send and clear, disable stop
        yield (
            read_latest_logs(),
            chatbox_msg,
            gr.update(value="", interactive=True),   # msg cleared and enabled
            gr.update(interactive=True),   # send_btn enabled
            gr.update(interactive=False),  # stop_btn disabled
            gr.update(interactive=True),   # clear_btn enabled
        )


def create_gradio_interface():
    """Create and configure the Gradio interface.

    Returns:
        Configured Gradio Blocks interface.
    """
    # Set Gradio temp directory and env var
    gradio_temp_dir = Path(__file__).parent.parent / "temp"
    gradio_temp_dir.mkdir(exist_ok=True)
    os.environ["GRADIO_TEMP_DIR"] = str(gradio_temp_dir)

    demo = gr.Blocks(
        theme=gr.themes.Soft(),
        css=".disclaimer{font-variant-caps:all-small-caps;}",
    )
    with demo:
        # Title
        gr.Markdown(
            "<h1><center>Multi-Agent Travel Assistant 🤖</center></h1>"
        )
        gr.Markdown(
            "<center>Powered by OpenVINO + MCP + A2A Tools</center>"
        )

        with gr.Row():
            # Left Column: Image Upload + Status
            with gr.Column(scale=2):
                image_file = gr.Image(
                    value=None,
                    label="Upload or choose an image",
                    interactive=True,
                )
                status = gr.Textbox(
                    "Ready", interactive=False, show_label=False, lines=3
                )

            # Right Column: Chat UI
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="Conversation", height=500, type="messages"
                )
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Type your message…",
                        show_label=False,
                        container=False,
                        interactive=True,
                    )
                with gr.Row():
                    stop_btn = gr.Button("Stop", interactive=False)
                    clr_btn = gr.Button("Clear")
                    send_btn = gr.Button("Send", variant="primary")
                gr.Examples(
                    EXAMPLES, inputs=[msg], label="Click example, then Send"
                )
            with gr.Column(scale=2):
                log_window = gr.Markdown(
                    value=read_latest_logs(),
                    label="Agent Workflow",
                    height=300
                )

        # Register listeners
        image_file.change(
            fn=image_captioning, inputs=[image_file], outputs=[status]
        )
        send_btn.click(
            fn=run_agent_workflow,
            inputs=[msg],
            outputs=[log_window, chatbot, msg, send_btn, stop_btn, clr_btn],
        )
        msg.submit(
            fn=run_agent_workflow,
            inputs=[msg],
            outputs=[log_window, chatbot, msg, send_btn, stop_btn, clr_btn],
        )
        stop_btn.click(
            fn=stop_query,
            inputs=[],
            outputs=[log_window, chatbot, send_btn, stop_btn, clr_btn]
        )
        clr_btn.click(fn=clear_all, inputs=[], outputs=[log_window, chatbot])

    return demo


def main():
    """Main entry point for the Gradio UI application."""
    try:
        # Get port from environment variable or use default
        port = int(os.getenv("GRADIO_SERVER_PORT", 7860))
        print(f"🌐 Starting on port {port}")

        demo = create_gradio_interface()
        demo.launch(
            server_name="0.0.0.0",
            server_port=port,
            inbrowser=True,
            share=False,
            debug=False,
            show_error=True,
            quiet=False
        )
    except Exception as e:
        print(f"❌ Error launching Gradio interface: {e}")
        print("🔧 If issues persist, check if port 7860 is available")
        traceback.print_exc()


if __name__ == "__main__":
    main()
