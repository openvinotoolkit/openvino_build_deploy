#!/usr/bin/env python3
"""Gradio UI for the agentic multimodal travel planner.

This module provides a web-based user interface for interacting with the
travel router agent, supporting both text queries and image analysis through
a chat interface.
"""

import asyncio
import concurrent.futures
import datetime
import os
import traceback
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

from beeai_framework.adapters.a2a.agents.agent import A2AAgent
from beeai_framework.adapters.a2a.agents.events import A2AAgentUpdateEvent
from beeai_framework.agents.requirement.events import (
    RequirementAgentFinalAnswerEvent
)
from beeai_framework.memory import UnconstrainedMemory

from utils.streaming_citation_parser import StreamingCitationParser
from utils.util import (
    extract_agent_activities,
    extract_response_text,
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

    async def chat_stream(self, query: str, abort_event=None, on_token=None):
        """Get response from the A2A travel router agent with streaming updates.

        Args:
            query: User query text to process.
            abort_event: asyncio.Event to signal cancellation.
            on_token: Callback function for streaming tokens.

        Returns:
            tuple: (response_text, is_final) where response_text is the final
                   response and is_final is always True for completed
                   responses.
        """
        if not self.initialized:
            await self.initialize()

        if not self.initialized:
            return ("Error: Travel Router not available.", True)

        try:
            # Check if already cancelled
            if abort_event and abort_event.is_set():
                return ("Query cancelled.", True)

            # Enable real A2A streaming with proper event observation
            streaming_events = []
            response = None
            citation_parser = StreamingCitationParser()
            last_processed_length = 0

            async def capture_events(data, event):
                """Capture A2A streaming events."""
                global workflow_steps, suppress_router_status
                nonlocal last_processed_length
                streaming_events.append((data, event))

                if event.name == "final_answer":
                    if isinstance(data, RequirementAgentFinalAnswerEvent):
                        if data.delta:
                            clean_text, _ = citation_parser.process_chunk(
                                data.delta
                            )
                            if clean_text and on_token:
                                on_token(clean_text)
                    return

                # Print debug info to UI logs
                if isinstance(data, A2AAgentUpdateEvent):
                    value = data.value

                    # Extract task and status info
                    if isinstance(value, tuple) and len(value) >= 2:
                        task, status_update = value

                        # Try to extract streaming text from task history
                        if hasattr(task, 'history') and task.history:
                            # Look at the last message
                            last_msg = task.history[-1]
                            if hasattr(last_msg, 'parts') and last_msg.parts:
                                current_text = ""
                                for part in last_msg.parts:
                                    if (hasattr(part, 'root') and
                                            hasattr(part.root, 'text')):
                                        current_text += part.root.text or ""

                                # If text grew, calculate delta
                                if len(current_text) > last_processed_length:
                                    delta = current_text[
                                        last_processed_length:
                                    ]
                                    last_processed_length = len(current_text)

                                    # Process delta through parser
                                    clean_text, _ = (
                                        citation_parser.process_chunk(delta)
                                    )
                                    if clean_text and on_token:
                                        on_token(clean_text)

                        # Check for handoffs in task history
                        if hasattr(task, 'history') and task.history:
                            for msg in task.history:
                                if hasattr(msg, 'parts') and msg.parts:
                                    for part in msg.parts:
                                        if hasattr(part, 'root'):
                                            # Check for handoff tool calls
                                            part_str = str(part.root)
                                            if ('HandoffTool' in part_str or
                                                    'handoff' in
                                                    part_str.lower()):
                                                print(
                                                    "Handoff detected in "
                                                    f"history: {part_str[:200]}"
                                                )

                        # Get agent name from metadata or default to Travel Router
                        agent_name = "Travel Router"
                        if hasattr(task, 'metadata') and task.metadata:
                            if isinstance(task.metadata, dict):
                                agent_name = task.metadata.get(
                                    'agent_name',
                                    task.metadata.get('name', 'Travel Router')
                                )

                        # Extract state
                        if hasattr(status_update, 'status') and hasattr(
                            status_update.status, 'state'
                        ):
                            state = status_update.status.state.value
                            is_final = getattr(status_update, 'final', False)

                            # Skip Travel Router working status if suppressed
                            if (suppress_router_status and
                                    agent_name == "Travel Router" and
                                    state in ['working', 'submitted']):
                                return

                            # Use animated icon for working/submitted states
                            if (state in ['working', 'submitted'] and
                                    not is_final):
                                icon = "🔄"
                                status_msg = f"{icon} {agent_name}: {state}"
                            elif is_final or state == 'completed':
                                icon = "✅"
                                status_msg = f"{icon} {agent_name}: {state}"
                                # Clear suppression when Travel Router completes
                                if agent_name == "Travel Router":
                                    suppress_router_status = False
                            else:
                                status_msg = f"🤖 {agent_name}: {state}"

                            # Only remove temporary working/submitted status
                            workflow_steps[:] = [
                                step for step in workflow_steps
                                if not (f"🔄 {agent_name}:" in step and
                                        (": working" in step or
                                         ": submitted" in step))
                            ]
                            add_workflow_step(status_msg)

            # Listen for streaming tokens using .on() pattern
            response = await self.client.run(query).on(
                "update",
                capture_events
            ).on(
                "final_answer",
                capture_events
            )

            # Flush any remaining text in parser
            final_chunk = citation_parser.finalize()
            if final_chunk and on_token:
                on_token(final_chunk)

            # Process captured streaming events
            final_text = None
            for event_data, _ in streaming_events:
                if isinstance(event_data, A2AAgentUpdateEvent):
                    value = event_data.value
                    if isinstance(value, tuple) and len(value) >= 2:
                        task, status_update = value

                        # Update workflow with real-time status
                        if (hasattr(status_update, 'status') and
                                hasattr(status_update.status, 'state')):
                            state = status_update.status.state.value
                            # Update global workflow state for UI
                            # Keep handoffs, remove old status
                            workflow_steps[:] = [
                                step for step in workflow_steps
                                if "Travel Router task" not in step
                            ]
                            workflow_steps.append(
                                f"🔄 Travel Router task {state}..."
                            )

                        # Extract final message when complete
                        if (hasattr(status_update, 'final') and
                                status_update.final and
                                hasattr(task, 'history') and task.history):
                            for message in task.history:
                                if (hasattr(message, 'role') and
                                        str(message.role) == "Role.agent" and
                                        hasattr(message, 'parts') and
                                        message.parts):
                                    for part in message.parts:
                                        if (hasattr(part, 'root') and
                                                hasattr(part.root, 'text') and
                                                part.root.text):
                                            final_text = part.root.text
                                            break
                            break

            # Return final result
            if response:
                text = extract_response_text(response)
                if text:
                    return (text, True)

            if final_text:
                return (final_text, True)
            else:
                return ("No response received", True)

        except Exception as e:
            error_msg = f"Error communicating with Travel Router: {e}"
            print(f"A2A Streaming Error: {error_msg}")
            print(f"Traceback: {traceback.format_exc()}")
            return (error_msg, True)


# Example prompts for the UI
EXAMPLES = [
    "Which city is shown in this image?",
    "Give me flights from Milan to Rome for March 1st to March 10th",
    "Give me hotels in Milan for March 1st to March 10th for 2 guests",
    (
        "Give me flights from Milan to Rome for March 1st to March 10th "
        "in business class"
    ),
]


# Global state variables
chatbox_msg = []
stop_requested = False
current_image_path = None
log_cache = {}  # Cache for log file positions
workflow_steps = []  # Track workflow steps
thinking_indicators = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
indicator_index = 0  # Global animation state
suppress_router_status = False  # Suppress router working status after handoff
abort_event = None  # Abort controller for cancelling agent


def initialize_travel_router_client():
    """Initialize the travel router client connection.

    Returns:
        Initialized TravelRouterClient instance or None on failure.
    """
    try:
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
    global current_image_path

    # Handle None input (e.g., when image is deleted in UI)
    if image_input is None:
        current_image_path = None
        return "Image cleared"

    project_root = Path(__file__).parent.parent
    tmp_dir = project_root / "tmp_files"

    try:
        saved_image_path = save_uploaded_image(
            image_input=image_input,
            destination_dir=tmp_dir,
            prefix="caption_image",
        )
        if saved_image_path is None:
            current_image_path = None
            return "Image cleared"
        print(f"✅ Image saved to {saved_image_path}", flush=True)
    except Exception as exc:
        return str(exc)

    current_image_path = str(saved_image_path)
    return "✅ Image uploaded successfully"


def stop_query():
    """Stop the current query execution by setting abort signal.

    This will propagate cancellation through the entire agent chain
    (A2A client → router agent → travel agents).

    Returns:
        Tuple of (status message, chatbox messages, send_btn state,
                  stop_btn state, clear_btn state).
    """
    global stop_requested, abort_event
    stop_requested = True
    if abort_event is not None:
        abort_event.set()  # This aborts all agents in the chain
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
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    workflow_steps.append(f"{timestamp} | {step_text}")
    # Keep only last 50 steps to show more workflow history
    if len(workflow_steps) > 50:
        workflow_steps.pop(0)


def read_latest_logs():
    """Display workflow steps with animation for handoff messages.

    Returns:
        Formatted markdown string with workflow steps.
    """
    global workflow_steps, indicator_index

    if not workflow_steps:
        return "### 🤖 Agent Workflow\n\n_Waiting for activity..._"

    # Format output with animation for handoff messages
    log_content = "### 🤖 Agent Workflow\n\n"

    # Get current indicator and increment for next call
    indicator = thinking_indicators[indicator_index % len(thinking_indicators)]
    indicator_index += 1

    animated_steps = []
    for step in workflow_steps:
        # Check if the step should be animated
        # (contains 🔄 but not arrows or completed)
        should_animate = "🔄" in step and "✓" not in step and "→" not in step

        if should_animate:
            if "|" in step:
                # Handle timestamp format: "HH:MM:SS | 🔄 Message..."
                parts = step.split("|", 1)
                timestamp = parts[0].strip()
                message = parts[1].strip()

                # Check if message part starts with the icon to replace
                if message.startswith("🔄"):
                    base_message = message[1:]  # Remove the 🔄
                    animated_steps.append(
                        f"{timestamp} | {indicator}{base_message}"
                    )
                else:
                    # Icon might be elsewhere or not at start of message part
                    animated_steps.append(step.replace("🔄", indicator))
            else:
                # No timestamp, direct replacement
                animated_steps.append(step.replace("🔄", indicator))
        else:
            animated_steps.append(step)

    log_content += '\n\n'.join(animated_steps)

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
    global current_image_path, workflow_steps, log_cache, suppress_router_status
    global abort_event

    # Create abort controller for this query (can cancel entire agent chain)
    abort_event = asyncio.Event()
    stop_requested = False  # Reset stop flag

    # Clear workflow steps for new query
    global workflow_steps, indicator_index
    workflow_steps = []
    indicator_index = 0  # Reset animation state
    suppress_router_status = False  # Reset suppression flag

    # Mark current log position for all logs to only capture new events
    logs_dir = Path(__file__).parent / "logs"

    # Reset cache for travel_router log (handoffs)
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

    # Reset cache for all agent logs (MCP tool calls)
    for log_file in logs_dir.glob("*.log"):
        if log_file.name == "travel_router.log":
            continue
        cache_key = str(log_file)
        try:
            current_size = log_file.stat().st_size
            log_cache[cache_key] = {
                'position': current_size,
                'seen_tools': set()
            }
        except Exception:
            pass

    # Create enhanced query that includes image path if available
    enhanced_query = query
    if current_image_path:
        # Include a clear text hint so router can parse image_path
        enhanced_query = f"{query} : <image_path> = <{current_image_path}> "
        add_workflow_step("📸 Image included in query")

    # Store original query in chatbox (not the enhanced version with metadata)
    chatbox_msg.append({"role": "user", "content": query})
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

        # Initialize UI state
        yield (
            read_latest_logs(),
            chatbox_msg,
            gr.update(value="", interactive=False),  # msg cleared and disabled
            gr.update(interactive=False),  # send_btn disabled
            gr.update(interactive=True),   # stop_btn enabled
            gr.update(interactive=False),  # clear_btn disabled
        )

        # Show thinking indicators while waiting for response

        # Add placeholder for streaming
        chatbox_msg.append({"role": "assistant", "content": "..."})

        stream_state = {"started": False}

        def on_token(text):
            # print(f"DEBUG on_token: {text}")
            if chatbox_msg and chatbox_msg[-1]["role"] == "assistant":
                if not stream_state["started"]:
                    chatbox_msg[-1]["content"] = ""
                    stream_state["started"] = True
                chatbox_msg[-1]["content"] += text

        # Start streaming in background
        streaming_task = asyncio.create_task(
            travel_router_client.chat_stream(
                enhanced_query, abort_event, on_token=on_token
            )
        )

        # Show animated thinking indicator while processing
        loading_idx = 0
        while not streaming_task.done():
            # Update chat bubble animation if streaming hasn't started
            if (not stream_state["started"] and chatbox_msg and
                    chatbox_msg[-1]["role"] == "assistant"):
                indicator = thinking_indicators[
                    loading_idx % len(thinking_indicators)
                ]
                chatbox_msg[-1]["content"] = f"{indicator}"
                loading_idx += 1

            # Check for cancellation
            if abort_event and abort_event.is_set():
                streaming_task.cancel()
                add_workflow_step("🛑 Query cancelled by user")
                break
            # Check for new agent activities (handoffs and MCP tool calls)
            new_activities = extract_agent_activities(logs_dir, log_cache)
            if new_activities:
                for activity in new_activities:
                    # Handle special markers
                    if activity == "CLEAR_ROUTER_STATUS":
                        # Remove Travel Router working/submitted status
                        workflow_steps[:] = [
                            step for step in workflow_steps
                            if not ("Travel Router:" in step and
                                    (": working" in step or
                                     ": submitted" in step))
                        ]
                    elif activity == "SUPPRESS_ROUTER_WORKING":
                        # Set flag to suppress future router working status
                        suppress_router_status = True
                    elif activity == "CLEAR_SUPPRESS_ROUTER":
                        # Clear suppression when router resumes
                        suppress_router_status = False
                    else:
                        add_workflow_step(activity)

            # Yield UI update with animation
            # Create a copy of chatbox_msg to ensure Gradio detects the change
            yield (
                read_latest_logs(),
                list(chatbox_msg),
                gr.update(value="", interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=True),
                gr.update(interactive=False),
            )

            # Wait a bit before next update
            await asyncio.sleep(0.15)

        # Get the result
        try:
            response_text, is_final = await streaming_task
        except asyncio.CancelledError:
            response_text = "Query cancelled by user."
            is_final = True

        # Check if cancelled
        if abort_event and abort_event.is_set():
            response_text = "Query cancelled by user."
            add_workflow_step("🛑 Query cancelled")
            if chatbox_msg and chatbox_msg[-1]["role"] == "assistant":
                if not stream_state["started"]:
                    chatbox_msg[-1]["content"] = response_text
                else:
                    chatbox_msg[-1]["content"] += f"\n\n{response_text}"
            else:
                chatbox_msg.append(
                    {"role": "assistant", "content": response_text}
                )
            yield (
                read_latest_logs(),
                chatbox_msg,
                gr.update(value="", interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=False),
                gr.update(interactive=True),
            )
            return

        # Show complete response when agent finishes
        if response_text:
            # Remove trailing colon if present (UI formatting issue)
            if (isinstance(response_text, str) and
                    response_text.rstrip().endswith(':')):
                response_text = response_text.rstrip()[:-1].rstrip()

            # Add or update complete response
            if chatbox_msg and chatbox_msg[-1]["role"] == "assistant":
                chatbox_msg[-1]["content"] = response_text
            else:
                chatbox_msg.append(
                    {"role": "assistant", "content": response_text}
                )

            yield (
                read_latest_logs(),
                chatbox_msg,
                gr.update(value="", interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=True),
                gr.update(interactive=False),
            )

        # Give it a moment for logs to flush
        await asyncio.sleep(0.3)

        # Check for any final activities multiple times (catch delayed log writes)
        for i in range(10):
            final_activities = extract_agent_activities(logs_dir, log_cache)
            if final_activities:
                for activity in final_activities:
                    # Handle special markers
                    if activity == "CLEAR_ROUTER_STATUS":
                        # Remove Travel Router working/submitted status
                        workflow_steps[:] = [
                            step for step in workflow_steps
                            if not ("Travel Router:" in step and
                                    (": working" in step or
                                     ": submitted" in step))
                        ]
                    elif activity == "SUPPRESS_ROUTER_WORKING":
                        # Set flag to suppress future router working status
                        suppress_router_status = True
                    elif activity == "CLEAR_SUPPRESS_ROUTER":
                        # Clear suppression when router resumes
                        suppress_router_status = False
                    else:
                        add_workflow_step(activity)
                yield (
                    read_latest_logs(),
                    chatbox_msg,
                    gr.update(value="", interactive=False),  # msg disabled
                    gr.update(interactive=False),  # send_btn disabled
                    gr.update(interactive=True),   # stop_btn enabled
                    gr.update(interactive=False),  # clear_btn disabled
                )
            await asyncio.sleep(0.2)  # Longer delay to catch activities

        # Response already added to chat history above

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
        if chatbox_msg and chatbox_msg[-1]["role"] == "assistant":
            # If we were showing the placeholder, overwrite it
            if not stream_state.get("started", False):
                chatbox_msg[-1]["content"] = error_msg
            else:
                chatbox_msg[-1]["content"] += f"\n\n{error_msg}"
        else:
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
                    height=600
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
