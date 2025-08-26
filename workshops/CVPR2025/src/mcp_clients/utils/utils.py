import mcp.types as types
from dotenv import load_dotenv
from pathlib import Path
from .logger import log
import sys
import os
from llama_index.llms.openvino import OpenVINOLLM
import openvino.properties.hint as hints
import openvino.properties.streams as streams
import openvino.properties as props
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager
from llama_index.core.agent.workflow import (
    AgentOutput,
    ToolCall,
    ToolCallResult,
)
from llama_index.core.workflow.events import StopEvent

class NotificationCollector:
    def __init__(self):
        self.progress_notifications: list = []
        self.log_messages: list = []
        self.resource_notifications: list = []
        self.tool_notifications: list = []

    async def handle_progress(self, params) -> None:
        self.progress_notifications.append(params)

    async def handle_log(self, params) -> None:
        self.log_messages.append(params)

    async def handle_resource_list_changed(self, params) -> None:
        self.resource_notifications.append(params)

    async def handle_tool_list_changed(self, params) -> None:
        self.tool_notifications.append(params)

    async def handle_generic_notification(self, message) -> None:
        # Check if this is a ServerNotification
        if isinstance(message, types.ServerNotification):
            # Check the specific notification type
            if isinstance(message.root, types.ProgressNotification):
                await self.handle_progress(message.root.params)
            elif isinstance(message.root, types.LoggingMessageNotification):
                await self.handle_log(message.root.params)
            elif isinstance(message.root, types.ResourceListChangedNotification):
                await self.handle_resource_list_changed(message.root.params)
            elif isinstance(message.root, types.ToolListChangedNotification):
                await self.handle_tool_list_changed(message.root.params)

async def message_handler(message, collector):
    print(f"Received message: {message}")
    await collector.handle_generic_notification(message)
    if isinstance(message, Exception):
        raise message

def load_env():
    """
    Load environment variables from a .env file.
    This function is used to load the environment variables required for the application.
    """
    load_dotenv()
    return

def setup_models():
    """
    Setup models for the mcp clients.
    """
    load_env()
    llm_model_path = os.getenv("LLM_MODEL_PATH", None)
    
    if not llm_model_path:
        log.error("LLM_MODEL_PATH environment variable is not set. Please set it to the path of the LLM model.")
        sys.exit(1)
    # Check if model paths exist
    if not Path(llm_model_path).exists():
        log.error(f"LLM model not found at {llm_model_path}. Please run convert_and_optimize_llm.py to download the model first.")
        sys.exit(1)
    
    llm_model_device = os.getenv("LLM_MODEL_DEVICE", "CPU")

    ov_config = {
        hints.performance_mode(): hints.PerformanceMode.LATENCY,
        streams.num(): "1",
        props.cache_dir(): ""
    }
        
    # Load LLM model locally    
    llm = OpenVINOLLM(
        model_id_or_path=str(llm_model_path),
        context_window=8192,
        max_new_tokens=1000,
        model_kwargs={"ov_config": ov_config},
        generate_kwargs={"do_sample": False, "temperature": 0.1, "top_p": 0.8},        
        device_map=llm_model_device,
    )

    return llm

# Don't initialize models at import time - this causes hanging
# llm = setup_models()
# Settings.llm = llm

# Initialize LLM lazily when needed
_llm_instance = None

def get_llm():
    """Get or initialize the LLM instance lazily"""
    global _llm_instance
    if _llm_instance is None:
        try:
            _llm_instance = setup_models()
            Settings.llm = _llm_instance
        except Exception as e:
            print(f"Warning: Failed to initialize LLM: {e}")
            _llm_instance = None
    return _llm_instance

def custom_handle_reasoning_failure(callback_manager: CallbackManager, exception: Exception):
    """
    Provides custom error handling for agent reasoning failures.
    
    Args:
        callback_manager: The callback manager instance for event handling
        exception: The exception that was raised during reasoning
    """
    return "Hmm...I didn't quite that. Could you please rephrase your question to be simpler?"

async def streaming_agent_workflow(handler):
    """
    Stream events from the agent workflow handler and print them to the console.
    
    Args:
        handler: The agent workflow handler to stream events from
    """
    # Initialize current agent and tool calls
    current_agent = None
    current_tool_calls = ""
    fn_response = ""
    async for event in handler.stream_events():
        if (
            hasattr(event, "current_agent_name")
            and event.current_agent_name != current_agent
        ):
            current_agent = event.current_agent_name
            print(f"\n{'='*50}")
            print(f"🤖 Agent: {current_agent}")
            print(f"{'='*50}\n")

        # if isinstance(event, AgentStream):
        #     if event.delta:
        #         print(event.delta, end="", flush=True)
        # elif isinstance(event, AgentInput):
        #     print("📥 Input:", event.input)
        elif isinstance(event, AgentOutput):
            if event.response.content:
                print("📤 Output:\n", event.response.content)
            if event.tool_calls:
                print(
                    "🛠️  Planning to use tools:",
                    [call.tool_name for call in event.tool_calls],
                )
        elif isinstance(event, ToolCallResult):
            print(f"🔧 Tool Result ({event.tool_name}):")
            print(f"  Arguments: {event.tool_kwargs}")
            print(f"  Output: {event.tool_output}")
        elif isinstance(event, ToolCall):
            print(f"🔨 Calling Tool: {event.tool_name}")
            print(f"  With arguments: {event.tool_kwargs}")

        if isinstance(event, StopEvent):
            fn_response = event.result
    return fn_response