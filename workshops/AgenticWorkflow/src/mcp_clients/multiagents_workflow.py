from llama_index.tools.mcp import BasicMCPClient, McpToolSpec
from .utils.utils import (
    load_env, get_llm, 
    # custom_handle_reasoning_failure, 
    streaming_agent_workflow
)
# from llama_index.core.agent import ReActChatFormatter
from llama_index.core.llms import MessageRole
from mcp import types
from llama_index.core.agent.workflow import ReActAgent, AgentWorkflow
from llama_index.core.workflow import Context
import os
from llama_index.core.memory import ChatMemoryBuffer
import asyncio
from llama_index.core.agent.workflow import (
    AgentOutput,
    ToolCall,
    ToolCallResult,
)
from llama_index.core.workflow.events import StopEvent
import base64
from io import BytesIO
from PIL import Image

class MultiAgentWorkflow:
    def __init__(self):
        self.agent_workflow = None
        self.video_search_agent_tool_memory = []
        self.initialized = False
        # Don't initialize immediately - let gradio_helper do it async
        
    async def initialize(self):
        """
        Initialize the multi-agent workflow.
        This function prepares the agents and their tools.
        """
        # Define callback functions first
        async def callback_search_from_video_fn(result):
            print(f"🔍 Callback triggered! Processing result with {len(result.content) if hasattr(result, 'content') else 0} content items")
            
            images = []
            text_content = None
            
            # Check if we have a single TextContent containing JSON
            if (hasattr(result, 'content') and len(result.content) == 1 and 
                isinstance(result.content[0], types.TextContent)):
                
                content_text = result.content[0].text
                print(f"📋 Single TextContent received with {len(content_text)} characters")
                
                try:
                    import json
                    # Try to parse the content as JSON
                    if content_text.strip().startswith('{'):
                        data = json.loads(content_text)
                        print("✅ Successfully parsed JSON from TextContent")
                        
                        # Extract proper content from parsed JSON
                        for item in data.get('content', []):
                            if item.get('type') == 'image':
                                base64_data = item.get('data', '')
                                images.append(base64_data)
                                print(f"    ✅ Found image content in JSON (data length: {len(base64_data)})")
                            elif item.get('type') == 'text':
                                text_content = item.get('text', '')
                                print(f"    ✅ Found text content in JSON: {text_content[:100]}...")
                                
                    else:
                        # If not JSON, treat as plain text
                        text_content = content_text
                        print(f"    📝 Treating as plain text: {text_content[:100]}...")
                        
                except json.JSONDecodeError:
                    # If JSON parsing fails, treat as plain text
                    text_content = content_text
                    print(f"    📝 JSON parsing failed, treating as plain text: {text_content[:100]}...")
                except Exception as e:
                    print(f"❌ Error processing TextContent: {e}")
                    text_content = content_text
            
            # Check if we have the fragmentation issue (many string items)
            elif hasattr(result, 'content') and len(result.content) > 100:
                print(f"📦 Content items: {len(result.content)}")
                print(f"  Item 0: {type(result.content[0])}")
                print(f"  Item 1: {type(result.content[1])}")
                print(f"  Item 2: {type(result.content[2])}")
                print(f"  Item 3: {type(result.content[3])}")
                print(f"  Item 4: {type(result.content[4])}")
                print(f"  ... and {len(result.content)-5} more items -> could be a image serialized a text format like base64 or others ..?")
                
                # Try to reconstruct from fragmented string
                if all(isinstance(item, str) for item in result.content):
                    fragmented_json = ''.join(result.content)
                    print(f"🔧 Reconstructed JSON length: {len(fragmented_json)} characters")
                    
                    try:
                        import json
                        # Look for the JSON structure
                        json_start = fragmented_json.find('{\n  "_meta"')
                        if json_start != -1:
                            # Find the matching closing brace
                            brace_count = 0
                            json_end = json_start
                            for i, char in enumerate(fragmented_json[json_start:]):
                                if char == '{':
                                    brace_count += 1
                                elif char == '}':
                                    brace_count -= 1
                                    if brace_count == 0:
                                        json_end = json_start + i + 1
                                        break
                            
                            json_part = fragmented_json[json_start:json_end]
                            data = json.loads(json_part)
                            
                            # Extract proper content
                            for item in data.get('content', []):
                                if item.get('type') == 'image':
                                    base64_data = item.get('data', '')
                                    images.append(base64_data)
                                    print(f"    ✅ Reconstructed image content (data length: {len(base64_data)})")
                                elif item.get('type') == 'text':
                                    text_content = item.get('text', '')
                                    print(f"    ✅ Reconstructed text content: {text_content[:100]}...")
                        
                        else:
                            print("❌ Could not find JSON structure in reconstructed content")
                            
                    except Exception as e:
                        print(f"❌ Failed to reconstruct from fragmented JSON: {e}")
            
            else:
                # Fallback to original logic for properly structured responses
                for i, res in enumerate(result.content):
                    print(f"  Content {i}: {type(res)}")
                    if isinstance(res, types.ImageContent):
                        images.append(res.data)
                        print(f"    ✅ Found image content (data length: {len(res.data)})")
                    elif isinstance(res, types.TextContent):
                        text_content = res.text
                        print(f"    ✅ Found text content: {res.text[:100]}...")
            
            # Store images in memory for later display
            if images:
                self.video_search_agent_tool_memory.append(images)
                print(f"🖼️ Stored {len(images)} images in memory for display")
            else:
                print("❌ No images found in callback")
            
            # Return the text content for the agent to process
            return text_content if text_content else "No text response found"
        
        async def callback_mcp_tool_fn(result):
            processed_result = []      
            for res in result.content:
                if isinstance(res, types.TextContent):
                    processed_result.append(res.text)
            return "\n".join(processed_result)

        try:
            print("🔄 Loading environment variables...")
            load_env()
            
            # Get LLM instance first
            print("🔄 Initializing LLM...")
            llm = get_llm()
            if llm is None:
                raise ValueError("Failed to initialize LLM - check model paths and configuration")
            print("✅ LLM initialized successfully")
            
            print("🔄 Connecting to video processing MCP server...")
            URL_VIDEO_PROCESSING_MCP_SERVER = os.getenv("URL_VIDEO_PROCESSING_MCP_SERVER")
            if not URL_VIDEO_PROCESSING_MCP_SERVER:
                raise ValueError("URL_VIDEO_PROCESSING_MCP_SERVER not set in environment")
                
            # Add timeout for MCP client connections
            video_search_client = BasicMCPClient(URL_VIDEO_PROCESSING_MCP_SERVER)
            video_search_tool_spec = McpToolSpec(client=video_search_client)
            
            # Try to get tools with timeout
            video_search_tools = await asyncio.wait_for(
                video_search_tool_spec.to_tool_list_async(), 
                timeout=10.0
            )
            print(f"✅ Connected to video processing server - {len(video_search_tools)} tools available")
            
            # Set up callbacks for video search tools
            for tool in video_search_tools:
                if tool.metadata.name == "search_from_video":
                    tool._async_callback = callback_search_from_video_fn

            print("🔄 Creating video search agent...")
            video_search_agent = ReActAgent(
                name="VideoSearchAgent",
                description="Useful for searching from video.",
                tools=video_search_tools, llm=llm, verbose=True)
            ctx_video_search_agent = Context(video_search_agent)
            print("✅ Video search agent created")

            # shopping cart agents
            print("🔄 Connecting to shopping cart MCP server...")
            URL_SHOPPING_CART_MCP_SERVER = os.getenv("URL_SHOPPING_CART_MCP_SERVER")
            if not URL_SHOPPING_CART_MCP_SERVER:
                raise ValueError("URL_SHOPPING_CART_MCP_SERVER not set in environment")
                
            shopping_cart_client = BasicMCPClient(URL_SHOPPING_CART_MCP_SERVER)
            shopping_cart_tool_spec = McpToolSpec(client=shopping_cart_client)
            shopping_cart_tools = await asyncio.wait_for(
                shopping_cart_tool_spec.to_tool_list_async(),
                timeout=10.0
            )
            print(f"✅ Connected to shopping cart server - {len(shopping_cart_tools)} tools available")

            # for tool in shopping_cart_tools:
            #     tool._async_callback = callback_mcp_tool_fn

            print("🔄 Creating shopping cart agent...")
            shopping_cart_agent = ReActAgent(
                name="ShoppingCartAgent",
                description="Useful for managing shopping cart and compute amount and cost.",
                tools=shopping_cart_tools, llm=llm, verbose=True)

            ctx_shopping_cart_agent = Context(shopping_cart_agent)
            print("✅ Shopping cart agent created")

            # Use the same LLM instance for router agent
            print("🔄 Creating router agent...")
            router_agent = ReActAgent(
                name="RouterAgent",
                description="Routes queries to the appropriate agent based on the query type.",
                tools=[],
                llm=llm,
                verbose=True,
                can_handoff_to=["VideoSearchAgent", "ShoppingCartAgent"],
                system_prompt=(
                    "You are the RouterAgent that can analyze the query to determine the appropriate agent to hand off control to by using the tool \"handoff\". "
                    "You should hand off control to the VideoSearchAgent to search from video. "
                    "You should hand off control to the ShoppingCartAgent when: \n"
                    "\t- Query asks about paint  for kitchen, recommendations, prices, or technical specifications. \n"
                    "\t- Query is about managing the shopping cart. "
                    # """IMPORTANT NOTE: Your answer MUST include "Action: handoff" in order to pass the query to either the VideoSearchAgent or the ShoppingCartAgent.\n"""
                    # "Do NOT respond to the user directly, only hand off control to the appropriate agent.\n"
                ),
            )
            print("✅ Router agent created")

            print("🔄 Creating agent workflow...")
            agent_workflow = AgentWorkflow(
                agents=[router_agent, video_search_agent, shopping_cart_agent],
                root_agent=router_agent.name,
            )
            self.agent_workflow = agent_workflow
            self.initialized = True
            print("✅ MultiAgentWorkflow initialization completed successfully")
            
        except asyncio.TimeoutError:
            print("❌ Timeout connecting to MCP servers - check if servers are running")
            raise
        except Exception as e:
            print(f"❌ Failed to initialize MultiAgentWorkflow: {e}")
            raise

    async def query(self, query: str, memory: ChatMemoryBuffer = None):
        """
        Run the agent workflow with the given query.
        """
        if not self.agent_workflow:
            raise ValueError("Agent workflow is not initialized.")
        
        
        handler = self.agent_workflow.run(
            user_msg=query,
            memory=memory,
            max_iterations=50,  # Increased from default 20 to handle complex queries
        )

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
                yield(f"\n{'='*50}", None)
                yield(f"🤖 Agent: {current_agent}", None)
                yield(f"{'='*50}\n", None)

            # if isinstance(event, AgentStream):
            #     if event.delta:
            #         print(event.delta, end="", flush=True)
            # elif isinstance(event, AgentInput):
            #     print("📥 Input:", event.input)
            elif isinstance(event, AgentOutput):
                if event.response.content:
                    yield((f"📤 Output:\n{event.response.content}", None))
                if event.tool_calls:
                    yield(
                        f"🛠️ Planning to use tools: {[call.tool_name for call in event.tool_calls]}", None
                    )
            elif isinstance(event, ToolCallResult):
                yield(f"🔧 Tool Result ({event.tool_name}):", None)
                yield(f"  Arguments: {event.tool_kwargs}", None)
                yield(f"  Output: {event.tool_output}", None)
            elif isinstance(event, ToolCall):
                yield(f"🔨 Calling Tool: {event.tool_name}", None)
                yield(f"  With arguments: {event.tool_kwargs}", None)

            if isinstance(event, StopEvent):
                fn_response = event.result
                partial_text = ""
                if self.video_search_agent_tool_memory:
                    partial_text = "According to audio and following frames from the video: \n"                    
                    for imgs in self.video_search_agent_tool_memory:
                        for base64_img in imgs:
                            partial_text += f'<img src="data:image/png;base64,{base64_img}">\n'
                yield (None, f"{partial_text}\n{fn_response}")


def shorten_memory(memory):   
    all_messages = memory.get_all()
    for i, message in enumerate(all_messages):
        if message.role == MessageRole.ASSISTANT:
            for block in message.blocks:
                if block.block_type == "text":
                    # Block text should include only Answer:                    
                    block.text = "Answer: " + block.text.split("Final Answer:")[-1].split("Answer:")[-1].strip()
    return memory
