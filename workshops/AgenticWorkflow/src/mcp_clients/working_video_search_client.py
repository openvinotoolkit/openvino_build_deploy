from llama_index.tools.mcp import BasicMCPClient, McpToolSpec
from .utils.utils import (
    load_env, llm, 
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

async def main(args=None):
    """
    Main function to run the video search client.
    This function initializes the video search client.
    """
    load_env()
    URL_VIDEO_PROCESSING_MCP_SERVER = os.getenv("URL_VIDEO_PROCESSING_MCP_SERVER")
    print("inside main")
    video_search_client = BasicMCPClient(URL_VIDEO_PROCESSING_MCP_SERVER)
    video_search_tool_spec = McpToolSpec(client=video_search_client)
    video_search_tools = await video_search_tool_spec.to_tool_list_async()
    # print("Video Search Tools:", video_search_tools)
    
    video_search_agent_tool_memory = []

    async def callback_search_from_video_fn(result):
        images = []
        for res in result.content:
            if isinstance(res, types.ImageContent):
                images.append(res.data)
        if images:
            video_search_agent_tool_memory.append(images)
        result.content = result.content[:1]
        # Limit to first result for simplicity
        return result.content[0].text
    
    async def callback_mcp_tool_fn(result):
        processed_result = []      
        for res in result.content:
            if isinstance(res, types.TextContent):
                processed_result.append(res.text)
        return "\n".join(processed_result)

    
    for tool in video_search_tools:
        if tool.metadata.name == "search_from_video":
            tool._async_callback = callback_search_from_video_fn
        # else:
        #     tool._async_callback = callback_mcp_tool_fn

    # Define LLM Agent for video search
    # video_search_agent = ReActAgent.from_tools(
    #     tools=video_search_tools,
    #     llm=llm,
    #     max_iterations=5,
    #     verbose=True,
    #     react_chat_formatter=ReActChatFormatter.from_defaults(
    #         observation_role=MessageRole.TOOL   
    #     ),
    #     handle_reasoning_failure_fn=custom_handle_reasoning_failure,
    # )
    
    # video_search_agent = ReActAgent(
    #     llm=llm,
    #     tools=video_search_tools,
        
    #     verbose=True,
    #     # react_chat_formatter=ReActChatFormatter.from_defaults(
    #     #     observation_role=MessageRole.TOOL
    #     # ),
    #     # handle_reasoning_failure_fn=custom_handle_reasoning_failure,
    # )

    video_search_agent = ReActAgent(
        name="VideoSearchAgent",
        description="Useful for searching from video.",
        tools=video_search_tools, llm=llm, verbose=True)
    ctx_video_search_agent = Context(video_search_agent)

    # shopping cart agents
    URL_SHOPPING_CART_MCP_SERVER = os.getenv("URL_SHOPPING_CART_MCP_SERVER")
    shopping_cart_client = BasicMCPClient(URL_SHOPPING_CART_MCP_SERVER)
    shopping_cart_tool_spec = McpToolSpec(client=shopping_cart_client)
    shopping_cart_tools = await shopping_cart_tool_spec.to_tool_list_async()
    # print("Shopping Cart Tools:", shopping_cart_tools)

    # for tool in shopping_cart_tools:
    #     tool._async_callback = callback_mcp_tool_fn

    shopping_cart_agent = ReActAgent(
        name="ShoppingCartAgent",
        description="Useful for managing shopping cart and compute amount and cost.",
        tools=shopping_cart_tools, llm=llm, verbose=True)

    ctx_shopping_cart_agent = Context(shopping_cart_agent)

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
            "\t- Query asks about paint products, recommendations, prices, or technical specifications. \n"
            "\t- Query is about managing the shopping cart. "
            # """IMPORTANT NOTE: Your answer MUST include "Action: handoff" in order to pass the query to either the VideoSearchAgent or the ShoppingCartAgent.\n"""
            # "Do NOT respond to the user directly, only hand off control to the appropriate agent.\n"
        ),
    )

    agent_workflow = AgentWorkflow(
        agents=[router_agent, video_search_agent, shopping_cart_agent],
        root_agent=router_agent.name,
    )
    
    memory = ChatMemoryBuffer.from_defaults(token_limit=40000)
    raw_memory = None
    while True:
        try:
            query = input("\nQuery: ").strip()

            if query.lower() == 'quit':
                break

            # response = video_search_agent.chat(query)
            # print("\n" + response.response)

            # response = await video_search_agent.run(input=query)
            # response = response['response']
            # print("\n" + response)
           
            # handler = await video_search_agent.run(query, ctx=ctx_video_search_agent)
            # print("\n" + handler.response.content)

            handler = agent_workflow.run(
                user_msg=query,
                memory=memory,
            )
            handler = await streaming_agent_workflow(handler)
            # state = await handler.ctx.get("state")
            fn_res = handler.response.content
            print("\n Response: " + fn_res)

            # print(ctx_video_search_agent.to_dict())
            # his = memory.get_all()
            # for mes in his:
            #     print("---------------")
            #     print(mes.content)

            memory = shorted_memory(memory)
                    
        except Exception as e:
            print(f"\nError: {str(e)}")
    # print()
    print(len(video_search_agent_tool_memory))
    for i, images in enumerate(video_search_agent_tool_memory):
        print(f" number of images from {i}: {len(images)}")

def shorted_memory(memory):   

    all_messages = memory.get_all()
    for i, message in enumerate(all_messages):
        if message.role == MessageRole.ASSISTANT:
            for block in message.blocks:
                if block.block_type == "text":
                    # Block text should include only Answer:                    
                    block.text = "Answer: " + block.text.split("Final Answer:")[-1].split("Answer:")[-1].strip()
    return memory


