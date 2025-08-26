from llama_index.tools.mcp import BasicMCPClient, McpToolSpec
from .utils.utils import load_env, llm, custom_handle_reasoning_failure
# from llama_index.core.agent import ReActAgent
from llama_index.core.agent import ReActChatFormatter
from llama_index.core.llms import MessageRole
from mcp import types
from .react_agent_workflow import ReActAgent
import os

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
    print("Video Search Tools:", video_search_tools)
    
    video_search_agent_tool_memory = []

    def call_back_search_from_video_fn(result):
        # print(len(result.content))
        # print(type(result.content))
        images = []
        for res in result.content:
            if isinstance(res, types.ImageContent):
                images.append(res.data)
        if images:
            video_search_agent_tool_memory.append(images)
        result.content = result.content[:1]
        # print(f"results: {result}")  # Limit to first result for simplicity
        return result
    
    for tool in video_search_tools:
        print(f"Tool Name: {tool.metadata.name}, Description: {tool.metadata.description}")
        if tool.metadata.name == "search_from_video":
            tool._callback = call_back_search_from_video_fn
    
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
    
    video_search_agent = ReActAgent(
        llm=llm,
        tools=video_search_tools,
        
        verbose=True,
        # react_chat_formatter=ReActChatFormatter.from_defaults(
        #     observation_role=MessageRole.TOOL
        # ),
        # handle_reasoning_failure_fn=custom_handle_reasoning_failure,
    )
    

    while True:
        try:
            query = input("\nQuery: ").strip()

            if query.lower() == 'quit':
                break

            # response = video_search_agent.chat(query)
            # print("\n" + response.response)

            response = await video_search_agent.run(input=query)
            response = response['response']
            print("\n" + response)
           
            

        except Exception as e:
            print(f"\nError: {str(e)}")
    print()
    print(len(video_search_agent_tool_memory[0]))




