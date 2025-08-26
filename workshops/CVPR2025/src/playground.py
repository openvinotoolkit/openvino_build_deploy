from langgraph.prebuilt import create_react_agent

# Example: Using a local LLM (replace with your actual local LLM interface)
def local_llm(prompt: str) -> str:
    # Replace this with your actual local LLM call
    # For example, call to llama.cpp, ollama, or a local HuggingFace model
    return "This is a mock response from the local LLM."

# Create the ReAct agent using the local LLM
agent = create_react_agent(
    model=local_llm,
    tools=[],  # Add any tools you want the agent to use
    name="abcd"
)

# Example usage
response = agent.invoke({"input" : "What is the capital of France?"})
print(response)

from unittest.mock import patch
from mcp.shared.memory import (
    create_connected_server_and_client_session as client_session,
)
from mcp.types import (
    BlobResourceContents,
    ImageContent,
    TextContent,
    TextResourceContents,
)
mock_log = None
client = None

from mcp.server.session import ServerSession

async def setup_logging_tool_test(mcp):
    global mock_log, client
    patcher = patch("mcp.server.session.ServerSession.send_log_message")
    mock_log = patcher.start()
    session = client_session(mcp._mcp_server)
    client = await session.__aenter__()
    return patcher, session

async def teardown_logging_tool_test(patcher, session):
    await session.__aexit__(None, None, None)
    patcher.stop()

async def test_logging_tool(mcp):
    patcher, session = await setup_logging_tool_test(mcp)
    try:
        result = await client.call_tool("logging_tool", {"msg": "test"})
        assert len(result.content) == 1
        content = result.content[0]
        assert isinstance(content, TextContent)
        assert "Logged messages for test" in content.text

        assert mock_log.call_count == 4
    finally:
        await teardown_logging_tool_test(patcher, session)

from llama_index.core.agent.workflow import ReActAgent

from mcp_servers.shopping_cart.utils import setup_models, load_documents
from utils.custom_embeddings import OpenVINOSentenceTransformerEmbedding
from dotenv import load_dotenv
load_dotenv()
import os
from llama_index.core import Settings
from pathlib import Path
# initialize embedding model
embedding_model_path = os.getenv("EMBEDDING_MODEL_PATH")

_embedding = OpenVINOSentenceTransformerEmbedding(model_path=str(embedding_model_path), device="CPU")
Settings.embed_model = _embedding

# load document
DOCUMENT_PATH = os.getenv("DOCUMENT_PATH")
index = load_documents(Path(DOCUMENT_PATH))

retriever = index.as_retriever(verbose=False)

Settings.llm = None
query_engine = index.as_query_engine(llm=None)
response = query_engine.query("what paint is the best for kitchens?")

print(response.response.strip())

retrieve_res = retriever.retrieve("what paint is the best for kitchens?")
for res in retrieve_res:
    print(res.node.get_content().strip())
    print("-------")

response = [res.node.get_content() for res in retrieve_res]
print("\n\n".join(response))