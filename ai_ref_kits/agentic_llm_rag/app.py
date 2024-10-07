from pathlib import Path
import huggingface_hub as hf_hub
from llama_index.llms.openvino import OpenVINOLLM
import openvino.properties as props
import openvino.properties.hint as hints
import openvino.properties.streams as streams
from llama_index.embeddings.huggingface_openvino import OpenVINOEmbedding
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex, Settings
from pathlib import Path
import requests
import io
from io import StringIO 
from create_tools import *
import sys

llm_model_id = "OpenVINO/Phi-3-mini-4k-instruct-int4-ov"
#TBD enable latest Llama model
llm_device = "GPU"
embedding_device = "AUTO"
#Load the model
llm_model_path = llm_model_id.split("/")[-1]
repo_name = llm_model_id.split("/")[0]

embedding_model_id = "BAAI/bge-small-en-v1.5"
embedding_model_path = "bge-small-en-v1.5"

text_example_en_path = Path("text_example_en.pdf")
text_example_en = "https://github.com/user-attachments/files/16171326/xeon6-e-cores-network-and-edge-brief.pdf"

if not text_example_en_path.exists():
    r = requests.get(url=text_example_en)
    content = io.BytesIO(r.content)
    with open("text_example_en.pdf", "wb") as f:
        f.write(content.read())

ov_config = {hints.performance_mode(): hints.PerformanceMode.LATENCY, streams.num(): "1", props.cache_dir(): ""}

def phi_completion_to_prompt(completion):
    return f"<|system|><|end|><|user|>{completion}<|end|><|assistant|>\n"

def llama3_completion_to_prompt(completion):
    return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{completion}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"


llm = OpenVINOLLM(
    model_id_or_path=str(llm_model_path),
    context_window=3900,
    max_new_tokens=1000,
    model_kwargs={"ov_config": ov_config},
    generate_kwargs={"do_sample": False, "temperature": None, "top_p": None},
    completion_to_prompt=phi_completion_to_prompt if llm_model_path == "Phi-3-mini-4k-instruct-int4-ov" else llama3_completion_to_prompt,
    device_map=llm_device,
)


embedding = OpenVINOEmbedding(model_id_or_path=embedding_model_path, device=embedding_device)


    
multiply_tool = FunctionTool.from_defaults(fn=multiply)



divide_tool = FunctionTool.from_defaults(fn=divide)

Settings.embed_model = embedding
Settings.llm = llm

reader = SimpleDirectoryReader(input_files=[text_example_en_path])
documents = reader.load_data()
index = VectorStoreIndex.from_documents(
    documents,
)

vector_tool = QueryEngineTool(
    index.as_query_engine(streaming=True),
    metadata=ToolMetadata(
        name="vector_search",
        description="Useful for searching for basic facts about 'Intel Xeon 6 processors'",
    ),
)

import nest_asyncio
nest_asyncio.apply()

agent = ReActAgent.from_tools([multiply_tool, divide_tool, vector_tool], llm=llm, verbose=True)


### Credit: Modified from llamaindex-packs: https://docs.llamaindex.ai/en/stable/api_reference/packs/gradio_agent_chat/
### We can remove this reference once we are finished with the app

class Capturing(list):
    #Define a class for capturing stdout from LlamaIndex React Agent
    #TBD - is there another option for obtaining info step by step from agent
    #Please see langchain example: https://www.gradio.app/guides/agents-and-tool-usage
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout
        
def _handle_user_message(user_message, history):
    """Handle the user submitted message. Clear message box, and append
    to the history.
    """
    return "", [*history, (user_message, "")]

def _generate_response(chat_history):
    """Generate the response from agent, and capture the stdout of the
    ReActAgent's thoughts.
    """

    with Capturing() as output:
        """
        TBD - instead of an stdout module, can we access the agent's "steps" directly? 
        Should we look at function calling?
        TBD - we could also directly append the prompt "Go step by step, 
        using a tool to do any math and a tool to gather information with the document"
        so user doesn't have to type it in
        """
        response = agent.stream_chat(chat_history[-1][0])
    #Note for dev team: Please see https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/agent/react/templates/system_header_template.md on the system template
    for token in response.response_gen:
        chat_history[-1][1] += token
        yield chat_history, str(output)

def _reset_chat():
    """Reset the agent's chat history. And clear all dialogue boxes."""
    # clear agent history
    agent.reset()
    return "", "", ""  # clear textboxes

def run():
    """Run the pipeline."""
    import gradio as gr
    
    with gr.Blocks() as demo:
        gr.Markdown("# Smart Retail Assistant 🤖: Agentic LLMs with RAG 💭")
        with gr.Row():
            chat_window = gr.Chatbot(
                label="Message History",
                scale=3,
                avatar_images=(
                None,
                "https://docs.openvino.ai/2024/_static/favicon.ico",
                ),
            )
            console = gr.Markdown()
        with gr.Row():
            message = gr.Textbox(label="Ask a question", scale=4)
            clear = gr.ClearButton()

        message.submit(
            _handle_user_message,
            [message, chat_window],
            [message, chat_window],
            queue=False,
        ).then(
            _generate_response,
            chat_window,
            [chat_window, console],
        )
        clear.click(_reset_chat, None, [message, chat_window, console])

    demo.launch()

run()