import argparse
import time
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
import requests
import io
from io import StringIO
from create_tools import Math
import sys
import gradio as gr
import nest_asyncio
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

llm_device = "GPU"
embedding_device = "GPU"

ov_config = {
    hints.performance_mode(): hints.PerformanceMode.LATENCY,
    streams.num(): "1",
    props.cache_dir(): ""
}

def phi_completion_to_prompt(completion):
    return f"<|system|><|end|><|user|>{completion}<|end|><|assistant|>\n"

def llama3_completion_to_prompt(completion):
    return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{completion}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

def setup_models(llm_model_path, embedding_model_path):
    # Load the Llama model locally
    llm = OpenVINOLLM(
        model_id_or_path=str(llm_model_path),
        context_window=3900,
        max_new_tokens=1000,
        model_kwargs={"ov_config": ov_config},
        generate_kwargs={"do_sample": False, "temperature": None, "top_p": None},
        completion_to_prompt=phi_completion_to_prompt if llm_model_path == "Phi-3-mini-4k-instruct-int4-ov" else llama3_completion_to_prompt,
        device_map=llm_device,
    )

    # Load the embedding model locally
    embedding = OpenVINOEmbedding(model_id_or_path=embedding_model_path, device=embedding_device)

    return llm, embedding

def setup_tools():
    multiply_tool = FunctionTool.from_defaults(fn=Math.multiply)
    divide_tool = FunctionTool.from_defaults(fn=Math.divide)

    return multiply_tool, divide_tool

def load_documents(text_example_en_path):
    # Check and download document if not present
    if not text_example_en_path.exists():
        text_example_en = "https://example.com/test_painting_llm_rag.pdf"  # Replace with valid URL
        r = requests.get(text_example_en)
        content = io.BytesIO(r.content)
        with open(text_example_en_path, "wb") as f:
            f.write(content.read())
    
    reader = SimpleDirectoryReader(input_files=[text_example_en_path])
    documents = reader.load_data()
    index = VectorStoreIndex.from_documents(documents)
    
    return index

def run_app(agent):
    class Capturing(list):
        def __enter__(self):
            self._stdout = sys.stdout
            sys.stdout = self._stringio = StringIO()
            return self
        def __exit__(self, *args):
            self.extend(self._stringio.getvalue().splitlines())
            del self._stringio
            sys.stdout = self._stdout

    def _handle_user_message(user_message, history):
        return "", [*history, (user_message, "")]

    def _generate_response(chat_history, log_history):
        if not isinstance(log_history, list):
            log_history = []

        # Capture time for thought process
        start_thought_time = time.time()

        # Capture the thought process output
        with Capturing() as output:
            response = agent.stream_chat(chat_history[-1][0])

        end_thought_time = time.time()
        thought_process_time = end_thought_time - start_thought_time
    
        # After response is complete, show the captured logs in the log area
        log_entries = "\n".join(output)
        thought_process_log = f"Thought Process Time: {thought_process_time:.2f} seconds"
        log_history.append(f"{log_entries}\n{thought_process_log}")

        yield chat_history, "\n".join(log_history)  # Yield after the thought process time is captured
            
        # Now capture response generation time
        start_response_time = time.time()

        # Gradually yield the response from the agent to the chat
        for token in response.response_gen:
            chat_history[-1][1] += token
            yield chat_history, "\n".join(log_history)  # Ensure log_history is a string

        end_response_time = time.time()
        response_time = end_response_time - start_response_time        

        # Log tokens per second along with the device information
        tokens = len(chat_history[-1][1].split(" ")) * 4 / 3  # Convert words to approx token count
        response_log = f"Response Time: {response_time:.2f} seconds ({tokens / response_time:.2f} tokens/s on {llm_device})"
    
        log.info(response_log)

        # Append the response time to log history
        log_history.append(response_log)
        yield chat_history, "\n".join(log_history)  # Join logs into a string for display

    def _reset_chat():
        agent.reset()
        return "", [], []  # Reset both chat and logs (initialize log as empty list)

    def purchase_click():
        return "Items are added to cart."

    def run():
        with gr.Blocks() as demo:

            gr.Markdown("# Smart Retail Assistant 🤖: Agentic LLMs with RAG 💭")
            gr.Markdown("Ask me about paint! 🎨")

            
            with gr.Row():
                chat_window = gr.Chatbot(
                    label="Paint Purchase Helper",
                    avatar_images=(None, "https://docs.openvino.ai/2024/_static/favicon.ico"),
		    height=400,  # Adjust height as per your preference
                    scale=3  # Set a higher scale value for Chatbot to make it wider
                   #autoscroll=True,  # Enable auto-scrolling for better UX
                )
                log_window = gr.Code(
                    label="Agent's Steps", 
                    language="python",
                    interactive=False,
                    scale=1  # Set lower scale to make it narrower than the Chatbot
                )

            
            with gr.Row():
                message = gr.Textbox(label="Ask the Paint Expert", scale=4)
                clear = gr.ClearButton()

            # Ensure that individual components are passed
            message.submit(
                _handle_user_message,
                inputs=[message, chat_window],
                outputs=[message, chat_window],
                queue=False,
            ).then(
                _generate_response,
                inputs=[chat_window, log_window],  # Pass individual components, including log_window
                outputs=[chat_window, log_window],  # Update chatbot and log window
            )
            clear.click(_reset_chat, None, [message, chat_window, log_window])
            
            gr.Markdown("------------------------------")
            gr.Markdown("### Purchase items")
            with gr.Row():
                gr.Dropdown(
                    ["Behr Premium Plus", "AwesomeSplash", "TheBrush", "PaintFinish"], 
                    multiselect=True, 
                    label="Items In-Stock", 
                    info="Which items would you like to purchase?"
                ),
                purchase = gr.Button(value="Purchase items")
                purchased_textbox = gr.Textbox()
                purchase.click(purchase_click, None, purchased_textbox)
        
        demo.launch(server_name='10.3.233.70', server_port=8694, share=True)

    run()


if __name__ == "__main__":
    
    # Define the argument parser at the end
    parser = argparse.ArgumentParser()
    parser.add_argument("--chat_model", type=str, default="model/llama3.1-8B-INT4", help="Path to the chat model directory")
    parser.add_argument("--embedding_model", type=str, default="model/bge-large-FP32", help="Path to the embedding model directory")
    args = parser.parse_args()

    # Load models and embedding based on parsed arguments
    llm, embedding = setup_models(args.chat_model, args.embedding_model)
    
    Settings.embed_model = embedding
    Settings.llm = llm

    # Set up tools
    multiply_tool, divide_tool = setup_tools()
    
    # Step 4: Load documents and create the VectorStoreIndex
    text_example_en_path = Path("test_painting_llm_rag.pdf")
    index = load_documents(text_example_en_path)

    vector_tool = QueryEngineTool(
        index.as_query_engine(streaming=True),
        metadata=ToolMetadata(
            name="vector_search",
            description="Useful for searching for facts and product recommendations about paint",
        ),
    )

    # Step 5: Initialize the agent with the loaded tools
    nest_asyncio.apply()
    agent = ReActAgent.from_tools([multiply_tool, divide_tool, vector_tool], llm=llm, verbose=True)

    # Step 6: Run the app
    run_app(agent)
