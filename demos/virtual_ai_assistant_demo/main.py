import argparse
import logging as log
import os
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional, Set, Tuple

import chromadb
import fitz
import gradio as gr
import numpy as np
import openvino as ov
import yaml
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core import Document, VectorStoreIndex, Settings, StorageContext
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.chat_engine.types import BaseChatEngine, ChatMode
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.node_parser import LangchainNodeParser
from llama_index.embeddings.huggingface_openvino import OpenVINOEmbedding
from llama_index.llms.openvino_genai import OpenVINOGenAILLM
from llama_index.postprocessor.openvino_rerank import OpenVINORerank
from llama_index.vector_stores.chroma import ChromaVectorStore
from openvino.runtime import opset10 as ops
from openvino.runtime import passes
from optimum.exporters.openvino.convert import export_tokenizer
from optimum.intel import OVModelForCausalLM, OVModelForFeatureExtraction, OVWeightQuantizationConfig, OVModelForSequenceClassification
from transformers import AutoTokenizer

SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils")
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils import demo_utils as utils

# Global variables initialization
MODEL_DIR = Path("model")
inference_lock = threading.Lock()

# Initialize Model variables
ov_llm: Optional[OpenVINOGenAILLM] = None
ov_embedding: Optional[OpenVINOEmbedding] = None
ov_reranker: Optional[OpenVINORerank] = None
ov_chat_engine: Optional[BaseChatEngine] = None

chatbot_config = {}


def get_available_devices() -> Set[str]:
    core = ov.Core()
    return {device.split(".")[0] for device in core.available_devices}


def load_chat_model(model_name: str, token: str = None) -> OpenVINOGenAILLM:
    model_path = MODEL_DIR / model_name    
    log.info(f"model_path: {model_path}")
    # tokenizers are disabled anyway, this allows to avoid warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if token is not None:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token

    ov_config = {"PERFORMANCE_HINT": "LATENCY", "CACHE_DIR": "", "NUM_STREAMS": 1, "INFERENCE_NUM_THREADS": 1}
    # load llama model and its tokenizer
    if not model_path.exists():
        log.info(f"Downloading {model_name}... It may take up to 1h depending on your Internet connection and model size.")     
        
        chat_tokenizer = AutoTokenizer.from_pretrained(model_name, token=token, use_fast=True)
        chat_tokenizer.save_pretrained(model_path)
        export_tokenizer(chat_tokenizer, model_path)

        # openvino models are used as is
        is_openvino_model = model_name.split("/")[0] == "OpenVINO"
        if is_openvino_model:
            chat_model = OVModelForCausalLM.from_pretrained(model_name, export=False, compile=False, token=token)
            chat_model.save_pretrained(model_path)
        else:
            log.info(f"Loading and quantizing {model_name} to INT4...")
            log.info(f"Quantizing {model_name} to INT4... It may take significant amount of time depending on your machine power.")
            quant_config = OVWeightQuantizationConfig(bits=8, sym=False)
            log.info(f"OVWeightQuantizationConfig is created")
            chat_model = OVModelForCausalLM.from_pretrained(model_name, export=True, compile=False, quantization_config=quant_config,
                                                            token=token, trust_remote_code=True, library_name="transformers")
            log.info(f"OVModelForCausalLM.from_pretrained is passed")
            chat_model.save_pretrained(model_path)
            log.info(f"save_pretrained is passed")

    device = "GPU" if "GPU" in get_available_devices() else "CPU"

    log.info(f"OpenVINOGenAILLM before")
    llm = OpenVINOGenAILLM(model_path=str(model_path), device=device, config=ov_config)
    log.info(f"OpenVINOGenAILLM passed")

    # change number of tokens to be generated in one step
    llm._streamer.tokens_len = 1

    llm.config.max_new_tokens = 1024
    llm.config.do_sample = True
    llm.config.temperature = 0.7
    llm.config.top_k = 50
    llm.config.top_p = 0.95
    log.info(f"load_chat_model passed")

    return llm


def optimize_model_for_npu(model: OVModelForFeatureExtraction):
    class ReplaceTensor(passes.MatcherPass):
        def __init__(self, packed_layername_tensor_dict_list):
            super().__init__()
            self.model_changed = False

            param = passes.WrapType("opset10.Multiply")

            def callback(matcher: passes.Matcher) -> bool:
                root = matcher.get_match_root()
                if root is None:
                    return False
                for y in packed_layername_tensor_dict_list:
                    root_name = root.get_friendly_name()
                    if root_name.find(y["name"]) != -1:
                        max_fp16 = np.array([[[[-np.finfo(np.float16).max]]]]).astype(np.float32)
                        new_tenser = ops.constant(max_fp16, ov.Type.f32, name="Constant_4431")
                        root.set_arguments([root.input_value(0).node, new_tenser])
                        packed_layername_tensor_dict_list.remove(y)

                return True

            self.register_matcher(passes.Matcher(param, "ReplaceTensor"), callback)

    packed_layer_tensor_dict_list = [{"name": "aten::mul/Multiply"}]

    manager = passes.Manager()
    manager.register_pass(ReplaceTensor(packed_layer_tensor_dict_list))
    manager.run_passes(model.model)
    model.reshape(1, 512)


def load_embedding_model(model_name: str) -> OpenVINOEmbedding:
    model_path = MODEL_DIR / model_name

    if not model_path.exists():
        embedding_model = OVModelForFeatureExtraction.from_pretrained(model_name, export=True, compile=False)
        optimize_model_for_npu(embedding_model)
        embedding_model.save_pretrained(model_path)
        embedding_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        embedding_tokenizer.save_pretrained(model_path)

    device = "NPU" if "NPU" in get_available_devices() else "CPU"
    return OpenVINOEmbedding(str(model_path), device=device, embed_batch_size=1, model_kwargs={"dynamic_shapes": False})


def load_reranker_model(model_name: str) -> OpenVINORerank:
    model_path = MODEL_DIR / model_name

    if not model_path.exists():
        reranker_model = OVModelForSequenceClassification.from_pretrained(model_name, export=True, compile=False)
        reranker_model.save_pretrained(model_path)
        reranker_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        reranker_tokenizer.save_pretrained(model_path)

    return OpenVINORerank(model_id_or_path=str(model_path), device="CPU", top_n=3)


def load_chat_models(chat_model_name: str, embedding_model_name: str, reranker_model_name: str, personality_file_path: Path, auth_token: str = None) -> None:
    global ov_llm, ov_chat_engine, ov_embedding, chatbot_config, ov_reranker

    with open(personality_file_path, "rb") as f:
        chatbot_config = yaml.safe_load(f)

    ov_llm = load_chat_model(chat_model_name, auth_token)
    log.info(f"Running {chat_model_name} on {ov_llm.device}")
    ov_embedding = load_embedding_model(embedding_model_name)
    log.info(f"Running {embedding_model_name} on {ov_embedding._model.request.get_property('EXECUTION_DEVICES')}")   
    ov_reranker = load_reranker_model(reranker_model_name)
    log.info(f"Running {reranker_model_name} on {','.join(ov_reranker._model.request.get_property('EXECUTION_DEVICES'))}")

    ov_chat_engine = SimpleChatEngine.from_defaults(llm=ov_llm, system_prompt=chatbot_config["system_configuration"],
                                                    memory=ChatMemoryBuffer.from_defaults())


def load_files(file_paths: List[str]) -> list[Document]:
    documents = []
    for file_path in map(lambda x: Path(x), file_paths):
        ext = file_path.suffix
        if ext == ".pdf":
            # Using PyMuPDF (fitz) to read PDF content
            text = ""
            with fitz.open(file_path) as pdf:
                for page in pdf:
                    text += page.get_text("text") + "\n"  # Extract text from each page
                # remove non-breaking space
                text.replace("\xa0", " ")
                documents.append(Document(text=text, metadata={"file_name": file_path.name}))

        elif ext == ".txt":
            # Reading text files as usual
            with open(file_path, "rb") as f:
                content = f.read()
                documents.append(Document(text=content, metadata={"file_name": file_path.name}))

        else:
            log.warning(f"{ext} file is not supported for now. Skipping {file_path.name}")

    return documents


def load_context(file_paths: List[str]) -> None:
    global ov_chat_engine

    # limit chat history to 1024 tokens
    memory = ChatMemoryBuffer.from_defaults(token_limit=2048)

    if not file_paths:
        ov_chat_engine = SimpleChatEngine.from_defaults(llm=ov_llm, system_prompt=chatbot_config["system_configuration"], memory=memory)
        return

    documents = load_files(file_paths)

    # a splitter to divide document into chunks
    splitter = LangchainNodeParser(RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100))

    chroma_client = chromadb.EphemeralClient()
    chroma_collection = chroma_client.create_collection("ai_assistant_collection")

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(documents, storage_context, embed_model=ov_embedding, transformations=[splitter])
    # create a RAG pipeline
    ov_chat_engine = index.as_chat_engine(llm=ov_llm, chat_mode=ChatMode.CONTEXT, system_prompt=chatbot_config["system_configuration"],
                                          memory=memory, node_postprocessors=[ov_reranker])


# this is necessary for thinking models e.g. deepseek
def emphasize_thinking_mode(token: str) -> str:
    return token + "<em><small>" if "<think>" in token else "</small></em>" + token if "</think>" in token else token


def generate_initial_greeting() -> str:
    response = ""
    for token in ov_chat_engine.stream_chat(chatbot_config["greet_the_user_prompt"]).response_gen:
        response += emphasize_thinking_mode(token)
    return response


def chat(history: List[List[str]]) -> Tuple[List[List[str]], float]:
    # get token by token and merge to the final response
    history[-1][1] = ""
    with inference_lock:
        chat_streamer = ov_chat_engine.stream_chat(history[-1][0]).response_gen

        # generate first token independently
        first_token = next(chat_streamer)
        history[-1][1] += emphasize_thinking_mode(first_token)
        yield history, 0.0

        # generate next tokens
        tokens = ov_chat_engine._llm._streamer.tokens_len
        start_time = time.time()
        for partial_text in chat_streamer:
            history[-1][1] += emphasize_thinking_mode(partial_text)
            processing_time = time.time() - start_time
            tokens += ov_chat_engine._llm._streamer.tokens_len
            # "return" partial response
            yield history, round(tokens / processing_time, 2)

        end_time = time.time()

        processing_time = end_time - start_time
        log.info(f"Chat model response time: {processing_time:.2f} seconds ({tokens / processing_time:.2f} tokens/s)")
        yield history, round(tokens / processing_time, 2)


def transcribe(prompt: str, conversation: List[List[str]]) -> List[List[str]]:
    conversation.append([prompt, None])
    return conversation


def extra_action(conversation: List) -> Tuple[str, float]:
    conversation.append([chatbot_config["extra_action_prompt"], None])
    for partial_summary, performance in chat(conversation):
        yield f"## Summary\n\n" + partial_summary[-1][1], performance


def create_UI(initial_message: str, action_name: str) -> gr.Blocks:
    qr_code = utils.get_qr_code("https://github.com/openvinotoolkit/openvino_build_deploy/tree/master/demos/virtual_ai_assistant_demo", size=200)

    with gr.Blocks(theme=utils.gradio_intel_theme(), title="Your Virtual AI Assistant") as demo:
        utils.gradio_intel_header(chatbot_config["assistant_name"])
        with gr.Row():
            with gr.Column(scale=7):
                gr.Markdown(chatbot_config["instructions"])
            gr.Image(qr_code, interactive=False, label="Get Demo Code", scale=1)

        with gr.Row():
            file_uploader_ui = gr.Files(label="Additional context", file_types=[".pdf", ".txt"], scale=1)
            with gr.Column(scale=4):
                chatbot_ui = gr.Chatbot(value=[[None, initial_message]], label="Chatbot", sanitize_html=False,
                                        avatar_images=(None, "https://docs.openvino.ai/2025/_static/favicon.ico"))
                with gr.Row():
                    input_text_ui = gr.Textbox(label="Your text input", scale=6)
                    submit_btn = gr.Button("Submit", variant="primary", interactive=False, scale=1)
                with gr.Row():
                    tps_text_ui = gr.Text("", label="Performance (tokens/s)", type="text", scale=6)
                    with gr.Column(scale=1):
                        clear_btn = gr.Button("Start over", variant="secondary")
                        extra_action_button = gr.Button(action_name, variant="primary", interactive=False)
        summary_ui = gr.Markdown(sanitize_html=False)

        # events
        # block submit button when no audio or text input
        gr.on(triggers=input_text_ui.change, inputs=input_text_ui, outputs=submit_btn,
              fn=lambda x: gr.Button(interactive=True) if bool(x) else gr.Button(interactive=False))

        file_uploader_ui.change(lambda: ([[None, initial_message]], None), outputs=[chatbot_ui, summary_ui]) \
            .then(load_context, inputs=file_uploader_ui)

        clear_btn.click(lambda: ([[None, initial_message]], None, None), outputs=[chatbot_ui, summary_ui, tps_text_ui]) \
            .then(load_context, inputs=file_uploader_ui) \
            .then(lambda: gr.Button(interactive=False), outputs=extra_action_button)

        # block buttons, do the transcription and conversation, clear audio, unblock buttons
        gr.on(triggers=[submit_btn.click, input_text_ui.submit], fn=lambda: gr.Button(interactive=False), outputs=submit_btn) \
            .then(lambda: gr.Button(interactive=False), outputs=extra_action_button) \
            .then(lambda: gr.Button(interactive=False), outputs=clear_btn) \
            .then(transcribe, inputs=[input_text_ui, chatbot_ui], outputs=chatbot_ui) \
            .then(lambda: None, outputs=input_text_ui) \
            .then(chat, inputs=chatbot_ui, outputs=[chatbot_ui, tps_text_ui]) \
            .then(lambda: gr.Button(interactive=True), outputs=clear_btn) \
            .then(lambda: gr.Button(interactive=True), outputs=extra_action_button)

        # block button, do the action, unblock button
        extra_action_button.click(lambda: gr.Button(interactive=False), outputs=extra_action_button) \
            .then(lambda: gr.Button(interactive=False), outputs=clear_btn) \
            .then(extra_action, inputs=chatbot_ui, outputs=[summary_ui, tps_text_ui]) \
            .then(lambda: gr.Button(interactive=True), outputs=clear_btn) \
            .then(lambda: gr.Button(interactive=True), outputs=extra_action_button)

        return demo


def run(chat_model_name: str, embedding_model_name: str, reranker_model_name: str, personality_file_path: Path, hf_token: str = None, local_network: bool = False, public_interface: bool = False) -> None:
    server_name = "0.0.0.0" if local_network else None

    # load chat models
    load_chat_models(chat_model_name, embedding_model_name, reranker_model_name, personality_file_path, hf_token)

    # get initial greeting
    initial_message = generate_initial_greeting()

    # create user interface
    demo = create_UI(initial_message, chatbot_config["extra_action_name"])
    # launch demo
    print("Demo is ready!", flush=True) # Required for the CI to detect readiness
    demo.queue().launch(server_name=server_name, share=public_interface)


if __name__ == "__main__":
    # set up logging
    log.getLogger().setLevel(log.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--chat_model", type=str, default="meta-llama/Llama-3.2-1B", help="Path/name of the chat model")
    parser.add_argument("--embedding_model", type=str, default="BAAI/bge-small-en-v1.5", help="Path/name of the model for embeddings")
    parser.add_argument("--reranker_model", type=str, default="BAAI/bge-reranker-base", help="Path/name of the reranker model")
    parser.add_argument("--personality", type=str, default="healthcare_personality.yaml", help="Path to the YAML file with chatbot personality")
    parser.add_argument("--hf_token", type=str, help="HuggingFace access token to get Llama3")
    parser.add_argument("--public", default=False, action="store_true", help="Whether interface should be available publicly")
    parser.add_argument("--local_network", action="store_true", help="Whether demo should be available in local network")

    args = parser.parse_args()
    run(args.chat_model, args.embedding_model, args.reranker_model, Path(args.personality), args.hf_token, args.local_network, args.public)
