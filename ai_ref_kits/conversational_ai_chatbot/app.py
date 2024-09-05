import argparse
import logging as log
import os
import threading
import time
from pathlib import Path
from threading import Thread
from typing import Tuple, List, Optional, Set

import faiss
import gradio as gr
import librosa
import numpy as np
import openvino as ov
import torch
from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core import Document, VectorStoreIndex, Settings, StorageContext
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.chat_engine.types import BaseChatEngine, ChatMode
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.node_parser import LangchainNodeParser
from llama_index.embeddings.huggingface_openvino import OpenVINOEmbedding
from llama_index.llms.openvino import OpenVINOLLM
from llama_index.postprocessor.openvino_rerank import OpenVINORerank
from llama_index.readers.file import PDFReader
from llama_index.vector_stores.faiss import FaissVectorStore
from optimum.intel import OVModelForSpeechSeq2Seq
from transformers import AutoProcessor, TextIteratorStreamer, pipeline

# Global variables initialization
TARGET_AUDIO_SAMPLE_RATE = 16000
SYSTEM_CONFIGURATION = (
    "You are Adrishuo - a helpful, respectful, and honest hotel concierge. "
    "Your role is talking to a guest who just came in to the hotel."
    "Your primary role is to answer questions about hotel rules and city. "
    "You must take into account the provided context information. "
    "If context information is empty, tell guest that hotel guide is missing and stop answering questions. "
)
GREET_THE_CUSTOMER = "Please introduce yourself and greet the hotel guest"

EXAMPLE_PDF_PATH = os.path.join(os.path.dirname(__file__), "Grand_Azure_Resort_Spa_Full_Guide.pdf")
MODEL_DIR = Path("model")
inference_lock = threading.Lock()

# Initialize Model variables
asr_model: Optional[OVModelForSpeechSeq2Seq] = None
asr_processor: Optional[AutoProcessor] = None
ov_llm: Optional[OpenVINOLLM] = None
ov_embedding: Optional[OpenVINOEmbedding] = None
ov_reranker: Optional[OpenVINORerank] = None
ov_chat_engine: Optional[BaseChatEngine] = None

# todo temporary
synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")
speaker_embedding = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")[2534]["xvector"]


def get_available_devices() -> Set[str]:
    """
    List all devices available for inference

    Returns:
        Set of available devices
    """
    core = ov.Core()
    return {device.split(".")[0] for device in core.available_devices}


def load_asr_model(model_dir: Path) -> None:
    """
    Load automatic speech recognition model and assign it to a global variable

    Params:
        model_dir: dir with the ASR model
    """
    global asr_model, asr_processor

    if not model_dir.exists():
        log.error(f"Cannot find {model_dir}")
        return

    device = "AUTO:GPU,CPU" if ov.__version__ < "2024.3" else "AUTO:CPU"
    # create a distil-whisper model and its processor
    asr_model = OVModelForSpeechSeq2Seq.from_pretrained(model_dir, device=device)
    asr_processor = AutoProcessor.from_pretrained(model_dir)

    model_name = model_dir.name
    log.info(f"Running {model_name} on {','.join(asr_model.encoder.request.get_property('EXECUTION_DEVICES'))}")


def load_chat_model(model_dir: Path) -> Optional[OpenVINOLLM]:
    """
    Load chat model

    Params:
        model_dir: dir with the chat model
    Returns:
        OpenVINO LLM model in LLama Index
    """
    if not model_dir.exists():
        log.error(f"Cannot find {model_dir}")
        return None

    ov_config = {'PERFORMANCE_HINT': 'LATENCY', 'NUM_STREAMS': '1', "CACHE_DIR": ""}
    # load llama model and its tokenizer in the format of Llama Index
    return OpenVINOLLM(context_window=2048, model_id_or_path=str(model_dir), max_new_tokens=512, device_map="AUTO:GPU,CPU",
                       model_kwargs={"ov_config": ov_config}, generate_kwargs={"do_sample": True, "temperature": 0.7, "top_k": 50, "top_p": 0.95})


def load_embedding_model(model_dir: Path) -> Optional[OpenVINOEmbedding]:
    """
    Load embedding model

    Params:
        model_dir: dir with the embedding model
    Returns:
        OpenVINO Embedding model in LLama Index
    """
    if not model_dir.exists():
        log.error(f"Cannot find {model_dir}")
        return None

    device = "AUTO:NPU" if "NPU" in get_available_devices() else "AUTO:CPU"
    # load embedding model in the format of Llama Index
    return OpenVINOEmbedding(str(model_dir), device=device, embed_batch_size=1, model_kwargs={"dynamic_shapes": False})


def load_reranker_model(model_dir: Path) -> Optional[OpenVINORerank]:
    """
    Load embedding model

    Params:
        model_dir: dir with the reranker model
    Returns:
        OpenVINO Reranker model in LLama Index
    """
    if not model_dir.exists():
        log.error(f"Cannot find {model_dir}")
        return None

    # load reranker model in the format of Llama Index
    return OpenVINORerank(model_id_or_path=str(model_dir), device="AUTO:CPU", top_n=3)


def load_rag_models(chat_model_dir: Path, embedding_model_dir: Path, reranker_model_dir: Path) -> None:
    """
    Load all models required in RAG pipeline

    Params:
        chat_model_dir: dir with the chat model
        embedding_model_dir: dir with the embedding model
        reranker_model_dir: dir with the reranker model
    """
    global ov_llm, ov_embedding, ov_reranker, ov_chat_engine

    # embedding model
    ov_embedding = load_embedding_model(embedding_model_dir)
    log.info(f"Running {embedding_model_dir} on {','.join(ov_embedding._model.request.get_property('EXECUTION_DEVICES'))}")

    # reranker model
    ov_reranker = load_reranker_model(reranker_model_dir)
    log.info(f"Running {reranker_model_dir} on {','.join(ov_reranker._model.request.get_property('EXECUTION_DEVICES'))}")

    # chat model
    ov_llm = load_chat_model(chat_model_dir)
    log.info(f"Running {chat_model_dir} on {','.join(ov_llm._model.request.get_compiled_model().get_property('EXECUTION_DEVICES'))}")

    # chat engine
    ov_chat_engine = SimpleChatEngine.from_defaults(llm=ov_llm, system_prompt=SYSTEM_CONFIGURATION,
                                                    memory=ChatMemoryBuffer.from_defaults())


def load_file(file_path: Path) -> Document:
    """
    Load text or pdf document

    Params:
        file_path: the path to the document
    Returns:
        A document in LLama Index format
    """
    # file extension
    ext = file_path.suffix
    if ext == ".pdf":
        reader = PDFReader()
        return reader.load_data(file_path)[0]
    elif ext == ".txt":
        with open(file_path) as f:
            content = f.read()
            return Document(text=content, metadata={"file_name": file_path.name})
    else:
        raise ValueError(f"{ext} file is not supported for now")


def load_context(file_path: str) -> None:
    """
    Load context (document) and create a RAG pipeline

    Params:
        file_path: the path to the document
    """
    global ov_chat_engine

    # limit chat history to 3000 tokens
    memory = ChatMemoryBuffer.from_defaults()

    # when context removed, no longer RAG pipeline is needed
    if not file_path:
        ov_chat_engine = SimpleChatEngine.from_defaults(llm=ov_llm, system_prompt=SYSTEM_CONFIGURATION, memory=memory)
        return

    document = load_file(Path(file_path))

    # a splitter to divide document into chunks
    splitter = LangchainNodeParser(RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100))

    dim = ov_embedding._model.request.outputs[0].get_partial_shape()[2].get_length()
    # a memory database to store chunks
    faiss_index = faiss.IndexFlatL2(dim)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # set embedding model
    Settings.embed_model = ov_embedding
    index = VectorStoreIndex.from_documents([document], storage_context, transformations=[splitter])
    # create a RAG pipeline
    ov_chat_engine = index.as_chat_engine(llm=ov_llm, chat_mode=ChatMode.CONTEXT, system_prompt=SYSTEM_CONFIGURATION,
                                          memory=memory, node_postprocessors=[ov_reranker])


def generate_initial_greeting() -> str:
    """
    Generates customer/patient greeting

    Returns:
        Generated greeting
    """
    return ov_chat_engine.chat(GREET_THE_CUSTOMER).response


def chat(history: List[List[str]]) -> List[List[str]]:
    """
    Chat function. It generates response based on a prompt

    Params:
        history: history of the messages (conversation) so far
    Returns:
        History with the latest chat's response (yields partial response)
    """
    # no document is loaded
    if isinstance(ov_chat_engine, SimpleChatEngine):
        history[-1][1] = "No guide is provided, so I cannot answer this question. Please upload the hotel guide."
        yield history
        return

    # get token by token and merge to the final response
    history[-1][1] = ""
    with inference_lock:
        start_time = time.time()

        chat_streamer = ov_chat_engine.stream_chat(history[-1][0]).response_gen
        for partial_text in chat_streamer:
            history[-1][1] += partial_text
            # "return" partial response
            yield history

        end_time = time.time()
        log.info(f"Chat model response time: {end_time - start_time:.2f} seconds")


def transcribe(audio: Tuple[int, np.ndarray], prompt: str, conversation: List[List[str]]) -> List[List[str]]:
    """
    Transcribe audio to text

    Params:
        audio: audio to transcribe text from
        conversation: conversation history with the chatbot
    Returns:
        User prompt as a text
    """
    # if audio is available, use audio, otherwise, use given text
    if audio:
        start_time = time.time()  # Start time for ASR process

        sample_rate, audio = audio
        # the whisper model requires 16000Hz, not 44100Hz
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sample_rate, target_sr=TARGET_AUDIO_SAMPLE_RATE)\
            .astype(np.int16)

        # get input features from the audio
        input_features = asr_processor(audio, sampling_rate=TARGET_AUDIO_SAMPLE_RATE, return_tensors="pt").input_features

        # use streamer to show transcription word by word
        text_streamer = TextIteratorStreamer(asr_processor, skip_prompt=True, skip_special_tokens=True)

        # transcribe in the background to deliver response token by token
        thread = Thread(target=asr_model.generate, kwargs={"input_features": input_features, "streamer": text_streamer})
        thread.start()

        conversation.append(["", None])
        # get token by token and merge to the final response
        for partial_text in text_streamer:
            conversation[-1][0] += partial_text
            # "return" partial response
            yield conversation

        end_time = time.time()  # End time for ASR process
        log.info(f"ASR model response time: {end_time - start_time:.2f} seconds")  # Print the ASR processing time

        # wait for the thread
        thread.join()
    else:
        conversation.append([prompt, None])
        yield conversation

    return conversation


def synthesize(conversation: List[List[str]]) -> Tuple[int, np.ndarray]:
    """
    Synthesizes speech from chatbot's response

    Params:
        conversation: conversation history with the chatbot
    Returns:
        Chatbot voice response (audio)
    """
    prompt = conversation[-1][1]

    start_time = time.time()

    # todo: replace with openvino pipeline
    speech = synthesiser(prompt, forward_params={"speaker_embeddings": torch.tensor(speaker_embedding).unsqueeze(0)})

    end_time = time.time()
    log.info(f"TTS model response time: {end_time - start_time:.2f} seconds")

    return speech["sampling_rate"], speech["audio"]


def create_UI(initial_message: str) -> gr.Blocks:
    """
    Create web user interface

    Params:
        initial_message: message to start with
    Returns:
        Demo UI
    """
    with gr.Blocks(title="Adrishuo - the Conversational AI Chatbot") as demo:
        gr.Markdown("""
        # Adrishuo: A Conversational AI Hotel Concierge running with OpenVINO

        Instructions for use:
        1. Attach the PDF or TXT file with the hotel guide (see "Grand_Azure_Resort_Spa_Full_Guide.pdf" as an example)
        2. Record your question/comment using the first audio widget ("Your voice input") or type it in the textbox ("Your text input"), then click Submit
        3. Wait for the chatbot to respond ("Chatbot") and say it aloud ("Chatbot voice response")
        4. Discuss with the chatbot and ask questions about the hotel rules and city places
        """)
        with gr.Row():
            with gr.Column(scale=1):
                file_uploader_ui = gr.File(label="Hotel guide", file_types=[".pdf", ".txt"], value=EXAMPLE_PDF_PATH)
                input_audio_ui = gr.Audio(sources=["microphone"], label="Your voice input")
                input_text_ui = gr.Textbox(label="Your text input")
                submit_btn = gr.Button("Submit", variant="primary", interactive=False)
            with gr.Column(scale=2):
                chatbot_ui = gr.Chatbot(value=[[None, initial_message]], label="Chatbot")
                output_audio_ui = gr.Audio(label="Chatbot voice response", autoplay=True)
                clear_btn = gr.Button("Start over", variant="secondary")

        # events
        # block submit button when no audio or text input
        gr.on(triggers=[input_audio_ui.change, input_text_ui.change], inputs=[input_audio_ui, input_text_ui], outputs=submit_btn,
              fn=lambda x, y: gr.Button(interactive=True) if bool(x) ^ bool(y) else gr.Button(interactive=False))

        file_uploader_ui.change(lambda: [[None, initial_message]], outputs=chatbot_ui) \
            .then(load_context, inputs=file_uploader_ui)

        clear_btn.click(lambda: [[None, initial_message]], outputs=chatbot_ui) \
            .then(lambda: gr.Button(interactive=False), outputs=clear_btn)

        # block buttons, clear output audio, do the transcription and conversation, clear input audio, unblock buttons
        gr.on(triggers=[submit_btn.click, input_text_ui.submit], fn=lambda: gr.Button(interactive=False), outputs=submit_btn) \
            .then(lambda: gr.Button(interactive=False), outputs=clear_btn) \
            .then(lambda: None, outputs=output_audio_ui) \
            .then(transcribe, inputs=[input_audio_ui, input_text_ui, chatbot_ui], outputs=chatbot_ui) \
            .then(lambda: None, outputs=input_text_ui) \
            .then(chat, chatbot_ui, chatbot_ui) \
            .then(synthesize, inputs=chatbot_ui, outputs=output_audio_ui) \
            .then(lambda: None, outputs=input_audio_ui) \
            .then(lambda: gr.Button(interactive=True), outputs=clear_btn)

        return demo


def run(asr_model_dir: Path, chat_model_dir: Path, embedding_model_dir: Path, reranker_model_dir: Path, public_interface: bool = False) -> None:
    """
    Run the chatbot application

    Params
        asr_model_dir: dir with the automatic speech recognition model
        chat_model_dir: dir with the chat model
        embedding_model_dir: dir with the embedding model
        reranker_model_dir: dir with the reranker model
        public_interface: whether UI should be available publicly
    """
    # set up logging
    log.getLogger().setLevel(log.INFO)

    # load whisper model
    load_asr_model(asr_model_dir)
    # load chat models
    load_rag_models(chat_model_dir, embedding_model_dir, reranker_model_dir)

    if asr_model is None or ov_llm is None or ov_embedding is None:
        log.error("Required models are not loaded. Exiting...")
        return

    # get initial greeting
    initial_message = generate_initial_greeting()

    # load initial context
    load_context(EXAMPLE_PDF_PATH)

    # create user interface
    demo = create_UI(initial_message)
    # launch demo
    demo.queue().launch(share=public_interface)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asr_model", type=str, default="model/distil-whisper-large-v3-INT8", help="Path of the automatic speech recognition model directory")
    parser.add_argument("--chat_model", type=str, default="model/llama3.1-8B-INT4", help="Path to the chat model directory")
    parser.add_argument("--embedding_model", type=str, default="model/bge-small-FP32", help="Path to the embedding model directory")
    parser.add_argument("--reranker_model", type=str, default="model/bge-reranker-large-FP32", help="Path to the reranker model directory")
    parser.add_argument("--public", default=False, action="store_true", help="Whether interface should be available publicly")

    args = parser.parse_args()
    run(Path(args.asr_model), Path(args.chat_model), Path(args.embedding_model), Path(args.reranker_model), args.public)
