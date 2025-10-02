import argparse
import logging as log
import threading
import time
from pathlib import Path
from threading import Thread
from typing import Tuple, List, Optional, Set

import fitz  # PyMuPDF
import gradio as gr
import librosa
import numpy as np
import openvino as ov
import torch
import yaml
import nltk
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.chat_engine.types import BaseChatEngine, ChatMode
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.node_parser import LangchainNodeParser
from llama_index.embeddings.huggingface_openvino import OpenVINOEmbedding
from llama_index.llms.openvino import OpenVINOLLM
from llama_index.postprocessor.openvino_rerank import OpenVINORerank
from optimum.intel import OVModelForSpeechSeq2Seq
from transformers import AutoProcessor, TextIteratorStreamer
from melo.api import TTS

# Global variables initialization
TARGET_AUDIO_SAMPLE_RATE = 16000
TARGET_AUDIO_SAMPLE_RATE_TTS = 44100

MODEL_DIR = Path("model")
inference_lock = threading.Lock()

# Initialize Model variables
asr_model: Optional[OVModelForSpeechSeq2Seq] = None
asr_processor: Optional[AutoProcessor] = None
ov_llm: Optional[OpenVINOLLM] = None
ov_embedding: Optional[OpenVINOEmbedding] = None
ov_reranker: Optional[OpenVINORerank] = None
ov_chat_engine: Optional[BaseChatEngine] = None
ov_tts_model: Optional[torch.Tensor] = None

chatbot_config = {}


def get_available_devices() -> Set[str]:
    """
    List all devices available for inference

    Returns:
        Set of available devices
    """
    core = ov.Core()
    return {device.split(".")[0] for device in core.available_devices}


def load_asr_model(model_dir: Path, device: str) -> None:
    """
    Load automatic speech recognition model and assign it to a global variable

    Params:
        model_dir: dir with the ASR model
        device: device to run the model inference on
    """
    global asr_model, asr_processor

    if not model_dir.exists():
        log.error(f"Cannot find {model_dir}. Did you run convert_and_optimize_asr.py first?")
        return

    # create a distil-whisper model and its processor
    asr_model = OVModelForSpeechSeq2Seq.from_pretrained(model_dir, device=device)
    asr_processor = AutoProcessor.from_pretrained(model_dir)

    model_name = model_dir.name
    log.info(f"Running {model_name} on {','.join(asr_model.encoder.request.get_property('EXECUTION_DEVICES'))}")


def load_chat_model(model_dir: Path, device: str) -> Optional[OpenVINOLLM]:
    """
    Load chat model

    Params:
        model_dir: dir with the chat model
        device: device to run the model inference on
    Returns:
        OpenVINO LLM model in LLama Index
    """
    if not model_dir.exists():
        log.error(f"Cannot find {model_dir}. Did you run convert_and_optimize_chat.py first?")
        return None

    ov_config = {'PERFORMANCE_HINT': 'LATENCY', 'NUM_STREAMS': '1', "CACHE_DIR": ""}
    # load llama model and its tokenizer in the format of Llama Index
    return OpenVINOLLM(context_window=2048, model_id_or_path=str(model_dir), max_new_tokens=512, device_map=device,
                       model_kwargs={"ov_config": ov_config}, generate_kwargs={"do_sample": True, "temperature": 0.7, "top_k": 50, "top_p": 0.95})


def load_embedding_model(model_dir: Path, device: str) -> Optional[OpenVINOEmbedding]:
    """
    Load embedding model

    Params:
        model_dir: dir with the embedding model
        device: device to run the model inference on
    Returns:
        OpenVINO Embedding model in LLama Index
    """
    if not model_dir.exists():
        log.error(f"Cannot find {model_dir}. Did you run convert_and_optimize_chat.py first?")
        return None

    # load embedding model in the format of Llama Index
    return OpenVINOEmbedding(str(model_dir), device=device, embed_batch_size=1, model_kwargs={"dynamic_shapes": False})


def load_reranker_model(model_dir: Path, device: str) -> Optional[OpenVINORerank]:
    """
    Load embedding model

    Params:
        model_dir: dir with the reranker model
        device: device to run the model inference on
    Returns:
        OpenVINO Reranker model in LLama Index
    """
    if not model_dir.exists():
        log.error(f"Cannot find {model_dir}. Did you run convert_and_optimize_chat.py first?")
        return None

    # load reranker model in the format of Llama Index
    return OpenVINORerank(model_id_or_path=str(model_dir), device=device, top_n=3)


def load_rag_models(chat_model_dir: Path, chat_model_device: str, embedding_model_dir: Path, embedding_model_device: str, reranker_model_dir: Path, reranker_model_device: str, personality_file_path: Path) -> None:
    """
    Load all models required in RAG pipeline

    Params:
        chat_model_dir: dir with the chat model
        chat_model_device: device to run chat model inference on
        embedding_model_dir: dir with the embedding model
        embedding_model_device: device to run embedding model inference on
        reranker_model_dir: dir with the reranker model
        reranker_model_device: device to run reranker model inference on
        personality_file_path: path to the chatbot personality specification file
    """
    global ov_llm, ov_embedding, ov_reranker, ov_chat_engine, chatbot_config

    with open(personality_file_path) as f:
        chatbot_config = yaml.safe_load(f)

    # embedding model
    ov_embedding = load_embedding_model(embedding_model_dir, embedding_model_device)
    log.info(f"Running {embedding_model_dir} on {','.join(ov_embedding._model.request.get_property('EXECUTION_DEVICES'))}")

    # reranker model
    ov_reranker = load_reranker_model(reranker_model_dir, reranker_model_device)
    log.info(f"Running {reranker_model_dir} on {','.join(ov_reranker._model.request.get_property('EXECUTION_DEVICES'))}")

    # chat model
    ov_llm = load_chat_model(chat_model_dir, chat_model_device)
    log.info(f"Running {chat_model_dir} on {','.join(ov_llm._model.request.get_compiled_model().get_property('EXECUTION_DEVICES'))}")

    # chat engine
    ov_chat_engine = SimpleChatEngine.from_defaults(llm=ov_llm, system_prompt=chatbot_config["system_configuration"],
                                                    memory=ChatMemoryBuffer.from_defaults())


def load_tts_model() -> None:
    """
    Load text-to-speech model (MeloTTS) and assign it to a global variable
    """
    global ov_tts_model

    nltk.download('punkt_tab', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng')

    # CPU is sufficient for real-time inference.
    ov_tts_model = TTS(language='EN', device='cpu')
    
    # Compile the model with OpenVINO backend for accelerated inference
    ov_tts_model.model.infer = torch.compile(ov_tts_model.model.infer, backend='openvino')

    log.info(f"Running {type(ov_tts_model).__name__} on {ov_tts_model.device.__str__().upper()}")


def load_file(file_path: Path) -> Document:
    """
    Load text or pdf document using PyMuPDF for PDFs and standard reading for text files.
    
    Params:
        file_path: the path to the document
    Returns:
        A document in LLama Index format
    """
    ext = file_path.suffix
    if ext == ".pdf":
        # Using PyMuPDF (fitz) to read PDF content
        text = ""
        with fitz.open(file_path) as pdf:
            for page in pdf:
                text += page.get_text("text") + "\n"  # Extract text from each page
            return Document(text=text, metadata={"file_name": file_path.name})
    
    elif ext == ".txt":
        # Reading text files as usual
        with open(file_path) as f:
            content = f.read()
            return Document(text=content, metadata={"file_name": file_path.name})
    
    else:
        raise ValueError(f"{ext} file is not supported for now")


def load_context(file_path: Path) -> None:
    """
    Load context (document) and create a RAG pipeline
    Params:
        file_path: the path to the document
    """
    global ov_chat_engine

    # Create memory buffer for chat history
    memory = ChatMemoryBuffer.from_defaults()

    # if no file is provided, use the default chat engine (not RAG based)
    if not file_path:
        ov_chat_engine = SimpleChatEngine.from_defaults(
            llm=ov_llm,
            system_prompt=chatbot_config["system_configuration"],
            memory=memory
        )
        return

    # load the document
    document = load_file(file_path)

    # create a splitter to split the document into chunks
    splitter = LangchainNodeParser(RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100))

    # set the embedding model
    
    # Create index using LlamaIndex's default vector store (in-memory) and the splitter
    index = VectorStoreIndex.from_documents(
        [document], 
        transformations=[splitter], 
        embed_model=ov_embedding
    )

    # Build RAG chat engine with reranker and memory
    ov_chat_engine = index.as_chat_engine(
        llm=ov_llm,
        chat_mode=ChatMode.CONTEXT,
        system_prompt=chatbot_config["system_configuration"],
        memory=memory,
        node_postprocessors=[ov_reranker]
    )

def generate_initial_greeting() -> str:
    """
    Generates customer/patient greeting

    Returns:
        Generated greeting
    """
    return ov_chat_engine.chat(chatbot_config["greet_the_user_prompt"]).response


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

        # 75 words ~= 100 tokens
        tokens = len(history[-1][1].split(" ")) * 4 / 3
        processing_time = end_time - start_time
        log.info(f"Chat model response time: {processing_time:.2f} seconds ({tokens / processing_time:.2f} tokens/s)")


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


def synthesize(conversation: List[List[str]], audio: Tuple[int, np.ndarray]) -> Optional[Tuple[int, np.ndarray]]:
    """
    Synthesizes speech from chatbot's response

    Params:
        conversation: conversation history with the chatbot
        audio: audio widget to check if used
    Returns:
        Chatbot voice response (audio)
    """
    # if audio wasn't used in the conversation, return None
    if not audio:
        return None

    prompt = conversation[-1][1]

    start_time = time.time()
    # English
    speech = ov_tts_model.tts_to_file(prompt, ov_tts_model.hps.data.spk2id['EN-US'], output_path=None, speed=1.0)
    end_time = time.time()

    log.info(f"TTS model response time: {end_time - start_time:.2f} seconds")

    return TARGET_AUDIO_SAMPLE_RATE_TTS, speech


def create_UI(initial_message: str, example_pdf_path: Path) -> gr.Blocks:
    """
    Create web user interface

    Params:
        initial_message: message to start with
        example_pdf_path: path to the pdf file
    Returns:
        Demo UI
    """
    with gr.Blocks(theme="base", title="Adrishuo - the Conversational AI Chatbot") as demo:
        gr.Markdown(chatbot_config["instructions"])
        with gr.Row():
            file_uploader_ui = gr.File(label="Hotel guide", file_types=[".pdf", ".txt"], value=str(example_pdf_path), scale=1)
            with gr.Column(scale=4):
                chatbot_ui = gr.Chatbot(value=[[None, initial_message]], label="Chatbot")
                with gr.Tab(label="Voice"):
                    with gr.Row():
                        input_audio_ui = gr.Audio(sources=["microphone"], label="Your voice input")
                        output_audio_ui = gr.Audio(label="Chatbot voice response", autoplay=True)
                with gr.Tab(label="Text"):
                    input_text_ui = gr.Textbox(label="Your text input")
                with gr.Row():
                    clear_btn = gr.Button("Start over", variant="secondary")
                    submit_btn = gr.Button("Submit", variant="primary", interactive=False)

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
            .then(synthesize, inputs=[chatbot_ui, input_audio_ui], outputs=output_audio_ui) \
            .then(lambda: None, outputs=input_audio_ui) \
            .then(lambda: gr.Button(interactive=True), outputs=clear_btn)

        return demo


def run(asr_model_dir: Path, asr_model_device: str, chat_model_dir: Path, chat_model_device: str, embedding_model_dir: Path, embedding_model_device: str,
        reranker_model_dir: Path, reranker_model_device: str, personality_file_path: Path, example_pdf_path: Path, public_interface: bool = False) -> None:
    """
    Run the chatbot application

    Params
        asr_model_dir: dir with the automatic speech recognition model
        asr_model_device: device to run ASR model inference on
        chat_model_dir: dir with the chat model
        chat_model_device: device to run chat model inference on
        embedding_model_dir: dir with the embedding model
        embedding_model_device: device to run embedding model inference on
        reranker_model_dir: dir with the reranker model
        reranker_model_device: device to run reranker model inference on
        personality_file_path: path to the chatbot personality specification file
        example_pdf_path: path to the pdf file
        public_interface: whether UI should be available publicly
    """
    # set up logging
    log.getLogger().setLevel(log.INFO)

    # load whisper model
    load_asr_model(asr_model_dir, asr_model_device)
    # load chat models
    load_rag_models(chat_model_dir, chat_model_device, embedding_model_dir, embedding_model_device, reranker_model_dir, reranker_model_device, personality_file_path)
    # load tts model
    load_tts_model()

    if asr_model is None or ov_llm is None or ov_embedding is None:
        log.error("Required models are not loaded. Exiting...")
        return

    # get initial greeting
    initial_message = generate_initial_greeting()

    # load initial context
    load_context(example_pdf_path)

    # create user interface
    demo = create_UI(initial_message, example_pdf_path)

    print("Demo is ready!", flush=True) # Required for the CI to detect readiness
    # launch demo
    demo.queue().launch(share=public_interface)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asr_model", type=str, default="model/distil-whisper-large-v3-FP16", help="Path of the automatic speech recognition model directory")
    parser.add_argument("--asr_model_device", type=str, default="CPU", choices=["AUTO", "GPU", "CPU", "NPU"], help="Device to run ASR model inference on")
    parser.add_argument("--chat_model", type=str, default="model/llama3.2-3B-INT4", help="Path to the chat model directory")
    parser.add_argument("--chat_model_device", type=str, default="GPU", choices=["AUTO", "GPU", "CPU", "NPU"], help="Device to run chat model inference on")
    parser.add_argument("--embedding_model", type=str, default="model/bge-small-FP32", help="Path to the embedding model directory")
    parser.add_argument("--embedding_model_device", type=str, default="CPU", choices=["AUTO", "GPU", "CPU", "NPU"], help="Device to run embedding model inference on")
    parser.add_argument("--reranker_model", type=str, default="model/bge-reranker-large-FP32", help="Path to the reranker model directory")
    parser.add_argument("--reranker_model_device", type=str, default="CPU", choices=["AUTO", "GPU", "CPU", "NPU"], help="Device to run reranker model inference on")
    parser.add_argument("--personality", type=str, default="config/concierge_personality.yaml", help="Path to the YAML file with chatbot personality")
    parser.add_argument("--example_pdf", type=str, default="data/Grand_Azure_Resort_Spa_Full_Guide.pdf", help="Path to the PDF file which is an additional context")
    parser.add_argument("--public", default=False, action="store_true", help="Whether interface should be available publicly")

    args = parser.parse_args()
    run(Path(args.asr_model), args.asr_model_device, Path(args.chat_model), args.chat_model_device, Path(args.embedding_model), args.embedding_model_device,
        Path(args.reranker_model), args.reranker_model_device, Path(args.personality), Path(args.example_pdf), args.public)
