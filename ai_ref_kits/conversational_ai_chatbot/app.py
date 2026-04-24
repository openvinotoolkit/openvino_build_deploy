import argparse
import logging as log
import platform
import threading
import time
from pathlib import Path
from threading import Thread
from dataclasses import dataclass
from typing import Any, Tuple, List, Optional, Set, Dict

import pymupdf as fitz  # PyPI: pymupdf
import gradio as gr
import librosa
import numpy as np
import openvino as ov
import torch
import yaml
import nltk
from langchain_text_splitters import RecursiveCharacterTextSplitter
from optimum.intel import (
    OVModelForCausalLM,
    OVModelForFeatureExtraction,
    OVModelForSequenceClassification,
    OVModelForSpeechSeq2Seq,
)
from transformers import AutoProcessor, AutoTokenizer, TextIteratorStreamer
from melo.api import TTS

# Global variables initialization
TARGET_AUDIO_SAMPLE_RATE = 16000
TARGET_AUDIO_SAMPLE_RATE_TTS = 44100

MODEL_DIR = Path("model")
# Optimum-Intel uses one InferRequest per compiled model; concurrent Gradio threads (e.g. chat
# streaming while file upload reloads RAG) must not call the same OV model at once.
inference_lock = threading.Lock()

# Initialize Model variables
asr_model: Optional[OVModelForSpeechSeq2Seq] = None
asr_processor: Optional[AutoProcessor] = None
chat_model: Optional[OVModelForCausalLM] = None
chat_tokenizer: Optional[Any] = None
embedding_model: Optional[OVModelForFeatureExtraction] = None
embedding_tokenizer: Optional[Any] = None
reranker_model: Optional[OVModelForSequenceClassification] = None
reranker_tokenizer: Optional[Any] = None

@dataclass(frozen=True)
class _Chunk:
    text: str
    embedding: np.ndarray  # (dim,)

rag_chunks: List[_Chunk] = []
ov_tts_model: Optional[torch.Tensor] = None

chatbot_config = {}


def _message_plain_text(content: Any) -> str:
    """Plain text from Gradio Chatbot message content (string or normalized multimodal list)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return ""


def _openvino_runtime_device(device: str) -> str:
    """Normalize device for OpenVINO on Linux/arm64 CI hosts.

    ``AUTO`` compiles through a scheduler path that can fail for Distil-Whisper FP16 IR on
    aarch64 (oneDNN matmul primitive); compiling directly for ``CPU`` avoids that failure.
    """
    if device.upper() == "AUTO" and platform.machine().lower() in ("aarch64", "arm64"):
        return "CPU"
    return device


def _asr_openvino_config() -> Optional[dict]:
    """Extra compile hints for Whisper on ARM64 where FP16 matmul kernels may be unavailable."""
    if platform.machine().lower() not in ("aarch64", "arm64"):
        return None
    return {
        "PERFORMANCE_HINT": "LATENCY",
        "NUM_STREAMS": "1",
        "INFERENCE_PRECISION_HINT": "f32",
        "CACHE_DIR": "",
    }


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

    device = _openvino_runtime_device(device)
    load_kw: dict = {"device": device}
    ov_cfg = _asr_openvino_config()
    if ov_cfg is not None:
        load_kw["ov_config"] = ov_cfg
    # create a distil-whisper model and its processor
    asr_model = OVModelForSpeechSeq2Seq.from_pretrained(model_dir, **load_kw)
    asr_processor = AutoProcessor.from_pretrained(model_dir)

    model_name = model_dir.name
    log.info(f"Running {model_name} on {','.join(asr_model.encoder.request.get_property('EXECUTION_DEVICES'))}")


def load_chat_model(model_dir: Path, device: str) -> Tuple[OVModelForCausalLM, Any]:
    """
    Load chat model

    Params:
        model_dir: dir with the chat model
        device: device to run the model inference on
    Returns:
        (OpenVINO causal LM, tokenizer)
    """
    if not model_dir.exists():
        log.error(f"Cannot find {model_dir}. Did you run convert_and_optimize_chat.py first?")
        raise FileNotFoundError(model_dir)

    device = _openvino_runtime_device(device)
    ov_config = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", "CACHE_DIR": ""}
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = OVModelForCausalLM.from_pretrained(model_dir, device=device, ov_config=ov_config)
    log.info(
        f"Running {model_dir} on {','.join(model.request.get_compiled_model().get_property('EXECUTION_DEVICES'))}"
    )
    return model, tokenizer


def load_embedding_model(model_dir: Path, device: str) -> Tuple[OVModelForFeatureExtraction, Any]:
    """
    Load embedding model

    Params:
        model_dir: dir with the embedding model
        device: device to run the model inference on
    Returns:
        (OpenVINO feature extractor, tokenizer)
    """
    if not model_dir.exists():
        log.error(f"Cannot find {model_dir}. Did you run convert_and_optimize_chat.py first?")
        raise FileNotFoundError(model_dir)

    device = _openvino_runtime_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = OVModelForFeatureExtraction.from_pretrained(model_dir, device=device)
    log.info(
        f"Running {model_dir} on {','.join(model.request.get_compiled_model().get_property('EXECUTION_DEVICES'))}"
    )
    return model, tokenizer


def load_reranker_model(model_dir: Path, device: str) -> Tuple[OVModelForSequenceClassification, Any]:
    """
    Load reranker model

    Params:
        model_dir: dir with the reranker model
        device: device to run the model inference on
    Returns:
        (OpenVINO sequence classifier, tokenizer)
    """
    if not model_dir.exists():
        log.error(f"Cannot find {model_dir}. Did you run convert_and_optimize_chat.py first?")
        raise FileNotFoundError(model_dir)

    device = _openvino_runtime_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = OVModelForSequenceClassification.from_pretrained(model_dir, device=device)
    log.info(
        f"Running {model_dir} on {','.join(model.request.get_compiled_model().get_property('EXECUTION_DEVICES'))}"
    )
    return model, tokenizer


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
    global chat_model, chat_tokenizer, embedding_model, embedding_tokenizer, reranker_model, reranker_tokenizer, chatbot_config

    with open(personality_file_path) as f:
        chatbot_config = yaml.safe_load(f)

    embedding_model, embedding_tokenizer = load_embedding_model(
        embedding_model_dir, embedding_model_device
    )
    reranker_model, reranker_tokenizer = load_reranker_model(
        reranker_model_dir, reranker_model_device
    )
    chat_model, chat_tokenizer = load_chat_model(chat_model_dir, chat_model_device)


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


def load_file(file_path: Path | str) -> str:
    """
    Load text or pdf document using PyMuPDF for PDFs and standard reading for text files.
    
    Params:
        file_path: the path to the document (``pathlib.Path`` or Gradio file value such as ``NamedString``)
    Returns:
        Full extracted text
    """
    path = Path(str(file_path))
    ext = path.suffix
    if ext == ".pdf":
        # Using PyMuPDF (fitz) to read PDF content
        text = ""
        with fitz.open(path) as pdf:
            for page in pdf:
                text += page.get_text("text") + "\n"  # Extract text from each page
            return text
    
    elif ext == ".txt":
        # Reading text files as usual
        with open(path) as f:
            return f.read()
    
    else:
        raise ValueError(f"{ext} file is not supported for now")


def _mean_pool(last_hidden_state: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    mask = attention_mask.astype(np.float32)
    mask = np.expand_dims(mask, axis=-1)  # (bs, seq, 1)
    summed = (last_hidden_state * mask).sum(axis=1)
    denom = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
    return summed / denom


def _embed_texts(texts: List[str]) -> np.ndarray:
    """Return embeddings shape (n, dim)."""
    assert embedding_model is not None and embedding_tokenizer is not None
    encoded = embedding_tokenizer(
        texts, padding=True, truncation=True, max_length=512, return_tensors="np"
    )
    # Optimum-Intel OpenVINO models accept numpy arrays and return numpy arrays
    outputs = embedding_model(**encoded)
    # Feature-extraction models return last_hidden_state-like array as first output
    last_hidden = outputs[0]
    pooled = _mean_pool(last_hidden, encoded["attention_mask"])
    # L2 normalize
    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    return pooled / np.clip(norms, 1e-12, None)


def _cosine_top_k(query_emb: np.ndarray, mat: np.ndarray, k: int) -> List[int]:
    # query_emb: (dim,), mat: (n, dim) already normalized => dot = cosine
    scores = mat @ query_emb
    if k >= scores.shape[0]:
        return list(np.argsort(-scores))
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return idx.tolist()


def _rerank(query: str, candidates: List[str], top_n: int = 3) -> List[str]:
    if reranker_model is None or reranker_tokenizer is None:
        return candidates[:top_n]
    pairs = [(query, c) for c in candidates]
    encoded = reranker_tokenizer(
        pairs, padding=True, truncation=True, max_length=512, return_tensors="np"
    )
    outputs = reranker_model(**encoded)
    logits = outputs[0]
    # assume higher logit => more relevant; handle (bs,1) or (bs,2)
    scores = logits.reshape(logits.shape[0], -1)
    scores = scores[:, -1]
    order = np.argsort(-scores)
    return [candidates[i] for i in order[:top_n]]


def load_context(file_path: Path | str | None) -> None:
    """
    Load context (document) and create a RAG pipeline
    Params:
        file_path: the path to the document
    """
    global rag_chunks
    rag_chunks = []

    if not file_path:
        return

    raw_text = load_file(file_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = [c.strip() for c in splitter.split_text(raw_text) if c.strip()]

    with inference_lock:
        embs = _embed_texts(chunks)

    rag_chunks = [_Chunk(text=t, embedding=embs[i]) for i, t in enumerate(chunks)]

def generate_initial_greeting() -> str:
    """
    Generates customer/patient greeting

    Returns:
        Generated greeting
    """
    # simple, non-RAG greeting
    return chatbot_config.get("greeting", "Hello! How can I help you today?")


def chat(history: List[dict]) -> List[dict]:
    """
    Chat function. It generates response based on a prompt

    Params:
        history: OpenAI-style chat messages (role / content) from the Chatbot
    Returns:
        History with the latest chat's response (yields partial response)
    """
    history = list(history)
    user_msg = (
        _message_plain_text(history[-1].get("content"))
        if history and history[-1].get("role") == "user"
        else ""
    )

    # retrieve context
    context_blocks: List[str] = []
    if rag_chunks and user_msg:
        with inference_lock:
            q = _embed_texts([user_msg])[0]
        mat = np.stack([c.embedding for c in rag_chunks], axis=0)
        top_idx = _cosine_top_k(q, mat, k=6)
        candidates = [rag_chunks[i].text for i in top_idx]
        context_blocks = _rerank(user_msg, candidates, top_n=3)

    sys_prompt = chatbot_config.get("system_configuration", "")
    context_text = "\n\n".join(
        f"[CONTEXT {i+1}]\n{t}" for i, t in enumerate(context_blocks)
    )
    if context_text:
        sys_prompt = f"{sys_prompt}\n\nUse the following context to answer:\n{context_text}".strip()

    messages: List[Dict[str, str]] = []
    if sys_prompt:
        messages.append({"role": "system", "content": sys_prompt})
    for m in history:
        role = m.get("role")
        if role in ("user", "assistant"):
            messages.append({"role": role, "content": _message_plain_text(m.get('content'))})

    assert chat_model is not None and chat_tokenizer is not None
    prompt = chat_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    streamer = TextIteratorStreamer(chat_tokenizer, skip_prompt=True, skip_special_tokens=True)

    gen_kwargs = dict(
        inputs=chat_tokenizer(prompt, return_tensors="np")["input_ids"],
        streamer=streamer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        top_k=50,
    )

    history.append({"role": "assistant", "content": ""})

    def _run_generate() -> None:
        with inference_lock:
            chat_model.generate(**gen_kwargs)

    thread = Thread(target=_run_generate)
    start_time = time.time()
    thread.start()
    for token in streamer:
        history[-1]["content"] += token
        yield history
    thread.join()
    end_time = time.time()
    reply = _message_plain_text(history[-1].get("content"))
    tokens = len(reply.split(" ")) * 4 / 3
    processing_time = end_time - start_time
    log.info(f"Chat model response time: {processing_time:.2f} seconds ({tokens / processing_time:.2f} tokens/s)")


def transcribe(audio: Tuple[int, np.ndarray], prompt: str, conversation: List[dict]) -> List[dict]:
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

        def _run_asr() -> None:
            with inference_lock:
                asr_model.generate(input_features=input_features, streamer=text_streamer)

        # transcribe in the background to deliver response token by token
        thread = Thread(target=_run_asr)
        thread.start()

        conversation = list(conversation)
        conversation.append({"role": "user", "content": ""})
        for partial_text in text_streamer:
            conversation[-1]["content"] += partial_text
            yield conversation

        end_time = time.time()  # End time for ASR process
        log.info(f"ASR model response time: {end_time - start_time:.2f} seconds")  # Print the ASR processing time

        # wait for the thread
        thread.join()
    else:
        conversation = list(conversation)
        conversation.append({"role": "user", "content": prompt})
        yield conversation

    return conversation


def synthesize(conversation: List[dict], audio: Tuple[int, np.ndarray]) -> Optional[Tuple[int, np.ndarray]]:
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

    prompt = _message_plain_text(conversation[-1].get("content")) if conversation else ""

    start_time = time.time()
    with inference_lock:
        speech = ov_tts_model.tts_to_file(
            prompt, ov_tts_model.hps.data.spk2id["EN-US"], output_path=None, speed=1.0
        )
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
    with gr.Blocks(title="Adrishuo - the Conversational AI Chatbot") as demo:
        gr.Markdown(chatbot_config["instructions"])
        with gr.Row():
            file_uploader_ui = gr.File(
                label="Hotel guide", file_types=[".pdf", ".txt"], value=str(example_pdf_path), scale=1
            )

            with gr.Column(scale=4):
                chatbot_ui = gr.Chatbot(
                    value=[{"role": "assistant", "content": initial_message}],
                    label="Chatbot",
                )
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

        reset_chat = lambda: [{"role": "assistant", "content": initial_message}]

        file_uploader_ui.change(reset_chat, outputs=chatbot_ui) \
            .then(load_context, inputs=file_uploader_ui)

        clear_btn.click(reset_chat, outputs=chatbot_ui) \
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

    if (
        asr_model is None
        or chat_model is None
        or chat_tokenizer is None
        or embedding_model is None
        or embedding_tokenizer is None
        or reranker_model is None
        or reranker_tokenizer is None
    ):
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
    demo.queue().launch(share=public_interface, theme="base")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asr_model", type=str, default="model/distil-whisper-large-v3-FP16", help="Path of the automatic speech recognition model directory")
    parser.add_argument("--asr_model_device", type=str, default="CPU", choices=["AUTO", "GPU", "CPU", "NPU"], help="Device to run ASR model inference on")
    parser.add_argument("--chat_model", type=str, default="model/llama3.2-3B-INT4", help="Path to the chat model directory")
    parser.add_argument("--chat_model_device", type=str, default="CPU", choices=["AUTO", "GPU", "CPU", "NPU"], help="Device to run chat model inference on")
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
