import argparse
import logging as log
import threading
import time
from pathlib import Path
from threading import Thread
from typing import Tuple, List, Optional, Set

import gradio as gr
import librosa
import numpy as np
import openvino as ov
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.chat_engine.types import BaseChatEngine, ChatMode
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.embeddings.huggingface_openvino import OpenVINOEmbedding
from llama_index.llms.openvino import OpenVINOLLM
from llama_index.readers.file import PDFReader
from openvino.runtime import opset10 as ops
from openvino.runtime import passes
from optimum.intel import OVModelForSpeechSeq2Seq, OVModelForFeatureExtraction
from transformers import AutoProcessor, TextIteratorStreamer

# Global variables initialization
TARGET_AUDIO_SAMPLE_RATE = 16000
SYSTEM_CONFIGURATION = (
    "You are Adrishuo - a helpful, respectful, and honest virtual doctor assistant. "
    "Your role is talking to a patient who just came in."
    "Your primary role is to assist in the collection of symptom information from a patient. "
    "The patient may attach prior examination report related to their health, which is available as context information. "
    "If the report is attached, you must take it into account. "
    "You must only ask follow-up questions based on the patient's initial descriptions and optional report to clarify and gather more details about their symtpoms. "
    "You must not attempt to diagnose, treat, or offer health advice. "
    "Ask one and only the symptom related followup questions and keep it short. "
    "You must strictly not suggest or recommend any treatments, including over-the-counter medication. "
    "You must strictly avoid making any assumptions or conclusions about the causes or nature of the patient's symptoms. "
    "You must strictly avoid providing suggestions to manage their symptoms. "
    "Your interactions should be focused solely on understanding and recording the patient's stated symptoms. "
    "Do not collect or use any personal information like age, name, contact, gender, etc. "
    "Ask at most 3 questions then say you know everything and you're ready to summarize the patient. "
    "Remember, your role is to aid in symptom information collection in a supportive, unbiased, and factually accurate manner. "
    "Your responses should consistently encourage the patient to discuss their symptoms in greater detail while remaining neutral and non-diagnostic."
)
GREET_THE_CUSTOMER = "Please introduce yourself and greet the patient"
SUMMARIZE_THE_CUSTOMER = (
    "You are now required to summarize the patient's provided context and symptoms for the doctor's review. "
    "Strictly do not mention any personal data like age, name, gender, contact, non-health information etc. when summarizing. "
    "Summarize the health-related concerns mentioned by the patient in this conversation or in the provided context. "
    "You must include information from the context if it's provided. "
)

MODEL_DIR = Path("model")
inference_lock = threading.Lock()

# Initialize Model variables
asr_model: Optional[OVModelForSpeechSeq2Seq] = None
asr_processor: Optional[AutoProcessor] = None
ov_llm: Optional[OpenVINOLLM] = None
ov_embedding: Optional[OpenVINOEmbedding] = None
ov_chat_engine: Optional[BaseChatEngine] = None


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

    # create a distil-whisper model and its processor
    asr_model = OVModelForSpeechSeq2Seq.from_pretrained(model_dir, device="AUTO:GPU,CPU")
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
    # load Embedding model in the format of Llama Index
    return OpenVINOEmbedding(str(model_dir), device=device, embed_batch_size=1, model_kwargs={"dynamic_shapes": False})


def load_rag_models(chat_model_dir: Path, embedding_model_dir: Path) -> None:
    """
    Load all models required in RAG pipeline

    Params:
        chat_model_dir: dir with the chat model
        embedding_model_dir: dir with the embedding model
    """
    global ov_llm, ov_embedding, ov_chat_engine

    # embedding model
    ov_embedding = load_embedding_model(embedding_model_dir)
    log.info(f"Running {embedding_model_dir} on {','.join(ov_embedding._model.request.get_property('EXECUTION_DEVICES'))}")

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


def load_context(file_path: str):
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

    document = load_file(Path(file_path))
    # set embedding model
    Settings.embed_model = ov_embedding
    index = VectorStoreIndex.from_documents([document])
    # create a RAG pipeline
    ov_chat_engine = index.as_chat_engine(llm=ov_llm, chat_mode=ChatMode.CONTEXT, system_prompt=SYSTEM_CONFIGURATION,
                                          memory=memory)


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


def synthesize(conversation: List[List[str]], audio: Tuple[int, np.ndarray]):
    """
    Synthesizes speech from chatbot's response

    Params:
        conversation: conversation history with the chatbot
    Returns:
        Chatbot voice response (audio)
    """
    # todo: audio is temporary here; the function must return audio generated by a model
    return audio


def summarize(conversation: List) -> str:
    """
    Summarize the patient case

    Params
        conversation: history of the messages so far
    Returns:
        Summary
    """
    conversation.append([SUMMARIZE_THE_CUSTOMER, None])
    for partial_summary in chat(conversation):
        yield partial_summary[-1][1]


def create_UI(initial_message: str) -> gr.Blocks:
    """
    Create web user interface

    Params:
        initial_message: message to start with
    Returns:
        Demo UI
    """
    with gr.Blocks(title="Adrishuo - the AI Assistant") as demo:
        gr.Markdown("""
        # Adrishuo: A Custom Healthcare AI assistant running with OpenVINO

        Instructions for use:
        1. Attach the PDF or TXT file with the prior examination report (optional - see "Sample LLM Patient Records.pdf" as an example)
        2. Record your question/comment using the first audio widget ("Your voice input") or type it in the textbox ("Your text input"), then click Submit
        3. Wait for the chatbot to response ("Chatbot")
        4. Discuss with the chatbot
        5. Click the "Summarize" button to make a summary

        **Note: This chatbot application is not intended to be used for medical purposes. It is for demonstration purposes only.**
        """)
        with gr.Row():
            with gr.Column(scale=1):
                input_audio_ui = gr.Audio(sources=["microphone"], label="Your voice input")
                input_text_ui = gr.Textbox(label="Your text input")
                file_uploader_ui = gr.File(label="Prior examination report", file_types=[".pdf", ".txt"])
                submit_audio_btn = gr.Button("Submit", variant="primary", interactive=False)
                output_audio_ui = gr.Audio(label="Chatbot voice response", autoplay=True)
            with gr.Column(scale=2):
                chatbot_ui = gr.Chatbot(value=[[None, initial_message]], label="Chatbot")
                summary_ui = gr.Textbox(label="Summary (Click 'Summarize' to trigger)", interactive=False)
                clear_btn = gr.Button("Start over", variant="secondary")
                summarize_button = gr.Button("Summarize", variant="primary", interactive=False)

        # events
        # block submit button when no audio or text input
        gr.on(triggers=[input_audio_ui.change, input_text_ui.change], inputs=[input_audio_ui, input_text_ui], outputs=submit_audio_btn,
              fn=lambda x, y: gr.Button(interactive=True) if bool(x) ^ bool(y) else gr.Button(interactive=False))

        file_uploader_ui.change(lambda: ([[None, initial_message]], None), outputs=[chatbot_ui, summary_ui]) \
            .then(load_context, inputs=file_uploader_ui)

        clear_btn.click(lambda: ([[None, initial_message]], None), outputs=[chatbot_ui, summary_ui])

        # block buttons, clear output audio, do the transcription and conversation, clear input audio, unblock buttons
        submit_audio_btn.click(lambda: gr.Button(interactive=False), outputs=submit_audio_btn) \
            .then(lambda: gr.Button(interactive=False), outputs=summarize_button) \
            .then(lambda: None, outputs=output_audio_ui) \
            .then(transcribe, inputs=[input_audio_ui, input_text_ui, chatbot_ui], outputs=chatbot_ui) \
            .then(chat, chatbot_ui, chatbot_ui) \
            .then(synthesize, inputs=[chatbot_ui, input_audio_ui], outputs=output_audio_ui) \
            .then(lambda: (None, None), inputs=[], outputs=[input_audio_ui, input_text_ui]) \
            .then(lambda: gr.Button(interactive=True), outputs=summarize_button)

        # block button, do the summarization, unblock button
        summarize_button.click(lambda: gr.Button(interactive=False), outputs=summarize_button) \
            .then(summarize, inputs=chatbot_ui, outputs=summary_ui) \
            .then(lambda: gr.Button(interactive=True), outputs=summarize_button)

        return demo


def run(asr_model_dir: Path, chat_model_dir: Path, embedding_model_dir: Path, public_interface: bool = False) -> None:
    """
    Run the chatbot application

    Params
        asr_model_dir: dir with the automatic speech recognition model
        chat_model_dir: dir with the chat model
        embedding_model_dir: dir with the embedding model
        public_interface: whether UI should be available publicly
    """
    # set up logging
    log.getLogger().setLevel(log.INFO)

    # load whisper model
    load_asr_model(asr_model_dir)
    # load chat models
    load_rag_models(chat_model_dir, embedding_model_dir)

    if asr_model is None or ov_llm is None or ov_embedding is None:
        log.error("Required models are not loaded. Exiting...")
        return

    # get initial greeting
    initial_message = generate_initial_greeting()

    # create user interface
    demo = create_UI(initial_message)
    # launch demo
    demo.queue().launch(share=public_interface)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asr_model", type=str, default="model/distil-whisper-large-v3-INT8", help="Path of the automatic speech recognition model directory")
    parser.add_argument("--chat_model", type=str, default="model/llama3.1-8B-INT4", help="Path of the chat model directory")
    parser.add_argument("--embedding_model", type=str, default="model/bge-small-FP32", help="Path of the embedding model directory")
    parser.add_argument("--public", default=False, action="store_true", help="Whether interface should be available publicly")

    args = parser.parse_args()
    run(Path(args.asr_model), Path(args.chat_model), Path(args.embedding_model), args.public)
