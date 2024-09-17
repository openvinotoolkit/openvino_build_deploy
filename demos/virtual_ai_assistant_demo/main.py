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
import yaml
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.chat_engine.types import BaseChatEngine, ChatMode
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.embeddings.huggingface_openvino import OpenVINOEmbedding
from llama_index.llms.openvino import OpenVINOLLM
from llama_index.readers.file import PDFReader
from openvino.runtime import opset10 as ops
from openvino.runtime import passes
from optimum.intel import OVModelForCausalLM, OVModelForSpeechSeq2Seq, OVModelForFeatureExtraction, \
    OVWeightQuantizationConfig, OVConfig, OVQuantizer
from transformers import AutoTokenizer, AutoProcessor, TextIteratorStreamer

# Global variables initialization
TARGET_AUDIO_SAMPLE_RATE = 16000

MODEL_DIR = Path("model")
inference_lock = threading.Lock()

# Initialize Model variables
asr_model: Optional[OVModelForSpeechSeq2Seq] = None
asr_processor: Optional[AutoProcessor] = None
ov_llm: Optional[OpenVINOLLM] = None
ov_embedding: Optional[OpenVINOEmbedding] = None
ov_chat_engine: Optional[BaseChatEngine] = None

chatbot_config = {}


def get_available_devices() -> Set[str]:
    core = ov.Core()
    return {device.split(".")[0] for device in core.available_devices}


def load_asr_model(model_name: str) -> None:
    global asr_model, asr_processor

    model_path = MODEL_DIR / model_name
    device = "GPU" if "GPU" in get_available_devices() else "CPU"

    # create a distil-whisper model and its processor
    if not model_path.exists():
        log.info(f"Downloading {model_name}... It may take up to 1h depending on your Internet connection.")
        asr_model = OVModelForSpeechSeq2Seq.from_pretrained(model_name, export=True, load_in_8bit=False, device=device)
        asr_model.save_pretrained(model_path)
        asr_processor = AutoProcessor.from_pretrained(model_name)
        asr_processor.save_pretrained(model_path)
    else:
        asr_model = OVModelForSpeechSeq2Seq.from_pretrained(str(model_path), device=device)
        asr_processor = AutoProcessor.from_pretrained(str(model_name))

    log.info(f"Running {model_name} on {','.join(asr_model.encoder.request.get_property('EXECUTION_DEVICES'))}")


def load_chat_model(model_name: str, token: str = None) -> OpenVINOLLM:
    model_path = MODEL_DIR / model_name

    ov_config = {'PERFORMANCE_HINT': 'LATENCY', 'NUM_STREAMS': '1', "CACHE_DIR": ""}
    # load llama model and its tokenizer
    if not model_path.exists():
        log.info(f"Downloading {model_name}... It may take up to 1h depending on your Internet connection.")
        chat_model = OVModelForCausalLM.from_pretrained(model_name, export=True, compile=False, load_in_8bit=False,
                                                        token=token)

        quant_config = OVWeightQuantizationConfig(bits=4, sym=False, ratio=0.8)
        config = OVConfig(quantization_config=quant_config)

        log.info(f"Quantizing {model_name} to INT4... It may take significant amount of time depending on your machine power.")
        quantizer = OVQuantizer.from_pretrained(chat_model, task="text-generation")
        quantizer.quantize(save_directory=model_path, weights_only=True, ov_config=config)

        chat_tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        chat_tokenizer.save_pretrained(model_path)

    device = "GPU" if "GPU" in get_available_devices() else "CPU"
    return OpenVINOLLM(context_window=2048, model_id_or_path=str(model_path), max_new_tokens=512, device_map=device,
                       model_kwargs={"ov_config": ov_config}, generate_kwargs={"do_sample": True, "temperature": 0.7, "top_k": 50, "top_p": 0.95})


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
        embedding_model = OVModelForFeatureExtraction.from_pretrained(model_name, export=True)
        optimize_model_for_npu(embedding_model)
        embedding_model.save_pretrained(model_path)
        embedding_tokenizer = AutoTokenizer.from_pretrained(model_name)
        embedding_tokenizer.save_pretrained(model_path)

    device = "NPU" if "NPU" in get_available_devices() else "CPU"
    return OpenVINOEmbedding(str(model_path), device=device, embed_batch_size=1, model_kwargs={"dynamic_shapes": False})


def load_chat_models(chat_model_name: str, embedding_model_name: str, personality_file_path: Path, auth_token: str = None) -> None:
    global ov_llm, ov_chat_engine, ov_embedding, chatbot_config

    with open(personality_file_path) as f:
        chatbot_config = yaml.safe_load(f)

    ov_embedding = load_embedding_model(embedding_model_name)
    log.info(f"Running {embedding_model_name} on {ov_embedding._model.request.get_property('EXECUTION_DEVICES')}")
    ov_llm = load_chat_model(chat_model_name, auth_token)
    log.info(f"Running {chat_model_name} on {','.join(ov_llm._model.request.get_compiled_model().get_property('EXECUTION_DEVICES'))}")

    ov_chat_engine = SimpleChatEngine.from_defaults(llm=ov_llm, system_prompt=chatbot_config["system_configuration"],
                                                    memory=ChatMemoryBuffer.from_defaults())


def load_file(file_path: Path) -> Document:
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
    global ov_chat_engine

    # limit chat history to 3000 tokens
    memory = ChatMemoryBuffer.from_defaults()

    if not file_path:
        ov_chat_engine = SimpleChatEngine.from_defaults(llm=ov_llm, system_prompt=chatbot_config["system_configuration"], memory=memory)
        return

    document = load_file(Path(file_path))
    Settings.embed_model = ov_embedding
    index = VectorStoreIndex.from_documents([document])
    ov_chat_engine = index.as_chat_engine(llm=ov_llm, chat_mode=ChatMode.CONTEXT, system_prompt=chatbot_config["system_configuration"],
                                          memory=memory)


def generate_initial_greeting() -> str:
    return ov_chat_engine.chat(chatbot_config["greet_the_user_prompt"]).response


def chat(history: List[List[str]]) -> List[List[str]]:
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


def summarize(conversation: List) -> str:
    conversation.append([chatbot_config["summarize_the_user_prompt"], None])
    for partial_summary in chat(conversation):
        yield partial_summary[-1][1]


def create_UI(initial_message: str) -> gr.Blocks:
    with gr.Blocks(title="Your Virtual AI Assistant") as demo:
        gr.Markdown(chatbot_config["instructions"])
        
        with gr.Row():
            with gr.Column(scale=1):
                input_audio_ui = gr.Audio(sources=["microphone"], label="Your voice input")
                input_text_ui = gr.Textbox(label="Your text input")
                file_uploader_ui = gr.File(label="Additional context", file_types=[".pdf", ".txt"])
                submit_btn = gr.Button("Submit", variant="primary", interactive=False)
            with gr.Column(scale=2):
                chatbot_ui = gr.Chatbot(value=[[None, initial_message]], label="Chatbot")
                summary_ui = gr.Textbox(label="Summary (Click 'Summarize' to trigger)", interactive=False)
                clear_btn = gr.Button("Start over", variant="secondary")
                summarize_button = gr.Button("Summarize", variant="primary", interactive=False)

        # events
        # block submit button when no audio or text input
        gr.on(triggers=[input_audio_ui.change, input_text_ui.change], inputs=[input_audio_ui, input_text_ui], outputs=submit_btn,
              fn=lambda x, y: gr.Button(interactive=True) if bool(x) ^ bool(y) else gr.Button(interactive=False))

        file_uploader_ui.change(lambda: ([[None, initial_message]], None), outputs=[chatbot_ui, summary_ui]) \
            .then(load_context, inputs=file_uploader_ui)

        clear_btn.click(lambda: ([[None, initial_message]], None), outputs=[chatbot_ui, summary_ui]) \
            .then(load_context, inputs=file_uploader_ui) \
            .then(lambda: gr.Button(interactive=False), outputs=summarize_button)

        # block buttons, do the transcription and conversation, clear audio, unblock buttons
        gr.on(triggers=[submit_btn.click, input_text_ui.submit], fn=lambda: gr.Button(interactive=False), outputs=submit_btn) \
            .then(lambda: gr.Button(interactive=False), outputs=summarize_button) \
            .then(lambda: gr.Button(interactive=False), outputs=clear_btn) \
            .then(transcribe, inputs=[input_audio_ui, input_text_ui, chatbot_ui], outputs=chatbot_ui) \
            .then(lambda: None, outputs=input_text_ui) \
            .then(chat, chatbot_ui, chatbot_ui) \
            .then(lambda: None, outputs=input_audio_ui) \
            .then(lambda: gr.Button(interactive=True), outputs=clear_btn) \
            .then(lambda: gr.Button(interactive=True), outputs=summarize_button)

        # block button, do the summarization, unblock button
        summarize_button.click(lambda: gr.Button(interactive=False), outputs=summarize_button) \
            .then(lambda: gr.Button(interactive=False), outputs=clear_btn) \
            .then(summarize, inputs=chatbot_ui, outputs=summary_ui) \
            .then(lambda: gr.Button(interactive=True), outputs=clear_btn) \
            .then(lambda: gr.Button(interactive=True), outputs=summarize_button)

        return demo


def run(asr_model_name: str, chat_model_name: str, embedding_model_name: str, personality_file_path: Path, hf_token: str = None, public_interface: bool = False) -> None:
    # set up logging
    log.getLogger().setLevel(log.INFO)

    # load whisper model
    load_asr_model(asr_model_name)
    # load chat models
    load_chat_models(chat_model_name, embedding_model_name, personality_file_path, hf_token)

    # get initial greeting
    initial_message = generate_initial_greeting()

    # create user interface
    demo = create_UI(initial_message)
    # launch demo
    demo.queue().launch(share=public_interface)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asr_model", type=str, default="distil-whisper/distil-large-v3", help="Path/name of the automatic speech recognition model")
    parser.add_argument("--chat_model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="Path/name of the chat model")
    parser.add_argument("--embedding_model", type=str, default="BAAI/bge-small-en-v1.5", help="Path/name of the model for embeddings")
    parser.add_argument("--personality", type=str, default="healthcare_personality.yaml", help="Path to the YAML file with chatbot personality")
    parser.add_argument("--hf_token", type=str, help="HuggingFace access token to get Llama3")
    parser.add_argument("--public", default=False, action="store_true", help="Whether interface should be available publicly")

    args = parser.parse_args()
    run(args.asr_model, args.chat_model, args.embedding_model, Path(args.personality), args.hf_token, args.public)
