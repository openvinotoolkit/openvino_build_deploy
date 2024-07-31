import argparse
import logging as log
import time
from pathlib import Path
from threading import Thread
from typing import Tuple, List, Optional, Set

import gradio as gr
import librosa
import numpy as np
import openvino as ov
from optimum.intel import OVModelForCausalLM, OVModelForSpeechSeq2Seq
from transformers import AutoConfig, AutoTokenizer, AutoProcessor, PreTrainedTokenizer, TextIteratorStreamer
from transformers.generation.streamers import BaseStreamer

# Global variables initialization
TARGET_AUDIO_SAMPLE_RATE = 16000
SYSTEM_CONFIGURATION = (
    "You are Adrishuo - a helpful, respectful, and honest virtual doctor assistant. "
    "Your role is talking to a patient who just came in."
    "Your primary role is to assist in the collection of Symptom information from patients. "
    "The patient may attach prior examination report related to their health, which is available after 'additional context' keywords. "
    "Even the report is attached, you still must continue the conversation. "
    "Focus solely on gathering symptom details without offering treatment or medical advice. "
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
    "You are now required to summarize the patient's exact provided symptoms for the doctor's review. "
    "Strictly do not mention any personal data like age, name, gender, contact, non-health information etc. when summarizing."
    "Warn the patients for immediate medical seeking in case they exhibit symptoms indicative of critical conditions such as heart attacks, strokes, severe allergic reactions, breathing difficulties, high fever with severe symptoms, significant burns, or severe injuries."
    "Summarize the health-related concerns mentioned by the patient in this conversation, focusing only on the information explicitly provided, without adding any assumptions or unrelated symptoms."
)
ADDITIONAL_CONTEXT_TEMPLATE = "{}\nAdditional context: {}"

MODEL_DIR = Path("model")

# Initialize Model variables
chat_model: Optional[OVModelForCausalLM] = None
chat_tokenizer: Optional[PreTrainedTokenizer] = None
asr_model: Optional[OVModelForSpeechSeq2Seq] = None
asr_processor: Optional[AutoProcessor] = None

context = ""


def get_available_devices() -> Set[str]:
    core = ov.Core()
    return {device.split(".")[0] for device in core.available_devices}


def load_asr_model(model_name: str) -> None:
    global asr_model, asr_processor

    model_path = MODEL_DIR / model_name

    device = "GPU" if "GPU" in get_available_devices() else "CPU"
    # create a distil-whisper model and its processor
    if not model_path.exists():
        asr_model = OVModelForSpeechSeq2Seq.from_pretrained(model_name, device=device, export=True, load_in_8bit=True)
        asr_model.save_pretrained(model_path)
        asr_processor = AutoProcessor.from_pretrained(model_name)
        asr_processor.save_pretrained(model_path)
    else:
        asr_model = OVModelForSpeechSeq2Seq.from_pretrained(str(model_path), device=device)
        asr_processor = AutoProcessor.from_pretrained(str(model_name))


def load_chat_model(model_name: str) -> None:
    global chat_model, chat_tokenizer

    model_path = MODEL_DIR / model_name

    device = "GPU" if "GPU" in get_available_devices() else "AUTO"
    ov_config = {'PERFORMANCE_HINT': 'LATENCY', 'NUM_STREAMS': '1', "CACHE_DIR": ""}
    # load llama model and its tokenizer
    if not model_path.exists():
        chat_model = OVModelForCausalLM.from_pretrained(model_name, device=device, config=AutoConfig.from_pretrained(model_name, trust_remote_code=True), ov_config=ov_config)
        chat_model.save_pretrained(model_path)
        chat_tokenizer = AutoTokenizer.from_pretrained(model_name)
        chat_tokenizer.save_pretrained(model_path)
    else:
        chat_model = OVModelForCausalLM.from_pretrained(str(model_path), device=device, config=AutoConfig.from_pretrained(model_name, trust_remote_code=True), ov_config=ov_config)
        chat_tokenizer = AutoTokenizer.from_pretrained(str(model_path))


def load_context(file_path: str) -> str:
    global context

    if not file_path:
        context = ""
        return "No report loaded"

    with open(file_path) as f:
        context = f.read()

    return "Report loaded!"


def respond(prompt: str, streamer: BaseStreamer | None = None) -> str:
    start_time = time.time()  # Start time
    # tokenize input text
    inputs = chat_tokenizer(prompt, return_tensors="pt").to(chat_model.device)
    input_length = inputs.input_ids.shape[1]
    # generate response tokens
    outputs = chat_model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.6, top_p=0.9, top_k=50, streamer=streamer)
    tokens = outputs[0, input_length:]
    end_time = time.time()  # End time
    log.info("Chat model response time: {:.2f} seconds".format(end_time - start_time))
    # decode tokens into text
    return chat_tokenizer.decode(tokens, skip_special_tokens=True)


def get_conversation(history: List[List[str]]) -> str:
    # the conversation must be in that format to use chat template
    conversation = [
        {"role": "system", "content": SYSTEM_CONFIGURATION},
        {"role": "user", "content": GREET_THE_CUSTOMER}
    ]
    # add prompts to the conversation
    for user_prompt, assistant_response in history:
        if user_prompt:
            user_prompt = ADDITIONAL_CONTEXT_TEMPLATE.format(user_prompt, context) if context else user_prompt
            conversation.append({"role": "user", "content": user_prompt})
        if assistant_response:
            conversation.append({"role": "assistant", "content": assistant_response})

    # use a template specific to the model
    return chat_tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)


def generate_initial_greeting() -> str:
    conv = get_conversation([[None, None]])
    return respond(conv)


def chat(history: List[List[str]]) -> List[List[str]]:
    # convert list of message to conversation string
    conversation = get_conversation(history)

    # use streamer to show response word by word
    chat_streamer = TextIteratorStreamer(chat_tokenizer, skip_prompt=True, skip_special_tokens=True)

    # generate response for the conversation in a new thread to deliver response token by token
    thread = Thread(target=respond, args=[conversation, chat_streamer])
    thread.start()

    # get token by token and merge to the final response
    history[-1][1] = ""
    for partial_text in chat_streamer:
        history[-1][1] += partial_text
        # "return" partial response
        yield history

    # wait for the thread
    thread.join()


def transcribe(audio: Tuple[int, np.ndarray], prompt: str, conversation: List[List[str]]) -> List[List[str]]:
    # if audio is available, use audio, otherwise, use given text
    if audio:
        start_time = time.time()  # Start time for ASR process

        sample_rate, audio = audio
        # the whisper model requires 16000Hz, not 44100Hz
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sample_rate, target_sr=TARGET_AUDIO_SAMPLE_RATE).astype(np.int16)

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
    conversation.append([SUMMARIZE_THE_CUSTOMER, None])
    for partial_summary in chat(conversation):
        yield partial_summary[-1][1]


def create_UI(initial_message: str) -> gr.Blocks:
    with gr.Blocks(title="Talk to Adrishuo - a custom AI assistant working as a healthcare assistant") as demo:
        gr.Markdown("""
        # Talk to Adrishuo - a custom AI assistant working today as a healthcare assistant

        Instructions for use:
        - attach the PDF or TXT file with the prior examination report (optional)
        - record your question/comment using the first audio widget ("Your voice input") or type it in the textbox ("Your text input"), then click Submit
        - wait for the chatbot to response ("Chatbot")
        - discuss with the chatbot
        - click summarize button to make a summary
        """)
        with gr.Row():
            with gr.Column(scale=1):
                file_uploader_ui = gr.File(label="Prior examination report", file_types=[".pdf", ".txt"])
                context_label = gr.Label(label="Report status", value="No report loaded")
            with gr.Column(scale=5):
                # user's inputs
                input_audio_ui = gr.Audio(sources=["microphone"], label="Your voice input")
                input_text_ui = gr.Textbox(label="Your text input")
            # submit button
            submit_audio_btn = gr.Button("Submit", variant="primary", scale=1, interactive=False)

        # chatbot
        chatbot_ui = gr.Chatbot(value=[[None, initial_message]], label="Chatbot")

        # summarize
        summarize_button = gr.Button("Summarize", variant="primary", interactive=False)
        summary_ui = gr.Textbox(label="Summary", interactive=False)

        # events
        # block submit button when no audio or text input
        gr.on(triggers=[input_audio_ui.change, input_text_ui.change], inputs=[input_audio_ui, input_text_ui], outputs=submit_audio_btn,
              fn=lambda x, y: gr.Button(interactive=True) if x or y else gr.Button(interactive=False))

        file_uploader_ui.change(load_context, inputs=file_uploader_ui, outputs=context_label)

        # block buttons, do the transcription and conversation, clear audio, unblock buttons
        submit_audio_btn.click(lambda: gr.Button(interactive=False), outputs=submit_audio_btn) \
            .then(lambda: gr.Button(interactive=False), outputs=summarize_button)\
            .then(transcribe, inputs=[input_audio_ui, input_text_ui, chatbot_ui], outputs=chatbot_ui)\
            .then(chat, chatbot_ui, chatbot_ui)\
            .then(lambda: (None, None), inputs=[], outputs=[input_audio_ui, input_text_ui])\
            .then(lambda: gr.Button(interactive=True), outputs=summarize_button)

        # block button, do the summarization, unblock button
        summarize_button.click(lambda: gr.Button(interactive=False), outputs=summarize_button) \
            .then(summarize, inputs=chatbot_ui, outputs=summary_ui) \
            .then(lambda: gr.Button(interactive=True), outputs=summarize_button)

    return demo


def run(asr_model_name: Path, chat_model_name: Path, public_interface: bool = False) -> None:
    # set up logging
    log.getLogger().setLevel(log.INFO)

    # load whisper model
    load_asr_model(asr_model_name)
    # load chat model
    load_chat_model(chat_model_name)

    if chat_model is None or asr_model is None:
        log.error("Required models are not loaded. Exiting...")
        return

    # get initial greeting
    initial_message = generate_initial_greeting()

    # create user interface
    demo = create_UI(initial_message)
    # launch demo
    demo.launch(share=public_interface)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asr_model", type=str, default="distil-whisper/distil-large-v2", help="Path/name of the automatic speech recognition model")
    parser.add_argument("--chat_model", type=str, default="OpenVINO/Phi-3-medium-4k-instruct-int8-ov", help="Path/name of the chat model")
    parser.add_argument("--public", default=False, action="store_true", help="Whether interface should be available publicly")

    args = parser.parse_args()
    run(Path(args.asr_model), Path(args.chat_model), args.public)
