import argparse
import asyncio
import logging as log
import os
import random
import sys
from pathlib import Path

import gradio as gr
import openvino_genai as genai
from huggingface_hub import snapshot_download

SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils")
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils import demo_utils as utils

MODEL_DIR = Path("model")

MODELS = [
    "OpenVINO/Qwen3-8B-int4-cw-ov",
    "OpenVINO/gemma-3-4b-it-int4-cw-ov",
]

stop_generating: bool = False
ov_pipelines: dict = {}
generated_text_buffer: str = ""


async def download_model(model_name: str) -> None:
    output_dir = MODEL_DIR / model_name
    if not output_dir.exists():
        snapshot_download(model_name, local_dir=output_dir, resume_download=True)


async def create_pipeline(model_name: str, device: str):
    ov_config = {"CACHE_DIR": "cache"}

    # Download model if it hasn't been downloaded yet
    model_dir = MODEL_DIR / model_name
    if not model_dir.exists():
        await download_model(model_name)

    return genai.LLMPipeline(model_dir, device=device, **ov_config)


async def load_pipeline(model_name: str, device: str) -> genai.LLMPipeline:
    if device not in ov_pipelines:
        ov_pipelines[device] = await create_pipeline(model_name, device)

    return ov_pipelines[device]


async def stop() -> None:
    global stop_generating
    stop_generating = True


def streamer(subword: str) -> genai.StreamingStatus:
    global generated_text_buffer, stop_generating

    if stop_generating:
        return genai.StreamingStatus.CANCEL

    if "<think>" not in subword and "</think>" not in subword:
        generated_text_buffer += subword

    return genai.StreamingStatus.RUNNING


async def generate_text(model_name: str, device: str, topic: str, endless_generation: bool):
    global stop_generating, generated_text_buffer
    stop_generating = False

    device = device.split(":")[0]  # Extract device type (e.g., "CPU", "GPU")

    config = genai.GenerationConfig()
    config.max_new_tokens = 1024
    config.do_sample = True

    prompt = f"/no_think Write a short story about {topic}. Don't use dialogues. Don't output anything else except the story."

    ov_pipeline = await load_pipeline(model_name, device)

    stories_to_generate = 2**31 if endless_generation else 1
    generated_stories_count = 0
    while generated_stories_count < stories_to_generate:
        if stop_generating:
            break

        generated_text_buffer = ""

        # run blocking code in a thread
        generation_task = asyncio.create_task(asyncio.to_thread(ov_pipeline.generate, [prompt], config, streamer))

        while not generation_task.done():
            if stop_generating:
                break

            yield generated_text_buffer

            # small delay necessary for streaming
            await asyncio.sleep(0.01)

        # wait for the generation to finish if it hasn't
        await generation_task

        yield generated_text_buffer  # final output

        generated_stories_count += 1


def build_ui():
    model_choices = MODELS
    initial_model = model_choices[0]

    examples = ["The AI That Took Over the Coffee Machine", "Robot Vacuum Rebellion", "AI Personal Trainer… Gone Wrong",
                "Talking Toaster Conspiracy", "Selfie-Crazy AI", "AI-Powered Babysitter", "Pet Translator... Sort Of",
                "The AI That Critiques Everything", "Dancing Robots Takeover", "The Overly Honest Smart Fridge"]

    available_devices = [f"{k}: {v}" for k, v in utils.available_devices().items()]

    qr_code = utils.get_qr_code("https://github.com/openvinotoolkit/openvino_build_deploy/tree/master/demos/infinite_quill_demo", size=384, with_embedded_image=True)
    with gr.Blocks(title="Infinite Quill by OpenVINO") as demo:
        # custom intel header
        utils.gradio_intel_header("Infinite Quill by OpenVINO")
        with gr.Group():
            with gr.Row(equal_height=True):
                topic_text = gr.Text(
                    label="Write a story about...",
                    placeholder="Enter your topic here",
                    value=examples[0],
                    scale=5
                )
                random_topic_button = gr.Button("Random topic", variant="secondary", scale=1)
            output_textbox = gr.Textbox(label="Story", lines=30, autoscroll=True, interactive=False)
            with gr.Row():
                result_time_label = gr.Text("", label="Inference time", type="text")
            with gr.Row(equal_height=True):
                model_dropdown = gr.Dropdown(choices=model_choices, value=initial_model, label="Model", interactive=True, scale=2)
                device_dropdown = gr.Dropdown(choices=available_devices, value=available_devices[0], interactive=True,
                                              label="Inference device", scale=2)
                endless_checkbox = gr.Checkbox(label="Generate endlessly", value=False)
                with gr.Column(scale=1):
                    start_button = gr.Button("Start generating", variant="primary")
                    stop_button = gr.Button("Stop generating", variant="secondary")

        with gr.Row():
            with gr.Column(scale=3):
                gr.Examples(label="Examples of topics", examples=examples, inputs=topic_text, outputs=output_textbox, cache_examples=False)
            gr.Image(qr_code, interactive=False, show_label=False, scale=1)

        def swap_buttons_highlighting():
            return gr.Button(variant="primary"), gr.Button(variant="secondary")

        # rand the topic
        random_topic_button.click(lambda: gr.Text(value=random.choice(examples)), outputs=topic_text)  #nosec B311

        # clicking stop
        stop_button.click(stop)

        # clicking run
        gr.on(
            triggers=[topic_text.submit, start_button.click],
            fn=swap_buttons_highlighting,
            outputs=[stop_button, start_button]
        ).then(
            generate_text,
            inputs=[model_dropdown, device_dropdown, topic_text, endless_checkbox],
            outputs=[output_textbox]
        ).then(swap_buttons_highlighting, outputs=[start_button, stop_button])

    return demo

def run_demo(local_network: bool, public: bool):
    server_name = "0.0.0.0" if local_network else None

    demo = build_ui()
    print("Demo is ready!", flush=True) # Required for the CI to detect readiness
    demo.launch(theme=utils.gradio_intel_theme(), server_name=server_name, share=public)


if __name__ == '__main__':
    # set up logging
    log.getLogger().setLevel(log.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--local_network", action="store_true", help="Whether demo should be available in local network")
    parser.add_argument("--public", default=False, action="store_true", help="Whether interface should be available publicly")

    args = parser.parse_args()
    run_demo(args.local_network, args.public)