import argparse
import asyncio
import logging as log
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

import gradio as gr
import numpy as np
import openvino as ov
import openvino_genai as genai
from PIL import Image
from huggingface_hub import snapshot_download
from transformers import Pipeline, pipeline

SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils")
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils import demo_utils as utils

MAX_SEED = np.iinfo(np.int32).max
MODEL_DIR = Path("model")

safety_checker: Optional[Pipeline] = None

ov_pipelines = {}

stop_generating: bool = True
hf_model_name: Optional[str] = None


def get_available_devices() -> list[str]:
    core = ov.Core()
    return list({device.split(".")[0] for device in core.available_devices})


def download_models(model_name: str, safety_checker_model: str) -> None:
    global safety_checker

    output_dir = MODEL_DIR / model_name
    if not output_dir.exists():
        snapshot_download(model_name, local_dir=output_dir)

    safety_checker_dir = MODEL_DIR / safety_checker_model
    if not safety_checker_dir.exists():
        snapshot_download(safety_checker_model, local_dir=safety_checker_dir)

    safety_checker = pipeline("image-classification", model=str(safety_checker_dir))


async def load_pipeline(model_name: str, device: str):
    if device not in ov_pipelines:
        model_dir = MODEL_DIR / model_name
        ov_config = {"CACHE_DIR": "cache"}

        ov_pipeline = genai.Text2ImagePipeline(model_dir, device, **ov_config)
        ov_pipelines[device] = ov_pipeline

    return ov_pipelines[device]


async def stop():
    global stop_generating
    stop_generating = True


async def generate_images(prompt: str, seed: int, size: int, guidance_scale: float, num_inference_steps: int, randomize_seed: bool, device: str, endless_generation: bool) -> tuple[np.ndarray, float]:
    global stop_generating
    stop_generating = not endless_generation

    ov_pipeline = await load_pipeline(hf_model_name, device)

    while True:
        if randomize_seed:
            seed = random.randint(0, MAX_SEED)

        start_time = time.time()
        result = ov_pipeline.generate(prompt=prompt, num_inference_steps=num_inference_steps, width=size, height=size,
                             guidance_scale=guidance_scale, generator=genai.CppStdGenerator(seed)).data[0]
        end_time = time.time()

        label = safety_checker(Image.fromarray(result), top_k=1)
        if label[0]["label"].lower() == "nsfw":
            result = np.zeros_like(result)
            h, w = result.shape[:2]
            utils.draw_text(result, "Potential NSFW content", (w // 2, h // 2), center=True, font_scale=3.0)

        utils.draw_ov_watermark(result, size=0.60)

        processing_time = end_time - start_time

        yield result, round(processing_time, 5)

        if stop_generating:
            break

        # small delay necessary for endless generation
        await asyncio.sleep(0.1)


def build_ui():
    examples = [
        "A sail boat on a grass field with mountains in the morning and sunny day",
        "Portrait photo of a girl, photograph, highly detailed face, depth of field, moody light, golden hour,"
        "Style by Dan Winters, Russell James, Steve McCurry, centered, extremely detailed, Nikon D850, award winning photography",
        "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
        "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
        "A photo of beautiful mountain with realistic sunset and blue lake, highly detailed, masterpiece",
    ]

    with gr.Blocks() as demo:
        with gr.Group():
            with gr.Row():
                prompt_text = gr.Text(
                    label="Prompt",
                    placeholder="Enter your prompt here",
                    value="A sail boat on a grass field with mountains in the morning and sunny day"
                )
            with gr.Row():
                with gr.Column():
                    result_img = gr.Image(label="Generated image", elem_id="output_image", format="png")
                    with gr.Row():
                        result_time_label = gr.Text("", label="Inference Time", type="text")
                    with gr.Row():
                        start_button = gr.Button("Start generation")
                        stop_button = gr.Button("Stop generation")
            with gr.Accordion("Advanced options", open=True):
                with gr.Row():
                    device_dropdown = gr.Dropdown(
                        choices=get_available_devices(),
                        value="CPU",
                        label="Inference device",
                        interactive=True,
                        scale=4
                    )
                    endless_checkbox = gr.Checkbox(label="Generate endlessly", value=False, scale=1)
                with gr.Row():
                    seed_slider = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0, randomize=True, scale=1)
                    randomize_seed_checkbox = gr.Checkbox(label="Randomize seed across runs", value=True, scale=0)
                    randomize_seed_button = gr.Button("Randomize seed", scale=0)
                with gr.Row():
                    guidance_scale_slider = gr.Slider(
                        label="Guidance scale for base",
                        minimum=2,
                        maximum=14,
                        step=0.1,
                        value=8.0,
                    )
                    num_inference_steps_slider = gr.Slider(
                        label="Number of inference steps for base",
                        minimum=1,
                        maximum=32,
                        step=1,
                        value=5,
                    )

                size_slider = gr.Slider(
                    label="Image size",
                    minimum=256,
                    maximum=1024,
                    step=64,
                    value=512
                )

        gr.Examples(
            examples=examples,
            inputs=prompt_text,
            outputs=result_img,
            cache_examples=False,
        )
        # clicking run
        gr.on(triggers=[prompt_text.submit, start_button.click],
              fn=generate_images,
              inputs=[prompt_text, seed_slider, size_slider, guidance_scale_slider, num_inference_steps_slider, randomize_seed_checkbox, device_dropdown, endless_checkbox],
              outputs=[result_img, result_time_label]
              )
        # clicking stop
        stop_button.click(stop)
        randomize_seed_button.click(lambda _: random.randint(0, MAX_SEED), inputs=seed_slider, outputs=seed_slider)

    return demo


def run_endless_lcm(model_name: str, safety_checker_model: str, local_network: bool = False, public_interface: bool = False):
    global hf_model_name
    hf_model_name = model_name
    server_name = "0.0.0.0" if local_network else None

    download_models(model_name, safety_checker_model)

    demo = build_ui()
    log.info("Demo is ready!")
    demo.launch(server_name=server_name, share=public_interface)


if __name__ == '__main__':
    # set up logging
    log.getLogger().setLevel(log.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="OpenVINO/LCM_Dreamshaper_v7-fp16-ov",
                        choices=["OpenVINO/LCM_Dreamshaper_v7-int8-ov", "OpenVINO/LCM_Dreamshaper_v7-fp16-ov"], help="Visual GenAI model to be used")
    parser.add_argument("--safety_checker_model", type=str, default="Falconsai/nsfw_image_detection",
                        choices=["Falconsai/nsfw_image_detection"], help="The model to verify if the generated image is NSFW")
    parser.add_argument("--local_network", action="store_true", help="Whether demo should be available in local network")
    parser.add_argument("--public", default=False, action="store_true", help="Whether interface should be available publicly")

    args = parser.parse_args()
    run_endless_lcm(args.model_name, args.safety_checker_model, args.local_network, args.public)
