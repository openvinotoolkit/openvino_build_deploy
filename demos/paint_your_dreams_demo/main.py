import argparse
import os
import random
import sys
import time
from functools import partial

import gradio as gr
import numpy as np
import openvino as ov
import torch
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from optimum.intel.openvino import OVLatentConsistencyModelPipeline

SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils")
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils import demo_utils as utils

stop_generating: bool = False
ov_pipeline: OVLatentConsistencyModelPipeline | None = None
safety_checker: StableDiffusionSafetyChecker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker")
ov_pipelines = {}

MAX_SEED = np.iinfo(np.int32).max


def get_available_devices() -> list[str]:
    core = ov.Core()
    return ["AUTO"] + list({device.split(".")[0] for device in core.available_devices})


def load_pipeline(model_name: str, device: str):
    global ov_pipeline

    if device in ov_pipelines:
        ov_pipeline = ov_pipelines[device]
    else:
        ov_config = {"CACHE_DIR": "cache"}
        ov_pipeline = OVLatentConsistencyModelPipeline.from_pretrained(model_name, compile=True, device=device,
                                                                       safety_checker=safety_checker, ov_config=ov_config)
        ov_pipelines[device] = ov_pipeline

    return device


def randomize_seed_fn(seed: int, randomize_seed: bool = True) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


def stop():
    global stop_generating
    stop_generating = True


def generate_images(prompt: str, seed: int, size: int, guidance_scale: float, num_inference_steps: int, randomize_seed: bool):
    global stop_generating

    stop_generating = False
    while True:
        if stop_generating:
            break

        if randomize_seed:
            seed = random.randint(0, MAX_SEED)

        torch.manual_seed(seed)
        np.random.seed(seed)

        start_time = time.time()
        result = ov_pipeline(prompt=prompt, num_inference_steps=num_inference_steps, width=size, height=size,
                             guidance_scale=guidance_scale).images
        end_time = time.time()

        result, nsfw = safety_checker(ov_pipeline.feature_extractor(result, return_tensors="pt").pixel_values, np.array(result))

        utils.draw_ov_watermark(result[0], size=0.60)

        processing_time = end_time - start_time
        yield result[0], round(processing_time, 5), seed


def build_ui(model_name: str):
    examples = [
        "portrait photo of a girl, photograph, highly detailed face, depth of field, moody light, golden hour,"
        "style by Dan Winters, Russell James, Steve McCurry, centered, extremely detailed, Nikon D850, award winning photography",
        "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
        "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
        "A photo of beautiful mountain with realistic sunset and blue lake, highly detailed, masterpiece",
    ]

    with gr.Blocks() as demo:
        with gr.Group():
            with gr.Row():
                prompt = gr.Text(
                    value=examples[0],
                    label="Prompt",
                    max_lines=1,
                    placeholder="Enter your prompt here",
                )
            with gr.Row():
                gr.Column(scale=1)
                with gr.Column(scale=6):
                    result = gr.Image(label="Generated image", elem_id="output_image", format="png")
                    with gr.Row():
                        result_time_label = gr.Text("", label="Processing Time", type="text")
                        result_device_label = gr.Text("AUTO", label="Device Name", type="text")
                    with gr.Row():
                        run_button = gr.Button("Start generation")
                        stop_button = gr.Button("Stop generation", interactive=False)
                gr.Column(scale=1)
            with gr.Accordion("Advanced options", open=False):
                device_dropdown = gr.Dropdown(
                    choices=get_available_devices(),
                    value="AUTO",
                    label="Inference device",
                    interactive=True
                )
                with gr.Row():
                    seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0, randomize=True,
                                     scale=1)
                    randomize_seed = gr.Checkbox(label="Randomize seed across runs", value=True, scale=0)
                    randomize_seed_button = gr.Button("Randomize seed", scale=0)
                with gr.Row():
                    guidance_scale = gr.Slider(
                        label="Guidance scale for base",
                        minimum=2,
                        maximum=14,
                        step=0.1,
                        value=8.0,
                    )
                    num_inference_steps = gr.Slider(
                        label="Number of inference steps for base",
                        minimum=1,
                        maximum=32,
                        step=1,
                        value=5,
                    )

                size = gr.Slider(
                    label="Image size",
                    minimum=256,
                    maximum=1024,
                    step=64,
                    value=512
                )

        gr.Examples(
            examples=examples,
            inputs=prompt,
            outputs=result,
            cache_examples=False,
        )
        gr.on(triggers=[prompt.submit, run_button.click], fn=lambda: gr.Button(interactive=False), outputs=run_button) \
            .then(lambda: gr.Button(interactive=True), outputs=stop_button) \
            .then(lambda: gr.Dropdown(interactive=False), outputs=device_dropdown) \
            .then(
            fn=generate_images,
            inputs=[
                prompt, seed, size, guidance_scale, num_inference_steps, randomize_seed
            ],
            outputs=[
                result, result_time_label, seed
            ],
        )
        stop_button.click(stop) \
            .then(lambda: gr.Button(interactive=True), outputs=run_button) \
            .then(lambda: gr.Button(interactive=False), outputs=stop_button) \
            .then(lambda: gr.Dropdown(interactive=True), outputs=device_dropdown)
        device_dropdown.change(lambda: gr.Button(interactive=False), outputs=run_button) \
            .then(partial(load_pipeline, model_name), inputs=device_dropdown, outputs=result_device_label) \
            .then(lambda: gr.Button(interactive=True), outputs=run_button)
        randomize_seed_button.click(lambda _: random.randint(0, MAX_SEED), inputs=seed, outputs=seed)

    return demo


def run_endless_lcm(model_name):
    load_pipeline(model_name, "AUTO")

    demo = build_ui(model_name)
    demo.launch(server_name="0.0.0.0")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="OpenVINO/LCM_Dreamshaper_v7-int8-ov",
                        help="Pose estimation model to be used")

    args = parser.parse_args()
    run_endless_lcm(args.model_name)
