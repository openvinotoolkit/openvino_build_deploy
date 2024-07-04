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

MAX_SEED = np.iinfo(np.int32).max

ov_pipeline: OVLatentConsistencyModelPipeline | None = None
safety_checker: StableDiffusionSafetyChecker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker")

ov_pipelines = {}

stop_generating: bool = True

prompt = ""
randomize_seed = True
seed = 0
guidance_scale = 8.0
num_inference_steps = 5
size = 512
images_to_generate = 1


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


def stop():
    global stop_generating
    stop_generating = True


def generate_images():
    global stop_generating

    stop_generating = False
    counter = 0
    while counter < images_to_generate:
        if stop_generating:
            break

        local_seed = seed
        if randomize_seed:
            local_seed = random.randint(0, MAX_SEED)

        torch.manual_seed(local_seed)
        np.random.seed(local_seed)

        start_time = time.time()
        result = ov_pipeline(prompt=prompt, num_inference_steps=num_inference_steps, width=size, height=size,
                             guidance_scale=guidance_scale).images
        end_time = time.time()

        result, nsfw = safety_checker(ov_pipeline.feature_extractor(result, return_tensors="pt").pixel_values, np.array(result))

        utils.draw_ov_watermark(result[0], size=0.60)

        processing_time = end_time - start_time
        yield result[0], round(processing_time, 5)

        counter += 1


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
                prompt_text = gr.Text(
                    label="Prompt",
                    max_lines=1,
                    placeholder="Enter your prompt here",
                )
            with gr.Row():
                with gr.Column():
                    result_img = gr.Image(label="Generated image", elem_id="output_image", format="png")
                    with gr.Row():
                        result_time_label = gr.Text("", label="Processing Time", type="text")
                        result_device_label = gr.Text("AUTO", label="Device Name", type="text")
                    with gr.Row() as endless_gen_row:
                        start_button = gr.Button("Start generation")
                        stop_button = gr.Button("Stop generation")
            with gr.Accordion("Advanced options", open=False):
                with gr.Row():
                    device_dropdown = gr.Dropdown(
                        choices=get_available_devices(),
                        value="AUTO",
                        label="Inference device",
                        interactive=True,
                        scale=4
                    )
                    endless_checkbox = gr.Checkbox(label="Generate endlessly", value=False, scale=1)
                with gr.Row():
                    seed_slider = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=seed, randomize=True, scale=1)
                    randomize_seed_checkbox = gr.Checkbox(label="Randomize seed across runs", value=randomize_seed, scale=0)
                    randomize_seed_button = gr.Button("Randomize seed", scale=0)
                with gr.Row():
                    guidance_scale_slider = gr.Slider(
                        label="Guidance scale for base",
                        minimum=2,
                        maximum=14,
                        step=0.1,
                        value=guidance_scale,
                    )
                    num_inference_steps_slider = gr.Slider(
                        label="Number of inference steps for base",
                        minimum=1,
                        maximum=32,
                        step=1,
                        value=num_inference_steps,
                    )

                size_slider = gr.Slider(
                    label="Image size",
                    minimum=256,
                    maximum=1024,
                    step=64,
                    value=size
                )

        gr.Examples(
            examples=examples,
            inputs=prompt_text,
            outputs=result_img,
            cache_examples=False,
        )
        # clicking run
        gr.on(triggers=[prompt_text.submit, start_button.click], fn=lambda: gr.Dropdown(interactive=False), outputs=device_dropdown) \
            .then(generate_images, outputs=[result_img, result_time_label])
        # clicking stop
        stop_button.click(stop) \
            .then(lambda: gr.Dropdown(interactive=True), outputs=device_dropdown)
        # changing device
        device_dropdown.change(lambda: gr.Row(visible=False), outputs=endless_gen_row) \
            .then(partial(load_pipeline, model_name), inputs=device_dropdown, outputs=result_device_label) \
            .then(lambda: gr.Row(visible=True), outputs=endless_gen_row)
        # changing parameters
        endless_checkbox.change(lambda x: globals().update(images_to_generate=MAX_SEED if x else 1), inputs=endless_checkbox)
        randomize_seed_button.click(lambda _: random.randint(0, MAX_SEED), inputs=seed_slider, outputs=seed_slider)
        randomize_seed_checkbox.change(lambda x: globals().update(randomize_seed=x), inputs=randomize_seed_checkbox)
        size_slider.change(lambda x: globals().update(size=x), inputs=size_slider)
        guidance_scale_slider.change(lambda x: globals().update(guidance_scale=x), inputs=guidance_scale_slider)
        seed_slider.change(lambda x: globals().update(seed=x), inputs=seed_slider)
        num_inference_steps_slider.change(lambda x: globals().update(num_inference_steps=x), inputs=num_inference_steps_slider)
        prompt_text.change(lambda x: globals().update(prompt=x), inputs=prompt_text)

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
