import argparse
import asyncio
import logging as log
import os
import random
import sys
import time
from functools import partial
from pathlib import Path
from typing import Optional

import cv2
import gradio as gr
import numpy as np
import openvino as ov
import openvino_genai as genai
import tqdm
from PIL import Image
from huggingface_hub import snapshot_download
from optimum.intel.openvino import OVModelForImageClassification
from transformers import Pipeline, pipeline, AutoProcessor

SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils")
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils import demo_utils as utils

MAX_SEED = np.iinfo(np.int32).max
MODEL_DIR = Path("model")

safety_checker: Optional[Pipeline] = None

ov_pipelines = {}

stop_generating: bool = True
hf_model_name: Optional[str] = None

dreamshaper = {
    "guidance_scale_value": 8,
    "num_inference_steps": 5,
    "strength_value": 0.8
}

dreamlike_anime = {
    "guidance_scale_value": 7.5,
    "num_inference_steps": 50,
    "strength_value": 0.2
}
flux = {
    "guidance_scale_value": 8,
    "num_inference_steps": 5,
    "strength_value": 0.8
}


def download_models(model_name, safety_checker_model: str) -> None:
    global safety_checker

    is_openvino_model = model_name.split("/")[0] == "OpenVINO"

    output_dir = MODEL_DIR / model_name
    if not output_dir.exists():
        if is_openvino_model:
            snapshot_download(model_name, local_dir=output_dir, resume_download=True)
        else:
            if model_name == "dreamlike-art/dreamlike-anime-1.0":
                output_dir_dream = MODEL_DIR / "dreamlike_anime_1_0_fp16_ov"
                global hf_model_name
                hf_model_name = "dreamlike_anime_1_0_fp16_ov"
                if not output_dir_dream.exists():
                    os.system(
                        'optimum-cli export openvino --model dreamlike-art/dreamlike-anime-1.0 --task stable-diffusion --weight-format fp16  ' + str(
                            MODEL_DIR) + '/dreamlike_anime_1_0_fp16_ov')
            else:
                raise ValueError(f"Model {model_name} is not supported")

    safety_checker_dir = MODEL_DIR / safety_checker_model
    if not safety_checker_dir.exists():
        model = OVModelForImageClassification.from_pretrained(safety_checker_model, export=True, compile=False)
        model.save_pretrained(safety_checker_dir)
        processor = AutoProcessor.from_pretrained(safety_checker_model)
        processor.save_pretrained(safety_checker_dir)

    safety_checker = pipeline("image-classification", model=OVModelForImageClassification.from_pretrained(safety_checker_dir),
                              image_processor=AutoProcessor.from_pretrained(safety_checker_dir))


async def create_pipeline(model_dir: Path, device: str, size: int, pipeline: str) -> genai.Text2ImagePipeline | genai.Image2ImagePipeline | genai.InpaintingPipeline:
    ov_config = {"CACHE_DIR": "cache"}

    if pipeline == "text2image":
        ov_pipeline = genai.Text2ImagePipeline(model_dir)
    elif pipeline == "image2image":
        ov_pipeline = genai.Image2ImagePipeline(model_dir)
    elif pipeline == "inpainting":
        ov_pipeline = genai.InpaintingPipeline(model_dir)
    else:
        raise ValueError(f"Unknown pipeline: {pipeline}")

    ov_pipeline.reshape(1, size, size, ov_pipeline.get_generation_config().guidance_scale)
    ov_pipeline.compile(device, config=ov_config)

    return ov_pipeline


async def load_pipeline(model_name: str, device: str, size: int, pipeline: str) -> genai.Text2ImagePipeline | genai.Image2ImagePipeline | genai.InpaintingPipeline:
    model_dir = MODEL_DIR / model_name

    if (device, pipeline) not in ov_pipelines:
        ov_pipelines[(device, pipeline)] = await create_pipeline(model_dir, device, size, pipeline)

    return ov_pipelines[(device, pipeline)]


async def stop():
    global stop_generating
    stop_generating = True


progress_bar = None
def progress(step, num_steps, latent) -> bool:
    global progress_bar
    if progress_bar is None:
        progress_bar = tqdm.tqdm(total=num_steps)

    progress_bar.update()

    if step == num_steps - 1:
        progress_bar = None

    return False


async def generate_images(input_image_mask: np.ndarray, prompt: str, seed: int, guidance_scale: float, num_inference_steps: int,
                          strength: float, randomize_seed: bool, device: str, endless_generation: bool, size: int) -> tuple[np.ndarray, float]:
    global stop_generating
    stop_generating = not endless_generation

    input_image = input_image_mask["background"][:, :, :3]
    image_mask = input_image_mask["layers"][0][:, :, 3:]

    # ensure image is square
    input_image = utils.crop_center(input_image)
    input_image = cv2.resize(input_image, (size, size))
    image_mask = cv2.resize(image_mask, (size, size), interpolation=cv2.INTER_NEAREST)
    image_mask = cv2.cvtColor(image_mask, cv2.COLOR_GRAY2BGR)

    while True:
        if randomize_seed:
            seed = random.randint(0, MAX_SEED)

        start_time = time.time()
        if input_image.any():

            # inpainting pipeline
            if image_mask.any():
                ov_pipeline = await load_pipeline(hf_model_name, device, size, "inpainting")
                result = ov_pipeline.generate(prompt=prompt, image=ov.Tensor(input_image[None]), mask_image=ov.Tensor(image_mask[None]), num_inference_steps=num_inference_steps,
                                              width=size, height=size, guidance_scale=guidance_scale, strength=1.0 - strength, rng_seed=seed, callback=progress).data[0]
            # image2image pipeline
            else:
                ov_pipeline = await load_pipeline(hf_model_name, device, size, "image2image")
                if "dreamlike" in hf_model_name:
                    # ensure image is square
                    input_image = utils.crop_center(input_image)
                    input_image = cv2.resize(input_image, (size, size))

                result = ov_pipeline.generate(prompt=prompt, image=ov.Tensor(input_image[None]),
                                              num_inference_steps=num_inference_steps, width=size, height=size,
                                              guidance_scale=guidance_scale, strength=1.0 - strength, rng_seed=seed,
                                              callback=progress).data[0]

        # text2image pipeline
        else:
            ov_pipeline = await load_pipeline(hf_model_name, device, size, "text2image")
            result = ov_pipeline.generate(prompt=prompt, num_inference_steps=num_inference_steps, width=size, height=size,
                                          guidance_scale=guidance_scale, rng_seed=seed, callback=progress).data[0]
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


def build_ui(image_size: int) -> gr.Interface:
    examples_t2i = [
        "A sail boat on a grass field with mountains in the morning and sunny day",
        "A beautiful sunset with a sail boat on the ocean, photograph, highly detailed, golden hour, Nikon D850",
        "Portrait photo of a girl, photograph, highly detailed face, depth of field, moody light, golden hour,"
        "Style by Dan Winters, Russell James, Steve McCurry, centered, extremely detailed, Nikon D850, award winning photography"
    ]

    examples_i2i = [
        "Make me a superhero, 8k",
        "Make me a beautiful cyborg with golden hair, 8k",
        "Make me an astronaut, cold color palette, muted colors, 8k"
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
                    with gr.Row(equal_height=True):
                        input_image = gr.ImageMask(label="Input image (leave blank for text2image generation)", sources=["webcam", "clipboard", "upload"])
                        result_img = gr.Image(label="Generated image", elem_id="output_image", format="png")
                    with gr.Row():
                        result_time_label = gr.Text("", label="Inference time", type="text")
                    with gr.Row(equal_height=True):
                        device_dropdown = gr.Dropdown(choices=utils.available_devices(), value="AUTO", label="Inference device", scale=4)
                        endless_checkbox = gr.Checkbox(label="Generate endlessly", value=False)
                        with gr.Column(scale=1):
                            start_button = gr.Button("Start generation", variant="primary")
                            stop_button = gr.Button("Stop generation", variant="secondary")

            with gr.Accordion("Advanced options", open=False):
                if "dreamlike" in hf_model_name:
                    dictionary = dreamlike_anime
                if "Dreamshaper" in hf_model_name:
                    dictionary = dreamshaper
                if "FLUX" in hf_model_name:
                    dictionary = flux
                with gr.Row():
                    seed_slider = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0, randomize=True, scale=1)
                    randomize_seed_checkbox = gr.Checkbox(label="Randomize seed across runs", value=True, scale=0)
                    randomize_seed_button = gr.Button("Randomize seed", scale=0)
                with gr.Row():
                    strength_slider = gr.Slider(label="Input image influence strength", minimum=0.0, maximum=1.0,
                                                step=0.01, value=dictionary["strength_value"])
                    guidance_scale_slider = gr.Slider(label="Guidance scale for base", minimum=2, maximum=14, step=0.1,
                                                      value=dictionary["guidance_scale_value"])
                    num_inference_steps_slider = gr.Slider(label="Number of inference steps for base", minimum=1,
                                                           maximum=32, step=1,
                                                           value=dictionary["num_inference_steps"], )

        gr.Examples(label="Examples for Text2Image", examples=examples_t2i, inputs=prompt_text, outputs=result_img, cache_examples=False)
        gr.Examples(label="Examples for Image2Image", examples=examples_i2i, inputs=prompt_text, outputs=result_img, cache_examples=False)

        # clicking run
        gr.on(
            triggers=[prompt_text.submit, start_button.click],
            fn=lambda: (gr.Button(variant="secondary"), gr.Button(variant="primary")),
            outputs=[start_button, stop_button]
        ).then(
            partial(generate_images, size=image_size),
            inputs=[input_image, prompt_text, seed_slider, guidance_scale_slider, num_inference_steps_slider,
                    strength_slider, randomize_seed_checkbox, device_dropdown, endless_checkbox],
            outputs=[result_img, result_time_label]
        ).then(
            lambda: (gr.Button(variant="primary"), gr.Button(variant="secondary")), outputs=[start_button, stop_button]
        )
        # clicking stop
        stop_button.click(stop)
        randomize_seed_button.click(lambda _: random.randint(0, MAX_SEED), inputs=seed_slider, outputs=seed_slider)

    return demo


def run_endless_lcm(model_name: str, safety_checker_model: str, image_size: int, local_network: bool = False, public_interface: bool = False) -> None:
    global hf_model_name
    hf_model_name = model_name
    server_name = "0.0.0.0" if local_network else None

    download_models(model_name, safety_checker_model)

    demo = build_ui(image_size)
    log.info("Demo is ready!")
    demo.launch(server_name=server_name, share=public_interface)


if __name__ == '__main__':
    # set up logging
    log.getLogger().setLevel(log.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="OpenVINO/LCM_Dreamshaper_v7-fp16-ov",
                        help="Visual GenAI model to be used",
                        choices=["OpenVINO/LCM_Dreamshaper_v7-int8-ov", "OpenVINO/LCM_Dreamshaper_v7-fp16-ov",
                                 "OpenVINO/FLUX.1-schnell-int4-ov",
                                 "OpenVINO/FLUX.1-schnell-int8-ov", "OpenVINO/FLUX.1-schnell-fp16-ov",
                                 "dreamlike-art/dreamlike-anime-1.0"])
    parser.add_argument("--safety_checker_model", type=str, default="Falconsai/nsfw_image_detection",
                        choices=["Falconsai/nsfw_image_detection"], help="The model to verify if the generated image is NSFW")
    parser.add_argument("--image_size", type=int, default=512, help="The image size to generate")
    parser.add_argument("--local_network", action="store_true", help="Whether demo should be available in local network")
    parser.add_argument("--public", default=False, action="store_true", help="Whether interface should be available publicly")

    args = parser.parse_args()
    run_endless_lcm(args.model_name, args.safety_checker_model, args.image_size, args.local_network, args.public)
