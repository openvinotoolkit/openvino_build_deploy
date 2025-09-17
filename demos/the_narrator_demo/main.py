import argparse
import logging as log
import os
import sys
import threading
import time
from collections import deque
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import openvino as ov
import torch
from transformers import BlipProcessor, BlipVisionModel, BlipTextLMHeadModel, BlipTextConfig, BlipForConditionalGeneration
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

import openvino_genai as ov_genai
from huggingface_hub import snapshot_download

SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils")
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils import demo_utils as utils

MODEL_DIR = Path("model")
TEXT_CONFIG = BlipTextConfig()

current_frames = deque(maxlen=1)
captions = deque(maxlen=100)   # keep history for summary

processing_times = deque(maxlen=100)

global_stop_event = threading.Event()

global_frame_lock = threading.Lock()
global_result_lock = threading.Lock()

VIDEO_SUMMARY_PROMPT = """
You are given captions generated independently for each frame of a webcam video stream. 
These captions may contain:
- Redundant or repeated descriptions of the same person or background.
- Small, frame-by-frame variations in appearance (e.g., shirt color, facial hair, glasses).
- Minor or trivial movements that are not meaningful (e.g., blinking, head tilts).
- Fragmented or inconsistent wording.

Your task:
1. Clean & Deduplicate: Merge repeated or overlapping frame-level descriptions into a single, coherent narrative.
2. Infer Context: Use the temporal sequence to reconstruct what is happening across the video.
3. Identify Main Events: Focus on meaningful actions, gestures, and scene changes rather than frame-by-frame noise.
4. Summarize Clearly: Produce a concise summary (1–2 sentences) describing what the webcam stream shows overall.
5. Ignore Noise: Do not include tiny repetitive movements, trivial frame changes, or redundant appearance details.

Input:
Webcam frame captions (in order):
{captions}

Output:
A concise, human-readable summary of the webcam video stream.
"""


def convert_vision_model(vision_model: BlipVisionModel, processor: BlipProcessor, output_dir: Path) -> None:
    vision_model.eval()
    inputs = processor(np.zeros((512, 512, 3), dtype=np.uint8), "sample string", return_tensors="pt")
    with torch.no_grad():
        ov_vision_model = ov.convert_model(vision_model, example_input=inputs["pixel_values"])
    ov.save_model(ov_vision_model, output_dir / "blip_vision_model.xml")


def convert_decoder_model(text_decoder: BlipTextLMHeadModel, output_dir: Path) -> None:
    text_decoder.eval()

    # prepare example inputs
    input_ids = torch.tensor([[30522]])  # begin of sequence token id
    attention_mask = torch.tensor([[1]])  # attention mask for input_ids
    encoder_hidden_states = torch.rand((1, 10, text_decoder.config.encoder_hidden_size))  # encoder last hidden state from text_decoder
    encoder_attention_mask = torch.ones((1, 10), dtype=torch.long)  # attention mask for encoder hidden states

    input_dict = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "encoder_hidden_states": encoder_hidden_states,
        "encoder_attention_mask": encoder_attention_mask,
    }
    text_decoder_outs = text_decoder(**input_dict)
    # extend input dictionary with hidden states from previous step
    input_dict["past_key_values"] = text_decoder_outs["past_key_values"]

    text_decoder.config.torchscript = True
    with torch.no_grad():
        ov_text_decoder = ov.convert_model(text_decoder, example_input=input_dict)
    ov.save_model(ov_text_decoder, output_dir / "blip_text_decoder_with_past.xml")


def download_and_convert_captioning_model(model_name: str) -> None:
    output_dir = MODEL_DIR / model_name
    
    processor = BlipProcessor.from_pretrained(model_name, use_fast=True)
    processor.save_pretrained(output_dir)

    blip_model = BlipForConditionalGeneration.from_pretrained(model_name)
    blip_model.text_decoder.save_pretrained(output_dir, safe_serialization=False)

    convert_vision_model(blip_model.vision_model, processor, output_dir)
    convert_decoder_model(blip_model.text_decoder, output_dir)


def load_models(model_name: str, device: str = "AUTO") -> tuple[ov.CompiledModel, BlipTextLMHeadModel, BlipProcessor]:
    model_dir = MODEL_DIR / model_name
    vision_model_path = model_dir / "blip_vision_model.xml"
    text_decoder_path = model_dir / "blip_text_decoder_with_past.xml"

    if not vision_model_path.exists() or not text_decoder_path.exists():
        download_and_convert_captioning_model(model_name)

    core = ov.Core()
    core.set_property({"CACHE_DIR": "cache"})

    # set up the models for inference
    vision_model = core.compile_model(vision_model_path, device)
    text_decoder = core.compile_model(text_decoder_path, device)

    processor = BlipProcessor.from_pretrained(model_dir, use_fast=True)
    text_model = BlipTextLMHeadModel.from_pretrained(model_dir)
    text_model.forward = partial(text_decoder_forward, ov_text_decoder_with_past=text_decoder)

    return vision_model, text_model, processor


def init_past_inputs(model_inputs: list) -> list[ov.Tensor]:
    past_inputs = []
    for input_tensor in model_inputs[4:]:
        partial_shape = input_tensor.partial_shape
        partial_shape[0] = 1
        partial_shape[2] = 0
        past_inputs.append(ov.Tensor(ov.Type.f32, partial_shape.get_shape()))
    return past_inputs


def text_decoder_forward(ov_text_decoder_with_past: ov.CompiledModel, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                         past_key_values: list[ov.Tensor], encoder_hidden_states: torch.Tensor, encoder_attention_mask: torch.Tensor,
                         **kwargs) -> CausalLMOutputWithCrossAttentions:
    inputs = [input_ids, attention_mask, encoder_hidden_states, encoder_attention_mask]
    if past_key_values is None:
        inputs.extend(init_past_inputs(ov_text_decoder_with_past.inputs))
    else:
        inputs.extend(past_key_values)

    outputs = ov_text_decoder_with_past(inputs)
    logits = torch.from_numpy(outputs[0])
    past_kv = list(outputs.values())[1:]

    return CausalLMOutputWithCrossAttentions(logits=logits, past_key_values=past_kv, hidden_states=None,
                                             attentions=None, cross_attentions=None)


def generate_caption(image: np.array, vision_model: ov.CompiledModel, text_decoder: BlipTextLMHeadModel, processor: BlipProcessor) -> str:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixel_values = np.array(processor(image).pixel_values)
    image_embeds = vision_model(np.array(pixel_values))[vision_model.output(0)]
    
    image_attention_mask = np.ones(image_embeds.shape[:-1], dtype=np.int64)
    input_ids = np.array([[TEXT_CONFIG.bos_token_id, TEXT_CONFIG.eos_token_id]], dtype=np.int64)

    outputs = text_decoder.generate(
        input_ids=torch.LongTensor(input_ids[:, :-1]),
        eos_token_id=TEXT_CONFIG.sep_token_id,
        pad_token_id=TEXT_CONFIG.pad_token_id,
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_attention_mask
    )
    return processor.decode(outputs[0], skip_special_tokens=True)


def inference_worker(vision_model, text_decoder, processor):
    global current_frames, captions, processing_times

    while not global_stop_event.is_set():
        if len(current_frames) == 0:
            time.sleep(0.01)
            continue

        with global_frame_lock:
            frame = current_frames[-1]

        start_time = time.perf_counter()
        caption = generate_caption(frame, vision_model, text_decoder, processor)
        elapsed = time.perf_counter() - start_time

        with global_result_lock:
            if len(captions) == 0 or caption != captions[-1]:
                captions.append(caption)
            processing_times.append(elapsed)


def download_summarization_model(model_name: str) -> Path:
    is_openvino_model = model_name.startswith("OpenVINO/")

    if not is_openvino_model:
        raise ValueError(f"Model '{model_name}' is not an OpenVINO pre-optimized model. Please provide a valid model from the OpenVINO organization on Hugging Face.")

    output_dir = MODEL_DIR / model_name
    if not output_dir.exists():
        log.info(f"Downloading {model_name}...")
        snapshot_download(model_name, local_dir=output_dir, resume_download=True)

    return output_dir


def summarize_session(captions_list: list[str], ov_model_path: Path, device: str) -> None:
    if not captions_list:
        log.warning("No captions collected.")
        return

    log.info("Summarization model loading...")

    pipe = ov_genai.LLMPipeline(ov_model_path, device)

    log.info("============================================================")
    log.info("Session Summary (OpenVINO GenAI LLMPipeline):")
    log.info("============================================================")

    buffer = []
    def streamer(subword: str):
        print(subword, end="", flush=True)
        buffer.append(subword)
        return ov_genai.StreamingStatus.RUNNING

    pipe.generate(VIDEO_SUMMARY_PROMPT.format(captions="\n".join(captions_list)), streamer=streamer)


def run(video_path: str, captioning_model: str, summary_model: str, flip: bool = True) -> None:
    global current_frames, captions, processing_times
    # set up logging
    log.getLogger().setLevel(log.INFO)

    qr_code = utils.get_qr_code("https://github.com/openvinotoolkit/openvino_build_deploy/tree/master/demos/the_narrator_demo", with_embedded_image=True)

    # NPU won't work with the dynamic shape models, so we exclude it
    device_mapping = utils.available_devices(exclude=["NPU"])
    device_type = "AUTO"

    # download the summary model
    local_summary_model = download_summarization_model(summary_model)

    vision_model, text_decoder, processor = load_models(captioning_model, device_type)

    # initialize video player to deliver frames
    if isinstance(video_path, str) and video_path.isnumeric():
        video_path = int(video_path)
    player = utils.VideoPlayer(video_path, size=(1920, 1080), fps=60, flip=flip)

    # keep at most 100 last times
    processing_times = deque(maxlen=100)

    title = "Press ESC to Exit"
    cv2.namedWindow(title, cv2.WINDOW_GUI_NORMAL)
    cv2.setWindowProperty(title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Start the inference thread
    worker = threading.Thread(
        target=inference_worker,
        args=(vision_model, text_decoder, processor),
        daemon=True
    )
    worker.start()

    # start a video stream
    player.start()
    t1 = time.time()
    caption = ""
    while True:
        # Grab the frame.
        frame = player.next()
        if frame is None:
            print("Source ended")
            break

        # Update the latest frame for inference
        with global_frame_lock:
            current_frames.append(frame)

        f_height, f_width = frame.shape[:2]

        # Get the latest caption
        with global_result_lock:
            t2 = time.time()
            # update the caption only if the time difference is significant, otherwise it will be flickering
            if t2 - t1 > 1 or not caption:
                caption = captions[-1] if len(captions) > 0 else ""
                t1 = t2

            # Get the mean processing time
            processing_time = np.mean(processing_times) * 1000 if processing_times else 0
            fps = 1000 / processing_time if processing_time > 0 else 0

        # Draw the results on the frame
        utils.draw_text(frame, text=caption, point=(f_width // 2, f_height - 50), center=True, font_scale=1.5, with_background=True)
        utils.draw_text(frame, text=f"Inference time: {processing_time:.0f}ms ({fps:.1f} FPS)", point=(10, 10))
        utils.draw_text(frame, text=f"Currently running {captioning_model} on {device_type}", point=(10, 50))
        utils.draw_text(frame, text=f"Press ESC to exit and get text summary", point=(10, 90))

        utils.draw_ov_watermark(frame)
        utils.draw_qr_code(frame, qr_code)

        # show the output live
        cv2.imshow(title, frame)
        key = cv2.waitKey(1)
        # escape = 27 or 'q' to close the app
        if key == 27 or key == ord('q'):
            break

        for i, dev in enumerate(device_mapping.keys()):
            if key == ord('1') + i:
                if device_type != dev:
                    device_type = dev
                    # Stop the current worker
                    global_stop_event.set()
                    worker.join(timeout=1)
                    global_stop_event.clear()

                    # Recompile models for the new device
                    vision_model, text_decoder, processor = load_models(captioning_model, device_type)
                    # Start a new inference worker
                    worker = threading.Thread(
                        target=inference_worker,
                        args=(vision_model, text_decoder, processor),
                        daemon=True
                    )
                    worker.start()
                    # Clear the processing times
                    with global_result_lock:
                        processing_times.clear()

    # stop the stream
    player.stop()
    global_stop_event.set()
    # wait 5s to finish inference - should be enough even for weak devices
    worker.join(timeout=5)
    # clean-up windows
    cv2.destroyAllWindows()

    # summarize the session
    summarize_session(list(captions), local_summary_model, device_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream", default="0", type=str, help="Path to a video file or the webcam number")
    parser.add_argument("--captioning_model", type=str, default="Salesforce/blip-image-captioning-base", help="Model to be used for captioning",
                        choices=["Salesforce/blip-image-captioning-base", "Salesforce/blip-image-captioning-large"])
    parser.add_argument("--summary_model", type=str, default="OpenVINO/qwen2.5-1.5b-instruct-int4-ov", help="Path or HF repo ID for OV pre-optimized GenAI LLM")
    parser.add_argument("--flip", type=bool, default=True, help="Mirror input video")
    args = parser.parse_args()
    run(args.stream, args.captioning_model, args.summary_model, args.flip)
