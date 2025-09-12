import argparse
import logging as log
import os
import sys
import threading
import time
from collections import deque
from functools import partial
from pathlib import Path
from typing import List

import cv2
import numpy as np
import openvino as ov
import torch
from transformers import BlipProcessor, BlipVisionModel, BlipTextLMHeadModel, BlipForConditionalGeneration
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
import openvino_genai as ov_genai  

SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils")
sys.path.append(os.path.dirname(SCRIPT_DIR))
from utils import demo_utils as utils  # noqa: E402

MODEL_DIR = Path("model")

# Frame/caption buffers
current_frames = deque(maxlen=1)          # latest frame for inference
captions = deque(maxlen=1)                # latest caption for on-screen display
session_captions = deque(maxlen=1000)     # ALL captions for final summary

# Perf buffer
processing_times = deque(maxlen=100)

# Global sync primitives
global_stop_event = threading.Event()
global_frame_lock = threading.Lock()
global_result_lock = threading.Lock()


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
    input_dict["past_key_values"] = text_decoder_outs["past_key_values"]

    text_decoder.config.torchscript = True
    with torch.no_grad():
        ov_text_decoder = ov.convert_model(text_decoder, example_input=input_dict)
    ov.save_model(ov_text_decoder, output_dir / "blip_text_decoder_with_past.xml")


def download_and_convert_model(model_name: str) -> None:
    output_dir = MODEL_DIR / model_name
    processor = BlipProcessor.from_pretrained(model_name, use_fast=True)
    processor.save_pretrained(output_dir)

    blip_model = BlipForConditionalGeneration.from_pretrained(model_name)
    blip_model.text_decoder.save_pretrained(output_dir, safe_serialization=False)

    convert_vision_model(blip_model.vision_model, processor, output_dir)
    convert_decoder_model(blip_model.text_decoder, output_dir)


def load_models(model_name: str, device: str = "AUTO"):
    model_dir = MODEL_DIR / model_name
    vision_model_path = model_dir / "blip_vision_model.xml"
    text_decoder_path = model_dir / "blip_text_decoder_with_past.xml"

    if not vision_model_path.exists() or not text_decoder_path.exists():
        download_and_convert_model(model_name)

    core = ov.Core()
    core.set_property({"CACHE_DIR": "cache"})

    # set up the models for inference
    vision_model = core.compile_model(vision_model_path, device)
    text_decoder = core.compile_model(text_decoder_path, device)

    processor = BlipProcessor.from_pretrained(model_dir, use_fast=True)
    text_model = BlipTextLMHeadModel.from_pretrained(model_dir)
    text_model.forward = partial(text_decoder_forward, ov_text_decoder_with_past=text_decoder)

    text_cfg = text_model.config  # tokenizer-aware config (ids, etc.)
    return vision_model, text_model, processor, text_cfg


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

    return CausalLMOutputWithCrossAttentions(
        logits=logits, past_key_values=past_kv, hidden_states=None, attentions=None, cross_attentions=None
    )


def generate_caption(image: np.ndarray, vision_model: ov.CompiledModel, text_decoder: BlipTextLMHeadModel, processor: BlipProcessor, text_cfg) -> str:
    # Preprocess
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixel_values = np.array(processor(image).pixel_values)
    image_embeds = vision_model(np.array(pixel_values))[vision_model.output(0)]
    image_attention_mask = np.ones(image_embeds.shape[:-1], dtype=np.int64)

    bos_id = getattr(text_cfg, "bos_token_id", None) or getattr(text_cfg, "decoder_start_token_id", None) or 30522
    eos_id = getattr(text_cfg, "eos_token_id", None)
    sep_id = getattr(text_cfg, "sep_token_id", None)
    pad_id = getattr(text_cfg, "pad_token_id", None) or 0

    input_ids = np.array([[bos_id, eos_id if eos_id is not None else bos_id]], dtype=np.int64)
    outputs = text_decoder.generate(
        input_ids=torch.LongTensor(input_ids[:, :-1]),          # only BOS goes in
        eos_token_id=sep_id if sep_id is not None else eos_id,  # BLIP often stops on SEP
        pad_token_id=pad_id,
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_attention_mask,
        # (intentionally no num_beams/do_sample/max_new_tokens here to mirror your original)
    )
    return processor.decode(outputs[0], skip_special_tokens=True).strip()


def inference_worker(vision_model, text_decoder, processor, text_cfg):
    global current_frames, captions, session_captions, processing_times

    last_caption = ""
    errored = False

    while not global_stop_event.is_set():
        with global_frame_lock:
            frame = current_frames[-1] if len(current_frames) > 0 else np.zeros((1080, 1920, 3), dtype=np.uint8)

        start_time = time.perf_counter()
        try:
            caption = generate_caption(frame, vision_model, text_decoder, processor, text_cfg)
        except Exception as e:
            # keep UI alive even if one step fails
            if not errored:
                print(f"[CAPTION ERROR] {type(e).__name__}: {e}", flush=True)
                errored = True
            elapsed = time.perf_counter() - start_time
            with global_result_lock:
                processing_times.append(elapsed)
            time.sleep(0.05)
            continue

        elapsed = time.perf_counter() - start_time

        if caption and caption != last_caption:
            with global_result_lock:
                captions.append(caption)
                session_captions.append(caption)
                processing_times.append(elapsed)
            last_caption = caption
        else:
            with global_result_lock:
                processing_times.append(elapsed)


def _deduplicate_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for it in items:
        key = (it or "").strip()
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out


def summarize_captions_with_ov_llmpipeline(captions_list: List[str], ov_model_path: str, device: str, max_new_tokens: int, stream: bool = True) -> str:
    texts = _deduplicate_keep_order(captions_list)
    if not texts:
        return "No captions were collected."
    joined = " ".join(texts)

    pipe = ov_genai.LLMPipeline(ov_model_path, device)
    gen_cfg = ov_genai.GenerationConfig()
    gen_cfg.max_new_tokens = max_new_tokens  # other params use OV GenAI defaults

    prompt = (
        "You are an expert live-caption session summarizer. "
        "Write a concise paragraph describing what happened.\n\n"
        f"CAPTIONS:\n{joined}\n\nSUMMARY:"
    )

    if stream:
        # Streaming generation (token-by-token printing) with fallback
        buf: List[str] = []

        def _cb(subword: str):
            print(subword, end="", flush=True)
            buf.append(subword)
            return ov_genai.StreamingStatus.RUNNING

        try:
            print("[STREAM START]", flush=True)
            pipe.generate(prompt, gen_cfg, _cb)
            print("\n[STREAM END]", flush=True)
        finally:
            # ensure cursor ends on a newline even if exceptions happen
            print("", flush=True)

        text = ("".join(buf)).strip()
        if not text:
            # Fallback once to guarantee visible output
            text = (pipe.generate(prompt, gen_cfg) or "").strip()
            print(text, flush=True)
        return text or "Summary unavailable."
    else:
        final = pipe.generate(prompt, gen_cfg)
        return (final or "").strip() or "Summary unavailable."


def run(video_path: str, model_name: str, flip: bool, summary_ov_model: str, summary_max_new_tokens: int) -> None:
    global current_frames, captions, session_captions, processing_times
    log.getLogger().setLevel(log.INFO)

    qr_code = utils.get_qr_code(
        "https://github.com/openvinotoolkit/openvino_build_deploy/tree/master/demos/the_narrator_demo",
        with_embedded_image=True,
    )

    # NPU won't work with dynamic shape models, so exclude it
    device_mapping = utils.available_devices(exclude=["NPU"])
    device_type = "AUTO"

    vision_model, text_decoder, processor, text_cfg = load_models(model_name, device_type)

    # initialize video player
    if isinstance(video_path, str) and video_path.isnumeric():
        video_path = int(video_path)
    player = utils.VideoPlayer(video_path, size=(1920, 1080), fps=60, flip=flip)

    processing_times = deque(maxlen=100)
    title = "Press ESC to Exit"
    cv2.namedWindow(title, cv2.WINDOW_GUI_NORMAL)
    cv2.setWindowProperty(title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Start the inference thread
    worker = threading.Thread(
        target=inference_worker, args=(vision_model, text_decoder, processor, text_cfg), daemon=True
    )
    worker.start()

    # start a video stream
    player.start()
    t1 = time.time()
    shown_caption = ""
    captions_snapshot: List[str] = []  # will hold a frozen copy at ESC time
    while True:
        frame = player.next()
        if frame is None:
            print("Source ended")
            break

        with global_frame_lock:
            current_frames.append(frame)

        f_height, f_width = frame.shape[:2]

        with global_result_lock:
            t2 = time.time()
            if t2 - t1 > 1:
                latest = captions[-1] if len(captions) > 0 else ""
                if latest and latest != shown_caption:
                    shown_caption = latest
                t1 = t2
            
            # Get the mean processing time
            processing_time = np.mean(processing_times) * 1000 if processing_times else 0
            fps = 1000 / processing_time if processing_time > 0 else 0

        # Draw results on the frame
        utils.draw_text(frame, text=shown_caption, point=(f_width // 2, f_height - 50), center=True, font_scale=1.5, with_background=True)
        utils.draw_text(frame, text=f"Inference time: {processing_time:.0f}ms ({fps:.1f} FPS)", point=(10, 10))
        utils.draw_text(frame, text=f"Currently running {model_name} on {device_type}", point=(10, 50))
        utils.draw_ov_watermark(frame)
        utils.draw_qr_code(frame, qr_code)

        esc_text = "Press ESC for summary"
        esc_scale = 0.9
        y_under_status = 50 + 50 
        utils.draw_text(frame, text=esc_text, point=(10, y_under_status), center=False, font_scale=esc_scale, with_background=True)

        # show the output live
        cv2.imshow(title, frame)
        key = cv2.waitKey(1) & 0xFF  # robust ESC handling

        # ESC or 'q' to exit
        if key == 27 or key == ord("q"):
            # Freeze captions right now, then close the window immediately
            with global_result_lock:
                captions_snapshot = list(session_captions)
            cv2.destroyAllWindows()       # <-- close UI instantly
            player.stop()                  # stop grabbing frames
            global_stop_event.set()        # signal worker to stop
            break

        # Device switching (1..N)
        for i, dev in enumerate(device_mapping.keys()):
            if key == (ord("1") + i):
                if device_type != dev:
                    device_type = dev
                    # Stop current worker
                    global_stop_event.set()
                    worker.join(timeout=1)
                    global_stop_event.clear()

                    # Recompile models for the new device
                    vision_model, text_decoder, processor, text_cfg = load_models(model_name, device_type)
                    # Start a new inference worker
                    worker = threading.Thread(
                        target=inference_worker, args=(vision_model, text_decoder, processor, text_cfg), daemon=True
                    )
                    worker.start()
                    # Clear the processing times
                    with global_result_lock:
                        processing_times.clear()

    # === Stream the final summary AFTER the UI is gone ===
    try:
        print("\n" + "=" * 60)
        print("Session Summary (OpenVINO GenAI LLMPipeline):")
        print("-" * 60)
        if summary_ov_model:
            if not captions_snapshot:
                # last resort if snapshot somehow empty
                with global_result_lock:
                    captions_snapshot = list(session_captions)
            print(f"[INFO] Captions collected: {len(captions_snapshot)}", flush=True)
            print(f"[INFO] LLM path: {summary_ov_model}", flush=True)
            print(f"[INFO] Device used: {device_type}", flush=True)
            _ = summarize_captions_with_ov_llmpipeline(
                captions_snapshot,
                ov_model_path=summary_ov_model,
                device=device_type,                   # use the current device_type
                max_new_tokens=summary_max_new_tokens,
                stream=True,
            )
        else:
            print("[WARN] No --summary_ov_model provided. Skipping summarization.")
        print("=" * 60 + "\n")
    except Exception as e:
        print(f"[WARN] Failed to summarize captions with OpenVINO GenAI LLMPipeline: {e}")

    # Let the worker finish gracefully after summary is shown
    worker.join(timeout=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream", default="0", type=str, help="Path to a video file or the webcam number")
    parser.add_argument("--model_name", type=str, default="Salesforce/blip-image-captioning-base", help="Model to be used for captioning",
                        choices=["Salesforce/blip-image-captioning-base", "Salesforce/blip-image-captioning-large"])
    parser.add_argument("--flip", type=bool, default=True, help="Mirror input video")
    parser.add_argument("--summary_ov_model", type=str, default="./model/Qwen2.5-1.5B-Instruct-int4-ov", help="Path to an OV GenAI LLM directory")
    parser.add_argument("--summary_max_new_tokens", type=int, default=160, help="Max new tokens for the summary")

    args = parser.parse_args()
    run(args.stream, args.model_name, args.flip, args.summary_ov_model, args.summary_max_new_tokens)
