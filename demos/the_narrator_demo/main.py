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
from transformers import AutoProcessor
from optimum.intel.openvino import OVModelForVisualCausalLM
from transformers import LlavaNextVideoProcessor
from optimum.intel import OVWeightQuantizationConfig, OVPipelineQuantizationConfig
from PIL import Image
import tempfile

SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils")
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils import demo_utils as utils

MODEL_DIR = Path("model")
TEXT_CONFIG = BlipTextConfig()

max_caption_length = 6
current_frames = deque(maxlen=max_caption_length)
captions = deque(maxlen=max_caption_length)

processing_times = deque(maxlen=100)

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
    # extend input dictionary with hidden states from previous step
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


def load_models(model_name: str, device: str = "AUTO") -> tuple[ov.CompiledModel, BlipTextLMHeadModel, BlipProcessor]:
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

    return vision_model, text_model, processor

def load_llava_video_models(model_name: str, device: str = "CPU") -> tuple:
    """Load LLaVA-NeXT-Video model with Intel HF Optimum (OpenVINO)"""
    model_dir = MODEL_DIR / model_name.replace("/", "_")
    
    # Download and convert if not exists
    if not model_dir.exists():
        download_and_convert_llava_video(model_name)
    
    print(f"Loading LLaVA-NeXT-Video Intel HF Optimum (OpenVINO) models from {model_dir}")
    
    # Load from the saved local directory, not from model_name
    model = OVModelForVisualCausalLM.from_pretrained(
        str(model_dir),
        trust_remote_code=True
    )

    processor = AutoProcessor.from_pretrained(str(model_dir))
    
    print(f"Successfully loaded LLaVA-NeXT-Video Intel HF Optimum (OpenVINO) model on {device}")
    return model, processor

def download_and_convert_llava_video(model_name: str) -> None:
    """Download pre-converted LLaVA-NeXT-Video OpenVINO model from Hugging Face"""
    model_dir = MODEL_DIR / model_name.replace("/", "_")
    output_dir = model_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading LLaVA-NeXT-Video OpenVINO model...")
    
    # Quantize and export directly from the model id
    q_model = OVModelForVisualCausalLM.from_pretrained(
        model_name,
        export=True,
        trust_remote_code=True,
        quantization_config=OVWeightQuantizationConfig(bits=8),
    )
    q_model.save_pretrained(output_dir)

    # Save processor
    processor = LlavaNextVideoProcessor.from_pretrained(model_name)
    processor.save_pretrained(output_dir)

    print(f"LLaVA-NeXT-Video model successfully downloaded and saved to {output_dir}")    


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

def generate_caption_video(current_frames,model, processor) -> str:
    print("Generating caption for video frames...")

    if not current_frames:
        return "No frames available"
    
    pil_frames=[]
    frames_copy=current_frames.copy()

    for frame in frames_copy:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
    
        #Resize
        pil_image = pil_image.resize((128, 96),Image.Resampling.LANCZOS)
        pil_frames.append(pil_image)

    bgr_frames = [cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) for img in pil_frames]

    # Write to temporary video file
    height, width = bgr_frames[0].shape[:2]
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_video_path = temp_video.name
    temp_video.close()
    
    try:
        # OpenCV VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(temp_video_path, fourcc, 5.0, (width, height))
        for frame in bgr_frames:
            out.write(frame)
        out.release()

        conversation_with_frames = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe what you see in the video in no more than 20 words."},
                    {"type": "video", "path": temp_video_path},
                ],
            },
        ]

        inputs_with_frames = processor.apply_chat_template(
            conversation_with_frames,
            num_frames=len(bgr_frames),
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
        )

        out_with_frames = model.generate(**inputs_with_frames, max_new_tokens=60)
    
        response = processor.batch_decode(
            out_with_frames,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]
        
        if "ASSISTANT:" in response:
            description = response.split("ASSISTANT:")[-1].strip()
        else:
            description = response.strip()
        return description
    
    finally:
        # Clean up the temporary video file
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

def generate_caption(image: np.array, vision_model: ov.CompiledModel, text_decoder: BlipTextLMHeadModel, processor: BlipProcessor) -> str:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixel_values = np.array(processor(image).pixel_values)
    print("Extracting image features...")
    image_embeds = vision_model(np.array(pixel_values))[vision_model.output(0)]

    image_attention_mask = np.ones(image_embeds.shape[:-1], dtype=np.int64)
    input_ids = np.array([[TEXT_CONFIG.bos_token_id, TEXT_CONFIG.eos_token_id]], dtype=np.int64)
    print("Generating caption for image...")
    outputs = text_decoder.generate(
        input_ids=torch.LongTensor(input_ids[:, :-1]),
        eos_token_id=TEXT_CONFIG.sep_token_id,
        pad_token_id=TEXT_CONFIG.pad_token_id,
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_attention_mask
    )
    return processor.decode(outputs[0], skip_special_tokens=True)


def inference_worker(model,processor,vision_model, text_decoder,video_input: bool = False) -> None:
    global current_frames, captions, processing_times
    if video_input == True:
        frames = max_caption_length
    else:
        frames = 0
    while not global_stop_event.is_set():
        with global_frame_lock:
            frame = current_frames.pop() if len(current_frames) > frames else np.zeros((1080, 1920, 3), dtype=np.uint8)

        start_time = time.perf_counter()

        if video_input ==True :
            if len(current_frames) >= 6:
                caption = generate_caption_video(current_frames, model, processor)
            else :
                caption = "Waiting.."
        else:
            caption = generate_caption(frame,vision_model,text_decoder, processor)
        
        elapsed = time.perf_counter() - start_time

        with global_result_lock:
            captions.append(caption)
            processing_times.append(elapsed)


def run(video_path: str, model_name: str, flip: bool = True, video_input: bool = False) -> None:
    global current_frames, captions, processing_times
    # set up logging
    log.getLogger().setLevel(log.INFO)

    qr_code = utils.get_qr_code("https://github.com/openvinotoolkit/openvino_build_deploy/tree/master/demos/the_narrator_demo", with_embedded_image=True)

    # NPU won't work with the dynamic shape models, so we exclude it
    device_mapping = utils.available_devices(exclude=["NPU"])
    device_type = "AUTO"

    #Downloadn and convert Image and Video models
    vision_model, text_decoder, processor = load_models(model_name, device_type)
    
    #For video captioning
    model_name_video = "llava-hf/LLaVA-NeXT-Video-7B-hf"
    device_type_video = "AUTO"
    
    # Load video input model and processor
    model_video, processor_video = load_llava_video_models(model_name_video, device_type_video)

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
            kwargs={
                "video_input": False,
                "model": None,
                "processor": processor,
                "vision_model": vision_model,
                "text_decoder": text_decoder
            },
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
                if video_input == True:
                    if len(captions) > 0:
                        caption=captions[-1]
                else:
                    caption = captions.pop() if len(captions) > 0 else ""
                t1 = t2

            # Get the mean processing time
            processing_time = np.mean(processing_times) * 1000 if processing_times else 0
            fps = 1000 / processing_time if processing_time > 0 else 0

        # Draw the results on the frame
        utils.draw_text(frame, text=caption, point=(f_width // 2, f_height - 50), center=True, font_scale=1.5, with_background=True)
        utils.draw_text(frame, text=f"Inference time: {processing_time:.0f}ms ({fps:.1f} FPS)", point=(10, 10))
        utils.draw_text(frame, text=f"Currently running {model_name} on {device_type}", point=(10, 50))
        utils.draw_text(frame, text=f"Press 2 to switch between Video and Image Captioning", point=(10, 90))


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
                    vision_model, text_decoder, processor = load_models(model_name, device_type)
                    
                    # Start a new inference worker
                    if video_input == True:
                        worker = threading.Thread(
                            target=inference_worker,
                            args=(video_input,model, processor),
                            daemon=True
                    )
                    else:
                        worker = threading.Thread(
                            target=inference_worker,
                            args=(video_input,vision_model, text_decoder, processor),
                            daemon=True
                    )
                    worker.start()
                    # Clear the processing times
                    with global_result_lock:
                        processing_times.clear()
        
        # For video captioning, allow switching back to image captioning with key '2'
        if key == ord('2'):
            video_input = not video_input

            if video_input == False:
                print("Switching to image_input mode...")

                model_name = "Salesforce/blip-image-captioning-base"
                device_type = "AUTO"
        
                # Stop current worker
                global_stop_event.set()
                worker.join(timeout=1)
                global_stop_event.clear()
        
                # Load image input models and processor
                vision_model, text_decoder, processor = load_models(model_name, device_type)
                
                # Start new inference worker with video_input=False
                worker = threading.Thread(
                    target=inference_worker,
                    kwargs={
                        "video_input": False,
                        "model": None,
                        "processor": processor,
                        "vision_model": vision_model,
                        "text_decoder": text_decoder
                    },
                    daemon=True
                )
                worker.start()
        
                # Clear frames, captions, processing times to avoid mix-up
                with global_frame_lock:
                    current_frames.clear()
                with global_result_lock:
                    captions.clear()
                    processing_times.clear()
        
                # Set caption so it shows on screen immediately
                caption = "Switching to image_input mode..."
            else: 
                print("Switching to video_input mode...")

               # Start new inference worker with video_input=True
                worker = threading.Thread(
                    target=inference_worker,
                    kwargs={
                        "video_input": True,
                        "model": model_video,
                        "processor": processor_video,
                        "vision_model": None,
                        "text_decoder": None
                    },
                    daemon=True
                )
                worker.start()
    
                # Clear frames, captions, processing times to avoid mix-up
                with global_frame_lock:
                    current_frames.clear()
                with global_result_lock:
                    captions.clear()
                    processing_times.clear()
    
                # Set caption so it shows on screen immediately
                caption = "Switching to video_input mode..."


    # stop the stream
    player.stop()
    global_stop_event.set()
    # wait 5s to finish inference - should be enough even for weak devices
    worker.join(timeout=5)
    # clean-up windows
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stream', default="0", type=str, help="Path to a video file or the webcam number")
    parser.add_argument("--model_name", type=str, default="Salesforce/blip-image-captioning-base", help="Model to be used for captioning",
                        choices=["Salesforce/blip-image-captioning-base", "Salesforce/blip-image-captioning-large"])
    parser.add_argument("--flip", type=bool, default=True, help="Mirror input video")

    args = parser.parse_args()
    run(args.stream, args.model_name, args.flip)