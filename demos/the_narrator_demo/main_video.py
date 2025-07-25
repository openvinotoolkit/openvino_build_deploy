import argparse
import logging as log
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration
from optimum.intel.openvino import OVModelForCausalLM

SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils")
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils import demo_utils as utils

MODEL_DIR = Path("model")

current_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)  # Placeholder for the current frame
current_caption = ""

processing_times = deque(maxlen=100)

global_stop_event = threading.Event()

global_frame_lock = threading.Lock()
global_result_lock = threading.Lock()


def download_and_convert_video_model(model_name: str) -> None:
    """Download and convert video captioning model to OpenVINO format"""
    output_dir = MODEL_DIR / model_name.replace("/", "_")
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Downloading and converting video model {model_name} to OpenVINO format...")
        
        try:
            # Use a video model that's compatible with OpenVINO export
            # VideoLlava is better supported than LlavaNextVideo for OpenVINO
            model = OVModelForCausalLM.from_pretrained(
                model_name,
                export=True,
                trust_remote_code=True,
                task="video-text-to-text",
                device="cpu",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            
            # Save the converted model
            model.save_pretrained(output_dir)
            
            # Download and save the processor  
            processor = VideoLlavaProcessor.from_pretrained(model_name)
            processor.save_pretrained(output_dir)
            
            print(f"Video model successfully converted and saved to {output_dir}")
            
        except Exception as e:
            print(f"Error during video model conversion: {e}")
            print("Trying alternative video model approach...")
            
            # Alternative: Use PyTorch model with manual video processing
            try:
                # Load PyTorch video model
                torch_model = VideoLlavaForConditionalGeneration.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                )
                
                # Manual conversion for video models
                # Create dummy video input for tracing
                processor = VideoLlavaProcessor.from_pretrained(model_name)
                
                # Save PyTorch model for now (can't convert video models yet)
                torch_model.save_pretrained(output_dir)
                processor.save_pretrained(output_dir)
                
                print(f"Video model saved in PyTorch format to {output_dir}")
                
            except Exception as e2:
                print(f"Alternative video model conversion failed: {e2}")
                print("Falling back to frame-based approach with video understanding")
                raise e2


def load_video_models(model_name: str, device: str = "CPU") -> tuple:
    """Load video captioning model and processor"""
    model_dir = MODEL_DIR / model_name.replace("/", "_")
    
    # Check if model exists, if not download and convert it
    if not model_dir.exists():
        download_and_convert_video_model(model_name)
    
    print(f"Loading video models from {model_dir}")
    
    try:
        # Try to load converted OpenVINO model
        model = OVModelForCausalLM.from_pretrained(
            model_dir,
            device=device,
            trust_remote_code=True
        )
        
        processor = VideoLlavaProcessor.from_pretrained(model_dir)
        
        print(f"Successfully loaded OpenVINO optimized video model on {device}")
        return model, processor, device, "openvino"
        
    except Exception as e:
        print(f"OpenVINO video model loading failed: {e}")
        print("Loading PyTorch video model...")
        
        try:
            # Load PyTorch video model  
            model = VideoLlavaForConditionalGeneration.from_pretrained(
                model_dir,
                torch_dtype=torch.float16 if device != "cpu" else torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            
            processor = VideoLlavaProcessor.from_pretrained(model_dir)
            
            print(f"Successfully loaded PyTorch video model")
            return model, processor, device, "pytorch"
            
        except Exception as e2:
            print(f"PyTorch video model loading failed: {e2}")
            print("Attempting direct download...")
            
            # Direct download and load
            model = VideoLlavaForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device != "cpu" else torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            
            processor = VideoLlavaProcessor.from_pretrained(model_name)
            
            # Save for future use
            model.save_pretrained(model_dir)
            processor.save_pretrained(model_dir)
            
            print(f"Video model downloaded and loaded successfully")
            return model, processor, device, "pytorch"


def generate_true_video_caption(video_frames: list, model, processor, model_type: str) -> str:
    """Generate true video caption using video understanding model"""
    try:
        if not video_frames or len(video_frames) == 0:
            return "No video frames to analyze"
        
        # Prepare video frames (RGB format)
        rgb_frames = []
        for frame in video_frames:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize for memory efficiency  
            height, width = rgb_frame.shape[:2]
            if height > 224 or width > 224:
                scale = min(224/height, 224/width)
                new_height, new_width = int(height * scale), int(width * scale)
                rgb_frame = cv2.resize(rgb_frame, (new_width, new_height))
            rgb_frames.append(rgb_frame)
        
        # Limit frames for memory efficiency
        if len(rgb_frames) > 8:
            # Sample 8 frames evenly from the video
            indices = np.linspace(0, len(rgb_frames) - 1, 8, dtype=int)
            rgb_frames = [rgb_frames[i] for i in indices]
        
        # Convert to numpy array for video processing
        video_array = np.array(rgb_frames)
        
        # Create video captioning prompt
        prompt = "USER: <video>\nDescribe what is happening in this video sequence, focusing on the actions and movement.\nASSISTANT:"
        
        # Process video with the video model
        inputs = processor(
            text=prompt,
            videos=video_array,
            return_tensors="pt"
        )
        
        # Move to device if needed
        if model_type == "pytorch" and hasattr(model, 'device'):
            inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
        
        # Generate video caption
        with torch.no_grad():
            if model_type == "openvino":
                # OpenVINO inference
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    temperature=0.7
                )
            else:
                # PyTorch inference
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    temperature=0.7,
                    pad_token_id=processor.tokenizer.eos_token_id
                )
        
        # Decode the output
        output_text = processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        # Extract the video caption
        if "ASSISTANT:" in output_text:
            caption = output_text.split("ASSISTANT:")[-1].strip()
        else:
            caption = output_text.replace(prompt.replace("<video>", ""), "").strip()
        
        # Clean up the caption
        caption = caption.replace("</s>", "").strip()
        
        if not caption or len(caption) < 5:
            caption = "Video sequence in progress"
            
        return caption
        
    except Exception as e:
        print(f"Error generating video caption: {e}")
        return f"Video processing error"


def video_inference_worker(model, processor, device, model_type):
    """Worker thread for true video caption inference"""
    global current_frame, current_caption, processing_times
    
    # Buffer for video frames (temporal context)
    video_buffer = deque(maxlen=16)  # Store 16 frames for video context
    last_processing_time = 0

    while not global_stop_event.is_set():
        with global_frame_lock:
            frame = current_frame.copy()
        
        # Add frame to video buffer
        video_buffer.append(frame)
        
        # Process video every 6 seconds when we have enough frames
        current_time = time.time()
        if (len(video_buffer) >= 8 and 
            (current_time - last_processing_time) > 6.0):
            
            start_time = time.perf_counter()
            
            # Use video frames for true video captioning
            video_frames = list(video_buffer)
            caption = generate_true_video_caption(video_frames, model, processor, model_type)
            
            elapsed = time.perf_counter() - start_time
            last_processing_time = current_time

            with global_result_lock:
                current_caption = caption
                processing_times.append(elapsed)
        
        # Video processing delay
        time.sleep(0.25)


def run(video_path: str, model_name: str, flip: bool = True) -> None:
    """Main execution function for video captioning"""
    global current_frame, current_caption, processing_times
    
    # Set up logging
    log.getLogger().setLevel(log.INFO)

    device_type = "CPU"  # Use CPU for stability
    
    print(f"Starting TRUE VIDEO CAPTIONING with OpenVINO optimized {model_name}")
    
    try:
        model, processor, device, model_type = load_video_models(model_name, device_type)
        print(f"Loaded video model using {model_type} backend")
    except Exception as e:
        print(f"Failed to load video model: {e}")
        return

    # Initialize video player
    if isinstance(video_path, str) and video_path.isnumeric():
        video_path = int(video_path)
    
    try:
        # Lower resolution for real-time video processing
        player = utils.VideoPlayer(video_path, size=(640, 480), fps=10, flip=flip)
    except Exception as e:
        print(f"Error initializing video player: {e}")
        return

    # Initialize processing times deque
    processing_times = deque(maxlen=50)

    title = "TRUE Video Captioning with OpenVINO - Press ESC to Exit"
    cv2.namedWindow(title, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)

    # Start the video inference thread
    worker = threading.Thread(
        target=video_inference_worker,
        args=(model, processor, device, model_type),
        daemon=True
    )
    worker.start()

    # Start video stream
    player.start()
    t1 = time.time()
    caption = current_caption
    
    print("TRUE VIDEO CAPTIONING started. Press ESC or 'q' to exit.")
    
    try:
        while True:
            # Grab the frame
            frame = player.next()
            if frame is None:
                print("Video source ended")
                break

            # Update the latest frame for inference
            with global_frame_lock:
                current_frame = frame

            f_height, f_width = frame.shape[:2]

            # Get the latest video caption
            with global_result_lock:
                t2 = time.time()
                # Update caption every 5 seconds for video processing
                if t2 - t1 > 5 or not caption:
                    caption = current_caption if current_caption else "Processing video sequence..."
                    t1 = t2

                # Get processing time statistics
                processing_time = np.mean(processing_times) * 1000 if processing_times else 0
                fps = 1000 / processing_time if processing_time > 0 else 0

            # Draw results on frame
            utils.draw_text(
                frame, 
                text=caption, 
                point=(f_width // 2, f_height - 30), 
                center=True, 
                font_scale=1.0, 
                with_background=True
            )
            utils.draw_text(
                frame, 
                text=f"VIDEO CAPTIONING: {processing_time:.0f}ms", 
                point=(10, 10)
            )
            utils.draw_text(
                frame, 
                text=f"Backend: {model_type.upper()}", 
                point=(10, 30)
            )

            utils.draw_ov_watermark(frame)
            
            # Show the output
            cv2.imshow(title, frame)
            key = cv2.waitKey(1)
            
            # Exit conditions
            if key == 27 or key == ord('q'):  # ESC or 'q'
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error during video captioning: {e}")
    finally:
        # Cleanup
        player.stop()
        global_stop_event.set()
        worker.join(timeout=2)
        cv2.destroyAllWindows()
        print("Video captioning stopped.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TRUE Video Captioning with OpenVINO')
    parser.add_argument(
        '--stream', 
        default="0", 
        type=str, 
        help="Path to video file or webcam number (0, 1, etc.)"
    )
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="LanguageBind/Video-LLaVA-7B-hf", 
        help="Video model to use for video captioning",
        choices=[
            "LanguageBind/Video-LLaVA-7B-hf",
            "microsoft/xclip-base-patch32-16-frames",
            "MCG-NJU/videomae-base-finetuned-kinetics"
        ]
    )
    parser.add_argument(
        "--flip", 
        action='store_true',
        help="Mirror input video"
    )

    args = parser.parse_args()
    run(args.stream, args.model_name, args.flip)
