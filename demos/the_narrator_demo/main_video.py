import argparse
import logging as log
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path
import getpass

import cv2
import numpy as np
import torch
from transformers import LlavaNextProcessor
from optimum.intel.openvino import OVModelForVisualCausalLM
from huggingface_hub import login

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


def setup_huggingface_auth():
    """Setup Hugging Face authentication if needed"""
    print("Hugging Face Authentication Setup")
    print("=" * 50)
    
    # Check if already authenticated
    try:
        from huggingface_hub import whoami
        user_info = whoami()
        print(f"Already authenticated as: {user_info['name']}")
        return True
    except Exception:
        pass
    
    print("Some models require Hugging Face authentication")
    print("You need a Hugging Face token to access gated models")
    print("Get your token at: https://huggingface.co/settings/tokens")
    print()
    
    choice = input("Do you have a Hugging Face token? (y/n): ").strip().lower()
    
    if choice in ['y', 'yes']:
        print("Please enter your Hugging Face token:")
        token = getpass.getpass("Token (hidden): ").strip()
        
        if token:
            try:
                login(token=token)
                print("Successfully authenticated with Hugging Face!")
                return True
            except Exception as e:
                print(f"Authentication failed: {e}")
                print("Continuing without authentication (may limit model access)")
                return False
        else:
            print("No token provided, continuing without authentication")
            return False
    else:
        print("Continuing without authentication")
        print("Note: Some models may not be accessible without authentication")
        return False


def download_and_convert_video_model(model_name: str) -> None:
    """Download and convert LlavaNext model to OpenVINO format for video captioning"""
    output_dir = MODEL_DIR / model_name.replace("/", "_")
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Downloading and converting {model_name} to OpenVINO format for video captioning...")
        
        try:
            # Use OpenVINO optimum with LlavaNext (fully supported)
            model = OVModelForVisualCausalLM.from_pretrained(
                model_name,
                export=True,
                trust_remote_code=True,
                task="image-text-to-text",  # LlavaNext uses this task
                device="cpu",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                # OpenVINO optimization settings
                load_in_8bit=True,  # Enable 8-bit quantization
                ov_config={"PERFORMANCE_HINT": "LATENCY"},
                use_auth_token=True  # Use HF authentication
            )
            
            # Save the converted OpenVINO model
            model.save_pretrained(output_dir)
            
            # Download and save the processor  
            processor = LlavaNextProcessor.from_pretrained(model_name, use_auth_token=True)
            processor.save_pretrained(output_dir)
            
            print(f"Model successfully converted to OpenVINO optimum and saved to {output_dir}")
            
        except Exception as e:
            print(f"Error during OpenVINO optimum conversion: {e}")
            print("This model may not be fully supported by OpenVINO optimum yet.")
            raise e


def load_video_models(model_name: str, device: str = "CPU") -> tuple:
    """Load LlavaNext model optimized for video captioning with OpenVINO optimum"""
    model_dir = MODEL_DIR / model_name.replace("/", "_")
    
    # Check if model exists, if not download and convert it
    if not model_dir.exists():
        download_and_convert_video_model(model_name)
    
    print(f"Loading OpenVINO optimum models from {model_dir}")
    
    try:
        # Load OpenVINO optimized model with memory optimizations
        model = OVModelForVisualCausalLM.from_pretrained(
            model_dir,
            device=device,
            trust_remote_code=True,
            # Memory optimization settings
            ov_config={
                "PERFORMANCE_HINT": "LATENCY",
                "NUM_STREAMS": "1",  # Reduce parallel streams
                "CACHE_DIR": ""  # Disable cache to save memory
            },
            # Additional memory optimizations
            use_cache=False,  # Disable KV cache for memory efficiency
            torch_dtype=torch.float16  # Use FP16 for memory efficiency
        )
        
        processor = LlavaNextProcessor.from_pretrained(model_dir)
        
        print(f"Successfully loaded OpenVINO OPTIMUM model on {device}")
        return model, processor, device, "openvino_optimum"
        
    except Exception as e:
        print(f"OpenVINO optimum model loading failed: {e}")
        print("Attempting direct OpenVINO optimum conversion...")
        
        try:
            # Direct conversion with OpenVINO optimum and memory optimizations
            model = OVModelForVisualCausalLM.from_pretrained(
                model_name,
                export=True,
                trust_remote_code=True,
                task="image-text-to-text",
                device=device,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                # Disable 8-bit to reduce memory during conversion
                # load_in_8bit=True,  
                ov_config={
                    "PERFORMANCE_HINT": "LATENCY",
                    "NUM_STREAMS": "1"
                },
                use_cache=False,
                use_auth_token=True  # Use HF authentication
            )
            
            processor = LlavaNextProcessor.from_pretrained(model_name, use_auth_token=True)
            
            # Save for future use
            model.save_pretrained(model_dir)
            processor.save_pretrained(model_dir)
            
            print(f"OpenVINO optimum model converted and loaded successfully")
            return model, processor, device, "openvino_optimum"
            
        except Exception as e2:
            print(f"Direct OpenVINO optimum conversion failed: {e2}")
            print("OpenVINO optimum is required - cannot fall back to PyTorch")
            raise e2


def generate_true_video_caption(video_frames: list, model, processor, model_type: str) -> str:
    """Generate TRUE video caption using OpenVINO optimum with temporal video analysis"""
    try:
        if not video_frames or len(video_frames) == 0:
            return "No video frames to analyze"
        
        print(f"Processing video sequence with {len(video_frames)} frames using OpenVINO optimum")
        
        # Create video analysis with temporal understanding
        # Select key frames that show the video progression
        num_frames = len(video_frames)
        if num_frames >= 8:
            # Sample key frames across the video timeline for temporal analysis
            indices = [0, num_frames//4, num_frames//2, 3*num_frames//4, num_frames-1]
            key_frames = [video_frames[i] for i in indices]
        else:
            key_frames = video_frames
        
        # Analyze the video sequence with OpenVINO optimum
        video_descriptions = []
        
        for i, frame in enumerate(key_frames):
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create video-aware prompt based on temporal position
            if i == 0:
                prompt = "USER: <image>\nDescribe what is happening at the beginning of this video sequence. Focus on the initial action or scene.\nASSISTANT:"
            elif i == len(key_frames) - 1:
                prompt = "USER: <image>\nDescribe what is happening at the end of this video sequence. How has the scene changed?\nASSISTANT:"
            else:
                prompt = "USER: <image>\nDescribe the action and movement happening in this part of the video sequence.\nASSISTANT:"
            
            # Process with OpenVINO optimum - fix image_sizes error
            try:
                inputs = processor(
                    text=prompt,
                    images=rgb_frame,
                    return_tensors="pt"
                )
            except Exception as e:
                if "image_sizes" in str(e):
                    # Handle image_sizes parameter issue
                    try:
                        # Alternative processing without problematic parameters
                        from PIL import Image
                        pil_image = Image.fromarray(rgb_frame)
                        inputs = processor(
                            text=prompt,
                            images=pil_image,
                            return_tensors="pt",
                            padding=True
                        )
                    except Exception as e2:
                        print(f"Processor compatibility issue: {e2}")
                        # Skip this frame if processing fails
                        continue
                else:
                    raise e
            
            # Generate with OpenVINO optimum inference
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=40,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.eos_token_id
                )
            
            # Decode the response
            output_text = processor.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            # Extract the description
            if "ASSISTANT:" in output_text:
                description = output_text.split("ASSISTANT:")[-1].strip()
            else:
                description = output_text.replace(prompt.replace("<image>", ""), "").strip()
            
            description = description.replace("</s>", "").strip()
            if description:
                video_descriptions.append(description)
        
        # Combine temporal descriptions into video narrative
        if len(video_descriptions) >= 3:
            video_caption = f"Video sequence: {video_descriptions[0]} Then, {video_descriptions[len(video_descriptions)//2]} Finally, {video_descriptions[-1]}"
        elif len(video_descriptions) == 2:
            video_caption = f"Video shows: {video_descriptions[0]} followed by {video_descriptions[1]}"
        elif len(video_descriptions) == 1:
            video_caption = f"Video: {video_descriptions[0]}"
        else:
            video_caption = "Video sequence showing continuous action and movement"
        
        # Clean up the video caption
        video_caption = video_caption.replace("</s>", "").strip()
        
        if not video_caption or len(video_caption) < 10:
            video_caption = "Video sequence showing continuous action and movement"
            
        print(f"Generated VIDEO caption with OpenVINO optimum: {video_caption}")
        return video_caption
        
    except Exception as e:
        print(f"Error in OpenVINO optimum video captioning: {e}")
        import traceback
        traceback.print_exc()
        return f"OpenVINO optimum video processing error: {str(e)[:50]}"


def video_inference_worker(model, processor, device, model_type):
    """Worker thread for video caption inference using OpenVINO optimum"""
    global current_frame, current_caption, processing_times
    
    # Smaller buffer for memory efficiency with OpenVINO optimum
    video_buffer = deque(maxlen=8)  # Store 8 frames for video context
    last_processing_time = 0

    while not global_stop_event.is_set():
        with global_frame_lock:
            frame = current_frame.copy()
        
        # Add frame to video buffer
        video_buffer.append(frame)
        
        # Process video every 6 seconds when we have enough frames (less frequent for memory)
        current_time = time.time()
        if (len(video_buffer) >= 4 and 
            (current_time - last_processing_time) > 6.0):
            
            start_time = time.perf_counter()
            
            # Use video frames for OpenVINO optimum video captioning
            video_frames = list(video_buffer)
            caption = generate_true_video_caption(video_frames, model, processor, model_type)
            
            elapsed = time.perf_counter() - start_time
            last_processing_time = current_time

            with global_result_lock:
                current_caption = caption
                processing_times.append(elapsed)
        
        # Longer delay for memory efficiency
        time.sleep(0.3)


def run(video_path: str, model_name: str, flip: bool = True) -> None:
    """Main execution function for OpenVINO optimum video captioning"""
    global current_frame, current_caption, processing_times
    
    # Set up logging
    log.getLogger().setLevel(log.INFO)

    device_type = "CPU"  # Use CPU for OpenVINO optimum
    
    print(f"Starting TRUE VIDEO CAPTIONING with OpenVINO OPTIMUM using {model_name}")
    
    # Setup Hugging Face authentication
    print("Setting up authentication...")
    setup_huggingface_auth()
    
    try:
        model, processor, device, model_type = load_video_models(model_name, device_type)
        print(f"Loaded video model using {model_type} backend")
        
        if model_type != "openvino_optimum":
            print("ERROR: OpenVINO optimum backend is required!")
            return
            
    except Exception as e:
        print(f"Failed to load OpenVINO optimum model: {e}")
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

    title = "TRUE Video Captioning with OpenVINO OPTIMUM - Press ESC to Exit"
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
    
    print("TRUE VIDEO CAPTIONING with OpenVINO OPTIMUM started. Press ESC or 'q' to exit.")
    
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
                # Update caption every 4 seconds for video processing
                if t2 - t1 > 4 or not caption:
                    caption = current_caption if current_caption else "Processing video sequence with OpenVINO optimum..."
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
        print("OpenVINO optimum video captioning stopped.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TRUE Video Captioning with OpenVINO OPTIMUM')
    parser.add_argument(
        '--stream', 
        default="0", 
        type=str, 
        help="Path to video file or webcam number (0, 1, etc.)"
    )
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="llava-hf/llava-1.5-7b-hf", 
        help="LlavaNext model for OpenVINO optimum video captioning",
        choices=[
            "llava-hf/llava-1.5-7b-hf",
            "llava-hf/LLaVA-NeXT-7B-hf", 
            "llava-hf/LLaVA-NeXT-13B-hf"
        ]
    )
    parser.add_argument(
        "--flip", 
        action='store_true',
        help="Mirror input video"
    )

    args = parser.parse_args()
    run(args.stream, args.model_name, args.flip)
