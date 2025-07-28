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
from transformers import LlavaNextVideoProcessor
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


def download_and_convert_llava_video(model_name: str) -> None:
    """Download pre-converted LLaVA-NeXT-Video OpenVINO model from Hugging Face"""
    model_dir = MODEL_DIR / model_name.replace("/", "_")
    output_dir = model_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading pre-converted LLaVA-NeXT-Video OpenVINO model...")
    
    # Use the pre-converted OpenVINO model from Hugging Face
    openvino_model_id = "ezelanza/llava-next-video-openvino-int8"
    
    print("Step 1: Downloading pre-converted OpenVINO model...")
    
    try:
        model = OVModelForVisualCausalLM.from_pretrained(openvino_model_id)
        print("Step 2: Saving OpenVINO model locally...")
        model.save_pretrained(output_dir)
        print("OpenVINO model saved successfully")
        
        print("Step 3: Loading processor...")
        processor = LlavaNextVideoProcessor.from_pretrained(model_name)
        processor.save_pretrained(output_dir)
        print("Processor loaded and saved successfully")
        
        print(f"LLaVA-NeXT-Video model successfully downloaded and saved to {output_dir}")
        
    except Exception as e:
        print(f"Failed to download pre-converted model: {e}")
        print("Falling back to local conversion...")
        
        # Fallback to local conversion if download fails
        print("Step 1: Loading LLaVA-NeXT-Video model...")
        
        model = OVModelForVisualCausalLM.from_pretrained(
                    model_name,
                    export=True,
                    trust_remote_code=True,
                    library_name="transformers"
                    )
        
        print("Step 2: Saving OpenVINO model...")
        model.save_pretrained(output_dir)

        print("OpenVINO model saved successfully")
            
        print("Step 3: Loading processor...")
            
        processor = LlavaNextVideoProcessor.from_pretrained(model_name)
        processor.save_pretrained(output_dir)

        print("Processor loaded and saved successfully")
            
        print(f"LLaVA-NeXT-Video model successfully converted to OpenVINO Optimum and saved to {output_dir}")


def load_llava_video_models(model_name: str, device: str = "CPU") -> tuple:
    """Load LLaVA-NeXT-Video model with Intel HF Optimum (OpenVINO)"""
    model_dir = MODEL_DIR / model_name.replace("/", "_")
    
    # Download and convert if not exists
    if not model_dir.exists():
        download_and_convert_llava_video(model_name)
    
    print(f"Loading LLaVA-NeXT-Video Intel HF Optimum (OpenVINO) models from {model_dir}")
    
    try:
        # Load Intel HF Optimum OpenVINO model with error handling
        print("Loading OpenVINO model...")
        try:
            model = OVModelForVisualCausalLM.from_pretrained(
                model_dir,
                device=device,
                ov_config={
                    "INFERENCE_PRECISION_HINT": "FP16",
                    "PERFORMANCE_HINT": "LATENCY",
                }
            )
            print("Model loaded with FP16 configuration")
        except Exception as e1:
            print(f"FP16 loading failed: {e1}")
            try:
                model = OVModelForVisualCausalLM.from_pretrained(
                    model_dir,
                    device=device,
                    ov_config={
                        "INFERENCE_PRECISION_HINT": "FP32",
                        "PERFORMANCE_HINT": "LATENCY",
                    }
                )
                print("Model loaded with FP32 configuration")
            except Exception as e2:
                print(f"FP32 loading failed: {e2}")
                model = OVModelForVisualCausalLM.from_pretrained(
                    model_dir,
                    device=device
                )
                print("Model loaded with default configuration")
        
        # Load processor with error handling
        print("Loading processor...")
        try:
            from transformers import LlavaNextVideoProcessor
            processor = LlavaNextVideoProcessor.from_pretrained(model_dir)
            print("LlavaNextVideoProcessor loaded successfully")
        except Exception as proc_error:
            print(f"LlavaNextVideoProcessor failed: {proc_error}")
            try:
                from transformers import LlavaNextProcessor
                processor = LlavaNextProcessor.from_pretrained(model_dir)
                print("LlavaNextProcessor loaded successfully")
            except Exception as proc_error2:
                print(f"LlavaNextProcessor failed: {proc_error2}")
                try:
                    from transformers import AutoProcessor
                    processor = AutoProcessor.from_pretrained(model_dir)
                    print("AutoProcessor loaded successfully")
                except Exception as proc_error3:
                    print(f"AutoProcessor failed: {proc_error3}")
                    # Try loading from original model
                    try:
                        from transformers import LlavaNextVideoProcessor
                        processor = LlavaNextVideoProcessor.from_pretrained(model_name)
                        print("Processor loaded from original model")
                    except Exception as proc_error4:
                        print(f"Original model processor failed: {proc_error4}")
                        raise Exception("Could not load any processor for Video-LLaVA")
        
        print(f"Successfully loaded LLaVA-NeXT-Video Intel HF Optimum (OpenVINO) model on {device}")
        return model, processor, device, "intel_optimum_openvino"
        
    except Exception as e:
        print(f"LLaVA-NeXT-Video Intel HF Optimum (OpenVINO) model loading failed: {e}")
        import traceback
        traceback.print_exc()
        print("Model may be corrupted, attempting fresh conversion...")
        
        # Clean up corrupted model and retry
        if model_dir.exists():
            import shutil
            try:
                shutil.rmtree(model_dir)
                print(f"Cleaned up corrupted model directory: {model_dir}")
            except:
                pass
        
        try:
            # Retry with fresh conversion
            download_and_convert_llava_video(model_name)
            return load_llava_video_models(model_name, device)
            
        except Exception as e2:
            print(f"Fresh LLaVA-NeXT-Video Intel HF Optimum (OpenVINO) conversion failed: {e2}")
            print("Intel HF Optimum is required - cannot fall back to PyTorch")
            raise e2


def caption_video_content(video_frames: list, model, processor, model_type: str) -> str:
    """Generate video captions using Video-LLaVA with Intel HF Optimum (OpenVINO)"""
    try:
        if not video_frames or len(video_frames) == 0:
            return "No video content to analyze"
        
        print(f"PROCESSING: Generating video caption with {len(video_frames)} frames using Video-LLaVA Intel HF Optimum (OpenVINO)")
        
        # For Video-LLaVA, process multiple frames as a video sequence
        # Select key frames from the video sequence for temporal understanding
        num_frames = len(video_frames)
        if num_frames >= 8:
            # Sample 8 frames evenly across the video sequence for Video-LLaVA
            indices = [int(i * (num_frames - 1) / 7) for i in range(8)]
            selected_frames = [video_frames[i] for i in indices]
        elif num_frames >= 4:
            # Sample 4 frames for shorter sequences
            indices = [int(i * (num_frames - 1) / 3) for i in range(4)]
            selected_frames = [video_frames[i] for i in indices]
        else:
            # Use all available frames
            selected_frames = video_frames
        
        # Convert frames to PIL Images for Video-LLaVA processing
        from PIL import Image
        pil_images = []
        for frame in selected_frames:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize for Video-LLaVA (standard size)
            rgb_frame = cv2.resize(rgb_frame, (336, 336))
            pil_image = Image.fromarray(rgb_frame)
            pil_images.append(pil_image)
        
        # Create video-specific prompt for Video-LLaVA using conversation format (matching working notebook)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is happening in this video?"},
                    {"type": "video", "images": pil_images},
                ],
            },
        ]
        
        try:
            if model_type == "intel_optimum_openvino":
                # Intel HF Optimum OpenVINO Video-LLaVA processing
                print("Using Intel HF Optimum OpenVINO Video-LLaVA inference")
                
                # Use processor with proper video format for Intel optimized OpenVINO model
                # Use apply_chat_template method (matching working notebook)
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What is happening in this video?"},
                            {"type": "video", "images": pil_images},
                        ],
                    },
                ]
                
                # Try direct processing approach
                inputs = processor(
                    text="What is happening in this video?",
                    images=pil_images,
                    return_tensors="pt"
                )
                
                # Convert lists to tensors and add batch dimension
                import torch
                for key in inputs:
                    if isinstance(inputs[key], list):
                        tensor = torch.tensor(inputs[key])
                        if tensor.dim() == 1:
                            tensor = tensor.unsqueeze(0)  # Add batch dimension
                        inputs[key] = tensor
                
                # Add pixel_values to inputs
                pixel_values = processor.image_processor(pil_images, return_tensors="pt")["pixel_values"]
                inputs["pixel_values"] = pixel_values
                # Add image_sizes to inputs (required by OpenVINO model)
                image_sizes = torch.tensor([[img.height, img.width] for img in pil_images])  # (height, width)
                inputs["image_sizes"] = image_sizes
                
                print(f"Successfully processed video sequence with {len(pil_images)} frames for Video-LLaVA Intel HF Optimum OpenVINO")
                print(f"Input keys: {list(inputs.keys())}")
                
                # Generate video caption with Intel optimized OpenVINO inference (matching working notebook)
                print(f"Input shapes: {[(k, v.shape if hasattr(v, 'shape') else type(v)) for k, v in inputs.items()]}")
                
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=150,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9
                    )[0]
                
                # Decode the output using processor (matching working notebook)
                response = processor.batch_decode(
                    [output_ids],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )[0]
                print(f"MODEL OUTPUT: {response}")
                
                # Extract the video caption from the response
                if "ASSISTANT:" in response:
                    description = response.split("ASSISTANT:")[-1].strip()
                else:
                    # If no ASSISTANT marker, use the full response
                    description = response.strip()
                
                # Clean up the description - remove any non-English characters or very short responses
                import re
                # Remove Chinese characters and other non-ASCII characters
                description = re.sub(r'[^\x00-\x7F]+', '', description)
                description = description.strip()
                
                # Only return caption if we have a meaningful description
                if description and len(description) > 10 and not description.isdigit():
                    print(f"CAPTION GENERATED: {description}")
                    return description
                else:
                    print(f"No meaningful caption generated (length: {len(description) if description else 0}), skipping display")
                    return None
                
        except Exception as vision_error:
            print(f"ERROR: Video captioning processing error: {vision_error}")
            import traceback
            traceback.print_exc()
            return None
        
    except Exception as e:
        print(f"ERROR: Error generating video caption: {e}")
        import traceback
        traceback.print_exc()
        return None


def video_captioning_worker(model, processor, device, model_type):
    """Worker thread for continuous video captioning using Video-LLaVA Intel HF Optimum (OpenVINO)"""
    global current_frame, current_caption, processing_times
    
    # Video buffer for Video-LLaVA analysis - larger buffer for video sequences
    video_buffer = deque(maxlen=12)  # Store 12 frames for better video context
    last_processing_time = 0

    while not global_stop_event.is_set():
        with global_frame_lock:
            frame = current_frame.copy()
        
        # Add frame to video buffer
        video_buffer.append(frame)
        
        # Generate video captions every 5 seconds when we have enough frames for Video-LLaVA
        current_time = time.time()
        if current_time - last_processing_time >= 5 and len(video_buffer) >= 4:
            start_time = time.time()
            
            # Create a copy of the video buffer for processing
            frames_to_process = list(video_buffer)
            
            caption = caption_video_content(frames_to_process, model, processor, model_type)
            
            elapsed = time.time() - start_time
            last_processing_time = current_time
            
            # Only update caption if we got a meaningful description
            if caption is not None:
                with global_result_lock:
                    current_caption = caption
                    processing_times.append(elapsed)
                
                print(f"CAPTION UPDATE: {caption}")
                print("Caption now available for display on video")
            else:
                print("No meaningful caption generated, keeping previous caption")
        
        # Very short delay for continuous responsiveness
        time.sleep(0.05)


def run(video_path: str, model_name: str, flip: bool = True) -> None:
    """Main execution function for Video-LLaVA video captioning with Intel HF Optimum (OpenVINO)"""
    global current_frame, current_caption, processing_times
    
    # Set up logging
    log.getLogger().setLevel(log.INFO)

    device_type = "CPU"  # Use CPU for Intel HF Optimum (OpenVINO)
    
    print(f"Starting continuous Video-LLaVA video captioning with Intel HF Optimum (OpenVINO) using {model_name}")
    print("=" * 80)
    print("This will generate continuous video captions describing what's happening")
    print("Using Video-LLaVA model specifically designed for video understanding with Intel optimizations")
    print("=" * 80)
    
    # Initialize models
    model, processor, device, model_type = None, None, None, None
    
    try:
        model, processor, device, model_type = load_llava_video_models(model_name, device_type)
        print(f"Loaded Video-LLaVA video captioning model using {model_type} backend")
        
        if model_type not in ["intel_optimum_openvino"]:
            print("ERROR: Supported backend is Intel HF Optimum (OpenVINO)!")
            print("This demo is designed to showcase Video-LLaVA performance with Intel optimizations")
            return
            
    except Exception as e:
        print(f"Failed to load Video-LLaVA Intel HF Optimum (OpenVINO) model: {e}")
        print("Intel HF Optimum (OpenVINO) is required for this demo")
        return

    # Initialize video player using utils
    if isinstance(video_path, str) and video_path.isnumeric():
        video_path = int(video_path)
    
    try:
        # Use utils VideoPlayer for proper video captioning
        player = utils.VideoPlayer(video_path, size=(640, 480), fps=15, flip=flip)
    except Exception as e:
        print(f"Error initializing video player: {e}")
        return

    # Initialize processing times deque
    processing_times = deque(maxlen=50)

    title = "Continuous Video-LLaVA Video Captioning with OpenVINO Optimum - Press ESC to Exit"
    cv2.namedWindow(title, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)

    # Start the video captioning inference thread
    worker = threading.Thread(
        target=video_captioning_worker,
        args=(model, processor, device, model_type),
        daemon=True
    )
    worker.start()

    # Start video stream
    player.start()
    t1 = time.time()
    caption = current_caption
    
    print("Continuous Video-LLaVA video captioning started. Press ESC or 'q' to exit.")
    
    try:
        while True:
            # Grab the frame using utils VideoPlayer
            frame = player.next()
            if frame is None:
                print("Video source ended")
                break

            # Update the latest frame for video captioning inference
            with global_frame_lock:
                current_frame = frame

            f_height, f_width = frame.shape[:2]

            # Get the latest video caption continuously
            with global_result_lock:
                t2 = time.time()
                # Update video caption continuously every 1 second
                if t2 - t1 > 1:
                    caption = current_caption
                    t1 = t2
                    if caption:
                        print(f"DISPLAY: Displaying caption: {caption[:50]}...")

                # Get processing time statistics
                processing_time = np.mean(processing_times) * 1000 if processing_times else 0

            # Only display caption if we have meaningful content
            display_caption = caption if caption else None
            
            # Better text wrapping for subtitles that fit on screen
            def wrap_text_for_screen(text, max_width):
                """Wrap text to fit screen width properly"""
                if not text:
                    return ""
                
                # Calculate max characters per line based on screen width - more conservative
                chars_per_line = max(25, min(45, max_width // 15))  # More conservative sizing
                
                words = text.split()
                lines = []
                current_line = ""
                
                for word in words:
                    test_line = current_line + " " + word if current_line else word
                    if len(test_line) <= chars_per_line:
                        current_line = test_line
                    else:
                        if current_line:
                            lines.append(current_line)
                            current_line = word
                        else:
                            # Word is too long, break it
                            lines.append(word[:chars_per_line])
                            current_line = word[chars_per_line:]
                
                if current_line:
                    lines.append(current_line)
                
                # Limit to 2 lines for better subtitle display
                return "\n".join(lines[:2])
            
            # Wrap subtitle text to fit screen (only if we have a caption)
            if display_caption:
                display_caption = wrap_text_for_screen(display_caption, f_width)

            # Calculate subtitle position - ensure it fits in the video
            subtitle_margin = 20  # Margin from edges
            subtitle_y = f_height - 80  # Less margin from bottom to fit better
            
            # Calculate font scale based on screen size - smaller and more conservative
            adaptive_font_scale = max(0.5, min(1.2, f_width / 600))  # Much smaller scale

            # Draw video subtitles with better fitting (only if we have a caption)
            if display_caption:
                utils.draw_text(
                    frame, 
                    text=display_caption, 
                    point=(f_width // 2, subtitle_y), 
                    center=True, 
                    font_scale=adaptive_font_scale,  # Smaller adaptive font scale
                    font_color=(255, 255, 255),  # White text
                    with_background=True  # Black background for readability
                )
            
            # Draw compact status information - smaller and positioned better
            utils.draw_text(
                frame, 
                text=f"LIVE: {processing_time:.0f}ms", 
                point=(10, 25),
                font_scale=0.4,  # Much smaller
                font_color=(0, 255, 0)
            )
            
            utils.draw_text(
                frame, 
                text=f"OpenVINO", 
                point=(10, 45),
                font_scale=0.4,  # Much smaller
                font_color=(0, 255, 255)
            )

            # Draw OpenVINO watermark using utils
            utils.draw_ov_watermark(frame)
            
            # Show the video with captions
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
        print("Video-LLaVA video captioning stopped.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video-LLaVA Video Captioning with OpenVINO Optimum or PyTorch Fallback')
    parser.add_argument(
        '--stream', 
        default="0", 
        type=str, 
        help="Path to video file or webcam number (0, 1, etc.)"
    )
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="llava-hf/LLaVA-NeXT-Video-7B-hf", 
        help="LLaVA-NeXT-Video model for Intel HF Optimum OpenVINO video captioning"
    )
    parser.add_argument(
        "--flip", 
        action='store_true',
        help="Mirror input video"
    )

    args = parser.parse_args()
    run(args.stream, args.model_name, args.flip) 



