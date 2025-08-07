import argparse
import logging as log
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path
import getpass
from PIL import Image
import cv2
import numpy as np
import re
from transformers import LlavaNextVideoProcessor
from optimum.intel.openvino import OVModelForVisualCausalLM
from huggingface_hub import login

SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils")
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils import demo_utils as utils

MODEL_DIR = Path(__file__).parent / "model"

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
        model = OVModelForVisualCausalLM.from_pretrained(
            openvino_model_id,
            trust_remote_code=False,
            library_name="transformers"
        )
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
        
        try:
            model = OVModelForVisualCausalLM.from_pretrained(
                        model_name,
                        export=True,
                        trust_remote_code=False,
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
            
        except Exception as e2:
            print(f"Local conversion also failed: {e2}")
            print("Please check your internet connection and Hugging Face authentication")
            raise e2


def load_llava_video_models(model_name: str, device: str = "CPU") -> tuple:
    """Load LLaVA-NeXT-Video model with Intel HF Optimum (OpenVINO)"""
    model_dir = MODEL_DIR / model_name.replace("/", "_")
    
    # Ensure MODEL_DIR exists
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Model directory: {model_dir}")
    print(f"Model directory exists: {model_dir.exists()}")
    
    # Download and convert if not exists
    if not model_dir.exists():
        print("Model directory does not exist, downloading and converting...")
        download_and_convert_llava_video(model_name)
    else:
        print("Model directory exists, checking contents...")
        if model_dir.exists():
            contents = list(model_dir.iterdir())
            print(f"Model directory contents: {[item.name for item in contents]}")
            
            # Check if we have the necessary files for OpenVINO model
            required_files = ["openvino_model.xml", "openvino_model.bin", "config.json"]
            missing_files = [f for f in required_files if not (model_dir / f).exists()]
            
            if missing_files:
                print(f"Missing required files: {missing_files}")
                print("Re-downloading and converting model...")
                download_and_convert_llava_video(model_name)
    
    print(f"Loading LLaVA-NeXT-Video Intel HF Optimum (OpenVINO) models from {model_dir}")
    
    try:
        # Load from the saved local directory, not from model_name
        model = OVModelForVisualCausalLM.from_pretrained(
            str(model_dir),
            trust_remote_code=False,
            library_name="transformers"
        )
        print("Model loaded successfully")

        processor = LlavaNextVideoProcessor.from_pretrained(str(model_dir))
        print("Processor loaded successfully")
        
        print(f"Successfully loaded LLaVA-NeXT-Video Intel HF Optimum (OpenVINO) model on {device}")
        return model, processor, device, "intel_optimum_openvino"
        
    except Exception as e:
        print(f"Error loading model from {model_dir}: {e}")
        print("Attempting to load from original model name...")
        
        # Fallback: try loading from original model name
        try:
            model = OVModelForVisualCausalLM.from_pretrained(
                model_name,
                trust_remote_code=False,
                library_name="transformers"
            )
            processor = LlavaNextVideoProcessor.from_pretrained(model_name)
            print(f"Successfully loaded LLaVA-NeXT-Video Intel HF Optimum (OpenVINO) model on {device}")
            return model, processor, device, "intel_optimum_openvino"
        except Exception as e2:
            print(f"Error loading from original model name: {e2}")
            raise e2
    


def caption_video_content(video_frames: list, model, processor, model_type: str) -> str:
    """Generate video captions using Video-LLaVA with Intel HF Optimum (OpenVINO)"""

    if not video_frames or len(video_frames) == 0:
            return "No video content to analyze"
        
    print(f"PROCESSING: Generating video caption with {len(video_frames)} frames using Video-LLaVA Intel HF Optimum (OpenVINO)")
        
        # For Video-LLaVA, process multiple frames as a video sequence
        # Select key frames from the video sequence for temporal understanding


    num_frames = len(video_frames)

    # Select 8 frames for better video context
    if num_frames >= 8:
        step = (num_frames - 1) / 7
        indices = [int(i * step) for i in range(8)]
    elif num_frames >= 4:
        step = (num_frames - 1) / 3
        indices = [int(i * step) for i in range(4)]
    else:
        indices = list(range(num_frames))

    selected_frames = [video_frames[i] for i in indices]

    # Convert to resized PIL images
    pil_images = [
        Image.fromarray(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (336, 336)))
        for frame in selected_frames
    ]

    # Intel HF Optimum OpenVINO Video-LLaVA processing
    print("Using Intel HF Optimum OpenVINO Video-LLaVA inference")
                
    # Create video-specific prompt for Video-LLaVA using conversation format
    conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe what you see happening in this video in detail. What objects, actions, and events are visible?"},
                            {"type": "video", "images": pil_images},
                        ],
                    },
                ]

    try:
        inputs = processor.apply_chat_template(
                        conversation,
                        num_frames=len(pil_images),
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True
                    )
        
        print(f"Input shape: {inputs['input_ids'].shape if 'input_ids' in inputs else 'No input_ids'}")
        
        out = model.generate(
            **inputs, 
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1
        )
        
        print(f"Output shape: {out.shape}")
        
    except Exception as e:
        print(f"Error during generation: {e}")
        return None

    response = processor.batch_decode(
                    out,
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

    print(f"CAPTION GENERATED: {description}")
                
                # Clean up the description - remove any non-English characters or very short responses
    # Remove Chinese characters and other non-ASCII characters
    description = re.sub(r'[^\x00-\x7F]+', '', description)
    description = description.strip()
                
                # Only return caption if we have a meaningful description
    if description and len(description) > 10 and not description.isdigit():
        print(f"CAPTION GENERATED: {description}")
        return description
    else:
        print(f"No meaningful caption generated (length: {len(description) if description else 0}), trying fallback prompt")
        
        # Try a simpler fallback prompt
        try:
            fallback_conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What do you see in this video?"},
                        {"type": "video", "images": pil_images},
                    ],
                },
            ]
            
            fallback_inputs = processor.apply_chat_template(
                fallback_conversation,
                num_frames=len(pil_images),
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True
            )
            
            fallback_out = model.generate(
                **fallback_inputs, 
                max_new_tokens=50,
                temperature=0.8,
                do_sample=True
            )
            
            fallback_response = processor.batch_decode(
                fallback_out,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )[0]
            
            if "ASSISTANT:" in fallback_response:
                fallback_description = fallback_response.split("ASSISTANT:")[-1].strip()
            else:
                fallback_description = fallback_response.strip()
            
            fallback_description = re.sub(r'[^\x00-\x7F]+', '', fallback_description)
            fallback_description = fallback_description.strip()
            
            if fallback_description and len(fallback_description) > 5:
                print(f"FALLBACK CAPTION: {fallback_description}")
                return fallback_description
                
        except Exception as e:
            print(f"Fallback prompt also failed: {e}")
        
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
    
    # Setup Hugging Face authentication if needed
    setup_huggingface_auth()
    
    # Initialize models
    model, processor, device, model_type = None, None, None, None
    
    try:
        model, processor, device, model_type = load_llava_video_models(model_name, device_type)
        print(f"Loaded Video-LLaVA video captioning model using {model_type} backend")
        
        if model_type not in ["intel_optimum_openvino"]:
            print("ERROR: Supported backend is Intel HF Optimum (OpenVINO)!")
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

    args = parser.parse_args()
    run(args.stream, args.model_name) 



