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
from transformers import AutoProcessor
from optimum.intel.openvino import OVModelForVisualCausalLM
from transformers import LlavaNextVideoProcessor
from huggingface_hub import login
from optimum.intel import OVQuantizationConfig, OVWeightQuantizationConfig, OVPipelineQuantizationConfig

SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils")
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils import demo_utils as utils

# Store and load models relative to this script's directory to avoid CWD issues
MODEL_DIR = Path(__file__).resolve().parent / "model"

current_frame = None  # Will be set when video starts
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
    return model, processor, device, "intel_optimum_openvino"
    

def capture_webcam_to_mp4(output_path: str = "output.mp4", duration_seconds: int = 4, fps: int = 2, use_existing: bool = False) -> bool:

    def _ensure_bgr(img):
        if img is None:
            return None
        if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    # Reuse already-open webcam stream via globals
    if use_existing:
        print("Reusing existing webcam stream")
        global current_frame, global_frame_lock
        # Wait briefly for a valid frame
        t0 = time.time()
        initial = None
        while time.time() - t0 < 2.0 and initial is None:
            with global_frame_lock:
                initial = current_frame.copy() if current_frame is not None else None
            if initial is None:
                time.sleep(0.02)
        if initial is None:
            print("Error: No frames available from existing stream")
            return False

        initial = _ensure_bgr(initial)
        h, w = initial.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        total_frames = int(duration_seconds * fps)
        frame_interval = 1.0 / max(fps, 1)
        start_time = time.time()
        print(f"Capturing {total_frames} frames over {duration_seconds} seconds...")
        for i in range(total_frames):
            with global_frame_lock:
                frame = current_frame.copy() if current_frame is not None else None
            if frame is not None:
                frame = _ensure_bgr(frame)
                if frame.shape[1] != w or frame.shape[0] != h:
                    frame = cv2.resize(frame, (w, h))
                out.write(frame)
                elapsed = time.time() - start_time
                print(f"Captured frame {i+1}/{total_frames} at {elapsed:.1f}s (existing stream)")
            time.sleep(frame_interval)
        out.release()
        print(f"Capture complete! Video saved to: {output_path}")
        return True


def caption_video_content(video_frames: list, model, processor, model_type: str) -> str:
    """Generate video captions using Video-LLaVA with Intel HF Optimum (OpenVINO)"""

    if not video_frames or len(video_frames) == 0:
            return "No video content to analyze"
        
    print(f"PROCESSING: Generating video caption with {len(video_frames)} frames using Video-LLaVA Intel HF Optimum (OpenVINO)")

    output_path="inference.mp4"
    capture_webcam_to_mp4(output_path, duration_seconds=3, fps=8, use_existing=True)
    
    conversation_webcam = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe what you see in the video,describe any actions or objects you see in no more than 10 words please!!"},
                {"type": "video", "path": output_path},
            ],
        },
    ]

    try:
        inputs_webcam = processor.apply_chat_template(
            conversation_webcam,
            num_frames=len(video_frames),
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True
            )

        out_webcam = model.generate(**inputs_webcam, max_new_tokens=60)

        response = processor.batch_decode(
                    out_webcam,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )[0]
    except Exception as e:
        print(f"Error during generation: {e}")
        return None
 
    print(f"MODEL OUTPUT: {response}")
                
    # Extract assistant text only if marker is present
    if "ASSISTANT:" in response:
        description = response.split("ASSISTANT:")[-1].strip()
    else:
        description = response.strip()

    print(f"CAPTION GENERATED: {description}")
    
    # Only return caption if we have a meaningful description
    if description and len(description) > 10 and not description.isdigit():
        return description


def caption_video_file(video_path: str, model, processor, prompt_text: str, num_frames: int = 16) -> str:
    """Generate a single caption for a provided video file path using Video-LLaVA.

    Args:
        video_path: Path to an existing video file on disk.
        model: Loaded OVModelForVisualCausalLM model.
        processor: Associated processor.
        prompt_text: Text prompt to drive the captioning.
        num_frames: Number of frames to sample from the video.

    Returns:
        The generated caption string, or None on failure.
    """
    if not os.path.isfile(video_path):
        print(f"Error: Video file not found: {video_path}")
        return None

    conversation_file = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "video", "path": video_path},
            ],
        },
    ]

    try:
        inputs_file = processor.apply_chat_template(
            conversation_file,
            num_frames=num_frames,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
        )

        out_file = model.generate(**inputs_file, max_new_tokens=60)
        response = processor.batch_decode(
            out_file,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]
    except Exception as e:
        print(f"Error during generation from file: {e}")
        return None

    if "ASSISTANT:" in response:
        description = response.split("ASSISTANT:")[-1].strip()
    else:
        description = response.strip()

    return description if description else None


def caption_video_file_in_batches(
    video_path: str,
    model,
    processor,
    prompt_text: str,
    batch_seconds: int = 9,
    max_new_tokens: int = 60,
    num_frames_per_batch: int = 16,
) -> None:
    """Process a video file in fixed-length batches and print a caption for each batch.

    The function reads the source video, segments it into windows of `batch_seconds`,
    writes each segment to a temporary mp4, and runs captioning on that segment.
    """
    if not os.path.isfile(video_path):
        print(f"Error: Video file not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    frames_per_batch = max(1, int(round(fps * batch_seconds)))
    batch_index = 0
    frame_index = 0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    print(f"Batch captioning: fps={fps:.2f}, total_frames={total_frames}, size=({width}x{height}), frames_per_batch={frames_per_batch}")

    while True:
        batch_start_time = time.time()
        
        # Prepare temp writer for this batch
        import tempfile
        fd, temp_path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)

        writer = cv2.VideoWriter(temp_path, fourcc, fps, (max(1, width), max(1, height)))
        if not writer.isOpened():
            print("Error: Could not open temporary writer for batch")
            os.remove(temp_path)
            break

        written = 0
        while written < frames_per_batch:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
            written += 1
            frame_index += 1

        writer.release()
        capture_time = time.time() - batch_start_time

        if written == 0:
            # No more frames
            os.remove(temp_path)
            break

        # Caption this batch
        start_sec = (batch_index * frames_per_batch) / fps
        end_sec = ((batch_index * frames_per_batch) + written) / fps
        print(f"\n[Batch {batch_index+1}] {start_sec:.1f}s to {end_sec:.1f}s (capture: {capture_time:.2f}s)")

        try:
            inference_start_time = time.time()
            
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "video", "path": temp_path},
                    ],
                },
            ]

            inputs = processor.apply_chat_template(
                conversation,
                num_frames=num_frames_per_batch,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
            )

            out = model.generate(**inputs, max_new_tokens=max_new_tokens)
            response = processor.batch_decode(
                out,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )[0]

            inference_time = time.time() - inference_start_time

            if "ASSISTANT:" in response:
                caption = response.split("ASSISTANT:")[-1].strip()
            else:
                caption = response.strip()

            print(f"Caption: {caption if caption else '<no caption>'}")
            print(f"Inference time: {inference_time:.2f}s")
        except Exception as e:
            print(f"Error during batch generation: {e}")
        finally:
            # Cleanup temp file
            try:
                os.remove(temp_path)
            except Exception:
                pass

        batch_index += 1

    cap.release()


def video_captioning_worker(model, processor, device, model_type):
    """Worker thread for continuous video captioning using Video-LLaVA Intel HF Optimum (OpenVINO)"""
    global current_frame, current_caption, processing_times
    
    # Video buffer for Video-LLaVA analysis - increased for better video analysis
    video_buffer = deque(maxlen=3)  # Increased to 5 frames for better video understanding
    last_processing_time = 0

    while not global_stop_event.is_set():
        with global_frame_lock:
            frame = current_frame.copy() if current_frame is not None else None
        
        # Check if frame is valid (not empty)
        if frame is not None and frame.size > 0:
            # Add frame to video buffer
            video_buffer.append(frame)
        
        # Generate video captions every 15 seconds when we have enough frames for Video-LLaVA
        current_time = time.time()
        if current_time - last_processing_time >= 15 and len(video_buffer) >= 3:  # Require at least 3 frames, increased delay
            start_time = time.time()
            
            # Create a copy of the video buffer for processing
            frames_to_process = list(video_buffer)
            
            try:
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
                    
            except Exception as e:
                print(f"Error in video captioning worker: {e}")
                last_processing_time = current_time  # Reset timer to avoid rapid retries
        
        # Very short delay for continuous responsiveness
        time.sleep(0.05)


def run(video_path: str, model_name: str, flip: bool = True) -> None:
    """Main execution function for Video-LLaVA video captioning with Intel HF Optimum (OpenVINO)"""
    global current_frame, current_caption, processing_times
    
    # Set up logging
    log.getLogger().setLevel(log.INFO)

    device_type = "CPU"  # Use CPU for Intel HF Optimum (OpenVINO)
    setup_huggingface_auth()
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

    # Start video stream first to ensure it's working
    player.start()
    
    # Wait a moment for video to initialize
    time.sleep(1)
    
    # Check if video is working by getting first frame
    test_frame = player.next()
    if test_frame is None:
        print("Error: Could not get video frames. Please check your webcam connection.")
        player.stop()
        return
    
    # Initialize current_frame with first frame
    global current_frame
    current_frame = test_frame

    # Start the video captioning inference thread
    worker = threading.Thread(
        target=video_captioning_worker,
        args=(model, processor, device, model_type),
        daemon=True
    )
    worker.start()
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
                lines = display_caption.split("\n")
                line_step = max(28, int(36 * adaptive_font_scale))
                start_y = subtitle_y - (len(lines) - 1) * line_step // 2
                for i, line in enumerate(lines):
                    utils.draw_text(
                        frame,
                        text=line,
                        point=(f_width // 2, start_y + i * line_step),
                        center=True,
                        font_scale=adaptive_font_scale,
                        font_color=(255, 255, 255),
                        with_background=True,
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
        '--video_input',
        type=str,
        default=None,
        help="Optional: path to a video file to caption once and exit"
    )
    parser.add_argument(
        '--batch_seconds',
        type=int,
        default=None,
        help="If set, process --video_input in fixed-size batches of this many seconds"
    )
    parser.add_argument(
        '--batch_num_frames',
        type=int,
        default=16,
        help="Number of frames to sample per batch when using --batch_seconds"
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default="Describe what you see in the video in no more than 20 words.",
        help="Prompt to guide the captioning when using --video_input"
    )
    parser.add_argument(
        '--num_frames',
        type=int,
        default=16,
        help="Number of frames to sample for captioning when using --video_input"
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
    
    # Non-interactive single video captioning path
    if args.video_input is not None:
        log.getLogger().setLevel(log.INFO)
        device_type = "CPU"
        setup_huggingface_auth()
        try:
            model, processor, device, model_type = load_llava_video_models(args.model_name, device_type)
        except Exception as e:
            print(f"Failed to load model: {e}")
            sys.exit(1)

        if model_type not in ["intel_optimum_openvino"]:
            print("ERROR: Supported backend is Intel HF Optimum (OpenVINO)!")
            sys.exit(1)

        # Batch mode
        if args.batch_seconds is not None and args.batch_seconds > 0:
            caption_video_file_in_batches(
                args.video_input,
                model,
                processor,
                args.prompt,
                batch_seconds=args.batch_seconds,
                max_new_tokens=60,
                num_frames_per_batch=args.batch_num_frames,
            )
            sys.exit(0)

        # Single caption mode
        caption = caption_video_file(args.video_input, model, processor, args.prompt, args.num_frames)
        if caption is None:
            print("No caption produced.")
            sys.exit(2)
        print(caption)
        sys.exit(0)

    # Interactive/live mode
    run(args.stream, args.model_name, args.flip) 



