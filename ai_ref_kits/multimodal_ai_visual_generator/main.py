from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pathlib import Path
from io import BytesIO
from PIL import Image
import numpy as np
import os
import base64
import sys
import yaml
import openvino_genai as ov_genai
import openvino as ov
import logging
import random
import json
import queue
import threading

# -------- Logging Setup --------
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Import watermark function
default_utils_path = (Path(__file__).resolve().parents[2] / "demos" / "utils")
external_utils_path = Path(os.getenv("UTILS_PATH", str(default_utils_path)))

if not external_utils_path.exists():
    raise RuntimeError(f"utils folder not found at {external_utils_path}")
    
sys.path.append(str(external_utils_path))

from demo_utils import draw_ov_watermark

app = FastAPI()

# Allow CORS (for frontend integration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Setup Project Paths ----------
PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config" / "illustration.yaml"

# Get model types from environment variables with defaults
IMAGE_MODEL_TYPE = os.getenv("IMAGE_MODEL_TYPE", "flux.1-schnell")
IMAGE_MODEL_DEVICE = os.getenv("IMAGE_MODEL_DEVICE", "AUTO")
LLM_MODEL_TYPE = os.getenv("LLM_MODEL_TYPE", "qwen2-7B")
LLM_MODEL_DEVICE = os.getenv("LLM_MODEL_DEVICE", "AUTO")
PRECISION = os.getenv("MODEL_PRECISION", "int4")

logger.info(f"Using Image Model Type: {IMAGE_MODEL_TYPE} on {IMAGE_MODEL_DEVICE}")
logger.info(f"Using LLM Model Type: {LLM_MODEL_TYPE} on {LLM_MODEL_DEVICE}")
logger.info(f"Using Model Precision: {PRECISION}")

image_model_dir = PROJECT_ROOT / "models" / f"{IMAGE_MODEL_TYPE}-{PRECISION.upper()}"
llm_model_dir = PROJECT_ROOT / "models" / f"{LLM_MODEL_TYPE}-{PRECISION.upper()}"

# ---------- Load Config ----------
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)
    
# ---------- Determine Device (GPU if available, else fallback) ----------
core = ov.Core()

# ---------- Load models ----------
image_pipe = None
llm_pipe = None

if image_model_dir.exists():
    try:
        image_pipe = ov_genai.Text2ImagePipeline(image_model_dir, device=IMAGE_MODEL_DEVICE)
        logger.info("Image model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load image model: {e}")
else:
    logger.warning(f"Image model not found at {image_model_dir}")

if llm_model_dir.exists():
    try:
        llm_pipe = ov_genai.LLMPipeline(str(llm_model_dir), device=LLM_MODEL_DEVICE)
        logger.info("LLM model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load LLM model: {e}")
else:
    logger.warning(f"LLM model not found at {llm_model_dir}")

llm_config = ov_genai.GenerationConfig()
llm_config.max_new_tokens = 256
llm_config.apply_chat_template = False

# ---------- Request/Response Schemas ----------
class PromptRequest(BaseModel):
    prompt: str

class StoryRequest(BaseModel):
    prompt: str

# --- Helper: Load YAML config ---
def load_story_config(req: Request) -> dict:
    config_type = req.query_params.get("config", "illustration")
    config_file = PROJECT_ROOT / "config" / f"{config_type}.yaml"
    if not config_file.exists():
        raise RuntimeError(f"Config file not found: {config_file}")
    with open(config_file, "r") as f:
        return yaml.safe_load(f)

# --- Helper: Parse LLM output into scenes ---
def parse_scenes(output_text: str, config: dict) -> list[str]:
    scenes = []
    current_scene = ""
    for line in output_text.split("\n"):
        clean = line.strip()
        if clean.lower().startswith(config["scene_prefix"].lower()):
            if current_scene:
                scenes.append(current_scene.strip())
            parts = clean.split(":", 1)
            if len(parts) == 2:
                current_scene = parts[1].strip()
        elif clean:
            current_scene += " " + clean
    if current_scene:
        scenes.append(current_scene.strip())
    return scenes

# --- Helper: Clean scenes and apply suffix ---
def finalize_scenes(scenes: list[str], config: dict) -> list[str]:
    suffixes = config["scene_suffixes"]
    max_words = config["max_words_per_scene"]
    final_scenes = []
    for i, scene in enumerate(scenes[:4]):
        words = scene.split()
        trimmed = " ".join(words[:max_words])
        suffix = suffixes[i] if i < len(suffixes) else suffixes[-1]
        if suffix.lower() not in trimmed.lower():
            if not trimmed.endswith((".", "!", "?")):
                trimmed += "."
            trimmed += " " + suffix
        final_scenes.append(trimmed.strip())
    while len(final_scenes) < 4:
        fallback = config["fallback_scene"]
        fallback_suffix = suffixes[len(final_scenes)] if len(suffixes) > len(final_scenes) else suffixes[-1]
        final_scenes.append(fallback + ". " + fallback_suffix)
    return final_scenes

# ---------- LLM Endpoint (Story Splitter) - Streaming ---------
@app.post("/generate_story_prompts")
def generate_story_prompts(request: StoryRequest, req: Request):
    if not llm_pipe:
        return JSONResponse(status_code=503, content={"error": "LLM model not available. Please export it before using this endpoint."})

    config = load_story_config(req)
    instruction = config["instruction_template"].replace("{user_prompt}", request.prompt)

    def stream_scenes():
        """Generator that yields tokens in real-time as they are generated."""
        # Use a queue to communicate between the streamer callback and the generator
        token_queue = queue.Queue()
        output = []
        generation_complete = threading.Event()
        
        def streamer(subword):
            """Callback that receives tokens from LLM as they're generated."""
            sys.stdout.write(subword)
            sys.stdout.flush()
            output.append(subword)
            token_queue.put(subword)  # Send token to queue for streaming
            return False
        
        def generate_llm():
            """Run LLM generation in a separate thread."""
            try:
                _ = llm_pipe.generate(instruction, llm_config, streamer)
            finally:
                generation_complete.set()
                token_queue.put(None)  # Signal completion
        
        # Start LLM generation in background thread
        generation_thread = threading.Thread(target=generate_llm)
        generation_thread.start()
        
        # Stream tokens as they arrive
        while True:
            token = token_queue.get()
            if token is None:  # Generation complete
                break
            # Yield each token as it arrives
            yield json.dumps({"token": token}) + "\n"
        
        # Wait for thread to complete
        generation_thread.join()
        
        # Now parse and send final scenes
        full_output = "".join(output)
        parsed_scenes = parse_scenes(full_output, config)
        final_scenes = finalize_scenes(parsed_scenes, config)
        
        # Send all parsed scenes
        for idx, scene in enumerate(final_scenes):
            yield json.dumps({"scene": scene, "index": idx}) + "\n"
        
        # Signal completion
        yield json.dumps({"done": True}) + "\n"
    
    return StreamingResponse(stream_scenes(), media_type="application/x-ndjson")

# ---------- Image Model Endpoint (Image Generator) ----------
@app.post("/generate_images")
def generate_image(request: PromptRequest):
    if not image_pipe:
        return JSONResponse(status_code=503, content={"error": "Image model not available. Please export it before using this endpoint."})

    prompt = request.prompt
    height = 512
    width = 512
    seed = random.randint(0, np.iinfo(np.int32).max)  #nosec B311
    steps = 4

    logger.info(f"Generating image for prompt: '{prompt}' with seed: {seed}")
    generator = ov_genai.TorchGenerator(seed)

    def callback(step, num_steps, latent):
        sys.stdout.write(f"Step {step+1}/{num_steps}\r")
        sys.stdout.flush()
        return False

    result = image_pipe.generate(
        prompt=prompt,
        num_inference_steps=steps,
        generator=generator,
        callback=callback,
        height=height,
        width=width
    )

    # Add watermark using OpenCV
    image_np = result.data[0]  # NumPy array
    draw_ov_watermark(image_np, size=0.6)

    # Convert to PIL and encode
    image = Image.fromarray(image_np)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return {"image": img_str}

# ---------- Server Start Print ----------
if image_pipe or llm_pipe:
    logger.info("FastAPI backend is running.")
    logger.info("In a separate terminal, start the Streamlit app using: streamlit run streamlit_app.py")
else:
    logger.warning("FastAPI backend is running, but no models were loaded.")
    logger.warning("Please export models before running the Streamlit app.")
