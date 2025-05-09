from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from io import BytesIO
from PIL import Image
import base64
import sys
import yaml
import openvino_genai as ov_genai


# Import watermark function
external_utils_path = (Path(__file__).resolve().parents[2] / "demos" / "utils").resolve()
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

flux_model_dir = PROJECT_ROOT / "models" / "flux.1-schnell-INT4"
qwen_model_dir = PROJECT_ROOT / "models" / "qwen2-7B-INT4"

# ---------- Load Config ----------
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)
    
# ---------- Load Flux Text2Image Model ----------
if not flux_model_dir.exists():
    print(f"Model path {flux_model_dir} not found.")
    sys.exit(0)

print("Loading Flux text-to-image model...")
flux_pipe = ov_genai.Text2ImagePipeline(flux_model_dir, device="GPU")

# ---------- Load Qwen LLM Model ----------
if not qwen_model_dir.exists():
    print(f"Model path {qwen_model_dir} not found.")
    sys.exit(0)

print("Loading Qwen LLM model...")
llm_pipe = ov_genai.LLMPipeline(str(qwen_model_dir), device="GPU")
llm_config = ov_genai.GenerationConfig()
llm_config.max_new_tokens = 256
llm_config.apply_chat_template = False

# ---------- Request/Response Schemas ----------
class PromptRequest(BaseModel):
    prompt: str

class StoryRequest(BaseModel):
    prompt: str

# ---------- LLM Endpoint (Story Splitter) ----------
from fastapi import Request

@app.post("/generate_story_prompts")
def generate_story_prompts(request: StoryRequest, req: Request):
    config_type = req.query_params.get("config", "illustration")
    config_file = PROJECT_ROOT / "config" / f"{config_type}.yaml"
    if not config_file.exists():
        raise RuntimeError(f"Config file not found: {config_file}")
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

# @app.post("/generate_story_prompts")
# def generate_story_prompts(request: StoryRequest):
    user_prompt = request.prompt

    instruction = config["instruction_template"].replace("{user_prompt}", user_prompt)

    output = []
    def streamer(subword):
        sys.stdout.write(subword)
        sys.stdout.flush()
        output.append(subword)
        return False

    _ = llm_pipe.generate(instruction, llm_config, streamer)
    full_output = "".join(output)

    # --- Parse scenes ---
    scenes = []
    current_scene = ""
    for line in full_output.split("\n"):
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

    # --- Clean & format output ---
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
        fallback_suffix = suffixes[len(final_scenes)] if len(suffixes) > len(final_scenes) else suffixes[-1]
        final_scenes.append(config["fallback_scene"] + ". " + fallback_suffix)
   
    return {"scenes": final_scenes}

# ---------- Flux Endpoint (Image Generator) ----------
@app.post("/generate_images")
def generate_image(request: PromptRequest):
    prompt = request.prompt
    height = 512
    width = 512
    seed = 191524753
    steps = 4

    generator = ov_genai.TorchGenerator(seed)

    def callback(step, num_steps, latent):
        sys.stdout.write(f"Step {step+1}/{num_steps}\r")
        sys.stdout.flush()
        return False

    result = flux_pipe.generate(
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
print("FastAPI backend is running.")
print("In a separate terminal, start the Streamlit app using:")
print("streamlit run app.py")