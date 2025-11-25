import subprocess
import time
import requests
import sys
import os
from pathlib import Path
import logging

# ----- Logging Setup -----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Add project root to path and import model converters
sys.path.append(str(Path(__file__).resolve().parent.parent))
from convert_and_optimize_llm import convert_chat_model
from convert_and_optimize_text2image import convert_image_model

# ----- Configuration -----

MODEL_DIR = Path("models")
LLM_MODEL_TYPE = "tiny-llama-1b-chat"
IMAGE_MODEL_TYPE = "lcm"
PRECISION = "int4"
LOG_FILE = Path("gradio_log.txt")

# ----- Step 1: Export Models if Needed (will handle download internally) -----
logger.info("Checking and exporting LLM + Text2Image models if necessary...")
convert_chat_model(LLM_MODEL_TYPE, PRECISION, MODEL_DIR)
convert_image_model(IMAGE_MODEL_TYPE, PRECISION, MODEL_DIR)

# ----- Step 2: Launch FastAPI Backend -----
logger.info("Launching FastAPI server...")
env = os.environ.copy()
env.update({
    "IMAGE_MODEL_TYPE": IMAGE_MODEL_TYPE,
    "LLM_MODEL_TYPE": LLM_MODEL_TYPE,
    "MODEL_PRECISION": PRECISION
})

with LOG_FILE.open("w") as lf:
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"],
        env=env,
        stdout=lf,
        stderr=subprocess.STDOUT
    )

try:
    # ----- Wait for Readiness from Logs -----
    logger.info("Waiting for FastAPI log to report readiness...")
    start_time = time.time()
    timeout = 130  # seconds

    while time.time() - start_time < timeout:
        if LOG_FILE.exists():
            content = LOG_FILE.read_text()
            if "Uvicorn running on" in content or "Application startup complete." in content:
                logger.info("FastAPI server is up.")
                break
        time.sleep(1)
    else:
        raise RuntimeError("FastAPI server did not start within timeout period.")

    # ----- Step 3: Test Story Prompt Generation -----
    logger.info("Testing /generate_story_prompts endpoint...")
    response1 = requests.post(
        "http://localhost:8000/generate_story_prompts",
        json={"prompt": "A flying whale in space"}
    )
    assert response1.status_code == 200, f"Story generation failed: {response1.text}"
    scenes = response1.json()["scenes"]
    logger.info("Scene prompt generation test passed. Example: %s", scenes)

    # ----- Step 4: Test Image Generation -----
    logger.info("Testing /generate_images endpoint...")
    response2 = requests.post(
        "http://localhost:8000/generate_images",
        json={"prompt": scenes[0]}
    )
    assert response2.status_code == 200, f"Image generation failed: {response2.text}"
    image = response2.json()["image"]
    logger.info("Image generation test passed. Base64 (truncated): %s", image[:100])
    print("Demo is ready!", flush=True) # Required for the CI to detect readiness

finally:
    logger.info("Shutting down FastAPI server...")
    process.terminate()
    process.wait()