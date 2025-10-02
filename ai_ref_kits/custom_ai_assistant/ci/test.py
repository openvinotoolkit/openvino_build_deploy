import os
import sys
from pathlib import Path

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PARENT_DIR)

import app
import convert_and_optimize_asr as asr
import convert_and_optimize_chat as chat

if __name__ == '__main__':
    model_dir = Path("model")
    asr_model_type = "distil-whisper-large-v3"
    asr_precision = "fp16"
    asr_model_device = "AUTO"
    chat_model_type = "llama3.2-1B"
    chat_precision = "int4"
    chat_model_device = "AUTO"
    personality_path = Path("../config/concierge_personality.yaml")
    example_pdf = Path("../data/Grand_Azure_Resort_Spa_Full_Guide.pdf")

    asr_model_dir = asr.convert_asr_model(asr_model_type, asr_precision, model_dir)
    chat_model_dir = chat.convert_chat_model(chat_model_type, chat_precision, model_dir)

    app.run(asr_model_dir, asr_model_device, chat_model_dir, chat_model_device)