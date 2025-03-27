import os
import sys
from pathlib import Path

PARENT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(os.path.dirname(PARENT_DIR))

from .. import app
from .. import convert_and_optimize_asr as asr
from .. import convert_and_optimize_chat as chat

if __name__ == '__main__':
    model_dir = "model"
    asr_model_type = "distil-whisper-large-v3"
    asr_precision = "fp16"
    embedding_model_type = "bge-small"
    reranker_model_type = "bge-reranker-base"
    chat_model_type = "llama3.2-1B"
    chat_precision = "int4"
    personality_path = Path("../config/concierge_personality.yaml")
    example_pdf = Path("../data/Grand_Azure_Resort_Spa_Full_Guide.pdf")

    asr_model_dir = asr.convert_asr_model(asr_model_type, asr_precision, Path(model_dir))

    embedding_model_dir = chat.convert_embedding_model(embedding_model_type, Path(model_dir))
    reranker_model_dir = chat.convert_reranker_model(reranker_model_type, Path(model_dir))
    chat_model_dir = chat.convert_chat_model(chat_model_type, chat_precision, Path(model_dir))

    app.run(asr_model_dir, chat_model_dir, embedding_model_dir, reranker_model_dir, personality_path, example_pdf)