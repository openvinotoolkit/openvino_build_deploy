import os
import sys
from pathlib import Path

PARENT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(os.path.dirname(PARENT_DIR))

from .. import app
from .. import convert_and_optimize_llm as chat

if __name__ == '__main__':
    model_dir = "model"
    chat_model_type = "llama3.2-1B"
    embedding_model_type = "bge-small"
    chat_precision = "int4"
    rag_pdf = "../data/test_painting_llm_rag.pdf"
    device = "AUTO"

    embedding_model_dir = chat.convert_embedding_model(embedding_model_type, Path(model_dir))
    chat_model_dir = chat.convert_chat_model(chat_model_type, chat_precision, Path(model_dir), None)

    app.run(str(chat_model_dir.parent), str(embedding_model_dir.parent), Path(rag_pdf), device)
