import os
import sys
from pathlib import Path

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PARENT_DIR)

import app
import convert_and_optimize_llm as chat

if __name__ == '__main__':
    model_dir = Path("model")
    chat_model_type = "llama3.2-1B"
    embedding_model_type = "bge-small"
    chat_precision = "int4"
    rag_pdf = Path(__file__).parent.parent / "data" / "test_painting_llm_rag.pdf"
    device = "AUTO"

    embedding_model_dir = chat.convert_embedding_model(embedding_model_type, model_dir)
    chat_model_dir = chat.convert_chat_model(chat_model_type, chat_precision, model_dir, None)

    app.run(chat_model_dir.parent, embedding_model_dir.parent, rag_pdf, device)
