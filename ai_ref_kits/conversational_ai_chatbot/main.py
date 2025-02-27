import argparse
from pathlib import Path

import app
import convert_and_optimize_asr as asr
import convert_and_optimize_chat as chat


def main(args):
    asr_model_dir = asr.convert_asr_model(args.asr_model_type, args.asr_precision, Path(args.model_dir))

    embedding_model_dir = chat.convert_embedding_model(args.embedding_model_type, Path(args.model_dir))
    reranker_model_dir = chat.convert_reranker_model(args.reranker_model_type, Path(args.model_dir))
    chat_model_dir = chat.convert_chat_model(args.chat_model_type, args.chat_precision, Path(args.model_dir), args.hf_token)

    app.run(asr_model_dir, chat_model_dir, embedding_model_dir, reranker_model_dir, Path(args.personality), Path(args.example_pdf), args.public)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--asr_model_type", type=str,
                        choices=["distil-whisper-large-v3", "belle-distil-whisper-large-v3-zh"], default="distil-whisper-large-v3",
                        help="Speech recognition model to be converted")
    parser.add_argument("--asr_precision", type=str, default="fp16", choices=["fp16", "int8"], help="ASR model precision")
    parser.add_argument("--chat_model_type", type=str, choices=["llama3.2-3B", "llama3.1-8B", "llama3-8B", "qwen2-7B"],
                        default="llama3.2-3B", help="Chat model to be converted")
    parser.add_argument("--embedding_model_type", type=str, choices=["bge-small", "bge-large", "bge-m3"],
                        default="bge-small", help="Embedding model to be converted")
    parser.add_argument("--reranker_model_type", type=str, choices=["bge-reranker-large", "bge-reranker-base", "bge-reranker-m3"],
                        default="bge-reranker-large", help="Reranker model to be converted")
    parser.add_argument("--chat_precision", type=str, default="int4", choices=["fp16", "int8", "int4"], help="Chat model precision")
    parser.add_argument("--hf_token", type=str, help="HuggingFace access token to get Llama3")
    parser.add_argument("--model_dir", type=str, default="model", help="Directory to place the model in")
    parser.add_argument("--personality", type=str, default="config/concierge_personality.yaml",
                        help="Path to the YAML file with chatbot personality")
    parser.add_argument("--example_pdf", type=str, default="data/Grand_Azure_Resort_Spa_Full_Guide.pdf",
                        help="Path to the PDF file which is an additional context")
    parser.add_argument("--public", default=False, action="store_true", help="Whether interface should be available publicly")

    main(parser.parse_args())