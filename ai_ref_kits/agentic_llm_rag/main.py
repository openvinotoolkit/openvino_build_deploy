import argparse
from pathlib import Path

import app
import convert_and_optimize_llm as chat


def main(args):  
    embedding_model_dir = chat.convert_embedding_model(args.embedding_model_type, Path(args.model_dir))    
    chat_model_dir = chat.convert_chat_model(args.chat_model_type, args.chat_precision, Path(args.model_dir), args.hf_token)

    app.run(chat_model_dir, embedding_model_dir, Path(args.rag_pdf), "GPU", args.public)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()    
    
    parser.add_argument("--chat_model_type", type=str, choices=["qwen2-7B"],
                        default="qwen2-7B", help="Chat model to be converted")
    parser.add_argument("--embedding_model_type", type=str, choices=["bge-small", "bge-large", "bge-m3"],
                        default="bge-small", help="Embedding model to be converted")
    parser.add_argument("--chat_precision", type=str, default="int4", choices=["fp16", "int8", "int4"], help="Chat model precision")
    parser.add_argument("--hf_token", type=str, help="HuggingFace access token to get Llama3")
    parser.add_argument("--model_dir", type=str, default="model", help="Directory to place the model in")    
    parser.add_argument("--rag_pdf", type=str, default="data/large.pdf",
                        help="Path to the PDF file which is an additional context")
    parser.add_argument("--public", default=False, action="store_true", help="Whether interface should be available publicly")

    main(parser.parse_args())