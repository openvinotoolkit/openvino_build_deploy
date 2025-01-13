import argparse
from pathlib import Path

import app
import convert_and_optimize_asr as asr
import convert_and_optimize_chat as chat


def main(args):
    asr_model_dir = asr.convert_asr_model(args.asr_model_type, args.asr_precision, Path(args.model_dir))
    chat_model_dir = chat.convert_chat_model(args.chat_model_type, args.chat_precision, Path(args.model_dir))

    app.run(asr_model_dir, chat_model_dir, args.public)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--asr_model_type", type=str, choices=["distil-whisper-large-v3", "belle-distilwhisper-large-v2-zh"],
                        default="distil-whisper-large-v3", help="Speech recognition model to be converted")
    parser.add_argument("--asr_precision", type=str, default="fp16", choices=["fp16", "int8"], help="ASR model precision")
    parser.add_argument("--chat_model_type", type=str, choices=["llama3.1-8B", "llama3-8B", "qwen2-7B", "llama3.2-3B"],
                        default="llama3.2-3B", help="Chat model to be converted")
    parser.add_argument("--chat_precision", type=str, default="int4", choices=["fp16", "int8", "int4"], help="Chat model precision")
    parser.add_argument("--model_dir", type=str, default="model", help="Directory to place the model in")
    parser.add_argument('--public_interface', default=False, action="store_true", help="Whether interface should be available publicly")

    main(parser.parse_args())
