import argparse
import json
from pathlib import Path
import os
import logging

from optimum.intel import (
    OVModelForCausalLM,
    OVWeightQuantizationConfig,
    OVConfig,
    OVQuantizer
)
from transformers import AutoTokenizer

# -------- Logging Setup --------
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Extend timeout to avoid HF read failures
os.environ["HF_HUB_TIMEOUT"] = "60"

MODEL_MAPPING = {
    "tiny-llama-1b-chat": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "neural-chat-7b-v3-3": "Intel/neural-chat-7b-v3-3",
    "mistral-7B-Instruct-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
    "zephyr-7b-beta": "HuggingFaceH4/zephyr-7b-beta",
    "qwen2-7B": "Qwen/Qwen2-7B-Instruct",
    "DeepSeek-R1-Distill-Qwen-1.5B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
}

CRITICAL_FILES = [
    "openvino_model.xml",
    "openvino_model.bin",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "model_index.json",
    "openvino_tokenizer.xml",
    "openvino_detokenizer.xml",
]

def convert_and_save_tokenizer(model_name: str, output_dir: Path):
    """
    Saves the tokenizer and optionally exports it to OpenVINO format if supported.
    Also generates fallback tokenizer_config.json and special_tokens_map.json if missing.
    """
    logger.info("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)

    try:
        from openvino_tokenizers import convert_tokenizer
        import openvino as ov
        logger.info("Converting and saving OpenVINO tokenizer...")
        ov_tokenizer, ov_detokenizer = convert_tokenizer(tokenizer, with_detokenizer=True)
        ov.save_model(ov_tokenizer, output_dir / "openvino_tokenizer.xml")
        ov.save_model(ov_detokenizer, output_dir / "openvino_detokenizer.xml")
    except ImportError:
        logger.warning("openvino_tokenizers not installed. Skipping OV tokenizer export.")

    if not (output_dir / "tokenizer_config.json").exists():
        logger.warning("tokenizer_config.json missing, creating fallback...")
        (output_dir / "tokenizer_config.json").write_text(
            json.dumps({"tokenizer_class": tokenizer.__class__.__name__}, indent=2)
        )

    if not (output_dir / "special_tokens_map.json").exists():
        logger.warning("special_tokens_map.json missing, creating fallback...")
        (output_dir / "special_tokens_map.json").write_text(json.dumps({}, indent=2))

def validate_export(output_dir: Path) -> list[str]:
    """
    Validates the presence of all required files in the exported model directory.
    Returns a list of any missing critical files.
    """
    return [file for file in CRITICAL_FILES if not (output_dir / file).exists()]

def quantize_model(model, precision: str, output_dir: Path):
    """
    Applies quantization to the given model and saves it to the output directory.
    Supports INT8 and INT4 quantization using optimum.intel.
    """
    if precision == "fp16":
        logger.info("Converting model to FP16...")
        model.half()
        model.save_pretrained(output_dir)
    else:
        logger.info(f"Quantizing model to {precision.upper()}...")
        quant_config = OVWeightQuantizationConfig(bits=4, sym=False, ratio=0.8) if precision == "int4" \
                        else OVWeightQuantizationConfig(bits=8, sym=False)
        config = OVConfig(quantization_config=quant_config)
        quantizer = OVQuantizer.from_pretrained(model, task="text-generation")
        quantizer.quantize(save_directory=output_dir, ov_config=config)
        
def convert_chat_model(model_type: str, precision: str, model_dir: Path) -> Path:
    """
    Main entrypoint to convert a llm model to OpenVINO format.
    Handles export, quantization, tokenizer conversion, and file validation.
    """
    base_output_dir = model_dir / model_type
    suffix = "-FP16" if precision == "fp16" else "-INT4" if precision == "int4" else "-INT8"
    output_dir = base_output_dir.with_name(base_output_dir.name + suffix)

    # Skip export if all critical files already exist
    if output_dir.exists():
        missing = [f for f in CRITICAL_FILES if not (output_dir / f).exists()]
        if not missing:
            logger.info(f"\n Model already exported at: {output_dir}")
            logger.info("Skipping re-export.\n")
            return output_dir
        else:
            logger.warning(f"\n Model folder exists but missing files: {missing}")
            logger.info("Proceeding to re-export...\n")

    model_name = MODEL_MAPPING[model_type]
    logger.info(f"\n Loading model: {model_name}")
    model = OVModelForCausalLM.from_pretrained(model_name, export=True, compile=False, load_in_8bit=False)

    quantize_model(model, precision, output_dir)
    convert_and_save_tokenizer(model_name, output_dir)

    logger.info("Writing model_index.json...")
    model_index = {
        "model_type": "text-generation",
        "precision": precision.upper(),
        "exported_with": "optimum-cli"
    }
    (output_dir / "model_index.json").write_text(json.dumps(model_index, indent=2))

    logger.info("Checking for missing critical files silently...")
    missing_files = validate_export(output_dir)
    if missing_files:
        logger.warning("Export completed with missing files.")
    else:
        logger.info("Export successful. All critical files are present.")

    logger.info(f"Final exported model at: {output_dir}\n")
    return output_dir

if __name__ == "__main__":
    model_keys = list(MODEL_MAPPING.keys())

    parser = argparse.ArgumentParser(description="Export and quantize a chat LLM with OpenVINO")
    parser.add_argument(
        "--chat_model_type",
        type=str,
        choices=model_keys,
        help="Model type to convert. If not provided, an interactive prompt will be shown."
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp16", "int8", "int4"],
        default="int4",
        help="Precision to use (default: int4)"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models",
        help="Output directory"
    )

    args = parser.parse_args()

    if not args.chat_model_type:
        print("Available Chat Models:")
        for i, key in enumerate(model_keys, start=1):
            print(f"  {i}. {key}")
        choice = input("Enter model number to export: ").strip()
        if not choice.isdigit() or not (1 <= int(choice) <= len(model_keys)):
            print("Invalid choice. Exiting.")
            exit(1)
        args.chat_model_type = model_keys[int(choice) - 1]

    convert_chat_model(args.chat_model_type, args.precision, Path(args.model_dir))
    logger.info("Conversion and optimization completed.")