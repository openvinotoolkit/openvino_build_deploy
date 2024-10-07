import argparse
from pathlib import Path

from optimum.intel import OVModelForCausalLM, OVWeightQuantizationConfig, OVConfig, OVQuantizer
from transformers import AutoTokenizer

#TBD Add Embedding Model Conversion


MODEL_MAPPING = {
    "llama3-8B": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama3.1-8B": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "llama3.2-3B": "meta-llama/Llama-3.2-3B-Instruct",
    "llama3.2-11B": "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "llama2-7B": "meta-llama/Llama-2-7b-chat-hf",
    "llama2-13B": "meta-llama/Llama-2-13b-chat-hf",
    "qwen2-7B": "Qwen/Qwen2-7B-Instruct", 
}


def convert_chat_model(model_type: str, precision: str, model_dir: Path, access_token: str) -> Path:
    """
    Convert chat model

    Params:
        model_type: selected mode type and size
        precision: model precision
        model_dir: dir to export model
        access_token: access token from Hugging Face to download gated models
    Returns:
       Path to exported model
    """
    output_dir = model_dir / model_type
    model_name = MODEL_MAPPING[model_type]

    # load model and convert it to OpenVINO
    model = OVModelForCausalLM.from_pretrained(model_name, export=True, compile=False, load_in_8bit=False, token=access_token)
    # change precision to FP16
    model.half()

    if precision != "fp16":
        # select quantization mode
        quant_config = OVWeightQuantizationConfig(bits=4, sym=False, ratio=0.8) if precision == "int4" else OVWeightQuantizationConfig(bits=8, sym=False)
        config = OVConfig(quantization_config=quant_config)

        suffix = "-INT4" if precision == "int4" else "-INT8"
        output_dir = output_dir.with_name(output_dir.name + suffix)

        # create a quantizer
        quantizer = OVQuantizer.from_pretrained(model, task="text-generation")
        # quantize weights and save the model to the output dir
        quantizer.quantize(save_directory=output_dir, weights_only=True, ov_config=config)
    else:
        output_dir = output_dir.with_name(output_dir.name + "-FP16")
        # save converted model
        model.save_pretrained(output_dir)

    # export also tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(output_dir)

    return Path(output_dir) / "openvino_model.xml"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chat_model_type", type=str, choices=["llama2-7B", "llama2-13B", "llama3-8B", "qwen2-7B", "llama3.2-3B", "llama3.1-8B", "llama3.2-11B"],
                        default="llama3.2-11B", help="Chat model to be converted")
    parser.add_argument("--precision", type=str, default="int4", choices=["fp16", "int8", "int4"], help="Model precision")
    parser.add_argument("--hf_token", type=str, help="HuggingFace access token to get Llama3")
    parser.add_argument("--model_dir", type=str, default="model", help="Directory to place the model in")

    args = parser.parse_args()
    convert_chat_model(args.chat_model_type, args.precision, Path(args.model_dir), args.hf_token)
