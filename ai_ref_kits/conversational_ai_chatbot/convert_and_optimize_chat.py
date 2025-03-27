import argparse
from pathlib import Path

import numpy as np
import openvino as ov
from openvino.runtime import opset10 as ops
from openvino.runtime import passes
from optimum.intel import OVModelForCausalLM, OVModelForFeatureExtraction, OVWeightQuantizationConfig, OVConfig, \
    OVQuantizer, OVModelForSequenceClassification
from transformers import AutoTokenizer

MODEL_MAPPING = {
    "llama3-8B": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama3.1-8B": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "llama3.2-1B": "meta-llama/Llama-3.2-1B-Instruct",
    "llama3.2-3B": "meta-llama/Llama-3.2-3B-Instruct",
    "qwen2-7B": "Qwen/Qwen2-7B-Instruct",
    "bge-small": "BAAI/bge-small-en-v1.5",
    "bge-large": "BAAI/bge-large-en-v1.5",
    "bge-m3": "BAAI/bge-m3",
    "bge-reranker-base": "BAAI/bge-reranker-base",
    "bge-reranker-large": "BAAI/bge-reranker-large",
    "bge-reranker-m3": "BAAI/bge-reranker-v2-m3"
}


def optimize_model_for_npu(model: OVModelForFeatureExtraction):
    """
    Fix some tensors to support NPU inference

     Params:
        model: model to fix
    """
    class ReplaceTensor(passes.MatcherPass):
        def __init__(self, packed_layer_name_tensor_dict_list):
            super().__init__()
            self.model_changed = False

            param = passes.WrapType("opset10.Multiply")

            def callback(matcher: passes.Matcher) -> bool:
                root = matcher.get_match_root()
                if root is None:
                    return False
                for y in packed_layer_name_tensor_dict_list:
                    root_name = root.get_friendly_name()
                    if root_name.find(y["name"]) != -1:
                        max_fp16 = np.array([[[[-np.finfo(np.float16).max]]]]).astype(np.float32)
                        new_tensor = ops.constant(max_fp16, ov.Type.f32, name="Constant_4431")
                        root.set_arguments([root.input_value(0).node, new_tensor])
                        packed_layer_name_tensor_dict_list.remove(y)

                return True

            self.register_matcher(passes.Matcher(param, "ReplaceTensor"), callback)

    packed_layer_tensor_dict_list = [{"name": "aten::mul/Multiply"}]

    manager = passes.Manager()
    manager.register_pass(ReplaceTensor(packed_layer_tensor_dict_list))
    manager.run_passes(model.model)
    model.reshape(1, 512)


def convert_chat_model(model_type: str, precision: str, model_dir: Path, access_token: str | None = None) -> Path:
    """
    Convert chat model

    Params:
        model_type: selected mode type and size
        precision: model precision
        model_dir: dir to export model
        access_token: access token from Hugging Face to download gated models
    Returns:
       Path to exported model dir
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
        quantizer.quantize(save_directory=output_dir, ov_config=config)
    else:
        output_dir = output_dir.with_name(output_dir.name + "-FP16")
        # save converted model
        model.save_pretrained(output_dir)

    # export also tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(output_dir)

    return Path(output_dir)


def convert_embedding_model(model_type: str, model_dir: Path) -> Path:
    """
    Convert embedding model

    Params:
        model_type: selected mode type and size
        model_dir: dir to export model
    Returns:
       Path to exported model dir
    """
    output_dir = model_dir / model_type
    output_dir = output_dir.with_name(output_dir.name + "-FP32")
    model_name = MODEL_MAPPING[model_type]

    # load model and convert it to OpenVINO
    model = OVModelForFeatureExtraction.from_pretrained(model_name, export=True, compile=False)
    optimize_model_for_npu(model)
    model.save_pretrained(output_dir)

    # export tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(output_dir)

    return Path(output_dir)


def convert_reranker_model(model_type: str, model_dir: Path) -> Path:
    """
    Convert reranker model

    Params:
        model_type: selected mode type and size
        model_dir: dir to export model
    Returns:
       Path to exported model dir
    """
    output_dir = model_dir / model_type
    output_dir = output_dir.with_name(output_dir.name + "-FP32")
    model_name = MODEL_MAPPING[model_type]

    # load model and convert it to OpenVINO
    model = OVModelForSequenceClassification.from_pretrained(model_name, export=True, compile=False)
    model.save_pretrained(output_dir)

    # export tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(output_dir)

    return Path(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chat_model_type", type=str, choices=["llama3.2-3B", "llama3.2-1B", "llama3.1-8B", "llama3-8B", "qwen2-7B"],
                        default="llama3.2-3B", help="Chat model to be converted")
    parser.add_argument("--embedding_model_type", type=str, choices=["bge-small", "bge-large", "bge-m3"],
                        default="bge-small", help="Embedding model to be converted")
    parser.add_argument("--reranker_model_type", type=str, choices=["bge-reranker-large", "bge-reranker-base", "bge-reranker-m3"],
                        default="bge-reranker-large", help="Reranker model to be converted")
    parser.add_argument("--precision", type=str, default="int4", choices=["fp16", "int8", "int4"], help="Model precision")
    parser.add_argument("--hf_token", type=str, help="HuggingFace access token to get Llama3")
    parser.add_argument("--model_dir", type=str, default="model", help="Directory to place the model in")

    args = parser.parse_args()
    convert_embedding_model(args.embedding_model_type, Path(args.model_dir))
    convert_reranker_model(args.reranker_model_type, Path(args.model_dir))
    convert_chat_model(args.chat_model_type, args.precision, Path(args.model_dir), args.hf_token)
