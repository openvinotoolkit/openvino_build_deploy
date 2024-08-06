from huggingface_hub import snapshot_download
from pathlib import Path

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import warnings
import argparse



parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="model", help="Directory to place the model in")
args = parser.parse_args()
model_local_dir = Path(args.model_dir + "/nanoLLaVA")

if not model_local_dir.exists():
    snapshot_download(repo_id="qnguyen3/nanoLLaVA", local_dir=model_local_dir)

modeling_file = model_local_dir / "modeling_llava_qwen2.py"
orig_modeling_file = model_local_dir / f"orig_{modeling_file.name}"


# model code depends from flash_attn package that may be problematic to load. Patch model code for avoiding import of this package
if not orig_modeling_file.exists():
    modeling_file.rename(orig_modeling_file)
with orig_modeling_file.open("r") as f:
    content = f.read()
replacement_lines = [
    ("from flash_attn import flash_attn_func, flash_attn_varlen_func", ""),
    ("from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input", ""),
    (' _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)', "pass"),
]

for replace_pair in replacement_lines:
    content = content.replace(*replace_pair)

with modeling_file.open("w") as f:
    f.write(content)


transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore")

model = AutoModelForCausalLM.from_pretrained(model_local_dir, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_local_dir, trust_remote_code=True)    

import gc
import warnings
import torch
import openvino as ov
import nncf
from typing import Optional, Tuple

warnings.filterwarnings("ignore")


def flattenize_inputs(inputs):
    """
    Helper function for making nested inputs flattens
    """
    flatten_inputs = []
    for input_data in inputs:
        if input_data is None:
            continue
        if isinstance(input_data, (list, tuple)):
            flatten_inputs.extend(flattenize_inputs(input_data))
        else:
            flatten_inputs.append(input_data)
    return flatten_inputs


def cleanup_torchscript_cache():
    """
    Helper for removing cached model representation
    """
    torch._C._jit_clear_class_registry()
    torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
    torch.jit._state._clear_class_state()


def postprocess_converted_model(
    ov_model,
    example_input=None,
    input_names=None,
    output_names=None,
    dynamic_shapes=None,
):
    """
    Helper function for appling postprocessing on converted model with updating input names, shapes and output names
    acording to requested specification
    """
    flatten_example_inputs = flattenize_inputs(example_input) if example_input else []

    if input_names:
        for inp_name, m_input, input_data in zip(input_names, ov_model.inputs, flatten_example_inputs):
            input_node = m_input.get_node()
            if input_node.element_type == ov.Type.dynamic:
                m_input.get_node().set_element_type(ov.Type.f32)
            shape = list(input_data.shape)
            if dynamic_shapes is not None and inp_name in dynamic_shapes:
                for k in dynamic_shapes[inp_name]:
                    shape[k] = -1
            input_node.set_partial_shape(ov.PartialShape(shape))
            m_input.get_tensor().set_names({inp_name})

    if output_names:
        for out, out_name in zip(ov_model.outputs, output_names):
            out.get_tensor().set_names({out_name})
    ov_model.validate_nodes_and_infer_types()
    return ov_model


ov_out_path = Path(args.model_dir + "/ov_nanollava/INT4_compressed_weights")
llava_wc_parameters = dict(mode=nncf.CompressWeightsMode.INT4_ASYM, group_size=128, ratio=0.8)

image_encoder_wc_parameters = dict(mode=nncf.CompressWeightsMode.INT8)

ov_out_path.mkdir(exist_ok=True, parents=True)
model.config.save_pretrained(ov_out_path)
vision_tower = model.get_vision_tower()
if not vision_tower.is_loaded:
    vision_tower.load_model()

image_encoder_path = ov_out_path / "image_encoder.xml"
token_embedding_model_path = ov_out_path / "token_embed.xml"
model_path = ov_out_path / "llava_with_past.xml"

model.eval()
model.config.use_cache = True
model.config.torchscript = True

if not image_encoder_path.exists():
    model.forward = model.encode_images
    with torch.no_grad():
        ov_model = ov.convert_model(
            model,
            example_input=torch.zeros((1, 3, 384, 384)),
            input=[(-1, 3, 384, 384)],
        )
    if image_encoder_wc_parameters is not None:
        print("Applying weight compression to image encoder")
        ov_model = nncf.compress_weights(ov_model, **image_encoder_wc_parameters)
    ov.save_model(ov_model, image_encoder_path)
    cleanup_torchscript_cache()
    del ov_model
    gc.collect()
    print("Image Encoder model successfully converted")

if not token_embedding_model_path.exists():
    with torch.no_grad():
        ov_model = ov.convert_model(model.get_model().embed_tokens, example_input=torch.ones((1, 10), dtype=torch.long))
    ov.save_model(ov_model, token_embedding_model_path)
    cleanup_torchscript_cache()
    del ov_model
    gc.collect()
    print("Token Embedding model successfully converted")

if not model_path.exists():
    model.forward = super(type(model), model).forward
    example_input = {"attention_mask": torch.ones([2, 10], dtype=torch.int64), "position_ids": torch.tensor([[8, 9], [8, 9]], dtype=torch.int64)}

    dynamic_shapes = {
        "input_embeds": {0: "batch_size", 1: "seq_len"},
        "attention_mask": {0: "batch_size", 1: "prev_seq_len + seq_len"},
        "position_ids": {0: "batch_size", 1: "seq_len"},
    }
    input_embeds = torch.zeros((2, 2, model.config.hidden_size))

    input_names = ["attention_mask", "position_ids"]
    output_names = ["logits"]

    past_key_values = []
    for i in range(model.config.num_hidden_layers):
        kv = [torch.randn([2, model.config.num_key_value_heads, 8, model.config.hidden_size // model.config.num_attention_heads]) for _ in range(2)]
        past_key_values.append(kv)
        input_names.extend([f"past_key_values.{i}.key", f"past_key_values.{i}.value"])
        output_names.extend([f"present.{i}.key", f"present.{i}.value"])
        dynamic_shapes[input_names[-2]] = {0: "batch_size", 2: "seq_len"}
        dynamic_shapes[input_names[-1]] = {0: "batch_size", 2: "seq_len"}

    example_input["past_key_values"] = past_key_values
    example_input["inputs_embeds"] = input_embeds
    input_names.append("inputs_embeds")
    dynamic_shapes["inputs_embeds"] = {0: "batch_size", 1: "seq_len"}
    ov_model = ov.convert_model(model, example_input=example_input)
    ov_model = postprocess_converted_model(
        ov_model, example_input=example_input.values(), input_names=input_names, output_names=output_names, dynamic_shapes=dynamic_shapes
    )

    if llava_wc_parameters is not None:
        print("Applying weight compression to second stage LLava model")
        ov_model = nncf.compress_weights(ov_model, **llava_wc_parameters)
    ov.save_model(ov_model, model_path)
    cleanup_torchscript_cache()
    del ov_model
    gc.collect()

    print("LLaVA model successfully converted")
del model
gc.collect();

