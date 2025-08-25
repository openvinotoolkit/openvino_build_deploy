# CMD TO RUN: TOKENIZERS_PARALLELISM=false python openvino_bridgetower_conversion.py
from bridgetower_custom import BridgeTowerTextFeatureExtractor, BridgeTowerForITC, BridgeTowerVisionFeatureExtractor
from transformers import BridgeTowerProcessor
import requests
from PIL import Image
import torch
import torch.nn.functional as F
import openvino as ov 
import os
from pathlib import Path

# prepare hyperparameters
static_shapes = True
max_length = 100
models_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "bridgetower_models"
Path(models_dir).mkdir(parents=True, exist_ok=True)

# prepare models
model_name = "BridgeTower/bridgetower-large-itm-mlm-itc"
processor = BridgeTowerProcessor.from_pretrained(model_name)
model = BridgeTowerForITC.from_pretrained(model_name)
vision_model = BridgeTowerVisionFeatureExtractor.from_pretrained(model_name)
text_model = BridgeTowerTextFeatureExtractor.from_pretrained(model_name)

# prepare input examples
image_urls = [
    "http://images.cocodataset.org/val2017/000000039769.jpg"]
texts = [
    "two cats sleeping on a couch"]
images = [Image.open(requests.get(url, stream=True).raw) for url in image_urls]

####################CONVERT BRIDGETOWER MODEL########################
print("Converting BridgeTower model for image-text pair input...")
# prepare image-text pair input examples
vision_text_encoding_inputs = processor(images[0], texts[0], return_tensors="pt")
if static_shapes:
    padding_len = max_length - vision_text_encoding_inputs.input_ids.shape[-1]
    vision_text_encoding_inputs["input_ids"] = F.pad(
        vision_text_encoding_inputs.input_ids,
        (0, padding_len),
        value=model.config.text_config.pad_token_id,
    )
    vision_text_encoding_inputs["attention_mask"] = F.pad(
        vision_text_encoding_inputs.attention_mask, (0, padding_len), value=0
    )

ov_model = ov.convert_model(model, example_input={**vision_text_encoding_inputs})
ov_model_name = models_dir / "bridgetower_large_itc.xml"
# save model to OpenVINO IR for later use
ov.save_model(ov_model, ov_model_name)

####################CONVERT BRIDGETOWER-FOR-TEXT MODEL########################
print("Converting BridgeTower model for text input...")
# prepare text input examples
text_encoding_inputs = processor.tokenizer(
            texts[0],
            return_tensors="pt",
            max_length=max_length,
            padding=True,
            truncation=True,
        )
if static_shapes:
    padding_len = max_length - text_encoding_inputs.input_ids.shape[-1]
    text_encoding_inputs["input_ids"] = F.pad(
        text_encoding_inputs.input_ids,
        (0, padding_len),
        value=model.config.pad_token_id,
    )
    text_encoding_inputs["attention_mask"] = F.pad(
        text_encoding_inputs.attention_mask, (0, padding_len), value=0
    )

ov_text_model = ov.convert_model(text_model, example_input={**text_encoding_inputs})
ov_text_model_name = models_dir / "custombridgetower_text_large_itc.xml"
# save model to OpenVINO IR for later use
ov.save_model(ov_text_model, ov_text_model_name)

####################CONVERT BRIDGETOWER-FOR-VISION MODEL########################
# prepare image input examples
# vision_encoding_inputs = processor.image_processor(
#     images[0], 
#     return_tensors="pt", 
#     do_resize=True, 
#     do_normalize=True, 
#     do_center_crop=True, 
#     do_pad=True,
# )

# ov_vision_model = ov.convert_model(vision_model, example_input={**vision_encoding_inputs})
# ov_vision_model_name = models_dir / "custombridgetower_vision_large_itc.xml"
# # save model to OpenVINO IR for later use
# ov.save_model(ov_vision_model, ov_vision_model_name)

# vision_encoding_inputs['pixel_values'] == image_text_encoding_inputs['pixel_values']

# TESTING
print("Testing the converted models...")
core = ov.Core()
model_name = "BridgeTower/bridgetower-large-itm-mlm-itc"
device = 'GPU' # BridgeTower runs on CPU by default, can be changed to 'AUTO' or other devices if needed.

model_path = Path("bridgetower_models") / "bridgetower_large_itc.xml"
text_model_path = Path("bridgetower_models") / "custombridgetower_text_large_itc.xml"
text_vision_model = core.compile_model(model=model_path, device_name=device)
text_model = core.compile_model(model=text_model_path, device_name=device)

model_cfg = BridgeTowerForITC.from_pretrained(model_name).config
processor = BridgeTowerProcessor.from_pretrained(model_name)

# prepare input example data
image_urls = ["http://images.cocodataset.org/val2017/000000039769.jpg"]
texts = ["two cats sleeping on a couch"]
images = [Image.open(requests.get(url, stream=True).raw) for url in image_urls]

# test image-text pair embedding computation
batch = processor(images, texts, return_tensors="pt",
                                   max_length=100,
                                   padding=True,
                                   truncation=True)
batch = dict(batch)
with torch.no_grad():
    outputs = text_vision_model(batch)
embeddings = [] # final embedding results
for k in range(len(texts)):
    embeddings.append(outputs[0][k, 2, :].tolist())

assert len(embeddings) == 1, "The number of embeddings should be equal to 1."
assert len(embeddings[0]) == model_cfg.contrastive_hidden_size, \
    f"The size of each embedding should be equal to {model_cfg.contrastive_hidden_size}."
print("Image-text pair embedding computed successfully.")
print(f"Size of each embedding: {len(embeddings[0])}")
# test text-only embedding computation
encoding = processor.tokenizer(
                texts,
                return_tensors="pt",
                max_length=100,
                padding=True,
                truncation=True,
            ) 
encoding = dict(encoding)
embeddings = []
with torch.no_grad():                
    outputs = text_model(encoding)
    embeddings += outputs[0].tolist()
assert len(embeddings) == 1, "The number of embeddings should be equal to 1."
assert len(embeddings[0]) == model_cfg.contrastive_hidden_size, \
    f"The size of each embedding should be equal to {model_cfg.contrastive_hidden_size}."
print("Text-only embedding computed successfully.")
print(f"Size of each embedding: {len(embeddings[0])}")