Assuming you are at this folder by
```bash
cd openvino_bridgetower
```
then run the following command:
```bash
TOKENIZERS_PARALLELISM=false python openvino_bridgetower_conversion.py
```
Afterward, this will create another folder `bridgetower_models` here that enclosed the 4 following files:
- bridgetower_large_itc.bin
- bridgetower_large_itc.xml
- custombridgetower_text_large_itc.bin
- custombridgetower_text_large_itc.xml

The first two files are for OpenVINO-compatible BridgeTower model that can be used to compute the embedding of image-text pair; the last two files are for OpenVINO-compatible BridgeTower model that is used to compute the embedding of text only.

These converted BridgeTower models can be used as the following example:

```python
import openvino as ov
from transformers import BridgeTowerProcessor
from bridgetower_custom import BridgeTowerTextFeatureExtractor, BridgeTowerForITC
import requests
from PIL import Image
from pathlib import Path

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
```