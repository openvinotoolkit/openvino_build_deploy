from pathlib import Path
from huggingface_hub import hf_hub_download
import datasets
import numpy as np
import nncf
import os
import cv2
import torch
import torch.nn.functional as F
from depth_anything_v2.dpt import DepthAnythingV2
import openvino as ov
from depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose
from huggingface_hub import snapshot_download
import warnings
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="model", help="Directory to place the model in")
args = parser.parse_args()
repo_dir = Path(args.model_dir + "/depth_anything_v2")

if not repo_dir.exists():
    snapshot_download(repo_id="depth-anything/Depth-Anything-V2", local_dir=repo_dir)
os.chdir(repo_dir)

attention_file_path = Path("./depth_anything_v2/dinov2_layers/attention.py")
orig_attention_path = attention_file_path.parent / ("orig_" + attention_file_path.name)

if not orig_attention_path.exists():
    attention_file_path.rename(orig_attention_path)

    with orig_attention_path.open("r") as f:
        data = f.read()
        data = data.replace("XFORMERS_AVAILABLE = True", "XFORMERS_AVAILABLE = False")
        with attention_file_path.open("w") as out_f:
            out_f.write(data)


encoder = "vits"
model_type = "Small"
model_id = f"depth_anything_v2_{encoder}"

model_path = hf_hub_download(repo_id=f"depth-anything/Depth-Anything-V2-{model_type}", filename=f"{model_id}.pth", repo_type="model")

model = DepthAnythingV2(encoder=encoder, features=64, out_channels=[48, 96, 192, 384])
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()


OV_DEPTH_ANYTHING_PATH = Path(f"{model_id}.xml")

if not OV_DEPTH_ANYTHING_PATH.exists():
    ov_model = ov.convert_model(model, example_input=torch.rand(1, 3, 518, 518), input=[1, 3, 518, 518])
    ov.save_model(ov_model, OV_DEPTH_ANYTHING_PATH)


transform = Compose(
    [
        Resize(
            width=518,
            height=518,
            resize_target=False,
            ensure_multiple_of=14,
            resize_method="lower_bound",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ]
)

OV_DEPTH_ANYTHING_INT8_PATH = Path(f"{model_id}_int8.xml")
if not OV_DEPTH_ANYTHING_INT8_PATH.exists():
    subset_size = 300
    calibration_data = []
    dataset = datasets.load_dataset("Nahrawy/VIDIT-Depth-ControlNet", split="train", streaming=True).shuffle(seed=42).take(subset_size)
    for batch in dataset:
        image = np.array(batch["image"])[...,:3]
        image = image / 255.0
        image = transform({'image': image})['image']
        image = np.expand_dims(image, 0)
        calibration_data.append(image)

core = ov.Core()

if not OV_DEPTH_ANYTHING_INT8_PATH.exists():
    model = core.read_model(OV_DEPTH_ANYTHING_PATH)
    quantized_model = nncf.quantize(
        model=model,
        subset_size=subset_size,
        model_type=nncf.ModelType.TRANSFORMER,
        calibration_dataset=nncf.Dataset(calibration_data),
    )
    ov.save_model(quantized_model, OV_DEPTH_ANYTHING_INT8_PATH)

fp16_ir_model_size = OV_DEPTH_ANYTHING_PATH.with_suffix(".bin").stat().st_size / 2**20
quantized_model_size = OV_DEPTH_ANYTHING_INT8_PATH.with_suffix(".bin").stat().st_size / 2**20

print(f"FP16 model size: {fp16_ir_model_size:.2f} MB")
print(f"INT8 model size: {quantized_model_size:.2f} MB")
print(f"Model compression rate: {fp16_ir_model_size / quantized_model_size:.3f}")