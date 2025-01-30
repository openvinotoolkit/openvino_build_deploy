from pathlib import Path
from huggingface_hub import hf_hub_download
import datasets
import numpy as np
import nncf
import os
import cv2
import torch
import torch.nn.functional as F
import openvino as ov
import subprocess
import sys

from torchvision.transforms import Compose
import warnings
import argparse

def clone_repo(repo_url: str, revision: str = None, add_to_sys_path: bool = True) -> Path:
    repo_path = Path(repo_url.split("/")[-1].replace(".git", ""))

    if not repo_path.exists():
        try:
            subprocess.run(["git", "clone", repo_url], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except Exception as exc:
            print(f"Failed to clone the repository: {exc.stderr}")
            raise

        if revision:
            subprocess.Popen(["git", "checkout", revision], cwd=str(repo_path))
    if add_to_sys_path and str(repo_path.resolve()) not in sys.path:
        sys.path.insert(0, str(repo_path.resolve()))

    return repo_path


parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="model", help="Directory to place the model in")
args = parser.parse_args()
repo_dir = Path(args.model_dir + "/Depth-Anything-V2")

if not repo_dir.exists():
    clone_repo("https://huggingface.co/spaces/depth-anything/Depth-Anything-V2")
sys.path.insert(0, str(repo_dir.resolve()))
  
print("adding this to the path: ",   repo_dir)
#sys.path.append(Path(repo_dir))
from depth_anything_v2.dpt import DepthAnythingV2

attention_file_path = Path("./Depth-Anything-V2/depth_anything_v2/dinov2_layers/attention.py")
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
    ov_model = ov.convert_model(model, example_input=torch.rand(1, 3, 434, 770), input=[1, 3, 434, 770])
    ov.save_model(ov_model, OV_DEPTH_ANYTHING_PATH)
