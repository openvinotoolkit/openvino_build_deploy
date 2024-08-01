import openvino as ov
from huggingface_hub import hf_hub_download
from depth_anything_v2.dpt import DepthAnythingV2
import cv2
import torch
import torch.nn.functional as F
import datasets
import nncf

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="model", help="Directory to place the model in")
args = parser.parse_args()
model_local_dir = Path(args.model_dir + "/depth_anything_v2")

encoder = "vits"
model_type = "Small"
model_id = f"depth_anything_v2_{encoder}"

model_path = hf_hub_download(repo_id=f"{model_local_dir}/Depth-Anything-V2-{model_type}", filename=f"{model_id}.pth", repo_type="model")

model = DepthAnythingV2(encoder=encoder, features=64, out_channels=[48, 96, 192, 384])
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

OV_DEPTH_ANYTHING_PATH = Path(f"{model_local_dir}/{model_id}.xml")

if not OV_DEPTH_ANYTHING_PATH.exists():
    ov_model = ov.convert_model(model, example_input=torch.rand(1, 3, 518, 518), input=[1, 3, 518, 518])
    ov.save_model(ov_model, OV_DEPTH_ANYTHING_PATH)

# Fetch `skip_kernel_extension` module
r = requests.get(
    url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/skip_kernel_extension.py",
)
open("skip_kernel_extension.py", "w").write(r.text)

OV_DEPTH_ANYTHING_INT8_PATH = Path(f"{model_local_dir}/{model_id}_int8.xml")

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