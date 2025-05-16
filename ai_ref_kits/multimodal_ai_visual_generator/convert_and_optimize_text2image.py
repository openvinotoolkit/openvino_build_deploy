import argparse
import subprocess
import platform
from pathlib import Path
import json
import os

# Optional: Extend timeout to avoid HF download errors
os.environ["HF_HUB_TIMEOUT"] = "60"

# Model mapping
MODEL_MAPPING = {
    "flux.1-schnell": "black-forest-labs/FLUX.1-schnell",
    "flux.1-dev": "black-forest-labs/FLUX.1-dev",
    "tiny-sd": "segmind/tiny-sd",
    "stable-diffusionv3-large": "stabilityai/stable-diffusion-3.5-large",
    "stable-diffusionv3-medium": "stabilityai/stable-diffusion-3.5-medium",
    "stable-diffusion-2-1": "stabilityai/stable-diffusion-2-1",
}

CRITICAL_FILES = [
    "model_index.json",
    "scheduler/scheduler_config.json",
    "tokenizer/tokenizer_config.json",
    "text_encoder/openvino_model.xml",
    "text_encoder_2/openvino_model.xml",
    "vae_encoder/openvino_model.xml",
    "vae_decoder/openvino_model.xml",
    "transformer/openvino_model.xml",
]

def run_optimum_export(model_id: str, output_dir: Path, precision: str):
    cmd = [
        "optimum-cli", "export", "openvino",
        "--model", model_id,
        "--task", "text-to-image",
        "--trust-remote-code",
    ]

    if precision == "int4":
        cmd += ["--weight-format", "int4", "--group-size", "64", "--ratio", "1.0"]
    else:
        cmd += ["--weight-format", "fp16"]

    cmd.append(str(output_dir))

    print(f"\n🚀 Exporting with command:\n{' '.join(cmd)}\n")
    subprocess.run(cmd, shell=(platform.system() == "Windows"), check=True)

def convert_image_model(model_type: str, precision: str, model_dir: Path) -> Path:
    output_dir = model_dir / f"{model_type}-{precision.upper()}"
    model_id = MODEL_MAPPING[model_type]

    # Check if export already exists
    if output_dir.exists():
        missing = [f for f in CRITICAL_FILES if not (output_dir / f).exists()]
        if not missing:
            print(f"Model already exported at: {output_dir}")
            print("Skipping re-export.\n")
            return output_dir
        else:
            print(f"Export folder exists but missing files: {missing}")
            print("Re-exporting model...\n")

    run_optimum_export(model_id, output_dir, precision)

    print("Verifying exported files:")
    missing = []
    for file in CRITICAL_FILES:
        if not (output_dir / file).exists():
            print(f"Missing: {file}")
            missing.append(file)
        else:
            print(f"Found: {file}")

    if missing:
        print("Export completed with missing files.")
    else:
        print("All critical files verified successfully.")

    print(f"Model exported to: {output_dir}\n")
    return output_dir

if __name__ == "__main__":
    model_keys = list(MODEL_MAPPING.keys())

    parser = argparse.ArgumentParser(description="Export and optimize a text-to-image model using Optimum + OpenVINO")
    parser.add_argument(
        "--image_model_type",
        type=str,
        choices=model_keys,
        help="Model to convert. If not provided, a list will be shown for interactive selection."
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp16", "int4"],
        default="int4",
        help="Desired weight format (default: int4)"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models",
        help="Base model output directory (default: models)"
    )

    args = parser.parse_args()

    if not args.image_model_type:
        print("Available Image Models:")
        for i, key in enumerate(model_keys, start=1):
            print(f"  {i}. {key}")
        choice = input("Enter model number to export: ").strip()
        if not choice.isdigit() or not (1 <= int(choice) <= len(model_keys)):
            print("Invalid choice. Exiting.")
            exit(1)
        args.image_model_type = model_keys[int(choice) - 1]

    convert_image_model(args.image_model_type, args.precision, Path(args.model_dir))
    print("Conversion and optimization completed.")