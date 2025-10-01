import os
import sys
import logging
import time
import argparse
import json
from pathlib import Path
import subprocess
import shutil
import requests
import zipfile
import io
import tempfile
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s.%(msecs)03d [%(name)s]: %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('model_downloader')

def download_file(url, output_path, chunk_size=8192):
    """
    Download a file from URL with progress bar.
    
    Args:
        url: URL to download from
        output_path: Path to save the file
        chunk_size: Size of chunks to download
        
    Returns:
        Path to the downloaded file
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get file size for progress bar
        total_size = int(response.headers.get('content-length', 0))
        
        # Show download progress
        desc = f"Downloading {os.path.basename(url)}"
        with open(output_path, 'wb') as f, tqdm(
            desc=desc,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    bar.update(len(chunk))
        
        return output_path
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        if os.path.exists(output_path):
            os.remove(output_path)
        raise e

def download_openvino_model(repo_id="ezelanza/llava-next-video-openvino", output_dir="./models"):
    """
    Download pre-converted OpenVINO model from Hugging Face.
    
    Args:
        repo_id: Hugging Face repository ID
        output_dir: Directory to save the downloaded model
        
    Returns:
        Path to the downloaded model directory
    """
    # Create a model-specific output directory
    model_id = repo_id.split('/')[-1].replace("-openvino", "")
    output_path = Path(output_dir) / f"{model_id}_openvino_model"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Check if model already exists (look for XML and BIN files)
    if list(output_path.glob("*.xml")) and list(output_path.glob("*.bin")):
        logger.info(f"OpenVINO model already exists at {output_path}")
        return output_path
    
    try:
        # Define files to download
        files_to_download = [
            "openvino_model.xml",
            "openvino_model.bin",
            "config.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "tokenizer.model"
        ]
        
        # Base URL for raw files on Hugging Face
        base_url = f"https://huggingface.co/{repo_id}/resolve/main"
        
        # Download each file
        for filename in files_to_download:
            file_url = f"{base_url}/{filename}"
            output_file = output_path / filename
            
            try:
                download_file(file_url, output_file)
                logger.info(f"Successfully downloaded {filename}")
            except Exception as e:
                logger.warning(f"Could not download {filename}: {str(e)}")
                # Continue with other files even if one fails
                continue
        
        # Create a model info file
        model_info = {
            "model_name": repo_id,
            "download_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "files": [f.name for f in output_path.glob("*") if f.is_file()]
        }
        
        with open(output_path / "model_info.json", "w") as f:
            json.dump(model_info, f, indent=2)
        
        logger.info(f"Model files downloaded to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        raise e

def download_vl_model(model_name="ezelanza/llava-next-video-openvino", output_dir="./models", precision=None):
    """
    Download a Vision-Language model.
    
    Args:
        model_name: Name of the model identifier on Hugging Face
        output_dir: Directory to save the model
        precision: Not used in this implementation but kept for API compatibility
    
    Returns:
        Path to the downloaded model
    """
    start_time = time.time()
    logger.info(f"Starting download of {model_name}...")
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Download the OpenVINO model
    model_path = download_openvino_model(model_name, output_dir)
    
    total_time = time.time() - start_time
    logger.info(f"Completed in {total_time:.2f} seconds. Model saved at {model_path}")
    
    return model_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download pre-converted Vision-Language models in OpenVINO format")
    parser.add_argument("--model_name", type=str, default="ezelanza/llava-next-video-openvino", 
                        help="Name of the model repository on Hugging Face Hub")
    parser.add_argument("--output_dir", type=str, default="./models", 
                        help="Directory to save the OpenVINO model")
    
    args = parser.parse_args()
    
    try:
        model_path = download_vl_model(args.model_name, args.output_dir)
        print(f"\nModel successfully downloaded to: {model_path}")
        print(f"Files downloaded:")
        for file in sorted(os.listdir(model_path)):
            file_path = os.path.join(model_path, file)
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"  - {file} ({size_mb:.2f} MB)")
    except Exception as e:
        print(f"\nError downloading model: {e}")
        sys.exit(1)
