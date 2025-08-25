#!/usr/bin/env python3
"""
Script to download and convert Whisper model to OpenVINO format.
"""

import os
from pathlib import Path
from optimum.intel import OVModelForSpeechSeq2Seq
from transformers import AutoProcessor
import sys

def download_and_convert_whisper_model(model_name="openai/whisper-small", output_dir=None):
    """
    Download and convert Whisper model to OpenVINO format.
    
    Args:
        model_name: The Hugging Face model name (default: "openai/whisper-small")
        output_dir: Directory to save the converted model
    """
    if output_dir is None:
        # Create whisper-models directory in the project root
        project_root = Path(__file__).parent.parent
        output_dir = project_root / "whisper-models" / model_name.split("/")[-1]
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading and converting {model_name} to OpenVINO format...")
    print(f"Output directory: {output_dir}")
    
    try:
        # Download and convert the model
        print("Step 1: Downloading and converting model...")
        model = OVModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            export=True,
            device="CPU"
        )
        
        # Download the processor
        print("Step 2: Downloading processor...")
        processor = AutoProcessor.from_pretrained(model_name)
        
        # Save the converted model
        print("Step 3: Saving converted model...")
        model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)
        
        print(f"✅ Successfully converted and saved model to: {output_dir}")
        print(f"✅ Model files created:")
        for file in output_dir.iterdir():
            print(f"   - {file.name}")
        
        print(f"\n🔧 Update your .env file with:")
        print(f'OPENVINO_WHISPER_MODEL_DIR="{output_dir}"')
        
        return str(output_dir)
        
    except Exception as e:
        print(f"❌ Error converting model: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # You can change the model size here: whisper-tiny, whisper-small, whisper-medium, whisper-large
    model_name = "openai/whisper-small"
    
    if len(sys.argv) > 1:
        model_name = f"openai/whisper-{sys.argv[1]}"
    
    print(f"Converting {model_name}...")
    result = download_and_convert_whisper_model(model_name)
    
    if result:
        print(f"\n✅ Conversion complete! Use this path in your .env file:")
        print(f'OPENVINO_WHISPER_MODEL_DIR="{result}"')
    else:
        print("❌ Conversion failed. Check the error messages above.")
