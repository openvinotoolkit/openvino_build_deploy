from pathlib import Path
from llama_index.core import VectorStoreIndex, Settings
import requests
from llama_index.core import SimpleDirectoryReader
import io
from dotenv import load_dotenv
from llama_index.llms.openvino import OpenVINOLLM
import openvino.properties.hint as hints
import openvino.properties.streams as streams
import openvino.properties as props
import os
import sys

def load_documents(text_example_en_path: Path) -> VectorStoreIndex:
    """
    Loads documents from the given path

    Args:
        text_example_en_path: Path to the document to load

    Returns:
        VectorStoreIndex for the loaded documents
    """    
    if not text_example_en_path.exists():
        text_example_en = "test_painting_llm_rag.pdf"
        r = requests.get(text_example_en)
        content = io.BytesIO(r.content)
        with open(text_example_en_path, "wb") as f:
            f.write(content.read())

    reader = SimpleDirectoryReader(input_files=[text_example_en_path])
    documents = reader.load_data()
    index = VectorStoreIndex.from_documents(documents)

    return index

def setup_models():
    """
    Setup models for the mcp clients.
    """
    load_dotenv()
    llm_model_path = os.getenv("LLM_MODEL_PATH", None)
    
    if not llm_model_path:
        print("LLM_MODEL_PATH environment variable is not set. Please set it to the path of the LLM model.")
        sys.exit(1)
    # Check if model paths exist
    if not Path(llm_model_path).exists():
        print(f"LLM model not found at {llm_model_path}. Please run convert_and_optimize_llm.py to download the model first.")
        sys.exit(1)
    
    llm_model_device = os.getenv("LLM_MODEL_DEVICE", "CPU")

    ov_config = {
        hints.performance_mode(): hints.PerformanceMode.LATENCY,
        streams.num(): "1",
        props.cache_dir(): ""
    }
        
    # Load LLM model locally
    print(f"[LLM MODEL] Initializing LLM model...")
    print(f"[LLM MODEL] Model path: {llm_model_path}")
    print(f"[LLM MODEL] Target device: {llm_model_device}")    
    llm = OpenVINOLLM(
        model_id_or_path=str(llm_model_path),
        context_window=8192,
        max_new_tokens=1000,
        model_kwargs={"ov_config": ov_config},
        generate_kwargs={"do_sample": False, "temperature": 0.1, "top_p": 0.8},        
        device_map=llm_model_device,
    )
    print(f"[LLM MODEL] LLM model loaded successfully on {llm_model_device} device")

    return llm