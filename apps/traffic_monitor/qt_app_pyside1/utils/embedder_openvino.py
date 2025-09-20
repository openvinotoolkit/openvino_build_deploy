"""
OpenVINO-based embedder for DeepSORT tracking.
"""

import os
import numpy as np
from pathlib import Path
import cv2
import time
from typing import List, Optional, Union

try:
    import openvino as ov
except ImportError:
    print("Installing openvino...")
    os.system('pip install --quiet "openvino>=2024.0.0"')
    import openvino as ov

class OpenVINOEmbedder:
    """
    OpenVINO embedder for DeepSORT tracking.
    
    This class provides an optimized version of the feature embedder used in DeepSORT,
    using OpenVINO for inference acceleration.
    """
    def __init__(
        self, 
        model_path: Optional[str] = None, 
        device: str = "AUTO",
        input_size: tuple = (128, 64),
        batch_size: int = 16,
        bgr: bool = True,
        half: bool = True
    ):
        """
        Initialize the OpenVINO embedder.
        
        Args:
            model_path: Path to the model file. If None, will use the default MobileNetV2 model.
            device: Device to run inference on ('CPU', 'GPU', 'AUTO', etc.)
            input_size: Input size for the model (height, width)
            batch_size: Batch size for inference
            bgr: Whether input images are BGR (True) or RGB (False)
            half: Whether to use half precision (FP16)
        """
        self.device = device
        self.input_size = input_size  # (h, w)
        self.batch_size = batch_size
        self.bgr = bgr
        self.half = half
        
        # Initialize OpenVINO Core
        self.core = ov.Core()
        
        # Find and load model
        if model_path is None:
            # Use MobileNetV2 converted to OpenVINO
            model_path = self._find_mobilenet_model()
            
            # If model not found, convert it
            if model_path is None:
                print("⚠️ MobileNetV2 OpenVINO model not found. Creating it...")
                model_path = self._convert_mobilenet()
        else:
            # When model_path is explicitly provided, verify it exists
            if not os.path.exists(model_path):
                print(f"⚠️ Specified model path does not exist: {model_path}")
                print("Falling back to default model search...")
                model_path = self._find_mobilenet_model()
                if model_path is None:
                    print("⚠️ Default model search also failed. Creating new model...")
                    model_path = self._convert_mobilenet()
            else:
                print(f"✅ Using explicitly provided model: {model_path}")
        
        print(f"📦 Loading embedder model: {model_path} on {device}")
        
        # Load and compile the model
        self.model = self.core.read_model(model_path)
        
        # Set up configuration for device
        ov_config = {}
        if device != "CPU":
            self.model.reshape({0: [self.batch_size, 3, self.input_size[0], self.input_size[1]]})
        if "GPU" in device or ("AUTO" in device and "GPU" in self.core.available_devices):
            ov_config = {"GPU_DISABLE_WINOGRAD_CONVOLUTION": "YES"}
            
        # Compile model for the specified device
        self.compiled_model = self.core.compile_model(model=self.model, device_name=self.device, config=ov_config)
        
        # Get input and output tensors
        self.input_layer = self.compiled_model.inputs[0]
        self.output_layer = self.compiled_model.outputs[0]
        
        # Create inference requests for async inference
        self.infer_requests = [self.compiled_model.create_infer_request() for _ in range(2)]
        self.current_request_idx = 0
        
        # Performance stats
        self.total_inference_time = 0
        self.inference_count = 0
        
    def _find_mobilenet_model(self) -> Optional[str]:
        """
        Find MobileNetV2 model converted to OpenVINO format.
        
        Returns:
            Path to the model file or None if not found
        """
        search_paths = [
            # Standard locations
            "mobilenetv2_embedder/mobilenetv2.xml",
            "../mobilenetv2_embedder/mobilenetv2.xml",
            "../../mobilenetv2_embedder/mobilenetv2.xml",
            # Look in models directory
            "../models/mobilenetv2.xml",
            "../../models/mobilenetv2.xml",
            # Look relative to DeepSORT location
            os.path.join(os.path.dirname(__file__), "models/mobilenetv2.xml"),
            # Look in openvino_models
            "../openvino_models/mobilenetv2.xml",
            "../../openvino_models/mobilenetv2.xml"
        ]
        
        for path in search_paths:
            if os.path.exists(path):
                return path
                
        return None
        
    def _convert_mobilenet(self) -> str:
        """
        Convert MobileNetV2 model to OpenVINO IR format.
        
        Returns:
            Path to the converted model
        """
        try:
            # Create directory for the model
            output_dir = Path("mobilenetv2_embedder")
            output_dir.mkdir(exist_ok=True)
            
            # First, we need to download the PyTorch model
            import torch
            import torch.nn as nn
            from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
            
            print("⬇️ Downloading MobileNetV2 model...")
            model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
            
            # Modify for feature extraction (remove classifier)
            class FeatureExtractor(nn.Module):
                def __init__(self, model):
                    super(FeatureExtractor, self).__init__()
                    self.features = nn.Sequential(*list(model.children())[:-1])
                    
                def forward(self, x):
                    return self.features(x).squeeze()
            
            feature_model = FeatureExtractor(model)
            feature_model.eval()
            
            # Save to ONNX
            onnx_path = output_dir / "mobilenetv2.onnx"
            print(f"💾 Converting to ONNX: {onnx_path}")
            dummy_input = torch.randn(1, 3, self.input_size[0], self.input_size[1])
            
            torch.onnx.export(
                feature_model,
                dummy_input,
                onnx_path,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
                opset_version=11
            )
            
            # Convert ONNX to OpenVINO IR
            ir_path = output_dir / "mobilenetv2.xml"
            print(f"💾 Converting to OpenVINO IR: {ir_path}")
            
            # Use the proper OpenVINO API to convert the model
            try:
                from openvino.tools.mo import convert_model
                
                print(f"Converting ONNX model using OpenVINO convert_model API...")
                print(f"Input model: {onnx_path}")
                print(f"Output directory: {output_dir}")
                print(f"Input shape: [{self.batch_size},3,{self.input_size[0]},{self.input_size[1]}]")
                print(f"Data type: {'FP16' if self.half else 'FP32'}")
                
                # Convert using the proper API
                convert_model(
                    model_path=str(onnx_path),
                    output_dir=str(output_dir),
                    input_shape=[self.batch_size, 3, self.input_size[0], self.input_size[1]],
                    data_type="FP16" if self.half else "FP32"
                )
                
                print(f"✅ Model successfully converted using OpenVINO convert_model API")
            except Exception as e:
                print(f"Error with convert_model: {e}, trying alternative approach...")
                
                # Fallback to subprocess with explicit path if needed
                import subprocess
                import sys
                import os
                
                # Try to find mo.py in the OpenVINO installation
                mo_paths = [
                    os.path.join(os.environ.get("INTEL_OPENVINO_DIR", ""), "tools", "mo", "mo.py"),
                    os.path.join(os.path.dirname(os.path.dirname(os.__file__)), "openvino", "tools", "mo", "mo.py"),
                    "C:/Program Files (x86)/Intel/openvino_2021/tools/mo/mo.py",
                    "C:/Program Files (x86)/Intel/openvino/tools/mo/mo.py"
                ]
                
                mo_script = None
                for path in mo_paths:
                    if os.path.exists(path):
                        mo_script = path
                        break
                
                if not mo_script:
                    raise FileNotFoundError("Cannot find OpenVINO Model Optimizer (mo.py)")
                
                cmd = [
                    sys.executable,
                    mo_script,
                    "--input_model", str(onnx_path),
                    "--output_dir", str(output_dir),
                    "--input_shape", f"[{self.batch_size},3,{self.input_size[0]},{self.input_size[1]}]",
                    "--data_type", "FP16" if self.half else "FP32"
                ]
                
                print(f"Running Model Optimizer: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    print(f"Error running Model Optimizer: {result.stderr}")
                    raise RuntimeError(f"Model Optimizer failed: {result.stderr}")
            
            print(f"✅ Model converted: {ir_path}")
            return str(ir_path)
            
        except Exception as e:
            print(f"❌ Error converting model: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    def preprocess(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Preprocess image crops for model input.
        
        Args:
            crops: List of image crops
            
        Returns:
            Preprocessed batch tensor
        """
        processed = []
        for crop in crops:
            # Resize to expected input size
            crop = cv2.resize(crop, (self.input_size[1], self.input_size[0]))
            
            # Convert BGR to RGB if needed
            if not self.bgr and crop.shape[2] == 3:
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                
            # Normalize (0-255 to 0-1)
            crop = crop.astype(np.float32) / 255.0
            
            # Change to NCHW format
            crop = crop.transpose(2, 0, 1)
            processed.append(crop)
            
        # Stack into batch
        batch = np.stack(processed)
        return batch
        
    def __call__(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Get embeddings for the image crops.
        
        Args:
            crops: List of image crops
            
        Returns:
            Embeddings for each crop
        """
        if not crops:
            return np.array([])
            
        # Preprocess crops
        batch = self.preprocess(crops)
        
        # Run inference
        start_time = time.time()
        
        # Use async inference to improve performance
        request = self.infer_requests[self.current_request_idx]
        self.current_request_idx = (self.current_request_idx + 1) % len(self.infer_requests)
        
        request.start_async({self.input_layer.any_name: batch})
        request.wait()
        
        # Get output
        embeddings = request.get_output_tensor().data
        
        # Track inference time
        inference_time = time.time() - start_time
        self.total_inference_time += inference_time
        self.inference_count += 1
        
        # Normalize embeddings
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        return embeddings
