from PySide6.QtCore import QObject, Signal, QThread, Qt, QMutex, QWaitCondition
from PySide6.QtWidgets import QApplication
import os
import sys
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, List, Tuple, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# OpenVINO GenAI imports
try:
    import openvino_genai as ov_genai
    import openvino as ov
    print("[VLM DEBUG] OpenVINO GenAI imported successfully")
    OPENVINO_AVAILABLE = True
except ImportError as e:
    print(f"[VLM DEBUG] Failed to import OpenVINO GenAI: {e}")
    OPENVINO_AVAILABLE = False

# PIL for image processing
try:
    from PIL import Image
    print("[VLM DEBUG] PIL imported successfully")
    PIL_AVAILABLE = True
except ImportError as e:
    print(f"[VLM DEBUG] Failed to import PIL: {e}")
    PIL_AVAILABLE = False


class VLMControllerThread(QThread):
    """Worker thread for VLM processing using OpenVINO GenAI."""
    result_ready = Signal(dict)
    error_occurred = Signal(str)
    progress_updated = Signal(int)

    def __init__(self, vlm_dir=None):
        print("[VLM DEBUG] >>> Entering VLMControllerThread.__init__")
        super().__init__()
        # Set VLM directory to the downloaded OpenVINO model
        if vlm_dir is None:
            current_dir = Path(__file__).parent.parent
            self.vlm_dir = current_dir / "llava_openvino_model"
        else:
            self.vlm_dir = Path(vlm_dir).resolve()
        print(f"[VLM DEBUG] vlm_dir resolved to: {self.vlm_dir}")
        self.mutex = QMutex()
        self.condition = QWaitCondition()
        self.abort = False
        self.image = None
        self.prompt = None
        self.vlm_pipeline = None
        self.device = "GPU"  # DEFAULT TO GPU FOR MAXIMUM PERFORMANCE
        print(f"[VLM DEBUG] VLMControllerThread initialized (OpenVINO GenAI)")
        print(f"[VLM DEBUG] VLM directory: {self.vlm_dir}")
        print(f"[VLM DEBUG] Directory exists: {self.vlm_dir.exists()}")
        print(f"[VLM DEBUG] 🚀 DEFAULT DEVICE: GPU (priority)")
        print("[VLM DEBUG] >>> Calling self._load_model()...")
        self._load_model()
        print("[VLM DEBUG] <<< Exiting VLMControllerThread.__init__")

    def _load_model(self):
        print("[VLM DEBUG] >>> Entering _load_model")
        try:
            print(f"[VLM DEBUG] Starting OpenVINO GenAI model loading...")
            # Check if OpenVINO GenAI is available
            if not OPENVINO_AVAILABLE:
                print(f"[VLM DEBUG] ❌ OpenVINO GenAI not available")
                return
            # Check if VLM directory exists
            if not self.vlm_dir.exists():
                print(f"[VLM DEBUG] ❌ VLM directory does not exist: {self.vlm_dir}")
                return
            # List files in VLM directory
            files_in_dir = list(self.vlm_dir.glob("*"))
            print(f"[VLM DEBUG] 📁 Files in VLM directory ({len(files_in_dir)}):")
            for file in sorted(files_in_dir):
                print(f"[VLM DEBUG]   - {file.name}")
            # Check for required OpenVINO files
            required_files = [
                "openvino_language_model.xml",
                "openvino_language_model.bin",
                "openvino_vision_embeddings_model.xml", 
                "openvino_vision_embeddings_model.bin",
                "openvino_text_embeddings_model.xml",
                "openvino_text_embeddings_model.bin"
            ]
            missing_files = []
            for file in required_files:
                if not (self.vlm_dir / file).exists():
                    missing_files.append(file)
            if missing_files:
                print(f"[VLM DEBUG] ⚠️ Missing files: {missing_files}")
            else:
                print(f"[VLM DEBUG] ✅ All required OpenVINO files found")
            # Detect available devices with GPU priority
            try:
                print("[VLM DEBUG] >>> Detecting OpenVINO devices...")
                core = ov.Core()
                available_devices = core.available_devices
                print(f"[VLM DEBUG] 🔍 Available OpenVINO devices: {available_devices}")
                gpu_available = "GPU" in available_devices
                print(f"[VLM DEBUG] GPU detected by OpenVINO: {gpu_available}")
                if not gpu_available:
                    print(f"[VLM DEBUG] ⚠️ GPU not detected by OpenVINO")
                if "GPU" in available_devices:
                    self.device = "GPU"
                    print(f"[VLM DEBUG] 🚀 PRIORITY: GPU selected for VLM inference")
                elif "CPU" in available_devices:
                    self.device = "CPU"
                    print(f"[VLM DEBUG] 🔧 FALLBACK: CPU selected (GPU not available)")
                else:
                    self.device = "AUTO"
                    print(f"[VLM DEBUG] 🤖 AUTO: Letting OpenVINO choose device")
            except Exception as e:
                print(f"[VLM DEBUG] ⚠️ Device detection failed: {e}")
                print(f"[VLM DEBUG] 🔄 Defaulting to GPU (will fallback to CPU if needed)")
                self.device = "GPU"
            # Load the VLM pipeline with GPU priority
            try:
                print(f"[VLM DEBUG] 🚀 Loading VLMPipeline from: {self.vlm_dir}")
                print(f"[VLM DEBUG] 🎯 Target device: {self.device}")
                self.vlm_pipeline = ov_genai.VLMPipeline(str(self.vlm_dir), self.device)
                print(f"[VLM DEBUG] ✅ VLMPipeline loaded successfully on {self.device}!")
            except Exception as e:
                print(f"[VLM DEBUG] ❌ Failed to load VLMPipeline: {e}")
                self.vlm_pipeline = None
        except Exception as e:
            print(f"[VLM DEBUG] ❌ Error in _load_model: {e}")
            self.vlm_pipeline = None
        print("[VLM DEBUG] <<< Exiting _load_model")

    def process_request(self, image, prompt):
        """Process a VLM request."""
        self.mutex.lock()
        try:
            self.image = image
            self.prompt = prompt
            self.condition.wakeOne()
        finally:
            self.mutex.unlock()

    def run(self):
        """Main thread loop."""
        while not self.abort:
            self.mutex.lock()
            try:
                if self.image is None or self.prompt is None:
                    self.condition.wait(self.mutex, 100)  # Wait for 100ms
                    continue
                
                # Process the request
                image = self.image
                prompt = self.prompt
                self.image = None
                self.prompt = None
                
            finally:
                self.mutex.unlock()
            
            # Process outside the lock
            result = self._process_request(image, prompt)
            self.result_ready.emit(result)

    def _process_request(self, image: np.ndarray, prompt: str) -> dict:
        """Process a single VLM request using OpenVINO GenAI."""
        try:
            if self.vlm_pipeline is None:
                return {
                    "status": "error",
                    "message": "VLM pipeline not loaded",
                    "response": "❌ VLM pipeline failed to load. Check logs for OpenVINO GenAI setup.",
                    "confidence": 0.0,
                    "timestamp": datetime.now().isoformat(),
                    "device": "none",
                    "processing_time": 0.0
                }
            
            return self._run_genai_inference(image, prompt)
                
        except Exception as e:
            print(f"[VLM DEBUG] Error in _process_request: {e}")
            return {
                "status": "error",
                "message": str(e),
                "response": f"❌ VLM processing error: {str(e)}",
                "confidence": 0.0,
                "timestamp": datetime.now().isoformat(),
                "device": getattr(self, 'device', 'unknown'),
                "processing_time": 0.0
            }

    def _run_genai_inference(self, image: np.ndarray, prompt: str) -> dict:
        """Run inference using OpenVINO GenAI VLMPipeline."""
        start_time = datetime.now()
        
        try:
            print(f"[VLM DEBUG] 🚀 Starting OpenVINO GenAI inference...")
            print(f"[VLM DEBUG] 📝 Prompt: {prompt}")
            print(f"[VLM DEBUG] 🖼️ Image shape: {image.shape}")
            print(f"[VLM DEBUG] 🎯 Device: {self.device}")
            
            # Convert numpy image to PIL Image
            if not PIL_AVAILABLE:
                return {
                    "status": "error",
                    "message": "PIL not available",
                    "response": "❌ PIL required for image processing",
                    "confidence": 0.0,
                    "timestamp": start_time.isoformat(),
                    "device": self.device,
                    "processing_time": 0.0
                }
            
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                if image.dtype == np.uint8:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image_rgb = image
            else:
                image_rgb = image
            
            # Convert to PIL Image
            pil_image = Image.fromarray(image_rgb.astype(np.uint8))
            print(f"[VLM DEBUG] 🖼️ PIL Image size: {pil_image.size}")
            
            # Convert PIL image to OpenVINO tensor
            image_array = np.array(pil_image)
            # Ensure NCHW format for OpenVINO
            if len(image_array.shape) == 3:
                image_array = np.transpose(image_array, (2, 0, 1))  # HWC to CHW
            image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
            
            image_tensor = ov.Tensor(image_array)
            print(f"[VLM DEBUG] 🔢 Image tensor shape: {image_tensor.shape}")
            
            # Start chat session
            print(f"[VLM DEBUG] 💬 Starting chat session...")
            self.vlm_pipeline.start_chat()
            
            # Generate response
            print(f"[VLM DEBUG] 🎲 Generating response...")
            response = self.vlm_pipeline.generate(
                prompt, 
                image=image_tensor,
                max_new_tokens=100,
                do_sample=False
            )
            
            # Finish chat session
            self.vlm_pipeline.finish_chat()
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            print(f"[VLM DEBUG] ✅ Generation complete!")
            print(f"[VLM DEBUG] 📝 Raw response type: {type(response)}")
            print(f"[VLM DEBUG] 📝 Raw response: {response}")
            
            # Extract text from VLMDecodedResults object
            response_text = ""
            try:
                if hasattr(response, 'texts'):
                    if isinstance(response.texts, list) and len(response.texts) > 0:
                        response_text = response.texts[0]
                    else:
                        response_text = str(response.texts)
                elif hasattr(response, 'text'):
                    response_text = response.text
                elif hasattr(response, '__str__'):
                    response_text = str(response)
                else:
                    response_text = f"Unable to extract text from response: {type(response)}"
                
                print(f"[VLM DEBUG] 📝 Extracted text: {response_text}")
                
            except Exception as text_extract_error:
                print(f"[VLM DEBUG] ❌ Error extracting text: {text_extract_error}")
                response_text = f"Text extraction failed: {str(text_extract_error)}"
            
            print(f"[VLM DEBUG] ⏱️ Processing time: {processing_time:.2f}s")
            print(f"[VLM DEBUG] 🎯 Used device: {self.device}")
            
            return {
                "status": "success",
                "message": "OpenVINO GenAI inference completed",
                "response": response_text,  # Return extracted text instead of raw object
                "raw_response": response,   # Keep raw response for debugging
                "confidence": 1.0,  # GenAI doesn't provide confidence scores
                "timestamp": start_time.isoformat(),
                "device": self.device,
                "processing_time": processing_time
            }
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"OpenVINO GenAI inference failed: {str(e)}"
            print(f"[VLM DEBUG] ❌ {error_msg}")
            
            # Try to finish chat session in case of error
            try:
                self.vlm_pipeline.finish_chat()
            except:
                pass
            
            return {
                "status": "error",
                "message": error_msg,
                "response": f"❌ VLM inference error: {str(e)}",
                "confidence": 0.0,
                "timestamp": start_time.isoformat(),
                "device": self.device,
                "processing_time": processing_time
            }

    def stop(self):
        """Stop the thread."""
        self.mutex.lock()
        self.abort = True
        self.condition.wakeOne()
        self.mutex.unlock()


class VLMController(QObject):
    """Main VLM controller class using OpenVINO GenAI."""
    result_ready = Signal(dict)
    error_occurred = Signal(str)

    def __init__(self, vlm_dir=None):
        super().__init__()
        print(f"[VLM DEBUG] Initializing VLM Controller (OpenVINO GenAI)")
        
        # Set VLM directory to the downloaded OpenVINO model
        if vlm_dir is None:
            current_dir = Path(__file__).parent.parent
            vlm_dir = current_dir / "llava_openvino_model"
            
        print(f"[VLM DEBUG] VLM directory: {vlm_dir}")
        print(f"[VLM DEBUG] VLM directory exists: {vlm_dir.exists()}")
        
        # Store comprehensive data for analysis
        self.data_context = {
            'detection_data': None,
            'frame_analysis': None,
            'scene_context': None,
            'traffic_state': None
        }
        
        # Create worker thread
        self.worker_thread = VLMControllerThread(vlm_dir)
        self.worker_thread.result_ready.connect(self.result_ready.emit)
        self.worker_thread.error_occurred.connect(self.error_occurred.emit)
        self.worker_thread.start()
        
        print(f"[VLM DEBUG] VLM Controller initialized successfully (OpenVINO GenAI)")

    def process_image(self, image: np.ndarray, prompt: str):
        """Process an image with VLM."""
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.process_request(image, prompt)
        else:
            self.error_occurred.emit("VLM worker thread not running")

    def stop(self):
        """Stop the VLM controller."""
        if self.worker_thread:
            self.worker_thread.stop()
            self.worker_thread.wait(5000)  # Wait up to 5 seconds for thread to finish
