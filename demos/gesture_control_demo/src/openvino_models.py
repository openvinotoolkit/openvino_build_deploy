import cv2
import numpy as np
import openvino as ov
from pathlib import Path
import threading
from typing import Optional, Tuple, Dict  

class ModelManager:
    """Manages OpenVINO models for gesture detection"""
    
    def __init__(self):
        self.core = ov.Core()
        self.compiled_models = {}
        self._lock = threading.Lock()
        self._initialized = False
        self.device = "CPU"  
        self.device_config = {} 

    def set_device(self, device: str):
        """Set the GLOBAL FALLBACK inference device."""
        available_devices = self.core.available_devices
        if device not in available_devices and device not in ["AUTO", "CPU"]:
            print(f"❌ Device '{device}' not available. Available: {available_devices}")
            return False

        if self.device != device:
            print(f"🔄 Global fallback device changed from '{self.device}' to '{device}'.")
            self.device = device
            if self._initialized:
                self._initialized = False 
        return True

    def set_device_configuration(self, config: Dict[str, str]):
        """Sets the per-model device configuration."""
        self.device_config = config
        print(f"🔧 ModelManager received device config: {self.device_config}")
        
        if self._initialized:
            self._initialized = False
            print("   Models will be re-initialized on next run with new device config.")
    
    def get_available_devices_with_descriptions(self):
        """Get available devices with user-friendly descriptions"""
        devices = []
        
        for device in self.core.available_devices:
            try:
                full_name = self.core.get_property(device, "FULL_DEVICE_NAME")
                description = f"{device}: {full_name}"
            except Exception:
                description = device
            
            devices.append({
                'device': device,
                'description': description,
                'available': True
            })
        
        return devices



    def initialize_models(self, model_paths: dict) -> bool:
        """Initialize all required models using the specified device configuration."""
        with self._lock:
            if self._initialized:
                return True
            
            try:
                
                if 'palm_detection' in model_paths:
                    device = self.device_config.get('palm_detection', self.device)
                    self._load_palm_detection_model(model_paths['palm_detection'], device)
                
                if 'hand_landmarks' in model_paths:
                    device = self.device_config.get('hand_landmarks', self.device)
                    self._load_landmark_model(model_paths['hand_landmarks'], device)
                
                if 'gesture_embedder' in model_paths:
                    device = self.device_config.get('gesture_embedder', self.device)
                    self._load_gesture_embedder(model_paths['gesture_embedder'], device)
                
                if 'gesture_classifier' in model_paths:
                    device = self.device_config.get('gesture_classifier', self.device)
                    self._load_gesture_classifier(model_paths['gesture_classifier'], device)
                
                self._initialized = True
                print(f"✅ All models initialized successfully using configured devices.")
                return True
                
            except Exception as e:
                print(f"❌ Model initialization failed: {e}")
                import traceback
                traceback.print_exc()
                return False
    
    def _load_palm_detection_model(self, model_path: str, device: str):
        """Load and compile palm detection model on a specific device"""
        from openvino.preprocess import PrePostProcessor, ColorFormat
        from openvino import Type, Layout
        
        model = self.core.read_model(model_path)
        
        
        ppp_pd = PrePostProcessor(model)
        ppp_pd.input().tensor() \
            .set_element_type(Type.u8) \
            .set_layout(Layout('NHWC')) \
            .set_color_format(ColorFormat.BGR)
        
        ppp_pd.input().model().set_layout(Layout('NHWC'))
        ppp_pd.input().preprocess() \
            .convert_element_type(Type.f32) \
            .convert_color(ColorFormat.RGB) \
            .scale([255.0, 255.0, 255.0])
        
        palm_detection_model = ppp_pd.build()
        
        print(f"  Compiling palm_detection on: {device}")
        compiled_model = self.core.compile_model(palm_detection_model, device)
        
        self.compiled_models['palm_detection'] = compiled_model
    
    def _load_landmark_model(self, model_path: str, device: str):
        """Load landmark detection model on a specific device"""
        model = self.core.read_model(model_path)
        print(f"  Compiling hand_landmarks on: {device}")
        compiled_model = self.core.compile_model(model, device)
        
        self.compiled_models['hand_landmarks'] = compiled_model
    
    def _load_gesture_embedder(self, model_path: str, device: str):
        """Load gesture embedding model on a specific device"""
        model = self.core.read_model(model_path)
        print(f"  Compiling gesture_embedder on: {device}")
        compiled_model = self.core.compile_model(model, device)
        
        self.compiled_models['gesture_embedder'] = compiled_model
    
    def _load_gesture_classifier(self, model_path: str, device: str):
        """Load gesture classification model on a specific device"""
        model = self.core.read_model(model_path)
        print(f"  Compiling gesture_classifier on: {device}")
        compiled_model = self.core.compile_model(model, device)
        
        self.compiled_models['gesture_classifier'] = compiled_model
    
    def get_compiled_model(self, model_name: str):
        """Get compiled model by name"""
        with self._lock:
            return self.compiled_models.get(model_name)
    
    def is_initialized(self) -> bool:
        """Check if models are initialized"""
        return self._initialized
    
    def get_model_info(self) -> dict:
        """Get information about loaded models"""
        with self._lock:
            return {
                'initialized': self._initialized,
                'available_models': list(self.compiled_models.keys()),
                'model_count': len(self.compiled_models),
                'device': self.device,
                'available_devices': self.core.available_devices
            }


model_manager = ModelManager()