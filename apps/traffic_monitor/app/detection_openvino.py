# # Detection logic using OpenVINO models (YOLO, etc.)

# import os
# import sys
# import time
# import cv2
# import numpy as np
# from pathlib import Path
# from typing import List, Dict, Tuple, Optional

# # --- Install required packages if missing ---
# try:
#     import openvino as ov
# except ImportError:
#     print("Installing openvino...")
#     os.system('pip install --quiet "openvino>=2024.0.0"')
#     import openvino as ov
# try:
#     from ultralytics import YOLO
# except ImportError:
#     print("Installing ultralytics...")
#     os.system('pip install --quiet "ultralytics==8.3.0"')
#     from ultralytics import YOLO
# try:
#     import nncf
# except ImportError:
#     print("Installing nncf...")
#     os.system('pip install --quiet "nncf>=2.9.0"')
#     import nncf

# # --- COCO dataset class names ---
# COCO_CLASSES = {
#     0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 
#     6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 
#     11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 
#     16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 
#     22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 
#     27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 
#     32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 
#     36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 
#     40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 
#     46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 
#     51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 
#     57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 
#     62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 
#     67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 
#     72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 
#     77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
# }

# # Traffic-related classes we're interested in (using standard COCO indices)
# TRAFFIC_CLASS_NAMES = [
#     'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck',
#     'traffic light', 'stop sign', 'parking meter'
# ]

# # --- Model Conversion and Quantization ---
# def convert_yolo_to_openvino(model_name: str = "yolo11x", half: bool = True) -> Path:
#     """Convert YOLOv11x PyTorch model to OpenVINO IR format."""
#     pt_path = Path(f"{model_name}.pt")
#     ov_dir = Path(f"{model_name}_openvino_model")
#     ov_xml = ov_dir / f"{model_name}.xml"
#     if not ov_xml.exists():
#         print(f"Exporting {pt_path} to OpenVINO IR...")
#         model = YOLO(str(pt_path))
#         model.export(format="openvino", dynamic=True, half=half)
#     else:
#         print(f"OpenVINO IR already exists: {ov_xml}")
#     return ov_xml

# def quantize_openvino_model(ov_xml: Path, model_name: str = "yolo11x") -> Path:
#     """Quantize OpenVINO IR model to INT8 using NNCF."""
#     int8_dir = Path(f"{model_name}_openvino_int8_model")
#     int8_xml = int8_dir / f"{model_name}.xml"
#     if int8_xml.exists():
#         print(f"INT8 model already exists: {int8_xml}")
#         return int8_xml
#     print("Quantization requires a calibration dataset. Skipping actual quantization in this demo.")
#     return ov_xml  # Return FP32 if no quantization

# class OpenVINOVehicleDetector:
#     def __init__(self, model_path: str = None, device: str = "AUTO", use_quantized: bool = False, enable_ocr: bool = False, confidence_threshold: float = 0.4):
#         import openvino as ov
#         self.device = device
#         self.confidence_threshold = confidence_threshold
#         self.ocr_reader = None
#         self.class_names = TRAFFIC_CLASS_NAMES
#         self.performance_stats = {
#             'fps': 0,
#             'avg_inference_time': 0,
#             'frames_processed': 0,
#             'backend': f"OpenVINO-{device}",
#             'total_detections': 0,
#             'detection_rate': 0
#         }
#         self._inference_times = []
#         self._start_time = time.time()
#         self._frame_count = 0
        
#         # Model selection logic
#         self.model_path = self._find_best_model(model_path, use_quantized)
#         print(f"🎯 OpenVINOVehicleDetector: Using model: {self.model_path}")
        
#         self.core = ov.Core()
#         self.model = self.core.read_model(self.model_path)
#         # Always reshape to static shape before accessing .shape
#         self.model.reshape({0: [1, 3, 640, 640]})
#         self.input_shape = self.model.inputs[0].shape
#         self.input_height = self.input_shape[2]
#         self.input_width = self.input_shape[3]
#         self.ov_config = {}
#         if device != "CPU":
#             # Already reshaped above, so nothing more needed here
#             pass
#         if "GPU" in device or ("AUTO" in device and "GPU" in self.core.available_devices):
#             self.ov_config = {"GPU_DISABLE_WINOGRAD_CONVOLUTION": "YES"}
#         self.compiled_model = self.core.compile_model(model=self.model, device_name=self.device, config=self.ov_config)

#         self.output_layer = self.compiled_model.output(0)

#     def _find_best_model(self, model_path, use_quantized):
#         # If a specific model path is provided, use it directly
#         if model_path and Path(model_path).exists():
#             print(f"🎯 Using provided model path: {model_path}")
#             return str(model_path)
        
#         # If no model path provided, extract model name from path or default to yolo11x
#         model_name = "yolo11x"  # Default fallback
#         if model_path:
#             # Try to extract model name from path
#             path_obj = Path(model_path)
#             if "yolo11n" in str(path_obj).lower():
#                 model_name = "yolo11n"
#             elif "yolo11s" in str(path_obj).lower():
#                 model_name = "yolo11s"
#             elif "yolo11m" in str(path_obj).lower():
#                 model_name = "yolo11m"
#             elif "yolo11l" in str(path_obj).lower():
#                 model_name = "yolo11l"
#             elif "yolo11x" in str(path_obj).lower():
#                 model_name = "yolo11x"
        
#         print(f"🔍 Searching for {model_name} model files...")
        
#         # Priority: quantized IR > IR > .pt
#         search_paths = [
#             Path(f"{model_name}_openvino_int8_model/{model_name}.xml") if use_quantized else None,
#             Path(f"{model_name}_openvino_model/{model_name}.xml"),
#             Path(f"rcb/{model_name}_openvino_model/{model_name}.xml"),
#             Path(f"{model_name}.xml"),
#             Path(f"rcb/{model_name}.xml"),
#             Path(f"{model_name}.pt"),
#             Path(f"rcb/{model_name}.pt")
#         ]
        
#         for p in search_paths:
#             if p and p.exists():
#                 print(f"✅ Found model: {p}")
#                 return str(p)
        
#         # Fallback to any yolo11x if specific model not found
#         fallback_paths = [
#             Path("openvino_models/yolo11x_openvino_model/yolo11x.xml"),
#         ]
        
#         for p in fallback_paths:
#             if p and p.exists():
#                 print(f"⚠️ Using fallback model: {p}")
#                 return str(p)
                
#         raise FileNotFoundError(f"No suitable {model_name} model found for OpenVINO.")

#     def detect_vehicles(self, frame: np.ndarray, conf_threshold: float = None) -> List[Dict]:
#         if conf_threshold is None:
#             conf_threshold = 0.1  # Lowered for debugging
#         start = time.time()
#         input_tensor = self._preprocess(frame)
#         output = self.compiled_model([input_tensor])[self.output_layer]
#         # Debug: print raw output shape
#         # print(f"[DEBUG] Model output shape: {output.shape}")
#         detections = self._postprocess(output, frame.shape, conf_threshold)
#         # print(f"[DEBUG] Detections after postprocess: {len(detections)}")
#         elapsed = time.time() - start
#         self._inference_times.append(elapsed)
#         self._frame_count += 1
#         self.performance_stats['frames_processed'] = self._frame_count
#         self.performance_stats['total_detections'] += len(detections)
#         if len(self._inference_times) > 100:
#             self._inference_times.pop(0)
#         self.performance_stats['avg_inference_time'] = float(np.mean(self._inference_times)) if self._inference_times else 0
#         total_time = time.time() - self._start_time
#         self.performance_stats['fps'] = self._frame_count / total_time if total_time > 0 else 0
#         return detections

#     def _preprocess(self, frame: np.ndarray) -> np.ndarray:
#         img = cv2.resize(frame, (self.input_width, self.input_height))
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = img.astype(np.float32) / 255.0
#         img = img.transpose(2, 0, 1)[None]
#         return img

#     def _postprocess(self, output: np.ndarray, frame_shape, conf_threshold: float) -> List[Dict]:
#         # Output: (1, 84, 8400) or (84, 8400) or (8400, 84)
#         if output.ndim == 3:
#             output = np.squeeze(output)
#         if output.shape[0] == 84:
#             output = output.T  # (8400, 84)
#         boxes = output[:, :4]
#         scores = output[:, 4:]
#         class_ids = np.argmax(scores, axis=1)
#         confidences = np.max(scores, axis=1)
#         detections = []
#         h, w = frame_shape[:2]
#         for i, (box, score, class_id) in enumerate(zip(boxes, confidences, class_ids)):
#             if score < conf_threshold:
#                 continue
#             x_c, y_c, bw, bh = box
#             # If normalized, scale to input size
#             if all(0.0 <= v <= 1.0 for v in box):
#                 x_c *= self.input_width
#                 y_c *= self.input_height
#                 bw *= self.input_width
#                 bh *= self.input_height
#             # Scale to original frame size
#             scale_x = w / self.input_width
#             scale_y = h / self.input_height
#             x_c *= scale_x
#             y_c *= scale_y
#             bw *= scale_x
#             bh *= scale_y
#             x1 = int(round(x_c - bw / 2))
#             y1 = int(round(y_c - bh / 2))
#             x2 = int(round(x_c + bw / 2))
#             y2 = int(round(y_c + bh / 2))
#             x1 = max(0, min(x1, w - 1))
#             y1 = max(0, min(y1, h - 1))
#             x2 = max(0, min(x2, w - 1))
#             y2 = max(0, min(y2, h - 1))
#             if x2 <= x1 or y2 <= y1:
#                 continue
#             # Only keep class 9 as traffic light, rename if found
#             if class_id == 9:
#                 class_name = "traffic light"
#             elif class_id < len(TRAFFIC_CLASS_NAMES):
#                 class_name = TRAFFIC_CLASS_NAMES[class_id]
#             else:
#                 continue  # Remove unknown/other classes
#             detections.append({
#                 'bbox': [x1, y1, x2, y2],
#                 'confidence': float(score),
#                 'class_id': int(class_id),
#                 'class_name': class_name
#             })
#         # print(f"[DEBUG] Raw detections before NMS: {len(detections)}")
#         # Apply NMS
#         if len(detections) > 0:
#             boxes = np.array([det['bbox'] for det in detections])
#             scores = np.array([det['confidence'] for det in detections])
#             indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_threshold, 0.5)
#             if isinstance(indices, (list, tuple)) and len(indices) > 0:
#                 indices = np.array(indices).flatten()
#             elif isinstance(indices, np.ndarray) and indices.size > 0:
#                 indices = indices.flatten()
#             else:
#                 indices = []
#             detections = [detections[int(i)] for i in indices] if len(indices) > 0 else []
#         # print(f"[DEBUG] Detections after NMS: {len(detections)}")
#         return detections

#     def draw(self, frame: np.ndarray, detections: List[Dict], box_thickness: int = 2) -> np.ndarray:
#         # 80+ visually distinct colors for COCO classes (BGR)
#         COCO_COLORS = [
#             (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29), (207, 210, 49),
#             (72, 249, 10), (146, 204, 23), (61, 219, 134), (26, 147, 52), (0, 212, 187),
#             (44, 153, 168), (0, 194, 255), (52, 69, 147), (100, 115, 255), (0, 24, 236),
#             (132, 56, 255), (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199),
#             (255, 255, 56), (255, 255, 151), (255, 255, 31), (255, 255, 29), (207, 255, 49),
#             (72, 255, 10), (146, 255, 23), (61, 255, 134), (26, 255, 52), (0, 255, 187),
#             (44, 255, 168), (0, 255, 255), (52, 255, 147), (100, 255, 255), (0, 255, 236),
#             (132, 255, 255), (82, 255, 133), (203, 255, 255), (255, 255, 200), (255, 255, 199),
#             (56, 255, 255), (157, 255, 151), (112, 255, 31), (178, 255, 29), (210, 255, 49),
#             (249, 255, 10), (204, 255, 23), (219, 255, 134), (147, 255, 52), (212, 255, 187),
#             (153, 255, 168), (194, 255, 255), (69, 255, 147), (115, 255, 255), (24, 255, 236),
#             (56, 132, 255), (157, 82, 151), (112, 203, 31), (178, 255, 29), (210, 255, 49),
#             (249, 72, 10), (204, 146, 23), (219, 61, 134), (147, 26, 52), (212, 0, 187),
#             (153, 44, 168), (194, 0, 255), (69, 52, 147), (115, 100, 255), (24, 0, 236),
#             (56, 132, 255), (157, 82, 151), (112, 203, 31), (178, 255, 29), (210, 255, 49),
#             (249, 72, 10), (204, 146, 23), (219, 61, 134), (147, 26, 52), (212, 0, 187),
#             (153, 44, 168), (194, 0, 255), (69, 52, 147), (115, 100, 255), (24, 0, 236),
#             (56, 132, 255), (157, 82, 151), (112, 203, 31), (178, 255, 29), (210, 255, 49)
#         ]
#         for det in detections:
#             x1, y1, x2, y2 = det['bbox']
#             label = f"{det['class_name']} {det['confidence']:.2f}"
#             color = COCO_COLORS[det['class_id'] % len(COCO_COLORS)]
#             cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)
#             cv2.putText(frame, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
#         return frame
    
#     def get_device(self):
#         """Get the device being used for inference"""
#         return self.device

# if __name__ == "__main__":
#     # Test the detector with YOLOv11n model
#     detector = OpenVINOVehicleDetector(model_path="yolo11n_openvino_model/yolo11n.xml")
#     print(f"Detector initialized with model: {detector.model_path}")
# Detection logic using OpenVINO models (YOLO, etc.)

import os
import time
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict

# --- Install required packages if missing ---
try:
    import openvino as ov
except ImportError:
    print("Installing openvino...")
    os.system('pip install --quiet "openvino>=2024.0.0"')
    import openvino as ov
try:
    from ultralytics import YOLO
except ImportError:
    print("Installing ultralytics...")
    os.system('pip install --quiet "ultralytics==8.3.0"')
    from ultralytics import YOLO
try:
    import nncf
except ImportError:
    print("Installing nncf...")
    os.system('pip install --quiet "nncf>=2.9.0"')
    import nncf

# --- Traffic-related classes (subset of COCO) ---
TRAFFIC_CLASS_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck',
    'traffic light', 'stop sign', 'parking meter'
]

# --- Model Conversion and Quantization ---
def convert_yolo_to_openvino(model_name: str = "yolo11x", half: bool = True) -> Path:
    """Convert YOLOv11x PyTorch model to OpenVINO IR format."""
    pt_path = Path(f"{model_name}.pt")
    ov_dir = Path(f"{model_name}_openvino_model")
    ov_xml = ov_dir / f"{model_name}.xml"
    if not ov_xml.exists():
        print(f"Exporting {pt_path} to OpenVINO IR...")
        model = YOLO(str(pt_path))
        model.export(format="openvino", dynamic=True, half=half)
    else:
        print(f"OpenVINO IR already exists: {ov_xml}")
    return ov_xml


def quantize_openvino_model(ov_xml: Path, model_name: str = "yolo11x") -> Path:
    """Quantize OpenVINO IR model to INT8 using NNCF."""
    int8_dir = Path(f"{model_name}_openvino_int8_model")
    int8_xml = int8_dir / f"{model_name}.xml"
    if int8_xml.exists():
        print(f"INT8 model already exists: {int8_xml}")
        return int8_xml
    print("Quantization requires a calibration dataset. Skipping actual quantization in this demo.")
    return ov_xml  # Return FP32 if no quantization


class OpenVINOVehicleDetector:
    def __init__(self, model_path: str = None, device: str = "AUTO", use_quantized: bool = False,
                 enable_ocr: bool = False, confidence_threshold: float = 0.4):
        import openvino as ov

        self.confidence_threshold = confidence_threshold
        self.ocr_reader = None
        self.class_names = TRAFFIC_CLASS_NAMES
        self.performance_stats = {
            'fps': 0,
            'avg_inference_time': 0,
            'frames_processed': 0,
            'backend': "OpenVINO",
            'total_detections': 0,
            'detection_rate': 0
        }
        self._inference_times = []
        self._start_time = time.time()
        self._frame_count = 0

        # ---------------------------
        # Select Model
        # ---------------------------
        self.model_path = self._find_best_model(model_path, use_quantized)
        print(f"🎯 OpenVINOVehicleDetector: Using model: {self.model_path}")

        # ---------------------------
        # Select Device (GPU > CPU)
        # ---------------------------
        self.core = ov.Core()
        if "GPU" in self.core.available_devices:
            self.device = "GPU"
            print("🚀 GPU detected, using GPU for inference.")
        else:
            self.device = "CPU"
            print("⚠️ GPU not found, falling back to CPU.")

        # ---------------------------
        # Load Model
        # ---------------------------
        self.model = self.core.read_model(self.model_path)
        self.model.reshape({0: [1, 3, 640, 640]})
        self.input_shape = self.model.inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

        # Config for GPU
        self.ov_config = {}
        if self.device == "GPU":
            self.ov_config = {"GPU_DISABLE_WINOGRAD_CONVOLUTION": "YES"}

        self.compiled_model = self.core.compile_model(
            model=self.model,
            device_name=self.device,
            config=self.ov_config
        )
        self.output_layer = self.compiled_model.output(0)

    def _find_best_model(self, model_path, use_quantized):
        # If provided path exists, use it
        if model_path and Path(model_path).exists():
            print(f"🎯 Using provided model path: {model_path}")
            return str(model_path)

        # Default model name
        model_name = "yolo11x"
        if model_path:
            path_obj = Path(model_path).name.lower()
            if "yolo11n" in path_obj:
                model_name = "yolo11n"
            elif "yolo11s" in path_obj:
                model_name = "yolo11s"
            elif "yolo11m" in path_obj:
                model_name = "yolo11m"
            elif "yolo11l" in path_obj:
                model_name = "yolo11l"

        print(f"🔍 Searching for {model_name} model files...")

        # Your base model directory
        base_dir = Path(r"D:\Downloads\project_t\traffic_monitor\app\openvino_models")

        # Priority search paths
        search_paths = [
            base_dir / f"{model_name}_openvino_int8_model/{model_name}.xml" if use_quantized else None,
            base_dir / f"{model_name}_openvino_model/{model_name}.xml",
            Path(f"{model_name}.xml"),  # fallback relative
            Path(f"{model_name}.pt")
        ]

        for p in search_paths:
            if p and p.exists():
                print(f"✅ Found model: {p}")
                return str(p)

        raise FileNotFoundError(f"No suitable {model_name} model found in {base_dir}")

    # -------------------- Detection --------------------
    def detect_vehicles(self, frame: np.ndarray, conf_threshold: float = None) -> List[Dict]:
        if conf_threshold is None:
            conf_threshold = self.confidence_threshold
        start = time.time()
        input_tensor = self._preprocess(frame)
        output = self.compiled_model([input_tensor])[self.output_layer]
        detections = self._postprocess(output, frame.shape, conf_threshold)

        elapsed = time.time() - start
        self._inference_times.append(elapsed)
        self._frame_count += 1
        self.performance_stats['frames_processed'] = self._frame_count
        self.performance_stats['total_detections'] += len(detections)

        if len(self._inference_times) > 100:
            self._inference_times.pop(0)
        self.performance_stats['avg_inference_time'] = float(np.mean(self._inference_times)) if self._inference_times else 0
        total_time = time.time() - self._start_time
        self.performance_stats['fps'] = self._frame_count / total_time if total_time > 0 else 0
        return detections

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        img = cv2.resize(frame, (self.input_width, self.input_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)[None]
        return img

    def _postprocess(self, output: np.ndarray, frame_shape, conf_threshold: float) -> List[Dict]:
        if output.ndim == 3:
            output = np.squeeze(output)
        if output.shape[0] == 84:
            output = output.T  # (8400, 84)
        boxes = output[:, :4]
        scores = output[:, 4:]
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)

        detections = []
        h, w = frame_shape[:2]
        for box, score, class_id in zip(boxes, confidences, class_ids):
            if score < conf_threshold:
                continue
            x_c, y_c, bw, bh = box
            if all(0.0 <= v <= 1.0 for v in box):
                x_c *= self.input_width
                y_c *= self.input_height
                bw *= self.input_width
                bh *= self.input_height

            scale_x = w / self.input_width
            scale_y = h / self.input_height
            x_c *= scale_x
            y_c *= scale_y
            bw *= scale_x
            bh *= scale_y

            x1 = int(round(x_c - bw / 2))
            y1 = int(round(y_c - bh / 2))
            x2 = int(round(x_c + bw / 2))
            y2 = int(round(y_c + bh / 2))
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(x2, w - 1), min(y2, h - 1)

            if x2 <= x1 or y2 <= y1:
                continue

            if class_id == 9:
                class_name = "traffic light"
            elif class_id < len(TRAFFIC_CLASS_NAMES):
                class_name = TRAFFIC_CLASS_NAMES[class_id]
            else:
                continue

            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': float(score),
                'class_id': int(class_id),
                'class_name': class_name
            })

        # NMS
        if detections:
            boxes = np.array([det['bbox'] for det in detections])
            scores = np.array([det['confidence'] for det in detections])
            indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_threshold, 0.5)
            if isinstance(indices, (list, tuple)) and len(indices) > 0:
                indices = np.array(indices).flatten()
            elif isinstance(indices, np.ndarray) and indices.size > 0:
                indices = indices.flatten()
            else:
                indices = []
            detections = [detections[int(i)] for i in indices] if len(indices) > 0 else []
        return detections

    def draw(self, frame: np.ndarray, detections: List[Dict], box_thickness: int = 2) -> np.ndarray:
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = f"{det['class_name']} {det['confidence']:.2f}"
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)
            cv2.putText(frame, label, (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame

    def get_device(self):
        return self.device


if __name__ == "__main__":
    detector = OpenVINOVehicleDetector(
        model_path=r"D:\Downloads\project_t\traffic_monitor\app\openvino_models\yolo11x_openvino_model\yolo11x.xml"
    )
    print(f"✅ Detector initialized with model: {detector.model_path}, device: {detector.get_device()}")
