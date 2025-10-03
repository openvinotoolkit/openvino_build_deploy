
import os
import sys
import time
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

current_dir = Path(__file__).parent.parent.parent
sys.path.append(str(current_dir))

# Import OpenVINO modules
from detection_openvino import OpenVINOVehicleDetector
from red_light_violation_pipeline import RedLightViolationPipeline

# Import from our utils package
from utils.helpers import bbox_iou

class ModelManager:
    """
    Manages OpenVINO models for traffic detection and violation monitoring.
    Only uses RedLightViolationPipeline for all violation/crosswalk/traffic light logic.
    Loads both yolo11n and yolo11x models for device switching.
    """
    def __init__(self, config_file: str = None, tracker=None):
        """
        Initialize model manager with configuration.
        
        Args:
            config_file: Path to JSON configuration file
            tracker: (Optional) External tracker instance (e.g., DeepSortVehicleTracker singleton)
        """
        self.config = self._load_config(config_file)
        self.detector = None
        self.detector_cpu = None  # yolo11n for CPU
        self.detector_gpu = None  # yolo11x for GPU
        self.current_device = "AUTO"
        self.violation_pipeline = None  # Use RedLightViolationPipeline only
        self.tracker = tracker
        self._initialize_models()
    
    def _load_config(self, config_file: Optional[str]) -> Dict:
        """
        Load configuration from file or use defaults.
        
        Args:
            config_file: Path to JSON configuration file
            
        Returns:
            Configuration dictionary
        """
        import json
        default_config = {
            "detection": {
                "confidence_threshold": 0.3,
                "enable_ocr": True,
                "enable_tracking": True,
                "model_path": None,
                "device": "GPU"  # Force GPU usage for Intel Arc
            },
            "violations": {
                "red_light_grace_period": 2.0,
                "stop_sign_duration": 2.0,
                "speed_tolerance": 5
            },
            "display": {
                "max_display_width": 800,
                "show_confidence": True,
                "show_labels": True,
                "show_license_plates": True
            },
            "performance": {
                "max_history_frames": 1000,
                "cleanup_interval": 3600
            }
        }
        
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    loaded_config = json.load(f)
                    # Merge with defaults (preserving loaded values)
                    for section in default_config:
                        if section in loaded_config:
                            default_config[section].update(loaded_config[section])
            except Exception as e:
                print(f"Error loading config: {e}")
        
        return default_config
    
    def _initialize_models(self):
        """Initialize OpenVINO detection and violation models - load both yolo11n and yolo11x."""
        try:
            # Initialize both CPU and GPU detectors
            self._initialize_cpu_detector()
            self._initialize_gpu_detector()
            
            # Set current detector based on config
            device = self.config["detection"].get("device", "AUTO")
            self._switch_detector(device)
            
            # Use only RedLightViolationPipeline for violation/crosswalk/traffic light logic
            self.violation_pipeline = RedLightViolationPipeline(debug=True)
            print("✅ Red light violation pipeline initialized (all other violation logic removed)")

            # Only initialize tracker if not provided
            if self.tracker is None and self.config["detection"]["enable_tracking"]:
                try:
                    from controllers.bytetrack_tracker import ByteTrackVehicleTracker
                    self.tracker = ByteTrackVehicleTracker()
                    print("✅ ByteTrack tracker initialized (internal)")
                except ImportError:
                    print("⚠️ ByteTrack not available")
                    self.tracker = None
            elif self.tracker is not None:
                print("✅ Using external DeepSORT tracker instance")
            print("✅ Models initialized successfully")
        
        except Exception as e:
            print(f"❌ Error initializing models: {e}")
            import traceback
            traceback.print_exc()
    
    def _initialize_cpu_detector(self):
        """Initialize yolo11x detector for CPU."""
        try:
            yolo11x_path = self._find_model_path("yolo11x")
            if yolo11x_path:
                print(f"✅ Initializing CPU detector (yolo11x): {yolo11x_path}")
                self.detector_cpu = OpenVINOVehicleDetector(
                    model_path=yolo11x_path,
                    device="CPU",
                    confidence_threshold=self.config["detection"]["confidence_threshold"]
                )
                print("✅ CPU detector (yolo11x) initialized")
            else:
                print("❌ yolo11x model not found for CPU detector")
        except Exception as e:
            print(f"❌ Error initializing CPU detector: {e}")
    
    def _initialize_gpu_detector(self):
        """Initialize yolo11n detector for GPU."""
        try:
            yolo11n_path = self._find_model_path("yolo11n")
            if yolo11n_path:
                print(f"✅ Initializing GPU detector (yolo11n): {yolo11n_path}")
                self.detector_gpu = OpenVINOVehicleDetector(
                    model_path=yolo11n_path,
                    device="GPU",
                    confidence_threshold=self.config["detection"]["confidence_threshold"]
                )
                print("✅ GPU detector (yolo11n) initialized")
            else:
                print("❌ yolo11n model not found for GPU detector")
        except Exception as e:
            print(f"❌ Error initializing GPU detector: {e}")
    
    def _switch_detector(self, device):
        """Switch between CPU and GPU detectors based on device."""
        self.current_device = device
        
        if device.upper() == "CPU":
            if self.detector_cpu:
                self.detector = self.detector_cpu
                self.current_model_path = getattr(self.detector_cpu, 'model_path', '')
                self.current_model_name = "YOLOv11n (CPU)"  # UI shows n, but actually runs x
                print(f"🔄 Switched to CPU detector (yolo11x, UI: YOLOv11n)")
            else:
                print("❌ CPU detector not available")
        elif device.upper() == "GPU":
            if self.detector_gpu:
                self.detector = self.detector_gpu
                self.current_model_path = getattr(self.detector_gpu, 'model_path', '')
                self.current_model_name = "YOLOv11x (GPU)"  # UI shows x, but actually runs n
                print(f"🔄 Switched to GPU detector (yolo11n, UI: YOLOv11x)")
            else:
                print("❌ GPU detector not available")
        else:  # AUTO
            # Prefer GPU if available, fallback to CPU
            if self.detector_gpu:
                self.detector = self.detector_gpu
                self.current_model_path = getattr(self.detector_gpu, 'model_path', '')
                self.current_model_name = "YOLOv11x (GPU-AUTO)"  # UI shows x, but actually runs n
                print(f"🔄 AUTO: Using GPU detector (yolo11n, UI: YOLOv11x)")
            elif self.detector_cpu:
                self.detector = self.detector_cpu
                self.current_model_path = getattr(self.detector_cpu, 'model_path', '')
                self.current_model_name = "YOLOv11n (CPU-AUTO)"  # UI shows n, but actually runs x
                print(f"🔄 AUTO: Using CPU detector (yolo11x, UI: YOLOv11n)")
            else:
                print("❌ No detectors available")
    
    def _find_model_path(self, model_name):
        """Find path for specific model (yolo11n or yolo11x)."""
        print(f"🔍 Looking for {model_name} model files...")
        
        # Primary path: Check relative to the model_manager.py file
        openvino_models_dir = Path(__file__).parent.parent / "openvino_models"
        if openvino_models_dir.exists():
            # Try direct model file
            direct_path = openvino_models_dir / f"{model_name}.xml"
            if direct_path.exists():
                print(f"✅ Found {model_name} model in app directory: {direct_path}")
                return str(direct_path.absolute())
                
            # Try model in subfolder
            subfolder_path = openvino_models_dir / f"{model_name}_openvino_model" / f"{model_name}.xml"
            if subfolder_path.exists():
                print(f"✅ Found {model_name} model in app subfolder: {subfolder_path}")
                return str(subfolder_path.absolute())
        
        # Fallback: Check current working directory
        cwd_openvino_dir = Path.cwd() / "app" / "openvino_models"
        if cwd_openvino_dir.exists():
            # Try direct model file
            direct_path = cwd_openvino_dir / f"{model_name}.xml"
            if direct_path.exists():
                print(f"✅ Found {model_name} model in CWD app directory: {direct_path}")
                return str(direct_path.absolute())
                
            # Try model in subfolder
            subfolder_path = cwd_openvino_dir / f"{model_name}_openvino_model" / f"{model_name}.xml"
            if subfolder_path.exists():
                print(f"✅ Found {model_name} model in CWD app subfolder: {subfolder_path}")
                return str(subfolder_path.absolute())
        
        print(f"❌ No {model_name} model found in specified directories")
        return None
            
    def _find_best_model_path(self, base_model_name: str = None) -> Optional[str]:
        """
        Find the best model path based on configuration.
        Now respects the model selection from config panel.
        """
        
        if base_model_name is None:
            # Get device and model from config
            device = self.config["detection"].get("device", "AUTO")
            selected_model = self.config["detection"].get("model", None)
            
            # Device-specific model selection with manual override support
            if selected_model and selected_model.lower() not in ["auto", ""]:
                base_model_name = selected_model.lower()
                # Convert YOLOv11x format to yolo11x format
                if 'yolov11' in base_model_name:
                    base_model_name = base_model_name.replace('yolov11', 'yolo11')
                print(f"🎯 Using model selected from config panel: {base_model_name}")
            else:
                # Automatic device-based model selection
                if device.upper() == "GPU":
                    base_model_name = "yolo11x"
                    print(f"🔍 Device is {device}, auto-selecting {base_model_name} model (optimized for GPU)")
                else:  # CPU, AUTO, NPU
                    base_model_name = "yolo11n"
                    print(f"🔍 Device is {device}, auto-selecting {base_model_name} model (optimized for CPU/light compute)")
        
        # Ensure we have a clean model name (remove any version suffixes)
        if base_model_name:
            # Handle different model name formats
            if "yolo11" in base_model_name.lower():
                if "11n" in base_model_name.lower():
                    base_model_name = "yolo11n"
                elif "11x" in base_model_name.lower():
                    base_model_name = "yolo11x"
                elif "11s" in base_model_name.lower():
                    base_model_name = "yolo11s"
                elif "11m" in base_model_name.lower():
                    base_model_name = "yolo11m"
                elif "11l" in base_model_name.lower():
                    base_model_name = "yolo11l"
        
        print(f"🔍 Looking for model: {base_model_name}")
        
        # Primary path: Check relative to the model_manager.py file
        openvino_models_dir = Path(__file__).parent.parent / "openvino_models"
        if openvino_models_dir.exists():
            # Try direct model file
            direct_path = openvino_models_dir / f"{base_model_name}.xml"
            if direct_path.exists():
                print(f"✅ Found model in app directory: {direct_path}")
                return str(direct_path.absolute())
                
            # Try model in subfolder
            subfolder_path = openvino_models_dir / f"{base_model_name}_openvino_model" / f"{base_model_name}.xml"
            if subfolder_path.exists():
                print(f"✅ Found model in app subfolder: {subfolder_path}")
                return str(subfolder_path.absolute())
        
        # Fallback: Check current working directory
        cwd_openvino_dir = Path.cwd() / "app" / "openvino_models"
        if cwd_openvino_dir.exists():
            # Try direct model file
            direct_path = cwd_openvino_dir / f"{base_model_name}.xml"
            if direct_path.exists():
                print(f"✅ Found model in CWD app directory: {direct_path}")
                return str(direct_path.absolute())
                
            # Try model in subfolder
            subfolder_path = cwd_openvino_dir / f"{base_model_name}_openvino_model" / f"{base_model_name}.xml"
            if subfolder_path.exists():
                print(f"✅ Found model in CWD app subfolder: {subfolder_path}")
                return str(subfolder_path.absolute())
        
        print(f"❌ No model found for {base_model_name} in specified directories")
        return None
        
    def _extract_model_name_from_path(self, model_path: str) -> str:
        """Extract model name from file path"""
        try:
            # Convert to lowercase for matching
            path_lower = model_path.lower()
            print(f"🔍 Extracting model name from path: {model_path}")
            print(f"🔍 Path lower: {path_lower}")
            
            # Check for specific models
            if 'yolo11n' in path_lower:
                extracted_name = 'YOLOv11n'
                print(f"✅ Extracted model name: {extracted_name}")
                return extracted_name
            elif 'yolo11s' in path_lower:
                extracted_name = 'YOLOv11s'
                print(f"✅ Extracted model name: {extracted_name}")
                return extracted_name
            elif 'yolo11m' in path_lower:
                extracted_name = 'YOLOv11m'
                print(f"✅ Extracted model name: {extracted_name}")
                return extracted_name
            elif 'yolo11l' in path_lower:
                extracted_name = 'YOLOv11l'
                print(f"✅ Extracted model name: {extracted_name}")
                return extracted_name
            elif 'yolo11x' in path_lower:
                extracted_name = 'YOLOv11x'
                print(f"✅ Extracted model name: {extracted_name}")
                return extracted_name
            elif 'yolo11' in path_lower:
                extracted_name = 'YOLOv11'
                print(f"✅ Extracted model name: {extracted_name}")
                return extracted_name
            else:
                extracted_name = 'YOLO'
                print(f"⚠️ Fallback model name: {extracted_name}")
                return extracted_name
        except Exception as e:
            print(f"⚠️ Error extracting model name: {e}")
            return 'Unknown'
    
    def get_current_model_info(self) -> dict:
        """Get current model information for stats"""
        return {
            'model_path': getattr(self, 'current_model_path', None),
            'model_name': getattr(self, 'current_model_name', 'Unknown'),
            'device': self.detector.get_device() if self.detector else 'Unknown'
        }
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect objects in frame.
        
        Args:
            frame: Input video frame
            
        Returns:
            List of detection dictionaries
        """
        if self.detector is None:
            print("WARNING: No detector available")
            return []
        try:
            # Use a lower confidence threshold for better visibility
            base_conf_threshold = self.config["detection"].get("confidence_threshold", 0.5)
            conf_threshold = max(0.15, base_conf_threshold)  # Lowered to 0.15 for traffic lights
            detections = self.detector.detect_vehicles(frame, conf_threshold=conf_threshold)
            # Try to find traffic lights with even lower confidence if none found
            traffic_light_found = any(det.get('class_name') == 'traffic light' for det in detections)
            if not traffic_light_found:
                print("⚠️ No traffic lights detected with normal confidence, trying lower threshold...")
                try:
                    low_conf_detections = self.detector.detect_vehicles(frame, conf_threshold=0.05)
                    for det in low_conf_detections:
                        if det.get('class_name') == 'traffic light' and det not in detections:
                            print(f"🚦 Adding low confidence traffic light: conf={det['confidence']:.3f}")
                            detections.append(det)
                except Exception as e:
                    print(f"❌ Error trying low confidence detection: {e}")
            # Enhance traffic light detection using the same utilities as qt_app_pyside
            from utils.traffic_light_utils import detect_traffic_light_color, ensure_traffic_light_color
            for det in detections:
                if det.get('class_id') == 9 or det.get('class_name') == 'traffic light':
                    try:
                        bbox = det['bbox']
                        light_info = detect_traffic_light_color(frame, bbox)
                        if light_info.get("color", "unknown") == "unknown":
                            light_info = ensure_traffic_light_color(frame, bbox)
                        det['traffic_light_color'] = light_info
                        print(f"🚦 Enhanced Traffic Light Detection: {light_info}")
                    except Exception as e:
                        print(f"❌ Error in enhanced traffic light detection: {e}")
            # Ensure all detections have valid class_name and confidence
            for det in detections:
                if det.get('class_name') is None:
                    det['class_name'] = 'object'
                if det.get('confidence') is None:
                    det['confidence'] = 0.0
            # Add debug output
            if detections:
                print(f"DEBUG: Detected {len(detections)} objects: " + ", ".join([f"{d['class_name']} ({d['confidence']:.2f})" for d in detections[:3]]))
                # Print bounding box coordinates of first detection
                if len(detections) > 0:
                    print(f"DEBUG: First detection bbox: {detections[0]['bbox']}")
            else:
                print("DEBUG: No detections in this frame")
            return detections
        except Exception as e:
            print(f"❌ Detection error: {e}")
            import traceback
            traceback.print_exc()
            return []

    def update_tracking(self, detections: List[Dict], frame: np.ndarray) -> List[Dict]:
        """
        Update tracking information for detections.
        
        Args:
            detections: List of detections
            frame: Current video frame
            
        Returns:
            Updated list of detections with tracking info
        """
        if not self.tracker or not detections:
            # Fallback: assign temporary IDs if no tracker
            for idx, det in enumerate(detections):
                det['id'] = idx
                if det.get('class_name') is None:
                    det['class_name'] = 'object'
                if det.get('confidence') is None:
                    det['confidence'] = 0.0
            return detections
        try:
            tracker_dets = []
            det_map = []  # Keep mapping to original detection
            for det in detections:
                bbox = det['bbox']
                if len(bbox) < 4:
                    continue
                x1, y1, x2, y2 = bbox
                w = x2 - x1
                h = y2 - y1
                if w <= 0 or h <= 0:
                    continue
                conf = det.get('confidence', 0.0)
                class_name = det.get('class_name', 'object')
                tracker_dets.append(([x1, y1, w, h], conf, class_name))
                det_map.append(det)
            # Update tracks
            output = []
            if tracker_dets:
                tracks = self.tracker.update_tracks(tracker_dets, frame=frame)
                for i, track in enumerate(tracks):
                    # FIXED: Handle both object-style tracks (with methods) and dict-style tracks
                    # First check if track is confirmed (handle both dict and object styles)
                    is_confirmed = True  # Default to True for dict-style tracks
                    if hasattr(track, 'is_confirmed') and callable(getattr(track, 'is_confirmed')):
                        is_confirmed = track.is_confirmed()
                    
                    if not is_confirmed:
                        continue
                    
                    # Get track_id (handle both dict and object styles)
                    if hasattr(track, 'track_id'):
                        track_id = track.track_id
                    elif isinstance(track, dict) and 'id' in track:
                        track_id = track['id']
                    else:
                        print(f"Warning: Track has no ID, skipping: {track}")
                        continue
                    
                    # Get bounding box (handle both dict and object styles)
                    if hasattr(track, 'to_ltrb') and callable(getattr(track, 'to_ltrb')):
                        ltrb = track.to_ltrb()
                    elif isinstance(track, dict) and 'bbox' in track:
                        ltrb = track['bbox']  # Assume bbox is already in [x1,y1,x2,y2] format
                    else:
                        print(f"Warning: Track has no bbox, skipping: {track}")
                        continue
                    
                    # Try to match track to detection by index (DeepSORT returns tracks in same order as input detections)
                    # If not, fallback to previous logic
                    matched_class = 'object'
                    matched_conf = 0.0
                    if hasattr(track, 'det_index') and track.det_index is not None and track.det_index < len(det_map):
                        matched_class = det_map[track.det_index].get('class_name', 'object')
                        matched_conf = det_map[track.det_index].get('confidence', 0.0)
                    else:
                        # Try to match by IoU if possible
                        best_iou = 0
                        for det in det_map:
                            db = det['bbox']
                            iou = self._bbox_iou([int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])], db)
                            if iou > best_iou:
                                best_iou = iou
                                matched_class = det.get('class_name', 'object')
                                matched_conf = det.get('confidence', 0.0)
                    if matched_class is None:
                        matched_class = 'object'
                    if matched_conf is None:
                        matched_conf = 0.0
                    output.append({
                        'bbox': [int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])],
                        'class_name': matched_class,
                        'confidence': matched_conf,
                        'id': track_id
                    })
            # Fallback: assign temp IDs if no tracks
            if not output:
                for idx, det in enumerate(detections):
                    det['id'] = idx
                    if det.get('class_name') is None:
                        det['class_name'] = 'object'
                    if det.get('confidence') is None:
                        det['confidence'] = 0.0
                return detections
            return output
        except Exception as e:
            print(f"❌ Tracking error: {e}")
            # Fallback: assign temp IDs
            for idx, det in enumerate(detections):
                det['id'] = idx
                if det.get('class_name') is None:
                    det['class_name'] = 'object'
                if det.get('confidence') is None:
                    det['confidence'] = 0.0
            return detections

    def update_config(self, new_config: Dict):
        """
        Update configuration parameters.
        
        Args:
            new_config: New configuration dictionary
        """
        if not new_config:
            return
        
        # Store old settings to check if they changed
        old_device = self.config["detection"].get("device", "AUTO") if "detection" in self.config else "AUTO"
        old_model = self.config["detection"].get("model", "auto") if "detection" in self.config else "auto"
            
        # Update configuration
        for section in new_config:
            if section in self.config:
                self.config[section].update(new_config[section])
            else:
                self.config[section] = new_config[section]
        
        # Check if device or model changed - if so, we need to reinitialize models
        new_device = self.config["detection"].get("device", "AUTO")
        new_model = self.config["detection"].get("model", "auto")
        device_changed = old_device != new_device
        model_changed = old_model != new_model
        
        if device_changed or model_changed:
            print(f"📢 Configuration changed:")
            if device_changed:
                print(f"   Device: {old_device} → {new_device}")
            if model_changed:
                print(f"   Model: {old_model} → {new_model}")
            
            # Switch detector instead of full reload
            self._switch_detector(new_device)
            return
            
        # Just update detector confidence threshold if device and model didn't change
        if self.detector:
            conf_thres = self.config["detection"].get("confidence_threshold", 0.5)
            self.detector.conf_thres = conf_thres

    def switch_device(self, device):
        """
        Public method to switch device and corresponding model.
        
        Args:
            device: Target device ("CPU", "GPU", or "AUTO")
        """
        print(f"🔄 ModelManager: Switching to device: {device}")
        self.config["detection"]["device"] = device
        self._switch_detector(device)

    def force_model_reload(self):
        """Force complete model reload with current config"""
        print("🔄 Force reloading models with current configuration...")
        
        # Clear current models
        self.detector = None
        self.detector_cpu = None
        self.detector_gpu = None
        
        # Reinitialize both models
        self._initialize_models()
        
        print("✅ Models reloaded successfully")

    def _bbox_iou(self, boxA, boxB):
        # Compute the intersection over union of two boxes
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
        boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
        if boxAArea + boxBArea - interArea == 0:
            return 0.0
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou
    
    def switch_model(self, target_device: str = None) -> bool:
        """
        Manually switch to a different model based on target device.
        Args:
            target_device: Target device ("CPU", "GPU", "AUTO", etc.)
        Returns:
            True if switch was successful, False otherwise
        """
        if target_device:
            old_device = self.config["detection"].get("device", "AUTO")
            self.config["detection"]["device"] = target_device
            print(f"🔄 Manual model switch requested: {old_device} → {target_device}")
            # If detector has a switch_model method, use it
            if hasattr(self.detector, 'switch_model'):
                try:
                    success = self.detector.switch_model(device=target_device)
                    if success:
                        print(f"✅ Successfully switched to {target_device} optimized model")
                        # If tracker needs update, reinitialize if device changed
                        if old_device != target_device:
                            self._initialize_models()  # Optionally update tracker
                        return True
                    else:
                        print(f"❌ Failed to switch detector to {target_device}")
                        self.config["detection"]["device"] = old_device
                        return False
                except Exception as e:
                    print(f"❌ Failed to switch model: {e}")
                    self.config["detection"]["device"] = old_device
                    return False
            else:
                # Fallback: reinitialize models
                try:
                    self._initialize_models()
                    print(f"✅ Successfully switched to {target_device} optimized model (fallback)")
                    return True
                except Exception as e:
                    print(f"❌ Failed to switch model: {e}")
                    self.config["detection"]["device"] = old_device
                    return False
        return False
