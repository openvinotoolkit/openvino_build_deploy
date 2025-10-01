"""
Smart Intersection Scene Analytics Adapter
Adapted for desktop PySide6 application with Intel Arc GPU acceleration
"""

import json
import logging
import math
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal, QTimer
from pathlib import Path

try:
    from influxdb_client import InfluxDBClient, Point
    INFLUXDB_AVAILABLE = True
except ImportError:
    INFLUXDB_AVAILABLE = False

class SceneAnalyticsAdapter(QObject):
    """
    Desktop-adapted scene analytics adapter for smart intersection functionality.
    Provides scene-based analytics without external dependencies like MQTT.
    """
    
    # Signals for desktop integration
    object_detected = Signal(dict)  # Emits object detection data
    scene_analytics_updated = Signal(dict)  # Emits scene analytics
    roi_event_detected = Signal(dict)  # Emits ROI-based events
    
    def __init__(self, camera_id: str = "desktop_cam", config_path: Optional[str] = None):
        super().__init__()
        self.camera_id = camera_id
        self.config = self._load_config(config_path)
        self.logger = self._setup_logger()
        
        # Analytics state
        self.frame_count = 0
        self.fps_calculator = FPSCalculator()
        self.object_tracker = ObjectTracker(self.config.get('tracker', {}))
        self.roi_analyzer = ROIAnalyzer()
        
        # Performance tracking
        self.processing_times = []
        self.last_analytics_update = time.time()
        
        # InfluxDB setup
        self.influxdb_enabled = INFLUXDB_AVAILABLE
        if self.influxdb_enabled:
            try:
                self.influxdb_url = "http://localhost:8086"
                self.influxdb_token = "kNFfXEpPQoWrk5Tteowda21Dzv6xD3jY7QHSHHQHb5oYW6VH6mkAgX9ZMjQJkaHHa8FwzmyVFqDG7qqzxN09uQ=="
                self.influxdb_org = "smart-intersection-org"
                self.influxdb_bucket = "traffic_monitoring"
                self.influxdb_client = InfluxDBClient(url=self.influxdb_url, token=self.influxdb_token, org=self.influxdb_org)
                self.influxdb_write_api = self.influxdb_client.write_api()
                print("[INFO] InfluxDB connection established successfully")
            except Exception as e:
                self.influxdb_enabled = False
                print(f"[ERROR] Failed to initialize InfluxDB: {e}")
        else:
            print("[WARNING] influxdb_client not installed. InfluxDB logging disabled.")
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration for scene analytics"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        
        # Default configuration
        return {
            "tracker": {
                "max_unreliable_frames": 10,
                "non_measurement_frames_dynamic": 8,
                "non_measurement_frames_static": 16,
                "baseline_frame_rate": 30
            },
            "analytics": {
                "enable_roi_detection": True,
                "enable_speed_estimation": True,
                "enable_direction_analysis": True
            },
            "performance": {
                "target_fps": 30,
                "max_processing_time_ms": 33
            }
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for scene analytics"""
        logger = logging.getLogger(f'SceneAnalytics_{self.camera_id}')
        logger.setLevel(logging.INFO)
        return logger
    
    def process_frame(self, frame: np.ndarray, detections: List[Dict]) -> Dict:
        """
        Process a frame with detections and generate scene analytics
        
        Args:
            frame: Input frame as numpy array
            detections: List of detection dictionaries from YOLO/detection model
            
        Returns:
            Dictionary containing scene analytics data
        """
        start_time = time.time()
        self.frame_count += 1
        
        # Update FPS calculation
        current_fps = self.fps_calculator.update()
        
        # Process detections into scene objects
        scene_objects = self._process_detections(detections, frame.shape)
        
        # Update object tracking
        tracked_objects = self.object_tracker.update(scene_objects)
        
        # Perform ROI analysis
        roi_events = self.roi_analyzer.analyze_objects(tracked_objects, frame.shape)
        
        # Generate analytics data
        analytics_data = {
            'timestamp': datetime.now().isoformat(),
            'camera_id': self.camera_id,
            'frame_number': self.frame_count,
            'fps': current_fps,
            'objects': tracked_objects,
            'roi_events': roi_events,
            'processing_time_ms': (time.time() - start_time) * 1000
        }
        
        # Emit signals for desktop integration
        self._emit_analytics_signals(analytics_data)
        
        # Track performance
        self.processing_times.append(analytics_data['processing_time_ms'])
        if len(self.processing_times) > 100:
            self.processing_times.pop(0)
            
        # Write to InfluxDB
        if self.influxdb_enabled:
            try:
                # Performance metrics
                perf_point = Point("performance") \
                    .tag("camera_id", self.camera_id) \
                    .field("fps", float(analytics_data['fps'])) \
                    .field("processing_time_ms", float(analytics_data['processing_time_ms'])) \
                    .time(datetime.utcnow())
                self.influxdb_write_api.write(bucket=self.influxdb_bucket, org=self.influxdb_org, record=perf_point)
                
                # Detection events - count vehicles and pedestrians
                vehicle_count = 0
                pedestrian_count = 0
                for obj in tracked_objects:
                    if obj.get('category') == 'vehicle':
                        vehicle_count += 1
                    elif obj.get('category') == 'pedestrian':
                        pedestrian_count += 1
                        
                detect_point = Point("detection_events") \
                    .tag("camera_id", self.camera_id) \
                    .field("vehicle_count", vehicle_count) \
                    .field("pedestrian_count", pedestrian_count) \
                    .time(datetime.utcnow())
                self.influxdb_write_api.write(bucket=self.influxdb_bucket, org=self.influxdb_org, record=detect_point)
                
                # Log any violations
                for event in roi_events:
                    if event.get('event_type') and 'violation' in event.get('event_type'):
                        violation_point = Point("violation_events") \
                            .tag("camera_id", self.camera_id) \
                            .tag("violation_type", event.get('event_type')) \
                            .field("count", 1) \
                            .time(datetime.utcnow())
                        self.influxdb_write_api.write(bucket=self.influxdb_bucket, 
                                                     org=self.influxdb_org, 
                                                     record=violation_point)
                
            except Exception as e:
                print(f"[ERROR] Failed to write to InfluxDB: {e}")
        
        return analytics_data
    
    def _process_detections(self, detections: List[Dict], frame_shape: Tuple[int, int, int]) -> List[Dict]:
        """Convert YOLO detections to scene objects"""
        scene_objects = []
        height, width = frame_shape[:2]
        
        for i, detection in enumerate(detections):
            # Extract detection data (adapt based on your detection format)
            if 'bbox' in detection:
                x, y, w, h = detection['bbox']
            elif 'box' in detection:
                x, y, w, h = detection['box']
            else:
                continue
                
            confidence = detection.get('confidence', detection.get('conf', 0.0))
            class_name = detection.get('class_name', detection.get('name', 'unknown'))
            class_id = detection.get('class_id', detection.get('cls', 0))
            
            # Create scene object
            scene_obj = {
                'id': f"{self.camera_id}_{self.frame_count}_{i}",
                'category': self._map_class_to_category(class_name),
                'confidence': float(confidence),
                'bounding_box_px': {
                    'x': int(x),
                    'y': int(y),
                    'width': int(w),
                    'height': int(h)
                },
                'center_of_mass': {
                    'x': int(x + w/2),
                    'y': int(y + h/2),
                    'width': w/3,
                    'height': h/4
                },
                'normalized_bbox': {
                    'x': x / width,
                    'y': y / height,
                    'w': w / width,
                    'h': h / height
                }
            }
            
            scene_objects.append(scene_obj)
        
        return scene_objects
    
    def _map_class_to_category(self, class_name: str) -> str:
        """Map detection class names to scene categories"""
        person_classes = ['person', 'pedestrian', 'cyclist']
        vehicle_classes = ['car', 'truck', 'bus', 'motorcycle', 'vehicle']
        
        class_lower = class_name.lower()
        
        if any(p in class_lower for p in person_classes):
            return 'person'
        elif any(v in class_lower for v in vehicle_classes):
            return 'vehicle'
        else:
            return 'other'
    
    def _emit_analytics_signals(self, analytics_data: Dict):
        """Emit Qt signals for desktop integration"""
        try:
            # Emit object detection signal
            for obj in analytics_data['objects']:
                self.object_detected.emit(obj)
            
            # Emit scene analytics
            self.scene_analytics_updated.emit({
                'timestamp': analytics_data['timestamp'],
                'camera_id': analytics_data['camera_id'],
                'fps': analytics_data['fps'],
                'object_count': len(analytics_data['objects']),
                'processing_time_ms': analytics_data['processing_time_ms']
            })
            
            # Emit ROI events
            for event in analytics_data['roi_events']:
                self.roi_event_detected.emit(event)
                
        except Exception as e:
            self.logger.error(f"Error emitting signals: {e}")
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        if not self.processing_times:
            return {}
        
        return {
            'avg_processing_time_ms': np.mean(self.processing_times),
            'max_processing_time_ms': np.max(self.processing_times),
            'min_processing_time_ms': np.min(self.processing_times),
            'current_fps': self.fps_calculator.get_current_fps(),
            'total_frames_processed': self.frame_count
        }


class FPSCalculator:
    """Calculate FPS with smoothing"""
    
    def __init__(self, alpha: float = 0.75):
        self.alpha = alpha
        self.fps = 30.0
        self.last_time = time.time()
        self.frame_count = 0
        
    def update(self) -> float:
        """Update FPS calculation"""
        current_time = time.time()
        self.frame_count += 1
        
        if self.frame_count > 1:
            frame_time = current_time - self.last_time
            if frame_time > 0:
                instant_fps = 1.0 / frame_time
                self.fps = self.fps * self.alpha + (1 - self.alpha) * instant_fps
        
        self.last_time = current_time
        return self.fps
    
    def get_current_fps(self) -> float:
        """Get current FPS"""
        return self.fps


class ObjectTracker:
    """Simple object tracker for desktop application"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.tracked_objects = {}
        self.next_track_id = 1
        self.max_unreliable_frames = config.get('max_unreliable_frames', 10)
        
    def update(self, detections: List[Dict]) -> List[Dict]:
        """Update object tracking"""
        # Simple tracking based on position proximity
        tracked = []
        
        for detection in detections:
            track_id = self._find_or_create_track(detection)
            detection['track_id'] = track_id
            tracked.append(detection)
        
        # Remove stale tracks
        self._cleanup_stale_tracks()
        
        return tracked
    
    def _find_or_create_track(self, detection: Dict) -> int:
        """Find existing track or create new one"""
        center = detection['center_of_mass']
        best_match = None
        best_distance = float('inf')
        
        # Find closest existing track
        for track_id, track_data in self.tracked_objects.items():
            if track_data['category'] == detection['category']:
                track_center = track_data['last_center']
                distance = math.sqrt(
                    (center['x'] - track_center['x'])**2 + 
                    (center['y'] - track_center['y'])**2
                )
                
                if distance < best_distance and distance < 100:  # Threshold
                    best_distance = distance
                    best_match = track_id
        
        if best_match:
            # Update existing track
            self.tracked_objects[best_match]['last_center'] = center
            self.tracked_objects[best_match]['frames_since_detection'] = 0
            return best_match
        else:
            # Create new track
            track_id = self.next_track_id
            self.next_track_id += 1
            self.tracked_objects[track_id] = {
                'category': detection['category'],
                'last_center': center,
                'frames_since_detection': 0,
                'created_frame': time.time()
            }
            return track_id
    
    def _cleanup_stale_tracks(self):
        """Remove tracks that haven't been updated"""
        stale_tracks = []
        for track_id, track_data in self.tracked_objects.items():
            track_data['frames_since_detection'] += 1
            if track_data['frames_since_detection'] > self.max_unreliable_frames:
                stale_tracks.append(track_id)
        
        for track_id in stale_tracks:
            del self.tracked_objects[track_id]


class ROIAnalyzer:
    """Analyze objects within regions of interest"""
    
    def __init__(self):
        self.roi_definitions = {}
        self.roi_events = []
    
    def add_roi(self, roi_id: str, roi_data: Dict):
        """Add a region of interest definition"""
        self.roi_definitions[roi_id] = roi_data
    
    def analyze_objects(self, objects: List[Dict], frame_shape: Tuple[int, int, int]) -> List[Dict]:
        """Analyze objects for ROI events"""
        events = []
        
        for obj in objects:
            for roi_id, roi_data in self.roi_definitions.items():
                if self._object_in_roi(obj, roi_data, frame_shape):
                    event = {
                        'type': 'object_in_roi',
                        'roi_id': roi_id,
                        'object_id': obj.get('track_id', obj['id']),
                        'object_category': obj['category'],
                        'timestamp': datetime.now().isoformat(),
                        'confidence': obj['confidence']
                    }
                    events.append(event)
        
        return events
    
    def _object_in_roi(self, obj: Dict, roi_data: Dict, frame_shape: Tuple[int, int, int]) -> bool:
        """Check if object is within ROI"""
        # Simple rectangular ROI check
        if roi_data.get('type') == 'rectangle':
            obj_center = obj['center_of_mass']
            roi_rect = roi_data.get('coordinates', {})
            
            return (roi_rect.get('x', 0) <= obj_center['x'] <= roi_rect.get('x', 0) + roi_rect.get('width', 0) and
                    roi_rect.get('y', 0) <= obj_center['y'] <= roi_rect.get('y', 0) + roi_rect.get('height', 0))
        
        return False
