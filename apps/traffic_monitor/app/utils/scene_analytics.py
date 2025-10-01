
from PySide6.QtCore import QObject, Signal
import time
import numpy as np

class SceneAnalyticsAdapter(QObject):
    """
    Scene analytics adapter for traffic monitoring.
    Provides object detection, scene analysis, and ROI event processing.
    """
    object_detected = Signal(dict)  # Emitted when an object is detected
    scene_analytics_updated = Signal(dict)  # Emitted when analytics data is updated
    roi_event_detected = Signal(dict)  # Emitted when an ROI event is detected

    def __init__(self, camera_id=None):
        super().__init__()
        self.camera_id = camera_id or "default_camera"
        
        # Analytics tracking
        self.frame_count = 0
        self.object_counts = {}
        self.roi_zones = []  # Define regions of interest
        
        # Performance tracking
        self.last_update_time = time.time()
        
        print(f"[SCENE ANALYTICS] Initialized for camera: {self.camera_id}")

    def process_frame(self, frame, detections):
        """
        Process a frame and detections, emit analytics signals as needed.
        
        Args:
            frame: The current video frame (numpy array)
            detections: List of detection dicts
            
        Returns:
            dict: Analytics results
        """
        try:
            self.frame_count += 1
            current_time = time.time()
            
            # Count objects by class
            object_counts = {}
            vehicle_count = 0
            person_count = 0
            traffic_light_count = 0
            
            for detection in detections:
                class_name = detection.get('class_name', 'unknown')
                confidence = detection.get('confidence', 0.0)
                
                # Only count high-confidence detections
                if confidence > 0.3:
                    object_counts[class_name] = object_counts.get(class_name, 0) + 1
                    
                    # Categorize objects
                    if class_name in ['car', 'truck', 'bus', 'motorcycle', 'bicycle']:
                        vehicle_count += 1
                        
                        # Emit object detection signal for vehicles
                        self.object_detected.emit({
                            'camera_id': self.camera_id,
                            'object_type': class_name,
                            'confidence': confidence,
                            'bbox': detection.get('bbox', []),
                            'timestamp': current_time
                        })
                        
                    elif class_name == 'person':
                        person_count += 1
                    elif class_name == 'traffic light':
                        traffic_light_count += 1
            
            # Update object counts tracking
            self.object_counts = object_counts
            
            # Prepare analytics data
            analytics_data = {
                'camera_id': self.camera_id,
                'frame_count': self.frame_count,
                'timestamp': current_time,
                'total_objects': len(detections),
                'vehicle_count': vehicle_count,
                'person_count': person_count,
                'traffic_light_count': traffic_light_count,
                'object_counts': object_counts,
                'fps': 1.0 / (current_time - self.last_update_time) if current_time > self.last_update_time else 0
            }
            
            # Emit analytics update every 30 frames or every 5 seconds
            if self.frame_count % 30 == 0 or (current_time - self.last_update_time) > 5.0:
                self.scene_analytics_updated.emit(analytics_data)
                self.last_update_time = current_time
            
            # Check for ROI events (can be enhanced based on specific requirements)
            self._check_roi_events(frame, detections, current_time)
            
            return analytics_data
            
        except Exception as e:
            print(f"[SCENE ANALYTICS] Error processing frame: {e}")
            return {}
    
    def _check_roi_events(self, frame, detections, current_time):
        """
        Check for events in regions of interest.
        This can be customized based on specific monitoring requirements.
        """
        try:
            # Example: Check for high vehicle density
            vehicle_detections = [d for d in detections if d.get('class_name') in ['car', 'truck', 'bus']]
            
            if len(vehicle_detections) > 5:  # More than 5 vehicles detected
                self.roi_event_detected.emit({
                    'camera_id': self.camera_id,
                    'event_type': 'high_vehicle_density',
                    'vehicle_count': len(vehicle_detections),
                    'timestamp': current_time,
                    'severity': 'medium'
                })
            
            # Example: Check for pedestrians near vehicles
            person_detections = [d for d in detections if d.get('class_name') == 'person']
            if person_detections and vehicle_detections:
                # Simple proximity check (this could be enhanced with actual distance calculation)
                self.roi_event_detected.emit({
                    'camera_id': self.camera_id,
                    'event_type': 'pedestrian_vehicle_proximity',
                    'person_count': len(person_detections),
                    'vehicle_count': len(vehicle_detections),
                    'timestamp': current_time,
                    'severity': 'low'
                })
                
        except Exception as e:
            print(f"[SCENE ANALYTICS] Error checking ROI events: {e}")
    
    def add_roi_zone(self, zone_name, coordinates, zone_type='detection'):
        """
        Add a region of interest for monitoring.
        
        Args:
            zone_name: Name of the zone
            coordinates: List of (x, y) points defining the zone
            zone_type: Type of zone ('detection', 'violation', 'counting')
        """
        roi_zone = {
            'name': zone_name,
            'coordinates': coordinates,
            'type': zone_type,
            'events': []
        }
        self.roi_zones.append(roi_zone)
        print(f"[SCENE ANALYTICS] Added ROI zone: {zone_name} ({zone_type})")
    
    def get_analytics_summary(self):
        """
        Get a summary of current analytics data.
        
        Returns:
            dict: Summary of analytics data
        """
        return {
            'camera_id': self.camera_id,
            'total_frames_processed': self.frame_count,
            'current_object_counts': self.object_counts,
            'roi_zones_count': len(self.roi_zones),
            'last_update': self.last_update_time
        }