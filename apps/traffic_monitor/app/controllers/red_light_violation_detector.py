"""
Red Light Violation Detector for traffic monitoring in Qt application
"""

import cv2
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import datetime
import os

# Import utilities
from utils.crosswalk_utils import (
    detect_crosswalk_and_violation_line,
    draw_violation_line
)
# Import traffic light utilities
try:
    from utils.traffic_light_utils import detect_traffic_light_color, draw_traffic_light_status
    print("✅ Imported traffic light utilities in violation detector")
except ImportError:
    def detect_traffic_light_color(frame, bbox):
        return {"color": "unknown", "confidence": 0.0}
    def draw_traffic_light_status(frame, bbox, color):
        return frame
    print("⚠️ Failed to import traffic light utilities")

class RedLightViolationDetector:
    """
    Detect red light violations based on traffic light status and vehicle positions.
    
    This class integrates crosswalk/stop line detection with traffic light color
    detection to identify vehicles that cross the line during a red light.
    """
    
    def __init__(self):
        """Initialize the detector with default settings."""
        # Detection state
        self.violation_line_y = None
        self.detection_enabled = True
        self.detection_mode = "auto"  # "auto", "crosswalk", "stopline"
        
        # Track vehicles for violation detection
        self.tracked_vehicles = {}  # id -> {position_history, violation_status}
        self.violations = []
        
        # Store frames for snapshots/video clips
        self.violation_buffer = deque(maxlen=30)  # Store ~1 second of frames
        
        # Settings
        self.confidence_threshold = 0.5
        self.save_snapshots = True
        self.snapshot_dir = os.path.join(os.path.expanduser("~"), "Documents", "TrafficViolations")
        os.makedirs(self.snapshot_dir, exist_ok=True)
        
    def detect_violation_line(self, frame: np.ndarray, traffic_light_bbox: Optional[List[int]] = None) -> int:
        """
        Detect the violation line in the frame using crosswalk or stop line detection.
        
        Args:
            frame: Input video frame
            traffic_light_bbox: Optional traffic light bounding box for context
            
        Returns:
            Y-coordinate of the violation line
        """
        frame_height = frame.shape[0]
        
        try:
            # Try to detect crosswalk first if mode is auto or crosswalk
            if self.detection_mode in ["auto", "crosswalk"]:
                # Use the new function for crosswalk and violation line detection
                result_frame, crosswalk_bbox, violation_line_y, crosswalk_debug = detect_crosswalk_and_violation_line(frame)
                print(f"Crosswalk detection result: bbox={crosswalk_bbox}, vline_y={violation_line_y}")
                frame = result_frame  # Use the frame with overlays for further processing or display
                if crosswalk_bbox:
                    # Use the top of the crosswalk as the violation line
                    self.violation_line_y = crosswalk_bbox[1] - 10  # 10px before crosswalk
                    self.detection_mode = "crosswalk"  # If auto and found crosswalk, switch to crosswalk mode
                    print(f"✅ Using crosswalk for violation line at y={self.violation_line_y}")
                    return self.violation_line_y
            
            # If traffic light is detected, position line below it
            if traffic_light_bbox:
                x1, y1, x2, y2 = traffic_light_bbox
                # Position the line a bit below the traffic light
                proposed_y = y2 + int(frame_height * 0.15)  # 15% of frame height below traffic light
                # Don't place too low in the frame
                if proposed_y < frame_height * 0.85:
                    self.violation_line_y = proposed_y
                    print(f"✅ Using traffic light position for violation line at y={self.violation_line_y}")
                    return self.violation_line_y
            
            # If nothing detected, use a default position based on frame height
            self.violation_line_y = int(frame_height * 0.75)  # Default position at 75% of frame height
            print(f"ℹ️ Using default violation line at y={self.violation_line_y}")
            
            return self.violation_line_y
            
        except Exception as e:
            print(f"❌ Error in detect_violation_line: {e}")
            # Fallback
            return int(frame_height * 0.75)
    
    def process_frame(self, frame: np.ndarray, detections: List[Dict], 
                      current_traffic_light_color: str) -> Tuple[np.ndarray, List[Dict]]:
        """
        Process a frame to detect red light violations.
        
        Args:
            frame: Input video frame
            detections: List of detection dictionaries with 'class_name', 'bbox', etc.
            current_traffic_light_color: Current traffic light color ('red', 'yellow', 'green', 'unknown')
            
        Returns:
            Tuple of (annotated frame, list of violation events)
        """
        if not self.detection_enabled:
            return frame, []
            
        # Store original frame for violation buffer
        self.violation_buffer.append(frame.copy())
        
        # Annotate frame for visualization
        annotated_frame = frame.copy()
          # Get traffic light position if available
        traffic_light_bbox = None
        for det in detections:
            # Check for both 'traffic light' and class_id 9 (COCO class for traffic light)
            if det.get('class_name') == 'traffic light' or det.get('class_id') == 9:
                traffic_light_bbox = det.get('bbox')
                print(f"Found traffic light with bbox: {traffic_light_bbox}")
                break
        
        # Detect violation line if not already detected
        if self.violation_line_y is None or self.violation_line_y <= 0:
            print(f"Detecting violation line with traffic light bbox: {traffic_light_bbox}")
            try:
                self.violation_line_y = self.detect_violation_line(frame, traffic_light_bbox)
                print(f"Successfully detected violation line at y={self.violation_line_y}")
            except Exception as e:
                print(f"❌ Error detecting violation line: {e}")
                # Fallback to default position
                self.violation_line_y = int(frame.shape[0] * 0.75)
                print(f"Using default violation line at y={self.violation_line_y}")
        
        # Draw violation line with enhanced visualization
        # Handle both string and dictionary return formats for compatibility
        if isinstance(current_traffic_light_color, dict):
            is_red = current_traffic_light_color.get("color") == "red"
            confidence = current_traffic_light_color.get("confidence", 0.0)
            confidence_text = f" (Conf: {confidence:.2f})"
        else:
            is_red = current_traffic_light_color == "red"
            confidence_text = ""
            
        line_color = (0, 0, 255) if is_red else (0, 255, 0)
        annotated_frame = draw_violation_line(
            annotated_frame, 
            self.violation_line_y, 
            line_color,
            f"VIOLATION LINE - {current_traffic_light_color.get('color', current_traffic_light_color).upper()}{confidence_text}"
        )
        
        # --- DEBUG: Always draw a hardcoded violation line for testing ---
        if self.violation_line_y is None or self.violation_line_y <= 0:
            frame_height = frame.shape[0]
            # Example: draw at 75% of frame height
            self.violation_line_y = int(frame_height * 0.75)
            print(f"[DEBUG] Drawing fallback violation line at y={self.violation_line_y}")
            import cv2
            cv2.line(annotated_frame, (0, self.violation_line_y), (frame.shape[1], self.violation_line_y), (0, 0, 255), 3)
        
        # Track vehicles and check for violations
        violations_this_frame = []
        
        # Process each detection
        for detection in detections:
            class_name = detection.get('class_name')
            confidence = detection.get('confidence', 0.0)
            bbox = detection.get('bbox')
            track_id = detection.get('track_id', -1)
              # Only process vehicles with sufficient confidence
            # Include both class_name and class_id checks for better compatibility
            is_vehicle = (class_name in ['car', 'truck', 'bus', 'motorcycle'] or 
                         detection.get('class_id') in [2, 3, 5, 7]) # COCO classes for vehicles
            
            if (is_vehicle and 
                confidence >= self.confidence_threshold and
                bbox is not None):
                  # Use object id or generate temporary one if tracking id not available
                if track_id < 0:
                    # Generate a temporary ID based on position and size
                    x1, y1, x2, y2 = bbox
                    temp_id = f"temp_{int((x1+x2)/2)}_{int((y1+y2)/2)}_{int((x2-x1)*(y2-y1))}"
                    track_id = temp_id
                
                # Initialize tracking if this is a new vehicle
                if track_id not in self.tracked_vehicles:
                    print(f"🚗 New vehicle detected with ID: {track_id}")
                    self.tracked_vehicles[track_id] = {
                        'positions': deque(maxlen=30),  # Store ~1 second of positions
                        'violated': False,
                        'first_detected': time.time()
                    }
                
                # Update position history
                vehicle_data = self.tracked_vehicles[track_id]
                vehicle_data['positions'].append((bbox, time.time()))
                
                # Check for violation only if traffic light is red
                # Handle both string and dictionary return formats
                is_red = False
                if isinstance(current_traffic_light_color, dict):
                    is_red = current_traffic_light_color.get("color") == "red"
                    confidence = current_traffic_light_color.get("confidence", 0.0)
                    # Only consider red if confidence is above threshold
                    is_red = is_red and confidence >= 0.4
                else:
                    is_red = current_traffic_light_color == "red"
                
                if (is_red and 
                    not vehicle_data['violated'] and 
                    check_vehicle_violation(bbox, self.violation_line_y)):
                    
                    # Mark as violated
                    vehicle_data['violated'] = True
                    
                    # Create violation record with enhanced information
                    violation = {
                        'id': len(self.violations) + 1,
                        'track_id': track_id,
                        'timestamp': datetime.datetime.now(),
                        'vehicle_type': class_name,
                        'confidence': detection.get('confidence', 0.0),
                        'bbox': bbox,
                        'violation_type': 'red_light',
                        'snapshot_path': None
                    }
                    
                    # Add traffic light information if available
                    if isinstance(current_traffic_light_color, dict):
                        violation['traffic_light'] = {
                            'color': current_traffic_light_color.get('color', 'red'),
                            'confidence': current_traffic_light_color.get('confidence', 0.0)
                        }
                    else:
                        violation['traffic_light'] = {
                            'color': current_traffic_light_color,
                            'confidence': 1.0
                        }
                    
                    # Save snapshot if enabled
                    if self.save_snapshots:
                        snapshot_path = os.path.join(
                            self.snapshot_dir, 
                            f"violation_{violation['id']}_{int(time.time())}.jpg"
                        )
                        cv2.imwrite(snapshot_path, frame)
                        violation['snapshot_path'] = snapshot_path
                    
                    # Add to violations list
                    self.violations.append(violation)
                    violations_this_frame.append(violation)
                    
                    # Draw violation box
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(
                        annotated_frame,
                        f"RED LIGHT VIOLATION #{violation['id']}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )
        
        # Clean up old tracked vehicles to prevent memory leaks
        current_time = time.time()
        old_ids = [tid for tid, data in self.tracked_vehicles.items() 
                  if current_time - data['first_detected'] > 30]  # Remove after 30 seconds
        for tid in old_ids:
            del self.tracked_vehicles[tid]
            
        return annotated_frame, violations_this_frame
    
    def reset(self):
        """Reset the detector state."""
        self.violation_line_y = None
        self.tracked_vehicles = {}
        # Keep violations history
        
    def get_violations(self) -> List[Dict]:
        """
        Get all detected violations.
        
        Returns:
            List of violation dictionaries
        """
        return self.violations
        
    def clear_violations(self):
        """Clear all violation records."""
        self.violations = []
