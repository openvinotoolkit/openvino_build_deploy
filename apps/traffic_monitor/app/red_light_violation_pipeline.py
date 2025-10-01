"""
Red Light Violation Detection Pipeline (Traditional CV, Rule-Based)
Integrates with detection and violation modules.
"""
import cv2
import numpy as np

class RedLightViolationPipeline:
    """
    Pipeline for detecting red light violations using computer vision.
    Integrates traffic light detection and vehicle tracking to identify violations.
    """
    def __init__(self, debug=False):
        """
        Initialize the pipeline.

        Args:
            debug (bool): If True, enables debug output for tracking and violation detection.
        """
        self.track_history = {}  # track_id -> list of (center, frame_idx)
        self.violation_events = []
        self.violation_line_y = None
        self.debug = debug
        self.last_known_light = 'unknown'

    def detect_violation_line(self, frame, traffic_light_bbox=None, crosswalk_bbox=None):
        """
        Detect the violation line (stop line or crosswalk) in the frame.
        Uses multiple approaches to find the most reliable stop line.
        
        Args:
            frame: Input video frame
            traffic_light_bbox: Optional bbox of detected traffic light [x1, y1, x2, y2]
            crosswalk_bbox: Optional bbox of detected crosswalk [x1, y1, x2, y2]
            
        Returns: 
            y-coordinate of the violation line
        """
        # Method 1: Use provided crosswalk if available
        if crosswalk_bbox is not None and len(crosswalk_bbox) == 4:
            self.violation_line_y = int(crosswalk_bbox[1]) - 15  # 15px before crosswalk
            if self.debug:
                print(f"Using provided crosswalk bbox, line_y={self.violation_line_y}")
            return self.violation_line_y

        # Method 2: Try to detect stop lines/crosswalk stripes
        height, width = frame.shape[:2]
        roi_height = int(height * 0.4)  # Look at bottom 40% of image for stop lines
        roi_y = height - roi_height
        roi = frame[roi_y:height, 0:width]
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding to handle varying lighting conditions
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 15, -2
        )
        
        # Enhance horizontal lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours based on width, aspect ratio, and location
        stop_line_candidates = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / max(h, 1)
            normalized_width = w / width
            
            # Good stop line: wide, thin, in lower part of ROI
            if (aspect_ratio > 5 and 
                normalized_width > 0.3 and 
                h < 15 and
                y > roi_height * 0.5):
                # y coordinate in full frame
                abs_y = y + roi_y  
                stop_line_candidates.append((abs_y, w))
        
        # Choose best stop line based on width and position
        if stop_line_candidates:
            # Sort by width (largest first)
            stop_line_candidates.sort(key=lambda x: x[1], reverse=True)
            self.violation_line_y = stop_line_candidates[0][0]
            if self.debug:
                print(f"Found stop line with CV, line_y={self.violation_line_y}")
            return self.violation_line_y
            
        # Method 3: If traffic light is detected, place line at reasonable distance
        if traffic_light_bbox is not None:
            # Position violation line at a reasonable distance from traffic light
            # Typically stop lines are below traffic lights
            traffic_light_bottom = traffic_light_bbox[3]
            traffic_light_height = traffic_light_bbox[3] - traffic_light_bbox[1]
            
            # Place line at approximately 4-6 times the height of traffic light below it
            estimated_distance = min(5 * traffic_light_height, height * 0.3)
            self.violation_line_y = min(int(traffic_light_bottom + estimated_distance), height - 20)
            
            if self.debug:
                print(f"Estimated line from traffic light position, line_y={self.violation_line_y}")
            return self.violation_line_y
        
        # Method 4: Fallback to fixed position in frame
        self.violation_line_y = int(height * 0.75)  # Lower 1/4 of the frame
        if self.debug:
            print(f"Using fallback position, line_y={self.violation_line_y}")
            
        return self.violation_line_y
        
    def detect_traffic_light_color(self, frame, traffic_light_bbox):
        """
        Detect the color of a traffic light using computer vision.
        
        Args:
            frame: Input video frame
            traffic_light_bbox: Bbox of detected traffic light [x1, y1, x2, y2]
            
        Returns:
            String: 'red', 'yellow', 'green', or 'unknown'
        """
        if traffic_light_bbox is None or len(traffic_light_bbox) != 4:
            return 'unknown'
        
        x1, y1, x2, y2 = traffic_light_bbox
        
        # Ensure bbox is within frame
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(0, min(x2, w-1))
        y2 = max(0, min(y2, h-1))
        
        if x2 <= x1 or y2 <= y1:
            return 'unknown'
        
        # Extract traffic light region
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return 'unknown'
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for traffic lights
        lower_red1 = np.array([0, 70, 60])
        upper_red1 = np.array([15, 255, 255])
        lower_red2 = np.array([160, 70, 60])  # Red wraps around in HSV
        upper_red2 = np.array([180, 255, 255])
        
        lower_yellow = np.array([15, 70, 70])
        upper_yellow = np.array([40, 255, 255])
        
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([95, 255, 255])
        
        # Create masks for each color
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        
        # Count pixels of each color
        red_pixels = cv2.countNonZero(mask_red)
        yellow_pixels = cv2.countNonZero(mask_yellow)
        green_pixels = cv2.countNonZero(mask_green)
        
        # Get the most dominant color
        max_pixels = max(red_pixels, yellow_pixels, green_pixels)
        min_required = 5  # Minimum number of pixels to confidently identify a color (reduced from 10)
        
        # Print debug info
        roi_area = roi.shape[0] * roi.shape[1] if roi.size > 0 else 1
        print(f"🔍 Traffic light color pixels: Red={red_pixels}/{roi_area}, Yellow={yellow_pixels}/{roi_area}, Green={green_pixels}/{roi_area}")
        
        if max_pixels < min_required:
            print("⚠️ No color has enough pixels, returning red as fallback")
            return 'red'  # safer to default to red
        elif red_pixels == max_pixels:
            return 'red'
        elif yellow_pixels == max_pixels:
            return 'yellow'
        elif green_pixels == max_pixels:
            return 'green'
        else:
            return 'red'  # safer to default to red

    def update_tracks(self, vehicle_detections, frame_idx):
        """
        Update track history with new vehicle detections.
        vehicle_detections: list of dicts with 'track_id' and 'bbox'
        """
        for det in vehicle_detections:
            track_id = det['track_id']
            x1, y1, x2, y2 = det['bbox']
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            if track_id not in self.track_history:
                self.track_history[track_id] = []
            self.track_history[track_id].append((center, frame_idx))
            # Keep only last 10 points
            self.track_history[track_id] = self.track_history[track_id][-10:]

    def is_moving_forward(self, track_id):
        """
        Returns True if the vehicle is moving forward (Y increasing).
        """
        history = self.track_history.get(track_id, [])
        if len(history) < 3:
            return False
        ys = [pt[0][1] for pt in history[-5:]]
        return ys[-1] - ys[0] > 15  # moved at least 15px forward

    def check_violations(self, vehicle_detections, traffic_light_state, frame_idx, timestamp):
        """
        For each vehicle, check if it crosses the violation line while the light is red.
        
        Args:
            vehicle_detections: List of dicts with 'track_id' and 'bbox'
            traffic_light_state: String 'red', 'yellow', 'green', or 'unknown'
            frame_idx: Current frame index
            timestamp: Current frame timestamp
            
        Returns:
            List of violation dictionaries
        """
        if self.violation_line_y is None:
            return []
            
        violations = []
        
        # Only check for violations if light is red or we're sure it's not green
        is_red_light_condition = (traffic_light_state == 'red' or 
                                 (traffic_light_state != 'green' and 
                                  traffic_light_state != 'yellow' and 
                                  self.last_known_light == 'red'))
                                 
        if not is_red_light_condition:
            # Update last known definitive state
            if traffic_light_state in ['red', 'yellow', 'green']:
                self.last_known_light = traffic_light_state
            return []
            
        # Check each vehicle
        for det in vehicle_detections:
            if not isinstance(det, dict):
                continue
                
            track_id = det.get('track_id')
            bbox = det.get('bbox')
            
            if track_id is None or bbox is None or len(bbox) != 4:
                continue
                
            x1, y1, x2, y2 = bbox
            
            # Check if the vehicle is at or below the violation line
            vehicle_bottom = y2
            
            # Get vehicle track history
            track_history = self.track_history.get(track_id, [])
            
            # Only consider vehicles with sufficient history
            if len(track_history) < 3:
                continue
            
            # Check if vehicle is crossing the line AND moving forward
            crossing_line = vehicle_bottom > self.violation_line_y
            moving_forward = self.is_moving_forward(track_id)
            
            # Check if this violation was already detected
            already_detected = False
            for v in self.violation_events:
                if v['track_id'] == track_id and frame_idx - v['frame_idx'] < 30:
                    already_detected = True
                    break
            
            if crossing_line and moving_forward and not already_detected:
                # Record violation
                violation = {
                    'type': 'red_light_violation',
                    'track_id': track_id,
                    'frame_idx': frame_idx,
                    'timestamp': timestamp,
                    'vehicle_bbox': bbox,
                    'violation_line_y': self.violation_line_y,
                    'traffic_light_state': traffic_light_state,
                    'confidence': 0.9,
                    'description': f'Vehicle ran red light at frame {frame_idx}'
                }
                
                violations.append(violation)
                self.violation_events.append(violation)
                
        return violations

    def draw_debug(self, frame, vehicle_detections, traffic_light_bbox, traffic_light_state):
        """
        Draw overlays for debugging: vehicle boxes, traffic light, violation line, violations.
        
        Args:
            frame: Input video frame
            vehicle_detections: List of dicts with vehicle detections
            traffic_light_bbox: Bbox of detected traffic light [x1, y1, x2, y2]
            traffic_light_state: String state of traffic light
            
        Returns:
            Annotated frame with debugging visualizations
        """
        # Create a copy to avoid modifying the original frame
        out = frame.copy()
        h, w = out.shape[:2]
        
        # Draw violation line
        if self.violation_line_y is not None:
            cv2.line(out, (0, self.violation_line_y), (w, self.violation_line_y), 
                    (0, 0, 255), 2)
            cv2.putText(out, "STOP LINE", (10, self.violation_line_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Draw traffic light
        if traffic_light_bbox is not None:
            x1, y1, x2, y2 = traffic_light_bbox
            
            # Color based on traffic light state
            if traffic_light_state == 'red':
                color = (0, 0, 255)  # Red (BGR)
            elif traffic_light_state == 'yellow':
                color = (0, 255, 255)  # Yellow (BGR)
            elif traffic_light_state == 'green':
                color = (0, 255, 0)  # Green (BGR)
            else:
                color = (255, 255, 255)  # White (BGR) for unknown
                
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, f"Traffic Light: {traffic_light_state}", 
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw vehicles and violations
        for det in vehicle_detections:
            if not isinstance(det, dict) or 'bbox' not in det:
                continue
                
            bbox = det['bbox']
            if len(bbox) != 4:
                continue
                
            x1, y1, x2, y2 = bbox
            track_id = det.get('track_id', '?')
            
            # Draw vehicle box
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Draw ID and center point
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            cv2.circle(out, center, 4, (0, 255, 255), -1)
            cv2.putText(out, f"ID:{track_id}", (x1, y1 - 5), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Check if this vehicle has a violation
            is_violating = False
            for violation in self.violation_events:
                if violation.get('track_id') == track_id:
                    is_violating = True
                    break
            
            # If vehicle is crossing line, check if it's a violation
            if y2 > self.violation_line_y:
                if traffic_light_state == 'red' and is_violating:
                    cv2.putText(out, "VIOLATION", (x1, y2 + 25), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                    # Draw a prominent red box around the violating vehicle
                    cv2.rectangle(out, (x1-5, y1-5), (x2+5, y2+5), (0, 0, 255), 3)
            
            # Draw track history
            track_history = self.track_history.get(track_id, [])
            if len(track_history) > 1:
                points = [pos for pos, _ in track_history]
                for i in range(1, len(points)):
                    # Gradient color from blue to red based on recency
                    alpha = i / len(points)
                    color = (int(255 * (1-alpha)), 0, int(255 * alpha))
                    cv2.line(out, points[i-1], points[i], color, 2)
        
        # Draw statistics
        cv2.putText(out, f"Total violations: {len(self.violation_events)}", 
                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Add timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(out, timestamp, (w - 230, h - 20), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return out

    def reset(self):
        """
        Reset the pipeline state, clearing all tracks and violation events.
        """
        self.track_history.clear()
        self.violation_events.clear()
        self.violation_line_y = None
