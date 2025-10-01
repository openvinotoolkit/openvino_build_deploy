"""
Enhanced ByteTrack implementation with all requested improvements:
1. Kalman Filter motion model for better prediction
2. Noise filtering and position smoothing (EMA)
3. Direction awareness with velocity gating
4. Geometric context with ROI filtering
5. Adaptive confidence thresholds
6. Hungarian algorithm for optimal associations
7. Adaptive track lifecycle management
8. Traffic-specific features (heading, speed estimation)
"""

import numpy as np
import cv2
import time
from collections import defaultdict, deque
from typing import List, Dict, Any, Tuple, Optional, Union
from scipy.optimize import linear_sum_assignment
from scipy.signal import savgol_filter
import math


class KalmanFilter:
    """
    2D Kalman Filter for constant velocity model
    State: [x, y, vx, vy] - position and velocity
    """
    
    def __init__(self, bbox):
        """Initialize Kalman filter with detection bbox"""
        # State vector: [x, y, vx, vy]
        x = (bbox[0] + bbox[2]) / 2.0  # center x
        y = (bbox[1] + bbox[3]) / 2.0  # center y
        self.x = np.array([x, y, 0.0, 0.0])
        
        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, 1, 0],  # x = x + vx
            [0, 1, 0, 1],  # y = y + vy
            [0, 0, 1, 0],  # vx = vx
            [0, 0, 0, 1]   # vy = vy
        ])
        
        # Measurement matrix (we observe position only)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Process noise covariance
        self.Q = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]) * 0.1
        
        # Measurement noise covariance
        self.R = np.array([
            [5, 0],
            [0, 5]
        ])
        
        # Initial covariance
        self.P = np.eye(4) * 1000
        
        # Store bbox dimensions
        self.w = bbox[2] - bbox[0]
        self.h = bbox[3] - bbox[1]
    
    def predict(self):
        """Predict next state"""
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.get_bbox()
    
    def update(self, bbox):
        """Update with measurement"""
        # Extract center position
        z = np.array([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0])
        
        # Update bbox dimensions (exponential moving average)
        alpha = 0.3
        self.w = alpha * (bbox[2] - bbox[0]) + (1 - alpha) * self.w
        self.h = alpha * (bbox[3] - bbox[1]) + (1 - alpha) * self.h
        
        # Kalman update
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)
    
    def get_bbox(self):
        """Get current bounding box"""
        x, y = self.x[0], self.x[1]
        return np.array([x - self.w/2, y - self.h/2, x + self.w/2, y + self.h/2])
    
    def get_velocity(self):
        """Get current velocity"""
        return self.x[2:4]


class AdaptiveThresholds:
    """Adaptive threshold management for confidence scores"""
    
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.confidence_history = deque(maxlen=window_size)
        self.high_thresh = 0.6
        self.low_thresh = 0.1
        
    def update(self, confidences):
        """Update thresholds based on confidence distribution"""
        self.confidence_history.extend(confidences)
        
        if len(self.confidence_history) >= 20:  # Minimum samples
            conf_array = np.array(list(self.confidence_history))
            mean_conf = np.mean(conf_array)
            std_conf = np.std(conf_array)
            
            # Adaptive thresholds based on distribution
            self.high_thresh = max(0.4, min(0.8, mean_conf + 0.5 * std_conf))
            self.low_thresh = max(0.05, min(0.3, mean_conf - std_conf))
    
    def get_thresholds(self):
        return self.high_thresh, self.low_thresh


class GeometricContext:
    """Geometric context for ROI filtering and direction constraints"""
    
    def __init__(self):
        self.roi_polygons = []
        self.lane_directions = {}  # lane_id -> expected_direction_vector
        self.enabled = False
    
    def add_roi(self, polygon_points):
        """Add a region of interest polygon"""
        self.roi_polygons.append(np.array(polygon_points))
        self.enabled = True
    
    def add_lane_direction(self, lane_id, direction_vector):
        """Add expected direction for a lane"""
        self.lane_directions[lane_id] = np.array(direction_vector)
    
    def is_inside_roi(self, bbox):
        """Check if bounding box center is inside any ROI"""
        if not self.enabled:
            return True
            
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        point = np.array([center_x, center_y])
        
        for polygon in self.roi_polygons:
            if cv2.pointPolygonTest(polygon, tuple(point), False) >= 0:
                return True
        return False
    
    def check_direction_consistency(self, velocity, lane_id=None):
        """Check if velocity is consistent with lane direction"""
        if not self.lane_directions or lane_id not in self.lane_directions:
            return True
            
        expected_dir = self.lane_directions[lane_id]
        velocity_norm = np.linalg.norm(velocity)
        
        if velocity_norm < 1e-6:  # Stationary
            return True
            
        # Cosine similarity between velocity and expected direction
        cos_sim = np.dot(velocity, expected_dir) / (velocity_norm * np.linalg.norm(expected_dir))
        return cos_sim > 0.3  # Allow some deviation


class EnhancedTrack:
    """Enhanced track with Kalman filter, noise filtering, and traffic features"""
    
    def __init__(self, detection, track_id, frame_id=0):
        self.track_id = track_id
        self.frame_id = frame_id
        self.start_frame = frame_id
        
        # Kalman filter for motion prediction
        bbox = detection[:4]
        self.kalman = KalmanFilter(bbox)
        
        # Track properties
        self.score = detection[4] if len(detection) > 4 else 0.5
        self.class_id = int(detection[5]) if len(detection) > 5 else 0
        
        # State management
        self.state = 'NEW'  # NEW -> TRACKED -> LOST -> REMOVED
        self.time_since_update = 0
        self.hits = 1
        self.age = 1
        self.tracklet_len = 1
        
        # Position and trajectory tracking
        self.raw_positions = deque(maxlen=50)  # Raw detector positions
        self.smooth_positions = deque(maxlen=50)  # Smoothed positions
        self.velocity_history = deque(maxlen=10)
        
        # Initialize with current detection
        self.raw_positions.append(bbox.copy())
        self.smooth_positions.append(bbox.copy())
        self.current_bbox = bbox.copy()
        
        # Appearance features (simple color histogram)
        self.appearance_features = None
        
        # Traffic-specific features
        self.heading_history = deque(maxlen=10)
        self.speed_history = deque(maxlen=10)
        
        # Noise filtering parameters
        self.ema_alpha = 0.3  # EMA smoothing factor
        
    def predict(self):
        """Predict next state using Kalman filter"""
        predicted_bbox = self.kalman.predict()
        self.age += 1
        self.time_since_update += 1
        return predicted_bbox
    
    def update(self, detection, frame=None):
        """Update track with new detection"""
        bbox = detection[:4]
        self.score = detection[4] if len(detection) > 4 else self.score
        
        # Update Kalman filter
        self.kalman.update(bbox)
        
        # Store raw position
        self.raw_positions.append(bbox.copy())
        
        # Apply EMA smoothing
        if len(self.smooth_positions) > 0:
            last_smooth = self.smooth_positions[-1]
            smooth_bbox = self.ema_alpha * bbox + (1 - self.ema_alpha) * last_smooth
        else:
            smooth_bbox = bbox.copy()
        
        self.smooth_positions.append(smooth_bbox)
        self.current_bbox = smooth_bbox
        
        # Update velocity
        velocity = self.kalman.get_velocity()
        self.velocity_history.append(velocity)
        
        # Update traffic features
        self._update_heading()
        self._update_speed()
        
        # Update appearance if frame provided
        if frame is not None:
            self._update_appearance(frame, bbox)
        
        # Update state
        self.hits += 1
        self.time_since_update = 0
        self.tracklet_len += 1
        
        if self.state == 'NEW' and self.hits >= 3:
            self.state = 'TRACKED'
    
    def _update_heading(self):
        """Update heading estimation from velocity"""
        if len(self.velocity_history) > 0:
            velocity = self.velocity_history[-1]
            if np.linalg.norm(velocity) > 1.0:  # Moving
                heading = math.atan2(velocity[1], velocity[0])
                self.heading_history.append(heading)
    
    def _update_speed(self):
        """Update speed estimation"""
        if len(self.velocity_history) > 0:
            velocity = self.velocity_history[-1]
            speed = np.linalg.norm(velocity)  # pixels/frame
            self.speed_history.append(speed)
    
    def _update_appearance(self, frame, bbox):
        """Update appearance features (simple color histogram)"""
        try:
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            if x2 > x1 and y2 > y1:
                roi = frame[y1:y2, x1:x2]
                # Simple HSV histogram
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
                self.appearance_features = cv2.normalize(hist, hist).flatten()
        except:
            pass
    
    def get_velocity(self):
        """Get smoothed velocity"""
        if len(self.velocity_history) >= 3:
            # Use median of recent velocities for robustness
            recent_vels = list(self.velocity_history)[-3:]
            return np.median(recent_vels, axis=0)
        elif len(self.velocity_history) > 0:
            return self.velocity_history[-1]
        return np.array([0.0, 0.0])
    
    def get_heading(self):
        """Get current heading in radians"""
        if len(self.heading_history) > 0:
            return self.heading_history[-1]
        return 0.0
    
    def get_speed(self):
        """Get current speed in pixels/frame"""
        if len(self.speed_history) > 0:
            return self.speed_history[-1]
        return 0.0
    
    def is_confirmed(self):
        """Check if track is confirmed"""
        return self.state == 'TRACKED'
    
    def mark_missed(self):
        """Mark track as missed"""
        self.time_since_update += 1
        if self.state == 'TRACKED' and self.time_since_update > 1:
            self.state = 'LOST'
    
    def should_remove(self, max_age):
        """Check if track should be removed"""
        # Adaptive removal based on track stability
        if self.state == 'NEW':
            return self.time_since_update > 3  # Remove unstable new tracks quickly
        elif self.state == 'TRACKED':
            # Stable tracks get longer recovery time
            recovery_time = min(max_age, max(10, self.tracklet_len // 3))
            return self.time_since_update > recovery_time
        else:  # LOST
            return self.time_since_update > max_age
    
    def to_dict(self):
        """Convert to dictionary format"""
        return {
            'id': self.track_id,
            'bbox': [float(x) for x in self.current_bbox],
            'confidence': float(self.score),
            'class_id': int(self.class_id),
            'velocity': [float(x) for x in self.get_velocity()],
            'heading': float(self.get_heading()),
            'speed': float(self.get_speed()),
            'state': self.state,
            'age': self.age,
            'hits': self.hits
        }


class EnhancedBYTETracker:
    """
    Enhanced ByteTrack with all requested improvements:
    - Kalman filter motion model
    - Noise filtering and smoothing
    - Direction awareness
    - Geometric context
    - Adaptive thresholds
    - Hungarian algorithm
    - Adaptive track lifecycle
    """
    
    def __init__(
        self,
        track_thresh=0.5,
        track_buffer=30,
        match_thresh=0.8,
        track_high_thresh=0.6,
        track_low_thresh=0.1,
        frame_rate=30,
        use_kalman=True,
        use_appearance=False,
        direction_thresh=0.3
    ):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.frame_rate = frame_rate
        self.use_kalman = use_kalman
        self.use_appearance = use_appearance
        self.direction_thresh = direction_thresh
        
        # Track management
        self.tracked_tracks = []
        self.lost_tracks = []
        self.removed_tracks = []
        
        self.frame_id = 0
        self.track_id_count = 0
        
        # Enhanced features
        self.adaptive_thresh = AdaptiveThresholds()
        self.geometric_context = GeometricContext()
        
        # Statistics
        self.stats = {
            'total_tracks': 0,
            'active_tracks': 0,
            'lost_tracks': 0,
            'id_switches': 0
        }
    
    def update(self, detections, frame=None):
        """
        Update tracker with new detections
        
        Args:
            detections: List of detection dicts or arrays
            frame: BGR image (optional, for appearance features)
            
        Returns:
            List of track dictionaries
        """
        self.frame_id += 1
        
        # Convert detections to consistent format
        det_arrays = self._convert_detections(detections)
        
        if len(det_arrays) == 0:
            # No detections - predict all tracks
            for track in self.tracked_tracks:
                track.predict()
                track.mark_missed()
            
            # Move tracks to lost if needed
            self._update_track_states()
            return self._get_output_tracks()
        
        # Update adaptive thresholds
        confidences = [det[4] for det in det_arrays if len(det) > 4]
        if confidences:
            self.adaptive_thresh.update(confidences)
        
        high_thresh, low_thresh = self.adaptive_thresh.get_thresholds()
        
        # Predict existing tracks
        for track in self.tracked_tracks:
            track.predict()
        
        # Separate detections by confidence
        high_conf_dets = [det for det in det_arrays if det[4] >= high_thresh]
        low_conf_dets = [det for det in det_arrays if low_thresh <= det[4] < high_thresh]
        
        # Stage 1: Associate high confidence detections with tracked tracks
        matches_a, unmatched_tracks_a, unmatched_high_dets = self._associate(
            high_conf_dets, self.tracked_tracks, self.match_thresh, frame
        )
        
        # Update matched tracks
        for track_idx, det_idx in matches_a:
            self.tracked_tracks[track_idx].update(high_conf_dets[det_idx], frame)
        
        # Stage 2: Associate remaining high confidence detections with lost tracks
        matches_b, unmatched_tracks_b, unmatched_high_dets = self._associate(
            [high_conf_dets[i] for i in unmatched_high_dets],
            self.lost_tracks,
            self.match_thresh,
            frame
        )
        
        # Recover lost tracks
        for track_idx, det_idx in matches_b:
            track = self.lost_tracks[track_idx]
            track.update([high_conf_dets[i] for i in unmatched_high_dets][det_idx], frame)
            track.state = 'TRACKED'
            self.tracked_tracks.append(track)
        
        # Remove recovered tracks from lost list
        for track_idx in sorted([m[0] for m in matches_b], reverse=True):
            self.lost_tracks.pop(track_idx)
        
        # Stage 3: Associate low confidence detections with unmatched tracks
        unmatched_tracks = [self.tracked_tracks[i] for i in unmatched_tracks_a]
        matches_c, unmatched_tracks_c, unmatched_low_dets = self._associate(
            low_conf_dets, unmatched_tracks, 0.5, frame  # Lower threshold for low conf
        )
        
        # Update tracks matched with low confidence detections
        track_mapping = {i: unmatched_tracks_a[i] for i in range(len(unmatched_tracks))}
        for track_idx, det_idx in matches_c:
            original_idx = track_mapping[track_idx]
            self.tracked_tracks[original_idx].update(low_conf_dets[det_idx], frame)
        
        # Mark unmatched tracks as missed
        all_unmatched = set(unmatched_tracks_a) - set(track_mapping[m[0]] for m in matches_c)
        for track_idx in all_unmatched:
            self.tracked_tracks[track_idx].mark_missed()
        
        # Create new tracks from unmatched high confidence detections
        remaining_high_dets = [high_conf_dets[i] for i in unmatched_high_dets if i not in [m[1] for m in matches_b]]
        for det in remaining_high_dets:
            if self.geometric_context.is_inside_roi(det[:4]):
                self._create_new_track(det)
        
        # Update track states and cleanup
        self._update_track_states()
        
        # Update statistics
        self._update_stats()
        
        return self._get_output_tracks()
    
    def _convert_detections(self, detections):
        """Convert various detection formats to [x1, y1, x2, y2, conf, class_id]"""
        det_arrays = []
        
        for det in detections:
            try:
                if isinstance(det, dict):
                    bbox = det.get('bbox', [])
                    conf = det.get('confidence', 0.5)
                    class_id = det.get('class_id', 0)
                    
                    if len(bbox) == 4:
                        det_arrays.append([bbox[0], bbox[1], bbox[2], bbox[3], conf, class_id])
                        
                elif isinstance(det, (list, tuple, np.ndarray)) and len(det) >= 4:
                    # Handle different array formats
                    if len(det) == 4:  # [x1, y1, x2, y2]
                        det_arrays.append([det[0], det[1], det[2], det[3], 0.5, 0])
                    elif len(det) >= 5:  # [x1, y1, x2, y2, conf, ...]
                        class_id = det[5] if len(det) > 5 else 0
                        det_arrays.append([det[0], det[1], det[2], det[3], det[4], class_id])
            except:
                continue
        
        return np.array(det_arrays) if det_arrays else np.empty((0, 6))
    
    def _associate(self, detections, tracks, thresh, frame=None):
        """Associate detections with tracks using Hungarian algorithm"""
        if len(detections) == 0 or len(tracks) == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))
        
        # Calculate cost matrix
        cost_matrix = self._calculate_cost_matrix(detections, tracks, frame)
        
        # Apply Hungarian algorithm
        if cost_matrix.size > 0:
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            matches = []
            
            for row, col in zip(row_indices, col_indices):
                if cost_matrix[row, col] <= (1 - thresh):  # Convert similarity to cost
                    matches.append([col, row])  # [track_idx, det_idx]
            
            unmatched_tracks = [i for i in range(len(tracks)) if i not in [m[0] for m in matches]]
            unmatched_dets = [i for i in range(len(detections)) if i not in [m[1] for m in matches]]
        else:
            matches = []
            unmatched_tracks = list(range(len(tracks)))
            unmatched_dets = list(range(len(detections)))
        
        return matches, unmatched_tracks, unmatched_dets
    
    def _calculate_cost_matrix(self, detections, tracks, frame=None):
        """Calculate cost matrix combining IoU and appearance"""
        if len(detections) == 0 or len(tracks) == 0:
            return np.empty((0, 0))
        
        # Get track bboxes (use Kalman prediction if available)
        track_bboxes = []
        for track in tracks:
            if hasattr(track, 'kalman') and self.use_kalman:
                bbox = track.kalman.get_bbox()
            else:
                bbox = track.current_bbox if hasattr(track, 'current_bbox') else track.tlbr
            track_bboxes.append(bbox)
        
        track_bboxes = np.array(track_bboxes)
        det_bboxes = np.array([det[:4] for det in detections])
        
        # Calculate IoU matrix
        iou_matrix = self._calculate_iou_matrix(det_bboxes, track_bboxes)
        
        # Apply direction gating
        iou_matrix = self._apply_direction_gating(detections, tracks, iou_matrix)
        
        # Apply geometric constraints
        iou_matrix = self._apply_geometric_constraints(detections, tracks, iou_matrix)
        
        # Combine with appearance if available
        if self.use_appearance and frame is not None:
            app_matrix = self._calculate_appearance_matrix(detections, tracks, frame)
            cost_matrix = 0.7 * (1 - iou_matrix) + 0.3 * (1 - app_matrix)
        else:
            cost_matrix = 1 - iou_matrix
        
        return cost_matrix
    
    def _calculate_iou_matrix(self, det_bboxes, track_bboxes):
        """Calculate IoU matrix between detections and tracks"""
        if len(det_bboxes) == 0 or len(track_bboxes) == 0:
            return np.empty((0, 0))
        
        # Calculate areas
        det_areas = (det_bboxes[:, 2] - det_bboxes[:, 0]) * (det_bboxes[:, 3] - det_bboxes[:, 1])
        track_areas = (track_bboxes[:, 2] - track_bboxes[:, 0]) * (track_bboxes[:, 3] - track_bboxes[:, 1])
        
        # Calculate IoU matrix
        ious = np.zeros((len(det_bboxes), len(track_bboxes)))
        
        for i, det in enumerate(det_bboxes):
            for j, track in enumerate(track_bboxes):
                # Intersection
                x1 = max(det[0], track[0])
                y1 = max(det[1], track[1])
                x2 = min(det[2], track[2])
                y2 = min(det[3], track[3])
                
                if x2 > x1 and y2 > y1:
                    intersection = (x2 - x1) * (y2 - y1)
                    union = det_areas[i] + track_areas[j] - intersection
                    ious[i, j] = intersection / union if union > 0 else 0
                else:
                    ious[i, j] = 0
        
        return ious
    
    def _apply_direction_gating(self, detections, tracks, iou_matrix):
        """Apply direction consistency gating"""
        if len(detections) == 0 or len(tracks) == 0:
            return iou_matrix
        
        gated_matrix = iou_matrix.copy()
        
        for i, det in enumerate(detections):
            det_center = np.array([(det[0] + det[2])/2, (det[1] + det[3])/2])
            
            for j, track in enumerate(tracks):
                if hasattr(track, 'get_velocity'):
                    track_velocity = track.get_velocity()
                    track_bbox = track.current_bbox if hasattr(track, 'current_bbox') else track.tlbr
                    track_center = np.array([(track_bbox[0] + track_bbox[2])/2, (track_bbox[1] + track_bbox[3])/2])
                    
                    # Calculate displacement vector
                    displacement = det_center - track_center
                    
                    # Check direction consistency
                    if np.linalg.norm(track_velocity) > 1.0 and np.linalg.norm(displacement) > 1.0:
                        cos_sim = np.dot(track_velocity, displacement) / (
                            np.linalg.norm(track_velocity) * np.linalg.norm(displacement)
                        )
                        
                        if cos_sim < self.direction_thresh:
                            gated_matrix[i, j] = 0  # Reject inconsistent direction
        
        return gated_matrix
    
    def _apply_geometric_constraints(self, detections, tracks, iou_matrix):
        """Apply geometric ROI constraints"""
        if not self.geometric_context.enabled:
            return iou_matrix
        
        constrained_matrix = iou_matrix.copy()
        
        for i, det in enumerate(detections):
            if not self.geometric_context.is_inside_roi(det[:4]):
                constrained_matrix[i, :] = 0  # Reject detections outside ROI
        
        return constrained_matrix
    
    def _calculate_appearance_matrix(self, detections, tracks, frame):
        """Calculate appearance similarity matrix"""
        if len(detections) == 0 or len(tracks) == 0:
            return np.empty((0, 0))
        
        app_matrix = np.zeros((len(detections), len(tracks)))
        
        for i, det in enumerate(detections):
            # Extract appearance features for detection
            det_features = self._extract_appearance_features(frame, det[:4])
            
            for j, track in enumerate(tracks):
                if hasattr(track, 'appearance_features') and track.appearance_features is not None:
                    if det_features is not None:
                        # Cosine similarity
                        similarity = np.dot(det_features, track.appearance_features) / (
                            np.linalg.norm(det_features) * np.linalg.norm(track.appearance_features) + 1e-6
                        )
                        app_matrix[i, j] = max(0, similarity)
        
        return app_matrix
    
    def _extract_appearance_features(self, frame, bbox):
        """Extract simple appearance features (HSV histogram)"""
        try:
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            if x2 > x1 and y2 > y1:
                roi = frame[y1:y2, x1:x2]
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
                return cv2.normalize(hist, hist).flatten()
        except:
            pass
        return None
    
    def _create_new_track(self, detection):
        """Create new track from detection"""
        self.track_id_count += 1
        new_track = EnhancedTrack(detection, self.track_id_count, self.frame_id)
        self.tracked_tracks.append(new_track)
        self.stats['total_tracks'] += 1
    
    def _update_track_states(self):
        """Update track states and move between lists"""
        # Move tracks from tracked to lost
        tracks_to_lost = []
        for i, track in enumerate(self.tracked_tracks):
            if track.state == 'LOST':
                tracks_to_lost.append(i)
        
        for i in reversed(tracks_to_lost):
            self.lost_tracks.append(self.tracked_tracks.pop(i))
        
        # Remove old tracks
        tracks_to_remove = []
        for i, track in enumerate(self.lost_tracks):
            if track.should_remove(self.track_buffer):
                tracks_to_remove.append(i)
        
        for i in reversed(tracks_to_remove):
            self.removed_tracks.append(self.lost_tracks.pop(i))
        
        # Limit removed tracks for memory management
        if len(self.removed_tracks) > 1000:
            self.removed_tracks = self.removed_tracks[-500:]
    
    def _update_stats(self):
        """Update tracking statistics"""
        self.stats['active_tracks'] = len(self.tracked_tracks)
        self.stats['lost_tracks'] = len(self.lost_tracks)
    
    def _get_output_tracks(self):
        """Get output tracks in required format"""
        output_tracks = []
        
        # Only return confirmed tracks
        for track in self.tracked_tracks:
            if track.is_confirmed():
                output_tracks.append(track.to_dict())
        
        return output_tracks
    
    def reset(self):
        """Reset tracker to initial state"""
        self.tracked_tracks = []
        self.lost_tracks = []
        self.removed_tracks = []
        self.frame_id = 0
        self.track_id_count = 0
        self.stats = {
            'total_tracks': 0,
            'active_tracks': 0,
            'lost_tracks': 0,
            'id_switches': 0
        }
    
    def add_roi(self, polygon_points):
        """Add region of interest for geometric filtering"""
        self.geometric_context.add_roi(polygon_points)
    
    def add_lane_direction(self, lane_id, direction_vector):
        """Add expected direction for a lane"""
        self.geometric_context.add_lane_direction(lane_id, direction_vector)
    
    def get_stats(self):
        """Get tracking statistics"""
        return self.stats.copy()


class EnhancedVehicleTracker:
    """
    Drop-in replacement for ByteTrackVehicleTracker with enhanced features
    Maintains exact same API for compatibility with video_controller_new.py
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            print("[ENHANCED TRACKER] Creating EnhancedVehicleTracker instance")
            cls._instance = super(EnhancedVehicleTracker, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if getattr(self, '_initialized', False):
            return
            
        print("[ENHANCED TRACKER] Initializing enhanced tracker")
        
        # Create enhanced tracker with optimized parameters
        self.tracker = EnhancedBYTETracker(
            track_thresh=0.4,
            track_buffer=30,
            match_thresh=0.8,
            track_high_thresh=0.6,
            track_low_thresh=0.2,
            frame_rate=30,
            use_kalman=True,
            use_appearance=True,
            direction_thresh=0.3
        )
        
        self._initialized = True
        self.debug = True
        
        # Setup default ROIs for traffic intersection (can be customized)
        self._setup_default_rois()
    
    def _setup_default_rois(self):
        """Setup default ROIs for traffic intersection"""
        # These can be customized based on your intersection layout
        # Example intersection ROI (adjust coordinates as needed)
        intersection_roi = [
            [200, 100],  # top-left
            [1000, 100], # top-right
            [1000, 600], # bottom-right
            [200, 600]   # bottom-left
        ]
        self.tracker.add_roi(intersection_roi)
        
        # Add lane directions (adjust as needed)
        self.tracker.add_lane_direction("north_south", [0, 1])    # Northbound
        self.tracker.add_lane_direction("south_north", [0, -1])   # Southbound
        self.tracker.add_lane_direction("east_west", [1, 0])      # Eastbound
        self.tracker.add_lane_direction("west_east", [-1, 0])     # Westbound
    
    def update(self, detections, frame=None):
        """
        Update tracker with new detections - EXACT API as original
        
        Args:
            detections: list of dicts with keys ['bbox', 'confidence', 'class_id', ...]
            frame: BGR image (optional)
            
        Returns:
            list of dicts with keys ['id', 'bbox', 'confidence', 'class_id', ...]
        """
        try:
            if self.debug:
                print(f"[ENHANCED TRACKER] Processing {len(detections)} detections")
            
            # Update tracker
            tracks = self.tracker.update(detections, frame)
            
            if self.debug:
                stats = self.tracker.get_stats()
                print(f"[ENHANCED TRACKER] Active: {stats['active_tracks']}, "
                      f"Total: {stats['total_tracks']}")
            
            return tracks
            
        except Exception as e:
            print(f"[ENHANCED TRACKER ERROR] Error updating tracker: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def update_tracks(self, detections, frame=None):
        """
        Compatibility method for DeepSORT interface
        """
        if self.debug:
            print(f"[ENHANCED TRACKER] update_tracks called with {len(detections)} detections")
        
        # Convert DeepSORT format to enhanced tracker format
        converted_dets = []
        
        for det in detections:
            try:
                if isinstance(det, (list, tuple)) and len(det) >= 2:
                    bbox_xywh, conf = det[:2]
                    class_name = det[2] if len(det) > 2 else 'vehicle'
                    
                    if isinstance(bbox_xywh, (list, tuple, np.ndarray)) and len(bbox_xywh) == 4:
                        x, y, w, h = map(float, bbox_xywh)
                        conf = float(conf)
                        
                        converted_dets.append({
                            'bbox': [x, y, x + w, y + h],
                            'confidence': conf,
                            'class_id': 0
                        })
            except Exception as e:
                if self.debug:
                    print(f"[ENHANCED TRACKER] Error converting detection: {e}")
        
        # Call the regular update method
        dict_tracks = self.update(converted_dets, frame)
        
        # Convert to DeepSORT-compatible objects
        ds_tracks = []
        for track_data in dict_tracks:
            ds_track = EnhancedTrackOutput(track_data)
            ds_tracks.append(ds_track)
        
        return ds_tracks
    
    def reset(self):
        """Reset the tracker"""
        print("[ENHANCED TRACKER] Resetting tracker state")
        self.tracker.reset()
    
    def add_roi(self, polygon_points):
        """Add custom ROI"""
        self.tracker.add_roi(polygon_points)
    
    def add_lane_direction(self, lane_id, direction_vector):
        """Add lane direction constraint"""
        self.tracker.add_lane_direction(lane_id, direction_vector)
    
    def get_stats(self):
        """Get tracking statistics"""
        return self.tracker.get_stats()


class EnhancedTrackOutput:
    """
    Enhanced adapter class for DeepSORT compatibility
    """
    
    def __init__(self, track_data):
        """Initialize from enhanced track dictionary"""
        self.track_id = track_data.get('id', -1)
        self.det_index = track_data.get('det_index', -1)
        
        # Convert bbox to [x, y, w, h] format
        bbox = track_data.get('bbox', [0, 0, 0, 0])
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            self.to_tlwh_ret = [x1, y1, x2 - x1, y2 - y1]
        else:
            self.to_tlwh_ret = [0, 0, 0, 0]
        
        self.bbox = track_data.get('bbox', [0, 0, 0, 0])
        self.confidence = track_data.get('confidence', 0.0)
        self.is_confirmed_flag = track_data.get('state', 'NEW') == 'TRACKED'
        
        # Enhanced features
        self.velocity = track_data.get('velocity', [0.0, 0.0])
        self.heading = track_data.get('heading', 0.0)
        self.speed = track_data.get('speed', 0.0)
        self.state = track_data.get('state', 'NEW')
        self.age = track_data.get('age', 1)
        self.hits = track_data.get('hits', 1)
        
        # Store original data
        self._track_data = track_data
    
    def to_tlwh(self):
        """Return bounding box in [x, y, w, h] format"""
        return self.to_tlwh_ret
    
    def is_confirmed(self):
        """Check if track is confirmed"""
        return self.is_confirmed_flag
    
    def __getattr__(self, name):
        """Fallback to original track data"""
        if name in self._track_data:
            return self._track_data[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
