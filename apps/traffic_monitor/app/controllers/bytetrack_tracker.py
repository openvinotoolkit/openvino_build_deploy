# ByteTrack implementation for vehicle tracking
# Efficient and robust multi-object tracking that works exactly like DeepSORT
import numpy as np
import cv2
import time
from collections import defaultdict, deque
from typing import List, Dict, Any, Tuple, Optional
import torch

class Track:
    """Track class for ByteTracker - Compatible with video_controller_new.py"""
    
    def __init__(self, detection, track_id):
        """Initialize a track from a detection
        
        Args:
            detection: Detection array [x1, y1, x2, y2, score, class_id]
            track_id: Unique track ID
        """
        self.track_id = track_id
        self.tlbr = detection[:4]  # [x1, y1, x2, y2]
        self.score = detection[4] if len(detection) > 4 else 0.5
        self.class_id = int(detection[5]) if len(detection) > 5 else 0
        
        self.time_since_update = 0
        self.hits = 1  # Number of times track was matched to a detection
        self.age = 1
        self.frame_id = 0  # Will be set by the tracker during update
        self.is_lost = False  # Flag to indicate if track is lost
        self.state = 'Tentative'  # Track state: Tentative, Confirmed, Deleted
        

        # Store position history for movement tracking
        self.history = deque(maxlen=30)
        self.history.append(self.tlbr.copy())
        self.center_history = deque(maxlen=30)
        self.bottom_history = deque(maxlen=30)
        center = self._get_center()
        self.center_history.append(center)
        self.bottom_history.append(self.tlbr[3])
        self.speed_history = deque(maxlen=10)
        self.velocity = np.array([0., 0.])
        self.is_moving = False
        self.moving_frames = 0
        self.stopped_frames = 0

        # Movement detection thresholds
        self.movement_threshold = 1.2  # px/frame, can be tuned
        self.creep_threshold = 0.5     # px/frame, for slow creep
        self.frames_for_moving = 3
        self.frames_for_stopped = 5
        self.high_speed_threshold = 6.0  # px/frame, for fast vehicles

    def _get_center(self):
        return np.array([(self.tlbr[0] + self.tlbr[2])/2, (self.tlbr[1] + self.tlbr[3])/2])


    def _update_movement(self):
        # Calculate rolling average speed (center and bottom)
        if len(self.center_history) >= 2:
            center_speeds = [np.linalg.norm(self.center_history[i] - self.center_history[i-1])
                             for i in range(1, len(self.center_history))]
        else:
            center_speeds = [0.0]
        if len(self.bottom_history) >= 2:
            bottom_speeds = [abs(self.bottom_history[i] - self.bottom_history[i-1])
                             for i in range(1, len(self.bottom_history))]
        else:
            bottom_speeds = [0.0]
        avg_center_speed = np.mean(center_speeds[-5:]) if len(center_speeds) >= 5 else np.mean(center_speeds)
        avg_bottom_speed = np.mean(bottom_speeds[-5:]) if len(bottom_speeds) >= 5 else np.mean(bottom_speeds)
        avg_speed = max(avg_center_speed, avg_bottom_speed)
        self.speed_history.append(avg_speed)
        # Fast movement detection: if any recent speed exceeds high threshold, set moving immediately
        recent_speeds = center_speeds[-3:] + bottom_speeds[-3:]
        if any(s > self.high_speed_threshold for s in recent_speeds):
            self.is_moving = True
            self.moving_frames = self.frames_for_moving  # Reset hysteresis
            self.stopped_frames = 0
            return
        # Movement state logic with hysteresis
        if avg_speed > self.movement_threshold:
            self.moving_frames += 1
            self.stopped_frames = 0
        elif avg_speed < self.creep_threshold:
            self.stopped_frames += 1
            self.moving_frames = 0
        # Only set moving if enough consecutive frames
        if self.moving_frames >= self.frames_for_moving:
            self.is_moving = True
        elif self.stopped_frames >= self.frames_for_stopped:
            self.is_moving = False

    def update(self, detection):
        self.tlbr = detection[:4]
        self.score = detection[4] if len(detection) > 4 else self.score
        self.class_id = int(detection[5]) if len(detection) > 5 else self.class_id
        self.hits += 1
        self.time_since_update = 0
        self.history.append(self.tlbr.copy())
        center = self._get_center()
        self.center_history.append(center)
        self.bottom_history.append(self.tlbr[3])
        self._update_movement()
        if self.state == 'Tentative' and self.hits >= 3:
            self.state = 'Confirmed'

    def to_dict(self):
        avg_speed = np.mean(self.speed_history) if self.speed_history else 0.0
        return {
            'id': self.track_id,
            'bbox': [float(self.tlbr[0]), float(self.tlbr[1]), float(self.tlbr[2]), float(self.tlbr[3])],
            'confidence': float(self.score),
            'class_id': int(self.class_id),
            'is_moving': self.is_moving,
            'avg_speed': avg_speed
        }
        
    def predict(self):
        """Predict the next state using simple motion model"""
        if len(self.history) >= 2:
            # Simple velocity estimation from last two positions
            curr_center = np.array([(self.tlbr[0] + self.tlbr[2])/2, (self.tlbr[1] + self.tlbr[3])/2])
            prev_tlbr = self.history[-2]
            prev_center = np.array([(prev_tlbr[0] + prev_tlbr[2])/2, (prev_tlbr[1] + prev_tlbr[3])/2])
            self.velocity = curr_center - prev_center
            
            # Predict next position
            next_center = curr_center + self.velocity
            w, h = self.tlbr[2] - self.tlbr[0], self.tlbr[3] - self.tlbr[1]
            self.tlbr = np.array([next_center[0] - w/2, next_center[1] - h/2, 
                                 next_center[0] + w/2, next_center[1] + h/2])
        
        self.age += 1
        self.time_since_update += 1
        
    def update(self, detection):
        """Update track with new detection"""
        self.tlbr = detection[:4]
        self.score = detection[4] if len(detection) > 4 else self.score
        self.class_id = int(detection[5]) if len(detection) > 5 else self.class_id
        
        self.hits += 1
        self.time_since_update = 0
        self.history.append(self.tlbr.copy())
        
        # Update state to confirmed after enough hits
        if self.state == 'Tentative' and self.hits >= 3:
            self.state = 'Confirmed'
    
    def mark_missed(self):
        """Mark track as missed (no detection matched)"""
        self.time_since_update += 1
        if self.time_since_update > 1:
            self.is_lost = True
            
    def is_confirmed(self):
        """Check if track is confirmed (has enough hits)"""
        return self.state == 'Confirmed'
    
    def to_dict(self):
        """Convert track to dictionary format for video_controller_new.py"""
        return {
            'id': self.track_id,
            'bbox': [float(self.tlbr[0]), float(self.tlbr[1]), float(self.tlbr[2]), float(self.tlbr[3])],
            'confidence': float(self.score),
            'class_id': int(self.class_id)
        }


class BYTETracker:
    """
    ByteTrack tracker implementation
    Designed to work exactly like DeepSORT with video_controller_new.py
    """
    def __init__(
        self,
        track_thresh=0.5,
        track_buffer=30,
        match_thresh=0.7,
        frame_rate=30,
        track_high_thresh=0.6,
        track_low_thresh=0.1
    ):
        self.tracked_tracks = []  # Active tracks being tracked
        self.lost_tracks = []     # Lost tracks (temporarily out of view)
        self.removed_tracks = []  # Removed tracks (permanently lost)
        
        self.frame_id = 0
        self.max_time_lost = int(frame_rate / 30.0 * track_buffer)
        
        self.track_thresh = track_thresh          # Threshold for high-confidence detections
        self.track_high_thresh = track_high_thresh  # Higher threshold for first association
        self.track_low_thresh = track_low_thresh    # Lower threshold for second association
        self.match_thresh = match_thresh          # IOU match threshold
        
        self.track_id_count = 0
        
        print(f"[BYTETRACK] Initialized with: high_thresh={track_high_thresh}, " +
              f"low_thresh={track_low_thresh}, match_thresh={match_thresh}, max_time_lost={self.max_time_lost}")

    def update(self, detections, frame=None):
        """Update tracks with new detections
        
        Args:
            detections: list of dicts with keys ['bbox', 'confidence', 'class_id', ...]
            frame: Optional BGR frame for debug visualization
        
        Returns:
            list of dicts with keys ['id', 'bbox', 'confidence', 'class_id', ...]
        """
        self.frame_id += 1
        
        # Convert detections to internal format
        converted_detections = self._convert_detections(detections)
        
        print(f"[BYTETRACK] Frame {self.frame_id}: Processing {len(converted_detections)} detections")
        print(f"[BYTETRACK] Current state: {len(self.tracked_tracks)} tracked, {len(self.lost_tracks)} lost")
        
        # Handle empty detections case
        if len(converted_detections) == 0:
            print(f"[BYTETRACK] No valid detections in frame {self.frame_id}")
            # Move all tracked to lost and update
            for track in self.tracked_tracks:
                track.mark_missed()
                track.predict()
                if track.time_since_update <= self.max_time_lost:
                    self.lost_tracks.append(track)
                else:
                    self.removed_tracks.append(track)
            
            # Update lost tracks
            updated_lost = []
            for track in self.lost_tracks:
                track.predict()
                if track.time_since_update <= self.max_time_lost:
                    updated_lost.append(track)
                else:
                    self.removed_tracks.append(track)
                    
            self.tracked_tracks = []
            self.lost_tracks = updated_lost
            return []
        
        # Split detections into high and low confidence
        confidence_values = converted_detections[:, 4].astype(float)
        high_indices = confidence_values >= self.track_high_thresh
        low_indices = (confidence_values >= self.track_low_thresh) & (confidence_values < self.track_high_thresh)
        
        high_detections = converted_detections[high_indices]
        low_detections = converted_detections[low_indices]
        
        print(f"[BYTETRACK] Split into {len(high_detections)} high-conf and {len(low_detections)} low-conf detections")
        
        # Predict all tracks
        for track in self.tracked_tracks + self.lost_tracks:
            track.predict()
        
        # First association: high-confidence detections with tracked tracks
        matches1, unmatched_tracks1, unmatched_dets1 = self._associate(
            high_detections, self.tracked_tracks, self.match_thresh)
        
        # Update matched tracks
        for match in matches1:
            track_idx, det_idx = match
            self.tracked_tracks[track_idx].update(high_detections[det_idx])
            self.tracked_tracks[track_idx].frame_id = self.frame_id
        
        # Move unmatched tracks to lost
        unmatched_tracked_tracks = []
        for idx in unmatched_tracks1:
            track = self.tracked_tracks[idx]
            track.mark_missed()
            if track.time_since_update <= self.max_time_lost:
                self.lost_tracks.append(track)
            else:
                self.removed_tracks.append(track)
        
        # Keep only matched tracks
        self.tracked_tracks = [self.tracked_tracks[i] for i in range(len(self.tracked_tracks)) if i not in unmatched_tracks1]
        
        # Second association: remaining high-conf detections with lost tracks
        if len(unmatched_dets1) > 0 and len(self.lost_tracks) > 0:
            remaining_high_dets = high_detections[unmatched_dets1]
            matches2, unmatched_tracks2, unmatched_dets2 = self._associate(
                remaining_high_dets, self.lost_tracks, self.match_thresh)
            
            # Reactivate matched lost tracks
            for match in matches2:
                track_idx, det_idx = match
                track = self.lost_tracks[track_idx]
                track.update(remaining_high_dets[det_idx])
                track.frame_id = self.frame_id
                track.is_lost = False
                self.tracked_tracks.append(track)
            
            # Remove reactivated tracks from lost
            self.lost_tracks = [self.lost_tracks[i] for i in range(len(self.lost_tracks)) if i not in [m[0] for m in matches2]]
            
            # Update unmatched detections indices
            final_unmatched_dets = [unmatched_dets1[i] for i in unmatched_dets2]
        else:
            final_unmatched_dets = unmatched_dets1
        
        # Third association: low-confidence detections with remaining lost tracks
        if len(low_detections) > 0 and len(self.lost_tracks) > 0:
            matches3, unmatched_tracks3, unmatched_dets3 = self._associate(
                low_detections, self.lost_tracks, self.match_thresh)
            
            # Reactivate matched lost tracks
            for match in matches3:
                track_idx, det_idx = match
                track = self.lost_tracks[track_idx]
                track.update(low_detections[det_idx])
                track.frame_id = self.frame_id
                track.is_lost = False
                self.tracked_tracks.append(track)
            
            # Remove reactivated tracks from lost
            self.lost_tracks = [self.lost_tracks[i] for i in range(len(self.lost_tracks)) if i not in [m[0] for m in matches3]]
        
        # Create new tracks for remaining unmatched high-confidence detections
        new_tracks_created = 0
        for det_idx in final_unmatched_dets:
            detection = high_detections[det_idx]
            if detection[4] >= self.track_thresh:  # Only create tracks for high-confidence detections
                self.track_id_count += 1
                new_track = Track(detection, self.track_id_count)
                new_track.frame_id = self.frame_id
                self.tracked_tracks.append(new_track)
                new_tracks_created += 1
        
        # Clean up lost tracks that have been lost too long
        updated_lost = []
        removed_count = 0
        for track in self.lost_tracks:
            if track.time_since_update <= self.max_time_lost:
                updated_lost.append(track)
            else:
                self.removed_tracks.append(track)
                removed_count += 1
        self.lost_tracks = updated_lost
        
        print(f"[BYTETRACK] Matched {len(matches1)} tracks, created {new_tracks_created} new tracks, removed {removed_count} expired tracks")
        print(f"[BYTETRACK] Final state: {len(self.tracked_tracks)} tracked, {len(self.lost_tracks)} lost")
        
        # Return confirmed tracks in dictionary format
        confirmed_tracks = []
        for track in self.tracked_tracks:
            if track.is_confirmed():
                confirmed_tracks.append(track.to_dict())
        
        print(f"[BYTETRACK] Returning {len(confirmed_tracks)} confirmed tracks")
        return confirmed_tracks
    
    def _convert_detections(self, detections):
        """Convert detection format to numpy array"""
        if len(detections) == 0:
            return np.empty((0, 6))
        
        converted = []
        for det in detections:
            bbox = det.get('bbox', [0, 0, 0, 0])
            conf = det.get('confidence', 0.0)
            class_id = det.get('class_id', 0)
            
            # Ensure bbox is valid
            if len(bbox) == 4 and bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                converted.append([float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]), float(conf), int(class_id)])
        
        return np.array(converted) if converted else np.empty((0, 6))
    
    def _associate(self, detections, tracks, iou_threshold):
        """Associate detections with tracks using IoU"""
        if len(detections) == 0 or len(tracks) == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))
        
        # Calculate IoU matrix
        iou_matrix = self._calculate_iou_matrix(detections[:, :4], np.array([track.tlbr for track in tracks]))
        
        # Use Hungarian algorithm (simplified greedy approach)
        matches, unmatched_tracks, unmatched_detections = self._linear_assignment(iou_matrix, iou_threshold)
        
        return matches, unmatched_tracks, unmatched_detections
    
    def _calculate_iou_matrix(self, detections, tracks):
        """Calculate IoU matrix between detections and tracks"""
        if len(detections) == 0 or len(tracks) == 0:
            return np.empty((0, 0))
        
        # Calculate areas
        det_areas = (detections[:, 2] - detections[:, 0]) * (detections[:, 3] - detections[:, 1])
        track_areas = (tracks[:, 2] - tracks[:, 0]) * (tracks[:, 3] - tracks[:, 1])
        
        # Calculate intersections
        ious = np.zeros((len(detections), len(tracks)))
        for i, det in enumerate(detections):
            for j, track in enumerate(tracks):
                # Intersection coordinates
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
    
    def _linear_assignment(self, cost_matrix, threshold):
        """Simple greedy assignment based on IoU threshold"""
        matches = []
        unmatched_tracks = list(range(cost_matrix.shape[1]))
        unmatched_detections = list(range(cost_matrix.shape[0]))
        
        if cost_matrix.size == 0:
            return matches, unmatched_tracks, unmatched_detections
        
        # Find matches above threshold
        for i in range(cost_matrix.shape[0]):
            for j in range(cost_matrix.shape[1]):
                if cost_matrix[i, j] >= threshold:
                    if i in unmatched_detections and j in unmatched_tracks:
                        matches.append([j, i])  # [track_idx, det_idx]
                        unmatched_tracks.remove(j)
                        unmatched_detections.remove(i)
                        break
        
        return matches, unmatched_tracks, unmatched_detections


class ByteTrackVehicleTracker:
    """
    ByteTrack-based vehicle tracker with exact same API as DeepSortVehicleTracker
    for drop-in replacement in video_controller_new.py
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            print("[BYTETRACK SINGLETON] Creating ByteTrackVehicleTracker instance")
            cls._instance = super(ByteTrackVehicleTracker, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, '_initialized', False):
            return
        print("[BYTETRACK INIT] Initializing ByteTrack tracker")
        
        # Parameters optimized for vehicle tracking in traffic scenes
        self.tracker = BYTETracker(
            track_thresh=0.4,           # Minimum confidence to create new tracks
            track_buffer=30,            # How many frames to keep lost tracks
            match_thresh=0.7,           # IoU threshold for matching
            track_high_thresh=0.5,      # High confidence threshold for first association
            track_low_thresh=0.2,       # Low confidence threshold for second association
            frame_rate=30               # Expected frame rate
        )
        
        self._initialized = True
        self.debug = True              # Enable debug output
        
        # Memory management
        self.max_removed_tracks = 100  # Limit removed tracks to prevent memory issues

    def update(self, detections, frame=None):
        """
        Update tracker with new detections - EXACT API as DeepSORT
        
        Args:
            detections: list of dicts with keys ['bbox', 'confidence', 'class_id', ...]
            frame: BGR image (optional)
            
        Returns:
            list of dicts with keys ['id', 'bbox', 'confidence', 'class_id', ...]
        """
        try:
            # Input validation
            if not isinstance(detections, list):
                print(f"[BYTETRACK ERROR] Invalid detections format: {type(detections)}")
                return []
            
            # Process detections
            valid_dets = []
            for i, det in enumerate(detections):
                if not isinstance(det, dict):
                    continue
                    
                bbox = det.get('bbox')
                conf = det.get('confidence', 0.0)
                class_id = det.get('class_id', 0)
                
                if bbox is not None and len(bbox) == 4:
                    x1, y1, x2, y2 = map(float, bbox)
                    conf = float(conf)
                    class_id = int(class_id)
                    
                    # Validate bbox dimensions
                    if x2 > x1 and y2 > y1 and conf > 0.1:
                        valid_dets.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': conf,
                            'class_id': class_id
                        })
            
            if self.debug:
                print(f"[BYTETRACK] Processing {len(valid_dets)} valid detections")
            
            # Update tracker
            tracks = self.tracker.update(valid_dets, frame)
            
            # Memory management - limit removed tracks
            if len(self.tracker.removed_tracks) > self.max_removed_tracks:
                self.tracker.removed_tracks = self.tracker.removed_tracks[-self.max_removed_tracks//2:]
                if self.debug:
                    print(f"[BYTETRACK] Cleaned up removed tracks, keeping last {len(self.tracker.removed_tracks)}")
            
            return tracks
            
        except Exception as e:
            print(f"[BYTETRACK ERROR] Error updating tracker: {e}")
            import traceback
            traceback.print_exc()
            return []

    def update_tracks(self, detections, frame=None):
        """
        Update method for compatibility with DeepSORT interface used by model_manager.py
        
        Args:
            detections: list of detection arrays in format [bbox_xywh, conf, class_name]
            frame: BGR image (optional)
        
        Returns:
            list of track objects with DeepSORT-compatible interface including is_confirmed() method
        """
        if self.debug:
            print(f"[BYTETRACK] update_tracks called with {len(detections)} detections")
        
        # Convert from DeepSORT format to ByteTrack format
        converted_dets = []
        
        for det in detections:
            try:
                # Handle different detection formats
                if isinstance(det, (list, tuple)) and len(det) >= 2:
                    # DeepSORT format: [bbox_xywh, conf, class_name]
                    bbox_xywh, conf = det[:2]
                    class_name = det[2] if len(det) > 2 else 'vehicle'
                    
                    # Convert [x, y, w, h] to [x1, y1, x2, y2] with type validation
                    if isinstance(bbox_xywh, (list, tuple, np.ndarray)) and len(bbox_xywh) == 4:
                        x, y, w, h = map(float, bbox_xywh)
                        conf = float(conf)
                        
                        converted_dets.append({
                            'bbox': [x, y, x + w, y + h],
                            'confidence': conf,
                            'class_id': 0  # Default vehicle class
                        })
                else:
                    if self.debug:
                        print(f"[BYTETRACK] Skipping invalid detection format: {det}")
            except Exception as e:
                if self.debug:
                    print(f"[BYTETRACK] Error converting detection: {e}")
        
        # Call the regular update method to get dictionary tracks
        dict_tracks = self.update(converted_dets, frame)
        
        if self.debug:
            print(f"[BYTETRACK] Converting {len(dict_tracks)} dict tracks to DeepSORT-compatible objects")
        
        # Create DeepSORT compatible track objects from dictionaries
        ds_tracks = []
        for track_data in dict_tracks:
            ds_track = ByteTrackOutput(track_data)
            ds_tracks.append(ds_track)
        
        return ds_tracks

    def reset(self):
        """
        Reset the tracker to clean state - starts track IDs from 1
        Call this when starting a new video or session
        """
        print("[BYTETRACK] Resetting tracker state")
        if hasattr(self, 'tracker') and self.tracker is not None:
            # Reset the internal BYTETracker
            self.tracker.tracked_tracks = []
            self.tracker.lost_tracks = []
            self.tracker.removed_tracks = []
            self.tracker.frame_id = 0
            self.tracker.track_id_count = 0  # Reset ID counter to start from 1
            
            print("[BYTETRACK] Reset complete - track IDs will start from 1")
        else:
            print("[BYTETRACK] Warning: Tracker not initialized, nothing to reset")


class ByteTrackOutput:
    """
    Adapter class to make ByteTrack output compatible with DeepSORT interface
    """
    
    def __init__(self, track_data):
        """Initialize from ByteTrack track dictionary"""
        self.track_id = track_data.get('id', -1)
        self.det_index = track_data.get('det_index', -1)
        self.to_tlwh_ret = track_data.get('bbox', [0, 0, 0, 0])  # [x, y, w, h]
        self.bbox = track_data.get('bbox', [0, 0, 0, 0])  # Add bbox property
        self.confidence = track_data.get('confidence', 0.0)
        self.is_confirmed = track_data.get('is_confirmed', True)
        # Store the original track data
        self._track_data = track_data
    
    def to_tlwh(self):
        """Return bounding box in [x, y, w, h] format"""
        return self.to_tlwh_ret
    
    def __getattr__(self, name):
        """Fallback to original track data"""
        if name in self._track_data:
            return self._track_data[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
