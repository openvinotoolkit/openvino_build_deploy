# DeepSORT integration for vehicle tracking
# You need to install deep_sort_realtime: pip install deep_sort_realtime
from deep_sort_realtime.deepsort_tracker import DeepSort

class DeepSortVehicleTracker:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            print("[DEEPSORT SINGLETON] Creating DeepSortVehicleTracker instance")
            cls._instance = super(DeepSortVehicleTracker, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, '_initialized', False):
            return
        print("[DEEPSORT INIT] Initializing DeepSort tracker (should only see this once)")
        # Use DeepSORT with better parameters to reduce duplicate IDs
        self.tracker = DeepSort(
            max_age=50,           # Keep tracks longer to avoid re-creating IDs
            n_init=3,             # Require 3 consecutive detections before confirming track
            nms_max_overlap=0.3,  # Stricter NMS to avoid duplicate detections
            max_cosine_distance=0.4,  # Stricter appearance matching
            nn_budget=100,        # Budget for appearance features
            gating_only_position=False  # Use both position and appearance for gating
        )
        self._initialized = True
        self.track_id_counter = {}  # Track seen IDs to detect duplicates

    def update(self, detections, frame=None):
        # detections: list of dicts with keys ['bbox', 'confidence', 'class_id', ...]
        # frame: BGR image (optional, for appearance embedding)
        # Returns: list of dicts with keys ['id', 'bbox', 'confidence', 'class_id', ...]
        
        # Convert detections to DeepSORT format with validation
        ds_detections = []
        for i, det in enumerate(detections):
            bbox = det.get('bbox')
            conf = det.get('confidence', 0.0)
            class_id = det.get('class_id', -1)
            
            if bbox is not None and len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                # Validate bbox dimensions
                if x2 > x1 and y2 > y1 and conf > 0.3:  # Higher confidence threshold
                    # Convert to [x1, y1, width, height] format expected by DeepSORT
                    bbox_xywh = [x1, y1, x2-x1, y2-y1]
                    ds_detections.append([bbox_xywh, conf, class_id])
                    print(f"[DEEPSORT] Added detection {i}: bbox={bbox_xywh}, conf={conf:.2f}")
                else:
                    print(f"[DEEPSORT] Rejected detection {i}: invalid bbox or low confidence")
            else:
                print(f"[DEEPSORT] Rejected detection {i}: invalid bbox format")

        print(f"[DEEPSORT] Processing {len(ds_detections)} valid detections")

        # Update tracker with frame for appearance features
        if frame is not None:
            tracks = self.tracker.update_tracks(ds_detections, frame=frame)
        else:
            tracks = self.tracker.update_tracks(ds_detections)
        
        # Process results and check for duplicate IDs
        results = []
        current_ids = []
        
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            track_id = track.track_id
            ltrb = track.to_ltrb()
            conf = track.det_conf if hasattr(track, 'det_conf') else 0.0
            class_id = track.det_class if hasattr(track, 'det_class') else -1
            
            # Check for duplicate IDs
            if track_id in current_ids:
                print(f"[DEEPSORT ERROR] DUPLICATE ID DETECTED: {track_id}")
                continue  # Skip this duplicate
            
            current_ids.append(track_id)
            
            # Convert back to [x1, y1, x2, y2] format
            x1, y1, x2, y2 = ltrb
            bbox_xyxy = [x1, y1, x2, y2]
            
            results.append({
                'id': track_id, 
                'bbox': bbox_xyxy, 
                'confidence': conf, 
                'class_id': class_id
            })
            
            conf_str = f"{conf:.2f}" if conf is not None else "None"
            print(f"[DEEPSORT] Track ID={track_id}: bbox={bbox_xyxy}, conf={conf_str}")
        
        # Update ID counter for statistics
        for track_id in current_ids:
            self.track_id_counter[track_id] = self.track_id_counter.get(track_id, 0) + 1
        
        print(f"[DEEPSORT] Returning {len(results)} confirmed tracks")
        return results
