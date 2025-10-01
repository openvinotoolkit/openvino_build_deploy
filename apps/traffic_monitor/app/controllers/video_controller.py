from PySide6.QtCore import QObject, Signal, QThread, Qt, QMutex, QWaitCondition, QTimer, Slot
from PySide6.QtGui import QImage, QPixmap
import cv2
import time
import numpy as np
from datetime import datetime
from collections import deque
from typing import Dict, List, Optional
import os
import sys
import math

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utilities
from utils.annotation_utils import (
    draw_detections, 
    draw_performance_metrics,
    resize_frame_for_display,
    convert_cv_to_qimage,
    convert_cv_to_pixmap,
    pipeline_with_violation_line
)

# Import enhanced annotation utilities
from utils.enhanced_annotation_utils import (
    enhanced_draw_detections,
    draw_performance_overlay,
    enhanced_cv_to_qimage,
    enhanced_cv_to_pixmap
)

# Import traffic light color detection utilities
from red_light_violation_pipeline import RedLightViolationPipeline
from utils.traffic_light_utils import detect_traffic_light_color, draw_traffic_light_status, ensure_traffic_light_color
from utils.crosswalk_utils import detect_crosswalk_and_violation_line, draw_violation_line, get_violation_line_y
from controllers.bytetrack_tracker import ByteTrackVehicleTracker
TRAFFIC_LIGHT_CLASSES = ["traffic light", "trafficlight", "tl"]
TRAFFIC_LIGHT_NAMES = ['trafficlight', 'traffic light', 'tl', 'signal']

def normalize_class_name(class_name):
    """Normalizes class names from different models/formats to a standard name"""
    if not class_name:
        return ""
    
    name_lower = class_name.lower()
    
    # Traffic light variants
    if name_lower in ['traffic light', 'trafficlight', 'traffic_light', 'tl', 'signal']:
        return 'traffic light'
    
    # Keep specific vehicle classes (car, truck, bus) separate
    # Just normalize naming variations within each class
    if name_lower in ['car', 'auto', 'automobile']:
        return 'car'
    elif name_lower in ['truck']:
        return 'truck'
    elif name_lower in ['bus']:
        return 'bus'
    elif name_lower in ['motorcycle', 'scooter', 'motorbike', 'bike']:
        return 'motorcycle'
    
    # Person variants
    if name_lower in ['person', 'pedestrian', 'human']:
        return 'person'
    
    # Other common classes can be added here
    
    return class_name

def is_traffic_light(class_name):
    """Helper function to check if a class name is a traffic light with normalization"""
    if not class_name:
        return False
    normalized = normalize_class_name(class_name)
    return normalized == 'traffic light'

class VideoController(QObject):      
    frame_ready = Signal(object, object, dict)  # QPixmap, detections, metrics
    raw_frame_ready = Signal(np.ndarray, list, float)  # frame, detections, fps
    frame_np_ready = Signal(np.ndarray)  # Direct NumPy frame signal for display
    stats_ready = Signal(dict)  # Dictionary with stats (fps, detection_time, traffic_light)
    violation_detected = Signal(dict)  # Signal emitted when a violation is detected
    progress_ready = Signal(int, int, float)  # value, max_value, timestamp
    device_info_ready = Signal(dict)  # Signal to emit device info to the UI
    auto_select_model_device = Signal()  # Signal for UI to request auto model/device selection
    performance_stats_ready = Signal(dict)  # NEW: Signal for performance tab (fps, inference, device, res)
    violations_batch_ready = Signal(list)  # NEW: Signal to emit a batch of violations
    pause_state_changed = Signal(bool)  # Signal emitted when pause state changes (True=paused, False=playing)
    
    def __init__(self, model_manager=None):
        """
        Initialize video controller.
        
        Args:
            model_manager: Model manager instance for detection and violation
        """        
        super().__init__()
        print("Loaded advanced VideoController from video_controller_new.py")  # DEBUG: Confirm correct controller
        
        self._running = False
        self._paused = False  # Add pause state
        self._last_frame = None  # Store last frame for VLM analysis during pause
        self._last_analysis_data = {}  # Store last analysis data for VLM
        self._reset_video_position = False  # Flag to reset video position on next start
        self.source = None
        self.source_type = None
        self.source_fps = 0
        self.performance_metrics = {}
        self.mutex = QMutex()
        self.pause_condition = QWaitCondition()  # Add wait condition for pause
        
        # Performance tracking
        self.processing_times = deque(maxlen=100)  # Store last 100 processing times
        self.fps_history = deque(maxlen=100)       # Store last 100 FPS values
        self.start_time = time.time()
        self.frame_count = 0
        self.actual_fps = 0.0
        
        self.model_manager = model_manager
        self.inference_model = None
        self.tracker = None
        
        # Initialize device tracking
        if self.model_manager and hasattr(self.model_manager, 'config'):
            self.current_device = self.model_manager.config.get("detection", {}).get("device", "CPU")
        else:
            self.current_device = "CPU"
        print(f"🔧 Video Controller: Initialized with device: {self.current_device}")
        
        self.current_frame = None
        self.current_detections = []
        
        # Traffic light state tracking
        self.latest_traffic_light = {"color": "unknown", "confidence": 0.0}
        
        # Vehicle tracking settings
        self.vehicle_history = {}  # Dictionary to store vehicle position history
        self.vehicle_statuses = {}  # Track stable movement status
        self.movement_threshold = 1.5  # ADJUSTED: More balanced movement detection (was 0.8)
        self.min_confidence_threshold = 0.3  # FIXED: Lower threshold for better detection (was 0.5)
        
        # Enhanced violation detection settings
        self.position_history_size = 20  # Increased from 10 to track longer history
        self.crossing_check_window = 8   # Check for crossings over the last 8 frames instead of just 2
        self.max_position_jump = 50      # Maximum allowed position jump between frames (detect ID switches)
        
        # Set up violation detection
        try:
            from controllers.red_light_violation_detector import RedLightViolationDetector
            self.violation_detector = RedLightViolationDetector()
            print("✅ Red light violation detector initialized")
        except Exception as e:
            self.violation_detector = None
            print(f"❌ Could not initialize violation detector: {e}")
            
        # Import crosswalk detection
        try:
            self.detect_crosswalk_and_violation_line = detect_crosswalk_and_violation_line
            # self.draw_violation_line = draw_violation_line
            print("✅ Crosswalk detection utilities imported")
        except Exception as e:
            print(f"❌ Could not import crosswalk detection: {e}")
            self.detect_crosswalk_and_violation_line = lambda frame, *args: (None, None, {})
            # self.draw_violation_line = lambda frame, *args, **kwargs: frame
        
        # Configure thread
        self.thread = QThread()
        self.moveToThread(self.thread)
        self.thread.started.connect(self._run)
          # Performance measurement
        self.mutex = QMutex()
        self.condition = QWaitCondition()
        self.performance_metrics = {
            'FPS': 0.0,
            'Detection (ms)': 0.0,
            'Total (ms)': 0.0
        }
        
        # Setup render timer with more aggressive settings for UI updates
        self.render_timer = QTimer()
        self.render_timer.timeout.connect(self._process_frame)
        
        # Frame buffer
        self.current_frame = None
        self.current_detections = []
        self.current_violations = []
        
        # Debug counter for monitoring frame processing
        self.debug_counter = 0
        self.violation_frame_counter = 0  # Add counter for violation processing
        
        # Initialize the traffic light color detection pipeline
        self.cv_violation_pipeline = RedLightViolationPipeline(debug=True)
        
        # Initialize vehicle tracker
        self.vehicle_tracker = ByteTrackVehicleTracker()
        
        # Add red light violation system
        # self.red_light_violation_system = RedLightViolationSystem()
        
        # Initialize scene analytics adapter
        try:
            from utils.scene_analytics import SceneAnalyticsAdapter
            self.scene_analytics = SceneAnalyticsAdapter(camera_id="desktop_main")
            self.scene_analytics.object_detected.connect(self._on_scene_object_detected)
            self.scene_analytics.scene_analytics_updated.connect(self._on_scene_analytics_updated)
            self.scene_analytics.roi_event_detected.connect(self._on_roi_event_detected)
            print("✅ Scene analytics adapter initialized")
        except Exception as e:
            self.scene_analytics = None
            print(f"❌ Could not initialize scene analytics: {e}")
        
    def refresh_model_info(self):
        """Force refresh of model information for performance display"""
        if hasattr(self, 'model_manager') and self.model_manager:
            print("🔄 Refreshing model information in video controller")
            # The model info will be refreshed in the next stats update
            # Force current device update from config
            if hasattr(self.model_manager, 'config') and 'detection' in self.model_manager.config:
                self.current_device = self.model_manager.config['detection'].get('device', 'CPU')
                print(f"🔄 Updated current device to: {self.current_device}")
        
    def set_source(self, source):
        """
        Set video source (file path, camera index, or URL)
        
        Args:
            source: Video source - can be a camera index (int), file path (str), 
                   or URL (str). If None, defaults to camera 0.
                   
        Returns:
            bool: True if source was set successfully, False otherwise
        """
        print(f"🎬 VideoController.set_source called with: {source} (type: {type(source)})")
        
        # Store current state
        was_running = self._running
        
        # Stop current processing if running
        if self._running:
            print("⏹️ Stopping current video processing")
            self.stop()
        
        try:
            # Handle source based on type with better error messages
            if source is None:
                print("⚠️ Received None source, defaulting to camera 0")
                self.source = 0
                self.source_type = "camera"
                
            elif isinstance(source, str) and source.strip():
                if os.path.exists(source):
                    # Valid file path
                    self.source = source
                    self.source_type = "file"
                    print(f"📄 Source set to file: {self.source}")
                elif source.lower().startswith(("http://", "https://", "rtsp://", "rtmp://")):
                    # URL stream
                    self.source = source
                    self.source_type = "url"
                    print(f"🌐 Source set to URL stream: {self.source}")
                elif source.isdigit():
                    # String camera index (convert to int)
                    self.source = int(source)
                    self.source_type = "camera"
                    print(f"📹 Source set to camera index: {self.source}")
                else:
                    # Try as device path or special string
                    self.source = source
                    self.source_type = "device"
                    print(f"📱 Source set to device path: {self.source}")
                    
            elif isinstance(source, int):
                # Camera index
                self.source = source
                self.source_type = "camera"
                print(f"📹 Source set to camera index: {self.source}")
                
            else:
                # Unrecognized - default to camera 0 with warning
                print(f"⚠️ Unrecognized source type: {type(source)}, defaulting to camera 0")
                self.source = 0
                self.source_type = "camera"
        except Exception as e:
            print(f"❌ Error setting source: {e}")
            self.source = 0
            self.source_type = "camera"
            return False
        
        # Get properties of the source (fps, dimensions, etc)
        print(f"🔍 Getting properties for source: {self.source}")
        success = self._get_source_properties()
        
        if success:
            print(f"✅ Successfully configured source: {self.source} ({self.source_type})")
            
            # Reset ByteTrack tracker for new source to ensure IDs start from 1
            if hasattr(self, 'vehicle_tracker') and self.vehicle_tracker is not None:
                try:
                    print("🔄 Resetting vehicle tracker for new source")
                    self.vehicle_tracker.reset()
                except Exception as e:
                    print(f"⚠️ Could not reset vehicle tracker: {e}")
            
            # Mark that we need to reset video position on next start (for video files)
            if self.source_type == "file":
                print("📍 Marking video file for position reset on next start")
                self._reset_video_position = True
            else:
                self._reset_video_position = False
            
            # Emit successful source change
            self.stats_ready.emit({
                'source_changed': True,
                'source_type': self.source_type,
                'fps': self.source_fps if hasattr(self, 'source_fps') else 0,
                'dimensions': f"{self.frame_width}x{self.frame_height}" if hasattr(self, 'frame_width') else "unknown"
            })
            
            # Restart if previously running
            if was_running:
                print("▶️ Restarting video processing with new source")
                self.start()
        else:
            print(f"❌ Failed to configure source: {self.source}")
            # Notify UI about the error
            self.stats_ready.emit({
                'source_changed': False,
                'error': f"Invalid video source: {self.source}",
                'source_type': self.source_type,
                'fps': 0,
                'detection_time_ms': "0",
                'traffic_light_color': {"color": "unknown", "confidence": 0.0}
            })
            
            return False
            
        # Return success status
        return success
    
    def _get_source_properties(self):
      
        try:
            print(f"🔍 Opening video source for properties check: {self.source}")
            cap = cv2.VideoCapture(self.source)
            
            
            if not cap.isOpened():
                print(f"❌ Failed to open video source: {self.source}")
                return False
                
            
            self.source_fps = cap.get(cv2.CAP_PROP_FPS)
             
            
            self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))                
            self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            
            ret, test_frame = cap.read()
            if not ret or test_frame is None:
                print("⚠️ Could not read test frame from source")
                
                if self.source_type == "camera":
                    print("🔄 Retrying camera initialization...")
                    time.sleep(1.0)  
                    ret, test_frame = cap.read()
                    if not ret or test_frame is None:
                        print("❌ Camera initialization failed after retry")
                        cap.release()
                        return False
                else:
                    print("❌ Could not read frames from video source")
                    cap.release()
                    return False
                
            # Release the capture
            cap.release()
            
            print(f"✅ Video source properties: {self.frame_width}x{self.frame_height}, {self.source_fps} FPS")
            return True
            
        except Exception as e:
            print(f"❌ Error getting source properties: {e}")
            return False
            return False
            
    def start(self):
        """Start video processing"""
        if not self._running:
            self._running = True
            self.start_time = time.time()
            self.frame_count = 0
            self.debug_counter = 0
            print("DEBUG: Starting video processing thread")
            
            # Reset notification flags for new session
            if hasattr(self, '_no_traffic_light_notified'):
                delattr(self, '_no_traffic_light_notified')
            
            # Reset ByteTrack tracker to ensure IDs start from 1
            if hasattr(self, 'vehicle_tracker') and self.vehicle_tracker is not None:
                try:
                    print("🔄 Resetting vehicle tracker for new session")
                    self.vehicle_tracker.reset()
                except Exception as e:
                    print(f"⚠️ Could not reset vehicle tracker: {e}")
            
            # Start the processing thread - add more detailed debugging
            if not self.thread.isRunning():
                print("🚀 Thread not running, starting now...")
                try:
                    self.thread.start()
                    print("✅ Thread started successfully")
                    print(f"🔄 Thread state: running={self.thread.isRunning()}, finished={self.thread.isFinished()}")
                except Exception as e:
                    print(f"❌ Failed to start thread: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("⚠️ Thread is already running!")
                print(f"🔄 Thread state: running={self.thread.isRunning()}, finished={self.thread.isFinished()}")
            
            # Start the render timer with a very aggressive interval (10ms = 100fps)
            # This ensures we can process frames as quickly as possible
            print("⏱️ Starting render timer...")
            self.render_timer.start(10)
            print("✅ Render timer started at 100Hz")
    
    def stop(self):
        """Stop video processing"""
        if self._running:
            print("DEBUG: Stopping video processing")
            self._running = False
            
            # If paused, wake up the thread so it can exit
            self.mutex.lock()
            try:
                self._paused = False
                self.pause_condition.wakeAll()
            finally:
                self.mutex.unlock()
            
            # Stop the render timer
            if hasattr(self, 'render_timer') and self.render_timer is not None:
                try:
                    self.render_timer.stop()
                    print("✅ Render timer stopped")
                except RuntimeError as e:
                    print(f"⚠️ Error stopping render timer: {e}")
            
            # Properly terminate the thread
            if hasattr(self, 'thread') and self.thread is not None:
                try:
                    if self.thread.isRunning():
                        print("🔄 Waiting for thread to finish...")
                        self.thread.quit()
                        if not self.thread.wait(3000):  # Wait 3 seconds max
                            print("⚠️ Thread didn't finish gracefully, forcing termination")
                            self.thread.terminate()
                            if not self.thread.wait(1000):  # Wait 1 more second
                                print("❌ Thread termination failed")
                            else:
                                print("✅ Thread terminated")
                        else:
                            print("✅ Thread finished gracefully")
                except RuntimeError as e:
                    print(f"⚠️ Error during thread cleanup: {e}")
            
            # Clear the current frame
            self.mutex.lock()
            try:
                self.current_frame = None
                self._last_frame = None
            finally:
                self.mutex.unlock()
            
            print("DEBUG: Video processing stopped")

    def __del__(self):
        print("[VideoController] __del__ called. Cleaning up thread and timer.")
        self.stop()
        if self.thread.isRunning():
            self.thread.quit()
            self.thread.wait(1000)
        self.render_timer.stop()
    
    def capture_snapshot(self) -> np.ndarray:
        """Capture current frame"""
        if self.current_frame is not None:
            return self.current_frame.copy()
        return None
        
    def _run(self):
        """Main processing loop (runs in thread)"""
        try:
            # Print the source we're trying to open
            print(f"DEBUG: Opening video source: {self.source} (type: {type(self.source)})")
            
            cap = None  # Initialize capture variable
            
            # Try to open source with more robust error handling
            max_retries = 3
            retry_delay = 1.0  # seconds
            
            # Function to attempt opening the source with multiple retries
            def try_open_source(src, retries=max_retries, delay=retry_delay):
                for attempt in range(1, retries + 1):
                    print(f"🎥 Opening source (attempt {attempt}/{retries}): {src}")
                    try:
                        capture = cv2.VideoCapture(src)
                        if capture.isOpened():
                            # Try to read a test frame to confirm it's working
                            ret, test_frame = capture.read()
                            if ret and test_frame is not None:
                                print(f"✅ Source opened successfully: {src}")
                                # Reset capture position for file sources
                                if isinstance(src, str) and os.path.exists(src):
                                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                                return capture
                            else:
                                print(f"⚠️ Source opened but couldn't read frame: {src}")
                                capture.release()
                        else:
                            print(f"⚠️ Failed to open source: {src}")
                            
                        # Retry after delay
                        if attempt < retries:
                            print(f"Retrying in {delay:.1f} seconds...")
                            time.sleep(delay)
                    except Exception as e:
                        print(f"❌ Error opening source {src}: {e}")
                        if attempt < retries:
                            print(f"Retrying in {delay:.1f} seconds...")
                            time.sleep(delay)
                
                print(f"❌ Failed to open source after {retries} attempts: {src}")
                return None
            
            # Handle different source types
            if isinstance(self.source, str) and os.path.exists(self.source):
                # It's a valid file path
                print(f"📄 Opening video file: {self.source}")
                cap = try_open_source(self.source)
                
            elif isinstance(self.source, int) or (isinstance(self.source, str) and self.source.isdigit()):
                # It's a camera index
                camera_idx = int(self.source) if isinstance(self.source, str) else self.source
                print(f"📹 Opening camera with index: {camera_idx}")
                
                # For cameras, try with different backend options if it fails
                cap = try_open_source(camera_idx)
                
                # If failed, try with DirectShow backend on Windows
                if cap is None and os.name == 'nt':
                    print("🔄 Trying camera with DirectShow backend...")
                    cap = try_open_source(camera_idx + cv2.CAP_DSHOW)
                    
            else:
                # Try as a string source (URL or device path)
                print(f"🌐 Opening source as string: {self.source}")
                cap = try_open_source(str(self.source))
                
            # Check if we successfully opened the source
            if cap is None:
                print(f"❌ Failed to open video source after all attempts: {self.source}")
                # Notify UI about the error
                self.stats_ready.emit({
                    'error': f"Could not open video source: {self.source}",
                    'fps': "0",
                    'detection_time_ms': "0",
                    'traffic_light_color': {"color": "unknown", "confidence": 0.0}
                })
                return
                    
            # Check again to ensure capture is valid
            if not cap or not cap.isOpened():
                print(f"ERROR: Could not open video source {self.source}")
                # Emit a signal to notify UI about the error
                self.stats_ready.emit({
                    'error': f"Failed to open video source: {self.source}",
                    'fps': "0",
                    'detection_time_ms': "0",
                    'traffic_light_color': {"color": "unknown", "confidence": 0.0}
                })
                return
                
            # Configure frame timing based on source FPS
            frame_time = 1.0 / self.source_fps if self.source_fps > 0 else 0.033
            prev_time = time.time()
            
            # Log successful opening
            print(f"SUCCESS: Video source opened: {self.source}")
            print(f"Source info - FPS: {self.source_fps}, Size: {self.frame_width}x{self.frame_height}")
            
            # Additional position reset check for video files when flag is set
            if self._reset_video_position and self.source_type == "file":
                print("🔄 Resetting video position to beginning due to new file selection")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self._reset_video_position = False  # Clear the flag
                print("✅ Video position reset to frame 0")
            
              # Main processing loop
            frame_error_count = 0
            max_consecutive_errors = 10
            
            # --- Violation Rule Functions ---
            def point_in_polygon(point, polygon):
                # Simple point-in-rect for now; replace with polygon logic if needed
                x, y = point
                x1, y1, w, h = polygon
                return x1 <= x <= x1 + w and y1 <= y <= y1 + h

            def calculate_speed(track, history_dict):
                # Use last two positions for speed
                hist = history_dict.get(track['id'], [])
                if len(hist) < 2:
                    return 0.0
                (x1, y1), t1 = hist[-2]
                (x2, y2), t2 = hist[-1]
                dist = ((x2-x1)**2 + (y2-y1)**2)**0.5
                dt = max(t2-t1, 1e-3)
                return dist / dt

            def check_vehicle_pedestrian_conflict(vehicle_track, pedestrian_tracks, crosswalk_poly, light_state):
                if light_state != 'green':
                    return False
                if not point_in_polygon(vehicle_track['center'], crosswalk_poly):
                    return False
                for ped in pedestrian_tracks:
                    if point_in_polygon(ped['center'], crosswalk_poly):
                        return True
                return False

            def check_stop_on_crosswalk(vehicle_track, crosswalk_poly, light_state, history_dict):
                if light_state != 'red':
                    return False
                is_inside = point_in_polygon(vehicle_track['center'], crosswalk_poly)
                speed = calculate_speed(vehicle_track, history_dict)
                return is_inside and speed < 0.5

            def check_amber_overspeed(vehicle_track, light_state, amber_start_time, stopline_poly, history_dict, speed_limit_px_per_sec):
                if light_state != 'amber':
                    return False
                if not point_in_polygon(vehicle_track['center'], stopline_poly):
                    return False
                current_time = time.time()
                speed = calculate_speed(vehicle_track, history_dict)
                if current_time > amber_start_time and speed > speed_limit_px_per_sec:
                    return True
                return False
            # --- End Violation Rule Functions ---
            
            while self._running and cap.isOpened():
                # Handle pause state with better error handling
                self.mutex.lock()
                try:
                    if self._paused:
                        print("[VideoController] Video paused, waiting...")
                        # Use a timeout to prevent infinite waiting
                        self.pause_condition.wait(self.mutex, 1000)  # 1 second timeout
                        if self._paused:  # Still paused after timeout
                            self.mutex.unlock()
                            continue  # Check again
                        print("[VideoController] Video resumed")
                finally:
                    self.mutex.unlock()
                
                # Exit if we're no longer running (could have stopped while paused)
                if not self._running:
                    break
                
                try:
                    ret, frame = cap.read()
                    # Add critical frame debugging
                    print(f"🟡 Frame read attempt: ret={ret}, frame={None if frame is None else frame.shape}")
                    
                    if not ret or frame is None:
                        frame_error_count += 1
                        print(f"⚠️ Frame read error ({frame_error_count}/{max_consecutive_errors})")
                        
                        if frame_error_count >= max_consecutive_errors:
                            print("❌ Too many consecutive frame errors, stopping video thread")
                            break
                            
                        # Skip this iteration and try again
                        time.sleep(0.1)  # Wait a bit before trying again
                        continue
                    
                    # Reset the error counter if we successfully got a frame
                    frame_error_count = 0
                    
                    # Store the last frame for VLM analysis during pause
                    self._last_frame = frame.copy()
                    print(f"🟢 Last frame stored for VLM: {frame.shape}")
                    
                except Exception as e:
                    print(f"❌ Critical error reading frame: {e}")
                    frame_error_count += 1
                    if frame_error_count >= max_consecutive_errors:
                        print("❌ Too many errors, stopping video thread")
                        break
                    continue
                    
                # Detection and violation processing
                process_start = time.time()
                
                # Process detections
                detection_start = time.time()
                detections = []
                if self.model_manager:
                    detections = self.model_manager.detect(frame)
                    
                    # Normalize class names for consistency and check for traffic lights
                    traffic_light_indices = []
                    for i, det in enumerate(detections):
                        if 'class_name' in det:
                            original_name = det['class_name']
                            normalized_name = normalize_class_name(original_name)
                            
                            # Keep track of traffic light indices
                            if normalized_name == 'traffic light' or original_name == 'traffic light':
                                traffic_light_indices.append(i)
                                
                            if original_name != normalized_name:
                                print(f"📊 Normalized class name: '{original_name}' -> '{normalized_name}'")
                                
                            det['class_name'] = normalized_name
                            
                    # Ensure we have at least one traffic light for debugging
                    if not traffic_light_indices and self.source_type == 'video':
                        print("⚠️ No traffic lights detected, checking for objects that might be traffic lights...")
                        
                        # Try lowering the confidence threshold specifically for traffic lights
                        # This is only for debugging purposes
                        if self.model_manager and hasattr(self.model_manager, 'detect'):
                            try:
                                low_conf_detections = self.model_manager.detect(frame, conf_threshold=0.2)
                                for det in low_conf_detections:
                                    if 'class_name' in det and det['class_name'] == 'traffic light':
                                        if det not in detections:
                                            print(f"🚦 Found low confidence traffic light: {det['confidence']:.2f}")
                                            detections.append(det)
                            except:
                                pass
                            
                detection_time = (time.time() - detection_start) * 1000
                
                # Violation detection is disabled
                violation_start = time.time()
                violations = []
                # if self.model_manager and detections:
                #     violations = self.model_manager.detect_violations(
                #         detections, frame, time.time()
                #     )
                violation_time = (time.time() - violation_start) * 1000
                
                # Update tracking if available
                if self.model_manager:
                    detections = self.model_manager.update_tracking(detections, frame)
                    # If detections are returned as tuples, convert to dicts for downstream code
                    if detections and isinstance(detections[0], tuple):
                        # Convert (id, bbox, conf, class_id) to dict
                        detections = [
                            {'id': d[0], 'bbox': d[1], 'confidence': d[2], 'class_id': d[3]}
                            for d in detections
                        ]
                
                # Calculate timing metrics
                process_time = (time.time() - process_start) * 1000
                self.processing_times.append(process_time)
                
                # Update FPS
                now = time.time()
                self.frame_count += 1
                elapsed = now - self.start_time
                if elapsed > 0:
                    self.actual_fps = self.frame_count / elapsed
                    
                fps_smoothed = 1.0 / (now - prev_time) if now > prev_time else 0
                prev_time = now
                  # Update metrics
                self.performance_metrics = {
                    'FPS': f"{fps_smoothed:.1f}",
                    'Detection (ms)': f"{detection_time:.1f}",
                    'Total (ms)': f"{process_time:.1f}"
                }
                
                # Store current frame data (thread-safe)
                self.mutex.lock()
                self.current_frame = frame.copy()
                self.current_detections = detections
                self.mutex.unlock()
                
                # --- SCENE ANALYTICS PROCESSING ---
                # Process detections through scene analytics if available
                if self.scene_analytics:
                    try:
                        scene_analytics_data = self.scene_analytics.process_frame(frame, detections)
                        # Scene analytics automatically emit signals that we handle above
                    except Exception as e:
                        print(f"Error in scene analytics processing: {e}")
                
                # Process frame with annotations before sending to UI
                annotated_frame = frame.copy()
                
                # --- VIOLATION DETECTION LOGIC (Run BEFORE drawing boxes) ---
                # First get violation information so we can color boxes appropriately
                violating_vehicle_ids = set()  # Track which vehicles are violating
                violations = []
                
                # Initialize traffic light variables
                traffic_lights = []
                has_traffic_lights = False
                
                # Handle multiple traffic lights with consensus approach
                traffic_light_count = 0
                for det in detections:
                    if is_traffic_light(det.get('class_name')):
                        has_traffic_lights = True
                        traffic_light_count += 1
                        if 'traffic_light_color' in det:
                            light_info = det['traffic_light_color']
                            traffic_lights.append({'bbox': det['bbox'], 'color': light_info.get('color', 'unknown'), 'confidence': light_info.get('confidence', 0.0)})
                
                print(f"[TRAFFIC LIGHT] Detected {traffic_light_count} traffic light(s), has_traffic_lights={has_traffic_lights}")
                if has_traffic_lights:
                    print(f"[TRAFFIC LIGHT] Traffic light colors: {[tl.get('color', 'unknown') for tl in traffic_lights]}")
                
                # Get traffic light position for crosswalk detection
                traffic_light_position = None
                if has_traffic_lights:
                    for det in detections:
                        if is_traffic_light(det.get('class_name')) and 'bbox' in det:
                            traffic_light_bbox = det['bbox']
                            # Extract center point from bbox for crosswalk utils
                            x1, y1, x2, y2 = traffic_light_bbox
                            traffic_light_position = ((x1 + x2) // 2, (y1 + y2) // 2)
                            break

                # Run crosswalk detection ONLY if traffic light is detected
                crosswalk_bbox, violation_line_y, debug_info = None, None, {}
                if has_traffic_lights and traffic_light_position is not None:
                    try:
                        print(f"[CROSSWALK] Traffic light detected at {traffic_light_position}, running crosswalk detection")
                        # Use new crosswalk_utils logic only when traffic light exists
                        annotated_frame, crosswalk_bbox, violation_line_y, debug_info = detect_crosswalk_and_violation_line(
                            annotated_frame,
                            traffic_light_position=traffic_light_position
                        )
                        print(f"[CROSSWALK] Detection result: crosswalk_bbox={crosswalk_bbox is not None}, violation_line_y={violation_line_y}")
                        # --- Draw crosswalk region if detected and close to traffic light ---
                        # (REMOVED: Do not draw crosswalk box or label)
                        # if crosswalk_bbox is not None:
                        #     x, y, w, h = map(int, crosswalk_bbox)
                        #     tl_x, tl_y = traffic_light_position
                        #     crosswalk_center_y = y + h // 2
                        #     distance = abs(crosswalk_center_y - tl_y)
                        #     print(f"[CROSSWALK DEBUG] Crosswalk bbox: {crosswalk_bbox}, Traffic light: {traffic_light_position}, vertical distance: {distance}")
                        #     if distance < 120:
                        #         cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                        #         cv2.putText(annotated_frame, "Crosswalk", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        #     # Top and bottom edge of crosswalk
                        #     top_edge = y
                        #     bottom_edge = y + h
                        #     if abs(tl_y - top_edge) < abs(tl_y - bottom_edge):
                        #         crosswalk_edge_y = top_edge
                        #     else:
                        #         crosswalk_edge_y = bottom_edge
                        if crosswalk_bbox is not None:
                            x, y, w, h = map(int, crosswalk_bbox)
                            tl_x, tl_y = traffic_light_position
                            crosswalk_center_y = y + h // 2
                            distance = abs(crosswalk_center_y - tl_y)
                            print(f"[CROSSWALK DEBUG] Crosswalk bbox: {crosswalk_bbox}, Traffic light: {traffic_light_position}, vertical distance: {distance}")
                            # Top and bottom edge of crosswalk
                            top_edge = y
                            bottom_edge = y + h
                            if abs(tl_y - top_edge) < abs(tl_y - bottom_edge):
                                crosswalk_edge_y = top_edge
                            else:
                                crosswalk_edge_y = bottom_edge
                    except Exception as e:
                        print(f"[ERROR] Crosswalk detection failed: {e}")
                        crosswalk_bbox, violation_line_y, debug_info = None, None, {}
                else:
                    print(f"[CROSSWALK] No traffic light detected (has_traffic_lights={has_traffic_lights}), skipping crosswalk detection")
                    # NO crosswalk detection without traffic light
                    violation_line_y = None
                
                # Check if crosswalk is detected
                crosswalk_detected = crosswalk_bbox is not None
                stop_line_detected = debug_info.get('stop_line') is not None
                
                # ALWAYS process vehicle tracking (moved outside violation logic)
                tracked_vehicles = []
                if hasattr(self, 'vehicle_tracker') and self.vehicle_tracker is not None:
                    try:
                        # Filter vehicle detections
                        vehicle_classes = ['car', 'truck', 'bus', 'motorcycle', 'van', 'bicycle']
                        vehicle_dets = []
                        h, w = frame.shape[:2]
                        
                        print(f"[TRACK DEBUG] Processing {len(detections)} total detections")
                        
                        for det in detections:
                            if (det.get('class_name') in vehicle_classes and 
                                'bbox' in det and 
                                det.get('confidence', 0) > self.min_confidence_threshold):
                                
                                # Check bbox dimensions
                                bbox = det['bbox']
                                x1, y1, x2, y2 = bbox
                                box_w, box_h = x2-x1, y2-y1
                                box_area = box_w * box_h
                                area_ratio = box_area / (w * h)
                                
                                print(f"[TRACK DEBUG] Vehicle {det.get('class_name')} conf={det.get('confidence'):.2f}, area_ratio={area_ratio:.4f}")
                                
                                if 0.001 <= area_ratio <= 0.25:
                                    vehicle_dets.append(det)
                                    print(f"[TRACK DEBUG] Added vehicle: {det.get('class_name')} conf={det.get('confidence'):.2f}")
                                else:
                                    print(f"[TRACK DEBUG] Rejected vehicle: area_ratio={area_ratio:.4f} not in range [0.001, 0.25]")
                        
                        print(f"[TRACK DEBUG] Filtered to {len(vehicle_dets)} vehicle detections")
                        
                        # Update tracker
                        if len(vehicle_dets) > 0:
                            print(f"[TRACK DEBUG] Updating tracker with {len(vehicle_dets)} vehicles...")
                            tracks = self.vehicle_tracker.update(vehicle_dets, frame)
                            # Filter out tracks without bbox to avoid warnings
                            valid_tracks = []
                            for track in tracks:
                                bbox = None
                                if isinstance(track, dict):
                                    bbox = track.get('bbox', None)
                                else:
                                    bbox = getattr(track, 'bbox', None)
                                if bbox is not None:
                                    valid_tracks.append(track)
                                else:
                                    print(f"Warning: Track has no bbox, skipping: {track}")
                            tracks = valid_tracks
                            print(f"[TRACK DEBUG] Tracker returned {len(tracks)} tracks (after bbox filter)")
                        else:
                            print(f"[TRACK DEBUG] No vehicles to track, skipping tracker update")
                            tracks = []
                        
                        # Process each tracked vehicle
                        tracked_vehicles = []
                        track_ids_seen = []
                        
                        for track in tracks:
                            track_id = track['id']
                            bbox = track['bbox']
                            x1, y1, x2, y2 = map(float, bbox)
                            center_y = (y1 + y2) / 2
                            
                            # Check for duplicate IDs
                            if track_id in track_ids_seen:
                                print(f"[TRACK ERROR] Duplicate ID detected: {track_id}")
                            track_ids_seen.append(track_id)
                            
                            print(f"[TRACK DEBUG] Processing track ID={track_id} bbox={bbox}")
                            
                            # Initialize or update vehicle history
                            if track_id not in self.vehicle_history:
                                from collections import deque
                                self.vehicle_history[track_id] = deque(maxlen=self.position_history_size)
                            
                            # Initialize vehicle status if not exists
                            if track_id not in self.vehicle_statuses:
                                self.vehicle_statuses[track_id] = {
                                    'recent_movement': [],
                                    'violation_history': [],
                                    'crossed_during_red': False,
                                    'last_position': None,  # Track last position for jump detection
                                    'suspicious_jumps': 0   # Count suspicious position jumps
                                }
                            
                            # Detect suspicious position jumps (potential ID switches)
                            if self.vehicle_statuses[track_id]['last_position'] is not None:
                                last_y = self.vehicle_statuses[track_id]['last_position']
                                center_y = (y1 + y2) / 2
                                position_jump = abs(center_y - last_y)
                                
                                if position_jump > self.max_position_jump:
                                    self.vehicle_statuses[track_id]['suspicious_jumps'] += 1
                                    print(f"[TRACK WARNING] Vehicle ID={track_id} suspicious position jump: {last_y:.1f} -> {center_y:.1f} (jump={position_jump:.1f})")
                                    
                                    # If too many suspicious jumps, reset violation status to be safe
                                    if self.vehicle_statuses[track_id]['suspicious_jumps'] > 2:
                                        print(f"[TRACK RESET] Vehicle ID={track_id} has too many suspicious jumps, resetting violation status")
                                        self.vehicle_statuses[track_id]['crossed_during_red'] = False
                                        self.vehicle_statuses[track_id]['suspicious_jumps'] = 0
                            
                            # Update position history and last position
                            self.vehicle_history[track_id].append(center_y)
                            self.vehicle_statuses[track_id]['last_position'] = center_y
                            
                            # BALANCED movement detection - detect clear movement while avoiding false positives
                            is_moving = False
                            movement_detected = False
                            
                            if len(self.vehicle_history[track_id]) >= 3:  # Require at least 3 frames for movement detection
                                recent_positions = list(self.vehicle_history[track_id])
                                
                                # Check movement over 3 frames for quick response
                                if len(recent_positions) >= 3:
                                    movement_3frames = abs(recent_positions[-1] - recent_positions[-3])
                                    if movement_3frames > self.movement_threshold:  # More responsive threshold
                                        movement_detected = True
                                        print(f"[MOVEMENT] Vehicle ID={track_id} MOVING: 3-frame movement = {movement_3frames:.1f}")
                                
                                # Confirm with longer movement for stability (if available)
                                if len(recent_positions) >= 5:
                                    movement_5frames = abs(recent_positions[-1] - recent_positions[-5])
                                    if movement_5frames > self.movement_threshold * 1.5:  # Moderate threshold for 5 frames
                                        movement_detected = True
                                        print(f"[MOVEMENT] Vehicle ID={track_id} MOVING: 5-frame movement = {movement_5frames:.1f}")
                            
                            # Store historical movement for smoothing - require consistent movement
                            self.vehicle_statuses[track_id]['recent_movement'].append(movement_detected)
                            if len(self.vehicle_statuses[track_id]['recent_movement']) > 4:  # Shorter history for quicker response
                                self.vehicle_statuses[track_id]['recent_movement'].pop(0)
                            
                            # BALANCED: Require majority of recent frames to show movement (2 out of 4)
                            recent_movement_count = sum(self.vehicle_statuses[track_id]['recent_movement'])
                            total_recent_frames = len(self.vehicle_statuses[track_id]['recent_movement'])
                            if total_recent_frames >= 2 and recent_movement_count >= (total_recent_frames * 0.5):  # 50% of frames must show movement
                                is_moving = True
                            
                            print(f"[TRACK DEBUG] Vehicle ID={track_id} is_moving={is_moving} (threshold={self.movement_threshold})")
                            
                            # Initialize as not violating
                            is_violation = False
                            
                            tracked_vehicles.append({
                                'id': track_id,
                                'bbox': bbox,
                                'center_y': center_y,
                                'is_moving': is_moving,
                                'is_violation': is_violation
                            })
                        
                        print(f"[DEBUG] ByteTrack tracked {len(tracked_vehicles)} vehicles")
                        for i, tracked in enumerate(tracked_vehicles):
                            print(f"  Vehicle {i}: ID={tracked['id']}, center_y={tracked['center_y']:.1f}, moving={tracked['is_moving']}, violating={tracked['is_violation']}")
                        
                        # DEBUG: Print all tracked vehicle IDs and their bboxes for this frame
                        if tracked_vehicles:
                            print(f"[DEBUG] All tracked vehicles this frame:")
                            for v in tracked_vehicles:
                                print(f"    ID={v['id']} bbox={v['bbox']} center_y={v.get('center_y', 'NA')}")
                        else:
                            print("[DEBUG] No tracked vehicles this frame!")
                        
                        # Clean up old vehicle data
                        current_track_ids = [tracked['id'] for tracked in tracked_vehicles]
                        self._cleanup_old_vehicle_data(current_track_ids)
                        
                    except Exception as e:
                        print(f"[ERROR] Vehicle tracking failed: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print("[WARN] ByteTrack vehicle tracker not available!")
                
                # Process violations - CHECK VEHICLES THAT CROSS THE LINE OVER A WINDOW OF FRAMES
                # IMPORTANT: Only process violations if BOTH traffic light is detected AND crosswalk is detected AND red light AND violation line exists
                
                # Handle case when no traffic light is detected in video
                if not has_traffic_lights:
                    print("[INFO] No traffic light detected in video - violation detection disabled")
                    # Emit status to UI (only once per session to avoid spam)
                    if not hasattr(self, '_no_traffic_light_notified'):
                        self.stats_ready.emit({
                            'status': 'monitoring_only',
                            'message': 'No traffic light detected - monitoring vehicles only',
                            'violation_detection_active': False,
                            'timestamp': time.time()
                        })
                        self._no_traffic_light_notified = True
                else:
                    # Check if traffic light is red (only when traffic light exists)
                    is_red_light = self.latest_traffic_light and self.latest_traffic_light.get('color') == 'red'
                    
                    # New condition: ALL of these must be true for violation line processing:
                    # 1. Traffic lights detected (has_traffic_lights)
                    # 2. Crosswalk detected (crosswalk_detected) 
                    # 3. Red light is currently active (is_red_light)
                    # 4. Violation line exists (violation_line_y is not None)
                    # 5. Vehicles are being tracked (tracked_vehicles)
                    if (has_traffic_lights and crosswalk_detected and is_red_light and 
                        violation_line_y is not None and tracked_vehicles):
                        print(f"[VIOLATION DEBUG] ALL CONDITIONS MET - Traffic light: {has_traffic_lights}, Crosswalk: {crosswalk_detected}, Red light: {is_red_light}, Line Y: {violation_line_y}, Vehicles: {len(tracked_vehicles)}")
                        
                        # Check each tracked vehicle for violations
                    for tracked in tracked_vehicles:
                        track_id = tracked['id']
                        center_y = tracked['center_y']
                        is_moving = tracked['is_moving']
                        
                        # Get position history for this vehicle
                        position_history = list(self.vehicle_history[track_id])
                        
                        # Enhanced crossing detection: check over a window of frames
                        line_crossed_in_window = False
                        crossing_details = None
                        if len(position_history) >= 2:
                            window_size = min(self.crossing_check_window, len(position_history))
                            for i in range(1, window_size):
                                prev_y = position_history[-(i+1)]  # Earlier position
                                curr_y = position_history[-i]     # Later position
                                # Check if vehicle crossed the line in this frame pair (both directions)
                                # From above to below: prev_y < violation_line_y and curr_y >= violation_line_y
                                # From below to above: prev_y > violation_line_y and curr_y <= violation_line_y
                                crossed_down = prev_y < violation_line_y and curr_y >= violation_line_y
                                crossed_up = prev_y > violation_line_y and curr_y <= violation_line_y
                                
                                if crossed_down or crossed_up:
                                    line_crossed_in_window = True
                                    crossing_direction = "down" if crossed_down else "up"
                                    crossing_details = {
                                        'frames_ago': i,
                                        'prev_y': prev_y,
                                        'curr_y': curr_y,
                                        'direction': crossing_direction,
                                        'window_checked': window_size
                                    }
                                    print(f"[VIOLATION DEBUG] Vehicle ID={track_id} crossed line {crossing_direction} {i} frames ago: {prev_y:.1f} -> {curr_y:.1f}")
                                    break
                        
                        print(f"[VIOLATION DEBUG] Vehicle ID={track_id}: latest_traffic_light={self.latest_traffic_light}")
                        print(f"[VIOLATION DEBUG] Vehicle ID={track_id}: position_history={[f'{p:.1f}' for p in position_history[-5:]]}");  # Show last 5 positions
                        print(f"[VIOLATION DEBUG] Vehicle ID={track_id}: line_crossed_in_window={line_crossed_in_window}, crossing_details={crossing_details}")
                        
                        # Enhanced violation detection: vehicle crossed the line while moving (red light already verified above)
                        actively_crossing = (line_crossed_in_window and is_moving)
                        
                        # Initialize violation status for new vehicles
                        if 'crossed_during_red' not in self.vehicle_statuses[track_id]:
                            self.vehicle_statuses[track_id]['crossed_during_red'] = False
                        
                        # Mark vehicle as having crossed during red if it actively crosses OR if it's already past the line
                        if actively_crossing:
                            # Additional validation: ensure it's not a false positive from ID switch
                            suspicious_jumps = self.vehicle_statuses[track_id].get('suspicious_jumps', 0)
                            if suspicious_jumps <= 1:  # Allow crossing if not too many suspicious jumps
                                self.vehicle_statuses[track_id]['crossed_during_red'] = True
                                print(f"[VIOLATION ALERT] Vehicle ID={track_id} CROSSED line during red light!")
                                print(f"  -> Crossing details: {crossing_details}")
                            else:
                                print(f"[VIOLATION IGNORED] Vehicle ID={track_id} crossing ignored due to {suspicious_jumps} suspicious jumps")
                        
                        # ADDITIONAL CHECK: If vehicle is already past the line during red light, mark as violation
                        # (for vehicles that were already past when we started tracking)
                        elif is_red_light and center_y < violation_line_y and not self.vehicle_statuses[track_id]['crossed_during_red']:
                            # Check if this vehicle has been consistently past the line for multiple frames
                            past_line_count = sum(1 for pos in position_history[-3:] if pos < violation_line_y)
                            if past_line_count >= 2 and len(position_history) >= 3:  # At least 2 out of last 3 frames
                                self.vehicle_statuses[track_id]['crossed_during_red'] = True
                                print(f"[VIOLATION ALERT] Vehicle ID={track_id} ALREADY PAST line during red light!")
                                print(f"  -> center_y={center_y:.1f}, line={violation_line_y}, past_frames={past_line_count}/3")
                        
                        # IMPORTANT: Reset violation status when light turns green (regardless of position)
                        if not is_red_light:
                            if self.vehicle_statuses[track_id]['crossed_during_red']:
                                print(f"[VIOLATION RESET] Vehicle ID={track_id} violation status reset (light turned green)")
                            self.vehicle_statuses[track_id]['crossed_during_red'] = False
                        
                        # Vehicle is violating ONLY if it crossed during red and light is still red
                        is_violation = (self.vehicle_statuses[track_id]['crossed_during_red'] and is_red_light)
                        
                        # Track current violation state for analytics - only actual crossings
                        self.vehicle_statuses[track_id]['violation_history'].append(actively_crossing)
                        if len(self.vehicle_statuses[track_id]['violation_history']) > 5:
                            self.vehicle_statuses[track_id]['violation_history'].pop(0)
                        
                        print(f"[VIOLATION DEBUG] Vehicle ID={track_id}: center_y={center_y:.1f}, line={violation_line_y}")
                        print(f"  history_window={[f'{p:.1f}' for p in position_history[-self.crossing_check_window:]]}")
                        print(f"  moving={is_moving}, red_light={is_red_light}")
                        print(f"  actively_crossing={actively_crossing}, crossed_during_red={self.vehicle_statuses[track_id]['crossed_during_red']}")
                        print(f"  suspicious_jumps={self.vehicle_statuses[track_id].get('suspicious_jumps', 0)}")
                        print(f"  FINAL_VIOLATION={is_violation}")
                        
                        # Update violation status
                        tracked['is_violation'] = is_violation
                        
                        if actively_crossing and self.vehicle_statuses[track_id].get('suspicious_jumps', 0) <= 1:  # Only add if not too many suspicious jumps
                            # Add to violating vehicles set
                            violating_vehicle_ids.add(track_id)
                            
                            # Add to violations list
                            timestamp = datetime.now()  # Keep as datetime object, not string
                            violations.append({
                                'track_id': track_id,
                                'id': track_id,
                                'bbox': [int(tracked['bbox'][0]), int(tracked['bbox'][1]), int(tracked['bbox'][2]), int(tracked['bbox'][3])],
                                'violation': 'line_crossing',
                                'violation_type': 'line_crossing',  # Add this for analytics compatibility
                                'timestamp': timestamp,
                                'line_position': violation_line_y,
                                'movement': crossing_details if crossing_details else {'prev_y': center_y, 'current_y': center_y},
                                'crossing_window': self.crossing_check_window,
                                'position_history': list(position_history[-10:])  # Include recent history for debugging
                            })
                            
                            print(f"[DEBUG] 🚨 VIOLATION DETECTED: Vehicle ID={track_id} CROSSED VIOLATION LINE")
                            print(f"    Enhanced detection: {crossing_details}")
                            print(f"    Position history: {[f'{p:.1f}' for p in position_history[-10:]]}")
                            print(f"    Detection window: {self.crossing_check_window} frames")
                            print(f"    while RED LIGHT & MOVING")
                    
                    else:
                        # Log why violation detection was skipped
                        reasons = []
                        if not crosswalk_detected:
                            reasons.append("No crosswalk detected")
                        if not is_red_light:
                            reasons.append(f"Light not red (current: {self.latest_traffic_light.get('color') if self.latest_traffic_light else 'None'})")
                        if violation_line_y is None:
                            reasons.append("No violation line")
                        if not tracked_vehicles:
                            reasons.append("No vehicles tracked")
                        
                        if reasons:
                            print(f"[INFO] Violation detection skipped: {', '.join(reasons)}")
                
                # --- ENHANCED VIOLATION DETECTION: Add new real-world scenarios ---
                # 1. Pedestrian right-of-way violation (blocking crosswalk during green)
                # 2. Improper stopping over crosswalk at red
                # 3. Accelerating through yellow/amber light
                pedestrian_dets = [det for det in detections if det.get('class_name') == 'person' and 'bbox' in det]
                pedestrian_tracks = []
                for ped in pedestrian_dets:
                    x1, y1, x2, y2 = ped['bbox']
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    pedestrian_tracks.append({'bbox': ped['bbox'], 'center': center})

                # Prepare crosswalk polygon for point-in-polygon checks
                crosswalk_poly = None
                if crosswalk_bbox is not None:
                    x, y, w, h = crosswalk_bbox
                    crosswalk_poly = (x, y, w, h)
                stopline_poly = crosswalk_poly  # For simplicity, use crosswalk as stopline

                # Track amber/yellow light start time
                amber_start_time = getattr(self, 'amber_start_time', None)
                latest_light_color = self.latest_traffic_light.get('color') if isinstance(self.latest_traffic_light, dict) else self.latest_traffic_light
                if latest_light_color == 'yellow' and amber_start_time is None:
                    amber_start_time = time.time()
                    self.amber_start_time = amber_start_time
                elif latest_light_color != 'yellow':
                    self.amber_start_time = None

                # Vehicle position history for speed calculation
                vehicle_position_history = {}
                for track in tracked_vehicles:
                    track_id = track['id']
                    bbox = track['bbox']
                    x1, y1, x2, y2 = bbox
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    # Store (center, timestamp)
                    if track_id not in vehicle_position_history:
                        vehicle_position_history[track_id] = []
                    vehicle_position_history[track_id].append((center, time.time()))
                    track['center'] = center

                # --- 1. Pedestrian right-of-way violation ---
                if crosswalk_poly and latest_light_color == 'green' and pedestrian_tracks:
                    for track in tracked_vehicles:
                        # Only consider moving vehicles - stopped vehicles aren't likely to be violating
                        if track.get('is_moving', False) and point_in_polygon(track['center'], crosswalk_poly):
                            # Check for pedestrians in the crosswalk
                            for ped in pedestrian_tracks:
                                if point_in_polygon(ped['center'], crosswalk_poly):
                                    # Vehicle is blocking crosswalk during green with pedestrian present
                                    violations.append({
                                        'track_id': track['id'],
                                        'id': track['id'],
                                        'bbox': [int(track['bbox'][0]), int(track['bbox'][1]), int(track['bbox'][2]), int(track['bbox'][3])],
                                        'violation': 'pedestrian_right_of_way',
                                        'violation_type': 'pedestrian_right_of_way',
                                        'timestamp': datetime.now(),
                                        'details': {
                                            'pedestrian_bbox': ped['bbox'],
                                            'crosswalk_bbox': crosswalk_bbox,
                                            'is_moving': track.get('is_moving', False),
                                            'traffic_light': latest_light_color
                                        }
                                    })
                                    print(f"[VIOLATION] Pedestrian right-of-way violation: Vehicle ID={track['id']} blocking crosswalk during green light with pedestrian present")
                                    
                                    # Mark vehicle as violating for drawing purposes
                                    for i, tracked_vehicle in enumerate(tracked_vehicles):
                                        if tracked_vehicle['id'] == track['id']:
                                            tracked_vehicles[i]['is_violation'] = True
                                            tracked_vehicles[i]['violation_type'] = 'pedestrian_right_of_way'
                                            print(f"[VIOLATION MARK] Vehicle ID={track['id']} marked as violating for pedestrian right-of-way")
                                            break

                # --- 2. Improper stopping over crosswalk at red ---
                if crosswalk_poly and latest_light_color == 'red':
                    for track in tracked_vehicles:
                        # Check if vehicle is not moving (to confirm it's stopped)
                        is_stopped = not track.get('is_moving', True)
                        
                        if is_stopped and point_in_polygon(track['center'], crosswalk_poly):
                            # Calculate overlap ratio between vehicle and crosswalk
                            vx1, vy1, vx2, vy2 = track['bbox']
                            cx, cy, cw, ch = crosswalk_poly
                            overlap_x1 = max(vx1, cx)
                            overlap_y1 = max(vy1, cy)
                            overlap_x2 = min(vx2, cx + cw)
                            overlap_y2 = min(vy2, cy + ch)
                            overlap_area = max(0, overlap_x2 - overlap_x1) * max(0, overlap_y2 - overlap_y1)
                            vehicle_area = (vx2 - vx1) * (vy2 - vy1)
                            overlap_ratio = overlap_area / max(vehicle_area, 1)
                            
                            # Double-verify that vehicle is stopped by checking explicit speed
                            speed = 0.0
                            hist = vehicle_position_history.get(track['id'], [])
                            if len(hist) >= 2:
                                (c1, t1), (c2, t2) = hist[-2], hist[-1]
                                dist = ((c2[0]-c1[0])**2 + (c2[1]-c1[1])**2)**0.5
                                dt = max(t2-t1, 1e-3)
                                speed = dist / dt
                            
                            # Vehicle must have significant overlap with crosswalk (>25%) and be stopped
                            if overlap_ratio > 0.25 and speed < 0.8:
                                violations.append({
                                    'track_id': track['id'],
                                    'id': track['id'],
                                    'bbox': [int(track['bbox'][0]), int(track['bbox'][1]), int(track['bbox'][2]), int(track['bbox'][3])],
                                    'violation': 'stop_on_crosswalk',
                                    'violation_type': 'stop_on_crosswalk',
                                    'timestamp': datetime.now(),
                                    'details': {
                                        'overlap_ratio': overlap_ratio,
                                        'speed': speed,
                                        'crosswalk_bbox': crosswalk_bbox,
                                        'traffic_light': latest_light_color,
                                        'is_moving_flag': track.get('is_moving', None)
                                    }
                                })
                                print(f"[VIOLATION] Improper stop on crosswalk: Vehicle ID={track['id']} stopped on crosswalk during red light (overlap={overlap_ratio:.2f}, speed={speed:.2f})")
                                
                                # NOTE: Don't mark crosswalk violations for red box drawing
                                # Only line crossing violations get red boxes
                                # for i, tracked_vehicle in enumerate(tracked_vehicles):
                                #     if tracked_vehicle['id'] == track['id']:
                                #         tracked_vehicles[i]['is_violation'] = True
                                #         tracked_vehicles[i]['violation_type'] = 'stop_on_crosswalk'
                                #         print(f"[VIOLATION MARK] Vehicle ID={track['id']} marked as violating for crosswalk stop")
                                #         break

                # --- 3. Accelerating through yellow/amber light ---
                if stopline_poly and latest_light_color == 'yellow' and amber_start_time:
                    # Calculate time since light turned yellow
                    current_time = time.time()
                    time_since_yellow = current_time - amber_start_time
                    
                    # Speed threshold (in pixels per second) - can be adjusted based on testing
                    speed_limit_px_per_sec = 8.0
                    
                    # Check each vehicle approaching the intersection
                    for track in tracked_vehicles:
                        # Check if vehicle is near the stop line/intersection
                        if point_in_polygon(track['center'], stopline_poly) or (
                            track['center'][1] < stopline_poly[1] + stopline_poly[3] + 50 and
                            track['center'][1] > stopline_poly[1] - 50
                        ):
                            # If the vehicle is moving (confirmed via tracker)
                            if track.get('is_moving', False):
                                # Calculate acceleration by looking at recent speed changes
                                hist = vehicle_position_history.get(track['id'], [])
                                if len(hist) >= 3:
                                    # Calculate speeds at different time points
                                    (c1, t1), (c2, t2), (c3, t3) = hist[-3], hist[-2], hist[-1]
                                    
                                    # Speed at earlier point
                                    v1 = ((c2[0]-c1[0])**2 + (c2[1]-c1[1])**2)**0.5 / max(t2-t1, 1e-3)
                                    
                                    # Speed at later point
                                    v2 = ((c3[0]-c2[0])**2 + (c3[1]-c2[1])**2)**0.5 / max(t3-t2, 1e-3)
                                    
                                    # Acceleration violation if:
                                    # 1. Speed increases significantly (>20%)
                                    # 2. Final speed exceeds threshold
                                    # 3. Yellow light is less than 3 seconds old (typical acceleration window)
                                    if v2 > v1 * 1.2 and v2 > speed_limit_px_per_sec and time_since_yellow < 3.0:
                                        violations.append({
                                            'track_id': track['id'],
                                            'id': track['id'],
                                            'bbox': [int(track['bbox'][0]), int(track['bbox'][1]), int(track['bbox'][2]), int(track['bbox'][3])],
                                            'violation': 'amber_acceleration',
                                            'violation_type': 'amber_acceleration',
                                            'timestamp': datetime.now(),
                                            'details': {
                                                'speed_before': v1,
                                                'speed_after': v2,
                                                'acceleration': (v2-v1)/max(t3-t2, 1e-3),
                                                'time_since_yellow': time_since_yellow,
                                                'traffic_light': latest_light_color
                                            }
                                        })
                                        print(f"[VIOLATION] Amber light acceleration: Vehicle ID={track['id']} accelerated from {v1:.2f} to {v2:.2f} px/sec {time_since_yellow:.1f}s after yellow light")
                                        
                                        # Mark vehicle as violating for drawing purposes
                                        for i, tracked_vehicle in enumerate(tracked_vehicles):
                                            if tracked_vehicle['id'] == track['id']:
                                                tracked_vehicles[i]['is_violation'] = True
                                                tracked_vehicles[i]['violation_type'] = 'amber_acceleration'
                                                print(f"[VIOLATION MARK] Vehicle ID={track['id']} marked as violating for amber acceleration")
                                                break
                
                # Emit progress signal after processing each frame
                if hasattr(self, 'progress_ready'):
                    self.progress_ready.emit(int(cap.get(cv2.CAP_PROP_POS_FRAMES)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), time.time())
                
                # Draw detections with bounding boxes - NOW with violation info
                # Only show traffic light and vehicle classes
                allowed_classes = ['traffic light', 'car', 'truck', 'bus', 'motorcycle', 'van', 'bicycle']
                filtered_detections = [det for det in detections if det.get('class_name') in allowed_classes]
                print(f"Drawing {len(filtered_detections)} detection boxes on frame (filtered)")
                # Statistics for debugging
                vehicles_with_ids = 0
                vehicles_without_ids = 0
                vehicles_moving = 0
                vehicles_violating = 0

                if detections and len(detections) > 0:
                    # Only show traffic light and vehicle classes
                    allowed_classes = ['traffic light', 'car', 'truck', 'bus', 'motorcycle', 'van', 'bicycle']
                    filtered_detections = [det for det in detections if det.get('class_name') in allowed_classes]
                    print(f"Drawing {len(filtered_detections)} detection boxes on frame (filtered)")
                    # Statistics for debugging
                    vehicles_with_ids = 0
                    vehicles_without_ids = 0
                    vehicles_moving = 0
                    vehicles_violating = 0
                    for det in filtered_detections:
                        if 'bbox' in det:
                            bbox = det['bbox']
                            x1, y1, x2, y2 = map(int, bbox)
                            label = det.get('class_name', 'object')
                            confidence = det.get('confidence', 0.0)
                            
                            # Robustness: ensure label and confidence are not None
                            if label is None:
                                label = 'object'
                            if confidence is None:
                                confidence = 0.0
                            class_id = det.get('class_id', -1)
                            
                            # Check if this detection corresponds to a violating or moving vehicle
                            det_center_x = (x1 + x2) / 2
                            det_center_y = (y1 + y2) / 2
                            is_violating_vehicle = False
                            is_moving_vehicle = False
                            vehicle_id = None
                            
                            # Match detection with tracked vehicles - IMPROVED MATCHING
                            if label in ['car', 'truck', 'bus', 'motorcycle', 'van', 'bicycle'] and len(tracked_vehicles) > 0:
                                print(f"[MATCH DEBUG] Attempting to match {label} detection at ({det_center_x:.1f}, {det_center_y:.1f}) with {len(tracked_vehicles)} tracked vehicles")
                                best_match = None
                                best_distance = float('inf')
                                best_iou = 0.0
                                
                                for i, tracked in enumerate(tracked_vehicles):
                                    track_bbox = tracked['bbox']
                                    track_x1, track_y1, track_x2, track_y2 = map(float, track_bbox)
                                    
                                    # Calculate center distance
                                    track_center_x = (track_x1 + track_x2) / 2
                                    track_center_y = (track_y1 + track_y2) / 2
                                    center_distance = ((det_center_x - track_center_x)**2 + (det_center_y - track_center_y)**2)**0.5
                                    
                                    # Calculate IoU (Intersection over Union)
                                    intersection_x1 = max(x1, track_x1)
                                    intersection_y1 = max(y1, track_y1)
                                    intersection_x2 = min(x2, track_x2)
                                    intersection_y2 = min(y2, track_y2)
                                    
                                    if intersection_x2 > intersection_x1 and intersection_y2 > intersection_y1:
                                        intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
                                        det_area = (x2 - x1) * (y2 - y1)
                                        track_area = (track_x2 - track_x1) * (track_y2 - track_y1)
                                        union_area = det_area + track_area - intersection_area
                                        iou = intersection_area / union_area if union_area > 0 else 0
                                    else:
                                        iou = 0
                                    
                                    print(f"[MATCH DEBUG] Track {i}: ID={tracked['id']}, center=({track_center_x:.1f}, {track_center_y:.1f}), distance={center_distance:.1f}, IoU={iou:.3f}")
                                    
                                    # Use stricter matching criteria - prioritize IoU over distance
                                    # Good match if: high IoU OR close center distance with some overlap
                                    is_good_match = (iou > 0.3) or (center_distance < 60 and iou > 0.1)
                                    
                                    if is_good_match:
                                        print(f"[MATCH DEBUG] Track {i} is a good match (IoU={iou:.3f}, distance={center_distance:.1f})")
                                        # Prefer higher IoU, then lower distance
                                        match_score = iou + (100 - min(center_distance, 100)) / 100  # Composite score
                                        if iou > best_iou or (iou == best_iou and center_distance < best_distance):
                                            best_distance = center_distance
                                            best_iou = iou
                                            best_match = tracked
                                    else:
                                        print(f"[MATCH DEBUG] Track {i} failed matching criteria (IoU={iou:.3f}, distance={center_distance:.1f})")
                                
                                if best_match:
                                    vehicle_id = best_match['id']
                                    is_moving_vehicle = best_match.get('is_moving', False)
                                    is_violating_vehicle = best_match.get('is_violation', False)
                                    print(f"[MATCH SUCCESS] Detection at ({det_center_x:.1f},{det_center_y:.1f}) matched with track ID={vehicle_id}")
                                    print(f"  -> STATUS: moving={is_moving_vehicle}, violating={is_violating_vehicle}, IoU={best_iou:.3f}, distance={best_distance:.1f}")
                                else:
                                    print(f"[MATCH FAILED] No suitable match found for {label} detection at ({det_center_x:.1f}, {det_center_y:.1f})")
                                    print(f"  -> Will draw as untracked detection with default color")
                            else:
                                if label not in ['car', 'truck', 'bus', 'motorcycle', 'van', 'bicycle']:
                                    print(f"[MATCH DEBUG] Skipping matching for non-vehicle label: {label}")
                                elif len(tracked_vehicles) == 0:
                                    print(f"[MATCH DEBUG] No tracked vehicles available for matching")
                                else:
                                    try:
                                        if len(tracked_vehicles) > 0:
                                            distances = [((det_center_x - (t['bbox'][0] + t['bbox'][2])/2)**2 + (det_center_y - (t['bbox'][1] + t['bbox'][3])/2)**2)**0.5 for t in tracked_vehicles[:3]]
                                            print(f"[DEBUG] No match found for detection at ({det_center_x:.1f},{det_center_y:.1f}) - distances: {distances}")
                                        else:
                                            print(f"[DEBUG] No tracked vehicles available to match detection at ({det_center_x:.1f},{det_center_y:.1f})")
                                    except NameError:
                                        print(f"[DEBUG] No match found for detection (coords unavailable)")
                                        if len(tracked_vehicles) > 0:
                                            print(f"[DEBUG] Had {len(tracked_vehicles)} tracked vehicles available")
                            
                            # Choose box color based on vehicle status 
                            # PRIORITY: 1. Violating (RED) - crossed during red light 2. Moving (ORANGE) 3. Stopped (GREEN)
                            if is_violating_vehicle and vehicle_id is not None:
                                box_color = (0, 0, 255)  # RED for violating vehicles
                                # Find violation type for better labeling
                                violation_type = None
                                for tracked in tracked_vehicles:
                                    if tracked['id'] == vehicle_id:
                                        violation_type = tracked.get('violation_type', 'crossing')
                                        break
                                
                                if violation_type == 'stop_on_crosswalk':
                                    label_text = f"{label}:ID{vehicle_id}⛔🚶"  # Stop on crosswalk
                                elif violation_type == 'pedestrian_right_of_way':
                                    label_text = f"{label}:ID{vehicle_id}⚠️🚶"  # Pedestrian violation
                                elif violation_type == 'amber_acceleration':
                                    label_text = f"{label}:ID{vehicle_id}🟡⚡"  # Amber acceleration
                                else:
                                    label_text = f"{label}:ID{vehicle_id}⚠️"  # Default violation
                                
                                thickness = 4
                                vehicles_violating += 1
                                print(f"[COLOR DEBUG] Drawing RED box for VIOLATING vehicle ID={vehicle_id} ({violation_type})")
                            elif is_moving_vehicle and vehicle_id is not None and not is_violating_vehicle:
                                box_color = (0, 165, 255)  # ORANGE for moving vehicles (not violating)
                                label_text = f"{label}:ID{vehicle_id}"
                                thickness = 3
                                vehicles_moving += 1
                                print(f"[COLOR DEBUG] Drawing ORANGE box for MOVING vehicle ID={vehicle_id} (not violating)")
                            elif label in ['car', 'truck', 'bus', 'motorcycle', 'van', 'bicycle'] and vehicle_id is not None:
                                box_color = (0, 255, 0)  # Green for stopped vehicles 
                                label_text = f"{label}:ID{vehicle_id}"
                                thickness = 2
                                print(f"[COLOR DEBUG] Drawing GREEN box for STOPPED vehicle ID={vehicle_id}")
                            elif is_traffic_light(label):
                                box_color = (0, 0, 255)  # Red for traffic lights
                                label_text = f"{label}"
                                thickness = 2
                            else:
                                box_color = (0, 255, 0)  # Default green for other objects
                                label_text = f"{label}"
                                thickness = 2
                            
                            # Update statistics
                            if label in ['car', 'truck', 'bus', 'motorcycle', 'van', 'bicycle']:
                                if vehicle_id is not None:
                                    vehicles_with_ids += 1
                                else:
                                    vehicles_without_ids += 1
                            
                            # Draw rectangle and label
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, thickness)
                            cv2.putText(annotated_frame, label_text, (x1, y1-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                            #     id_text = f"ID: {det['id']}"
                            #     # Calculate text size for background
                            #     (tw, th), baseline = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                            #     # Draw filled rectangle for background (top-left of bbox)
                            #     cv2.rectangle(annotated_frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 0, 0), -1)
                            #     # Draw the ID text in bold yellow
                            #     cv2.putText(annotated_frame, id_text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                            #     print(f"[DEBUG] Detection ID: {det['id']} BBOX: {bbox} CLASS: {label} CONF: {confidence:.2f}")
                           
                            if class_id == 9 or is_traffic_light(label):
                                try:
                                    light_info = detect_traffic_light_color(annotated_frame, [x1, y1, x2, y2])
                                    if light_info.get("color", "unknown") == "unknown":
                                        light_info = ensure_traffic_light_color(annotated_frame, [x1, y1, x2, y2])
                                    det['traffic_light_color'] = light_info
                                    # Draw enhanced traffic light status
                                    annotated_frame = draw_traffic_light_status(annotated_frame, bbox, light_info)
                                    
                                    # --- Update latest_traffic_light for UI/console ---
                                    self.latest_traffic_light = light_info
                                    
                                    # Add a prominent traffic light status at the top of the frame
                                    color = light_info.get('color', 'unknown')
                                    confidence = light_info.get('confidence', 0.0)
                                    
                                    if color == 'red':
                                        status_color = (0, 0, 255)  # Red
                                        status_text = f"Traffic Light: RED ({confidence:.2f})"
                                        
                                        # Draw a prominent red banner across the top
                                        banner_height = 40
                                        cv2.rectangle(annotated_frame, (0, 0), (annotated_frame.shape[1], banner_height), (0, 0, 150), -1)
                                        
                                        # Add text
                                        font = cv2.FONT_HERSHEY_DUPLEX
                                        font_scale = 0.9
                                        font_thickness = 2
                                        cv2.putText(annotated_frame, status_text, (10, banner_height-12), font, 
                                                  font_scale, (255, 255, 255), font_thickness)
                                except Exception as e:
                                    print(f"[WARN] Could not detect/draw traffic light color: {e}")

                # Print statistics summary
                print(f"[STATS] Vehicles: {vehicles_with_ids} with IDs, {vehicles_without_ids} without IDs")
                
                # Handle multiple traffic lights with consensus approach
                for det in detections:
                    if is_traffic_light(det.get('class_name')):
                        has_traffic_lights = True
                        if 'traffic_light_color' in det:
                            light_info = det['traffic_light_color']
                            traffic_lights.append({'bbox': det['bbox'], 'color': light_info.get('color', 'unknown'), 'confidence': light_info.get('confidence', 0.0)})
                
                # Determine the dominant traffic light color based on confidence
                if traffic_lights:
                    # Filter to just red lights and sort by confidence
                    red_lights = [tl for tl in traffic_lights if tl.get('color') == 'red']
                    if red_lights:
                        # Use the highest confidence red light for display
                        highest_conf_red = max(red_lights, key=lambda x: x.get('confidence', 0))
                        # Update the global traffic light status for consistent UI display
                        self.latest_traffic_light = {
                            'color': 'red',
                            'confidence': highest_conf_red.get('confidence', 0.0)
                        }

                # Emit all violations as a batch for UI (optional)
                if violations:
                    if hasattr(self, 'violations_batch_ready'):
                        self.violations_batch_ready.emit(violations)
                    # Emit individual violation signals for each violation
                    try:
                        for violation in violations:
                            print(f"🚨 Emitting RED LIGHT VIOLATION: Track ID {violation['track_id']}")
                            print(f"[VIOLATION DEBUG] Full violation data: {violation}")
                            
                            # Convert datetime to timestamp for Qt signal compatibility
                            if 'timestamp' in violation and hasattr(violation['timestamp'], 'timestamp'):
                                violation['timestamp'] = violation['timestamp'].timestamp()
                                print(f"[VIOLATION DEBUG] Converted datetime to timestamp: {violation['timestamp']}")
                            
                            violation['frame'] = frame
                            violation['violation_line_y'] = violation_line_y
                            print(f"[VIOLATION DEBUG] About to emit violation_detected signal")
                            print(f"[VIOLATION DEBUG] Signal object: {self.violation_detected}")
                            self.violation_detected.emit(violation)
                            print(f"[VIOLATION DEBUG] Successfully emitted violation for Track ID {violation['track_id']}")
                        print(f"[DEBUG] Emitted {len(violations)} violation signals successfully")
                    except Exception as e:
                        print(f"❌ Error emitting violation signals: {e}")
                        import traceback
                        traceback.print_exc()
                
                # Add FPS display directly on frame
                # cv2.putText(annotated_frame, f"FPS: {fps_smoothed:.1f}", (10, 30), 
                #            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # # --- Always draw detected traffic light color indicator at top ---
                # color = self.latest_traffic_light.get('color', 'unknown') if isinstance(self.latest_traffic_light, dict) else str(self.latest_traffic_light)
                # confidence = self.latest_traffic_light.get('confidence', 0.0) if isinstance(self.latest_traffic_light, dict) else 0.0
                # indicator_size = 30
                # margin = 10
                # status_colors = {
                #     "red": (0, 0, 255),
                #     "yellow": (0, 255, 255),
                #     "green": (0, 255, 0),
                #     "unknown": (200, 200, 200)
                # }
                # draw_color = status_colors.get(color, (200, 200, 200))
                # # Draw circle indicator
                # cv2.circle(
                #     annotated_frame,
                #     (annotated_frame.shape[1] - margin - indicator_size, margin + indicator_size),
                #     indicator_size,
                #     draw_color,
                #     -1
                # )
                # # Add color text
                # cv2.putText(
                #     annotated_frame,
                #     f"{color.upper()} ({confidence:.2f})",
                #     (annotated_frame.shape[1] - margin - indicator_size - 120, margin + indicator_size + 10),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.7,
                #     (0, 0, 0),
                #     2
                # )

                # Signal for raw data subscribers (now without violations)
                # Emit with correct number of arguments
                try:
                    if hasattr(self, 'raw_frame_ready') and self.raw_frame_ready is not None:
                        self.raw_frame_ready.emit(frame.copy(), detections, fps_smoothed)
                        print(f"✅ raw_frame_ready signal emitted with {len(detections)} detections, fps={fps_smoothed:.1f}")
                    else:
                        print("⚠️ raw_frame_ready signal not available")
                except RuntimeError as e:
                    if "Internal C++ object" in str(e) or "already deleted" in str(e):
                        print(f"⚠️ Object deleted during raw_frame_ready emission: {e}")
                        self._running = False  # Stop processing
                        break
                    else:
                        print(f"❌ Runtime error emitting raw_frame_ready: {e}")
                except Exception as e:
                    print(f"❌ Error emitting raw_frame_ready: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Emit the NumPy frame signal for direct display - annotated version for visual feedback
                print(f"🔴 Emitting frame_np_ready signal with annotated_frame shape: {annotated_frame.shape}")
                try:
                    # Check if signal is still valid before emission
                    if hasattr(self, 'frame_np_ready') and self.frame_np_ready is not None:
                        # Make sure the frame can be safely transmitted over Qt's signal system
                        # Create a contiguous copy of the array
                        frame_copy = np.ascontiguousarray(annotated_frame)
                        print(f"🔍 Debug - Before emission: frame_copy type={type(frame_copy)}, shape={frame_copy.shape}, is_contiguous={frame_copy.flags['C_CONTIGUOUS']}")
                        self.frame_np_ready.emit(frame_copy)
                        print("✅ frame_np_ready signal emitted successfully")
                    else:
                        print("⚠️ frame_np_ready signal not available")
                except RuntimeError as e:
                    if "Internal C++ object" in str(e) or "already deleted" in str(e):
                        print(f"⚠️ Object deleted during signal emission: {e}")
                        self._running = False  # Stop processing
                        break
                    else:
                        print(f"❌ Runtime error emitting frame: {e}")
                except Exception as e:
                    print(f"❌ Error emitting frame: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Emit QPixmap for video detection tab (frame_ready)
                try:
                    if hasattr(self, 'frame_ready') and self.frame_ready is not None:
                        from PySide6.QtGui import QImage, QPixmap
                        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                        h, w, ch = rgb_frame.shape
                        bytes_per_line = ch * w
                        qimg = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                        pixmap = QPixmap.fromImage(qimg)
                        metrics = {
                            'FPS': fps_smoothed,
                            'Detection (ms)': detection_time
                        }
                        self.frame_ready.emit(pixmap, detections, metrics)
                        print("✅ frame_ready signal emitted for video detection tab")
                    else:
                        print("⚠️ frame_ready signal not available")
                except RuntimeError as e:
                    if "Internal C++ object" in str(e) or "already deleted" in str(e):
                        print(f"⚠️ Object deleted during frame_ready emission: {e}")
                        self._running = False  # Stop processing
                        break
                    else:
                        print(f"❌ Runtime error emitting frame_ready: {e}")
                except Exception as e:
                    print(f"❌ Error emitting frame_ready: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Emit stats signal for performance monitoring
                # Count traffic lights for UI (confidence >= 0.5)
                traffic_light_count = 0
                for det in detections:
                    if is_traffic_light(det.get('class_name')):
                        tl_conf = 0.0
                        if 'traffic_light_color' in det and isinstance(det['traffic_light_color'], dict):
                            tl_conf = det['traffic_light_color'].get('confidence', 0.0)
                        if tl_conf >= 0.5:
                            traffic_light_count += 1
                # Count cars for UI (confidence >= 0.5)
                car_count = 0
                for det in detections:
                    if det.get('class_name') == 'car' and det.get('confidence', 0.0) >= 0.5:
                        car_count += 1
                # Get model information from model manager
                model_info = {}
                if self.model_manager and hasattr(self.model_manager, 'get_current_model_info'):
                    model_info = self.model_manager.get_current_model_info()
                    print(f"🔧 DEBUG: Model info from manager: {model_info}")
                
                stats = {
                    'fps': fps_smoothed,
                    'detection_fps': fps_smoothed,  # Numeric value for analytics
                    'detection_time': detection_time,
                    'detection_time_ms': detection_time,  # Numeric value for analytics
                    'traffic_light_color': self.latest_traffic_light,
                    'tlights': traffic_light_count,  # Only confident traffic lights
                    'cars': car_count,  # Only confident cars
                    'model_path': model_info.get('model_path', ''),  # Add model path for UI
                    'model_name': model_info.get('model_name', 'Unknown')  # Add model name for UI
                }
                print(f"🔧 DEBUG: Stats with model info: model_name={stats.get('model_name')}, model_path={stats.get('model_path')}")
                
                # Print detailed stats for debugging
                tl_color = "unknown"
                if isinstance(self.latest_traffic_light, dict):
                    tl_color = self.latest_traffic_light.get('color', 'unknown')
                elif isinstance(self.latest_traffic_light, str):
                    tl_color = self.latest_traffic_light
                
                print(f"🟢 Stats Updated: FPS={fps_smoothed:.2f}, Inference={detection_time:.2f}ms, Traffic Light={tl_color}")
                      
                # Emit stats signal
                try:
                    if hasattr(self, 'stats_ready') and self.stats_ready is not None:
                        self.stats_ready.emit(stats)
                        print(f"✅ stats_ready signal emitted successfully")
                    else:
                        print("⚠️ stats_ready signal not available")
                except RuntimeError as e:
                    if "Internal C++ object" in str(e) or "already deleted" in str(e):
                        print(f"⚠️ Object deleted during stats emission: {e}")
                        self._running = False  # Stop processing
                        break
                    else:
                        print(f"❌ Runtime error emitting stats_ready: {e}")
                except Exception as e:
                    print(f"❌ Error emitting stats_ready: {e}")

                # Emit performance stats for performance graphs
                try:
                    if hasattr(self, 'performance_stats_ready') and self.performance_stats_ready is not None:
                        perf_stats = {
                            'frame_idx': self.frame_count,
                            'fps': fps_smoothed,
                            'inference_time': detection_time,
                            'device': getattr(self, 'current_device', 'CPU'),
                            'resolution': getattr(self, 'current_resolution', f'{frame.shape[1]}x{frame.shape[0]}' if frame is not None else '-'),
                            'model_name': model_info.get('model_name', 'Unknown'),  # Add model name for performance graphs
                            'is_spike': False,  # TODO: Add spike logic if needed
                            'is_res_change': False,  # TODO: Add res change logic if needed
                            'cpu_spike': False,  # TODO: Add cpu spike logic if needed
                        }
                        print(f"[PERF] Emitting performance_stats_ready: {perf_stats}")
                        self.performance_stats_ready.emit(perf_stats)
                        print(f"✅ performance_stats_ready signal emitted successfully")
                    else:
                        print("⚠️ performance_stats_ready signal not available")
                except RuntimeError as e:
                    if "Internal C++ object" in str(e) or "already deleted" in str(e):
                        print(f"⚠️ Object deleted during performance stats emission: {e}")
                        self._running = False  # Stop processing
                        break
                    else:
                        print(f"❌ Runtime error emitting performance_stats_ready: {e}")
                except Exception as e:
                    print(f"❌ Error emitting performance_stats_ready: {e}")
                
                # --- Update last analysis data for VLM ---
                self._last_analysis_data = {
                    'detections': detections,
                    'tracked_vehicles': tracked_vehicles if 'tracked_vehicles' in locals() else [],
                    'traffic_light': self.latest_traffic_light,
                    'crosswalk_bbox': crosswalk_bbox if 'crosswalk_bbox' in locals() else None,
                    'violation_line_y': violation_line_y if 'violation_line_y' in locals() else None,
                    'crosswalk_detected': crosswalk_bbox is not None if 'crosswalk_bbox' in locals() else False,
                    'traffic_light_position': traffic_light_position if has_traffic_lights else None,
                    'frame_shape': frame.shape if frame is not None else None,
                    'timestamp': time.time()
                }
                
                # --- Ensure analytics update every frame ---
                # Always add traffic_light_color to each detection dict for analytics
                for det in detections:
                    if is_traffic_light(det.get('class_name')):
                        if 'traffic_light_color' not in det:
                            det['traffic_light_color'] = self.latest_traffic_light if hasattr(self, 'latest_traffic_light') else {'color': 'unknown', 'confidence': 0.0}
                if hasattr(self, 'analytics_controller') and self.analytics_controller is not None:
                    try:
                        self.analytics_controller.process_frame_data(frame, detections, stats)
                        print("[DEBUG] Called analytics_controller.process_frame_data for analytics update")
                    except Exception as e:
                        print(f"[ERROR] Could not update analytics: {e}")
                
                # Control processing rate for file sources
                if isinstance(self.source, str) and self.source_fps > 0:
                    frame_duration = time.time() - process_start
                    if frame_duration < frame_time:
                        time.sleep(frame_time - frame_duration)
            
            cap.release()
        except Exception as e:
            print(f"Video processing error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._running = False
    def _process_frame(self):
        """Process current frame for display with improved error handling"""
        try:
            self.mutex.lock()
            if self.current_frame is None:
                print("⚠️ No frame available to process")
                self.mutex.unlock()
                
                # Check if we're running - if not, this is expected behavior
                if not self._running:
                    return
                
                # If we are running but have no frame, create a blank frame with error message
                h, w = 480, 640  # Default size
                blank_frame = np.zeros((h, w, 3), dtype=np.uint8)
                cv2.putText(blank_frame, "No video input", (w//2-100, h//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Emit this blank frame
                try:
                    self.frame_np_ready.emit(blank_frame)
                except Exception as e:
                    print(f"Error emitting blank frame: {e}")
                
                return
            
            # Make a copy of the data we need
            try:
                frame = self.current_frame.copy()
                detections = self.current_detections.copy() if self.current_detections else []
                metrics = self.performance_metrics.copy()
            except Exception as e:
                print(f"Error copying frame data: {e}")
                self.mutex.unlock()
                return
                
            self.mutex.unlock()
        except Exception as e:
            print(f"Critical error in _process_frame initialization: {e}")
            import traceback
            traceback.print_exc()
            try:
                self.mutex.unlock()
            except:
                pass
            return
        
        try:
            # --- Simplified frame processing for display ---
            # The violation logic is now handled in the main _run thread
            # This method just handles basic display overlays
            
            annotated_frame = frame.copy()

            # Add performance overlays and debug markers - COMMENTED OUT for clean video display
            # annotated_frame = draw_performance_overlay(annotated_frame, metrics)
            # cv2.circle(annotated_frame, (20, 20), 10, (255, 255, 0), -1)

            # Convert BGR to RGB before display (for PyQt/PySide)
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            # Display the RGB frame in the UI (replace with your display logic)
            # Example: self.image_label.setPixmap(QPixmap.fromImage(QImage(frame_rgb.data, w, h, QImage.Format_RGB888)))
        except Exception as e:
            print(f"Error in _process_frame: {e}")
            import traceback
            traceback.print_exc()

    def _cleanup_old_vehicle_data(self, current_track_ids):
        """
        Clean up tracking data for vehicles that are no longer being tracked.
        This prevents memory leaks and improves performance.
        
        Args:
            current_track_ids: Set of currently active track IDs
        """
        # Find IDs that are no longer active
        old_ids = set(self.vehicle_history.keys()) - set(current_track_ids)
        
        if old_ids:
            print(f"[CLEANUP] Removing tracking data for {len(old_ids)} old vehicle IDs: {sorted(old_ids)}")
            for old_id in old_ids:
                # Remove from history and status tracking
                if old_id in self.vehicle_history:
                    del self.vehicle_history[old_id]
                if old_id in self.vehicle_statuses:
                    del self.vehicle_statuses[old_id]
            print(f"[CLEANUP] Now tracking {len(self.vehicle_history)} active vehicles")

    # --- Removed unused internal violation line detection methods and RedLightViolationSystem usage ---
    def play(self):
        """Alias for start(), for UI compatibility."""
        self.start()
    
    def pause(self):
        """Pause video processing."""
        print("[VideoController] Pause requested")
        
        # Use mutex safely
        self.mutex.lock()
        try:
            self._paused = True
        finally:
            self.mutex.unlock()
        
        # Emit the last captured frame for VLM analysis if available
        if hasattr(self, '_last_frame') and self._last_frame is not None:
            print("[VideoController] Emitting last frame for VLM analysis during pause")
            try:
                if hasattr(self, 'raw_frame_ready') and self.raw_frame_ready is not None:
                    # Emit the last frame with empty detections for VLM
                    self.raw_frame_ready.emit(self._last_frame.copy(), [], 0.0)
                    print("✅ Last frame emitted for VLM analysis")
                else:
                    print("⚠️ raw_frame_ready signal not available for pause emission")
            except RuntimeError as e:
                if "Internal C++ object" in str(e) or "already deleted" in str(e):
                    print(f"⚠️ Object deleted during pause frame emission: {e}")
                else:
                    print(f"❌ Runtime error emitting pause frame: {e}")
            except Exception as e:
                print(f"❌ Error emitting last frame: {e}")
        else:
            print("[VideoController] No last frame available for VLM analysis")
        
        # Emit pause state signal safely
        try:
            self.pause_state_changed.emit(True)
            print("[VideoController] Pause state signal emitted: True")
        except Exception as e:
            print(f"❌ Error emitting pause state signal: {e}")
    
    def resume(self):
        """Resume video processing from pause."""
        print("[VideoController] Resume requested") 
        
        # Use mutex safely
        self.mutex.lock()
        try:
            self._paused = False
            self.pause_condition.wakeAll()  # Wake up any waiting threads
        finally:
            self.mutex.unlock()
        
        # Emit pause state signal safely
        try:
            self.pause_state_changed.emit(False)
            print("[VideoController] Pause state signal emitted: False")
        except Exception as e:
            print(f"❌ Error emitting resume state signal: {e}")
    
    def is_paused(self):
        """Check if video processing is currently paused."""
        self.mutex.lock()
        try:
            return self._paused
        finally:
            self.mutex.unlock()

    def get_current_analysis_data(self):
        """Get current analysis data for VLM insights."""
        return self._last_analysis_data.copy() if self._last_analysis_data else {}

    def _on_scene_object_detected(self, obj_data: dict):
        """Handle scene object detection signal"""
        try:
            # Forward scene object detection to analytics
            print(f"[SCENE] Object detected: {obj_data.get('category', 'unknown')} "
                  f"(confidence: {obj_data.get('confidence', 0):.2f})")
        except Exception as e:
            print(f"Error handling scene object detection: {e}")
    
    def _on_scene_analytics_updated(self, analytics_data: dict):
        """Handle scene analytics update signal"""
        try:
            # Forward scene analytics to performance stats
            camera_id = analytics_data.get('camera_id', 'unknown')
            fps = analytics_data.get('fps', 0)
            processing_time = analytics_data.get('processing_time_ms', 0)
            object_count = analytics_data.get('object_count', 0)
            
            # Update performance metrics with scene analytics
            self.performance_metrics['Scene_FPS'] = fps
            self.performance_metrics['Scene_Objects'] = object_count
            self.performance_metrics['Scene_Processing_ms'] = processing_time
            
            print(f"[SCENE] Analytics updated - FPS: {fps:.1f}, Objects: {object_count}, "
                  f"Processing: {processing_time:.1f}ms")
                  
        except Exception as e:
            print(f"Error handling scene analytics update: {e}")
    
    def _on_roi_event_detected(self, event_data: dict):
        """Handle ROI event detection signal"""
        try:
            event_type = event_data.get('type', 'unknown')
            roi_id = event_data.get('roi_id', 'unknown')
            object_category = event_data.get('object_category', 'unknown')
            
            print(f"[SCENE] ROI Event: {event_type} in {roi_id} - {object_category}")
            
            # Emit as violation if it's a safety-related event
            if 'safety' in event_type.lower() or 'violation' in event_type.lower():
                violation_data = {
                    'type': 'roi_violation',
                    'roi_id': roi_id,
                    'object_category': object_category,
                    'timestamp': event_data.get('timestamp'),
                    'confidence': event_data.get('confidence', 1.0),
                    'source': 'scene_analytics'
                }
                
                # Convert datetime to timestamp for Qt signal compatibility
                if 'timestamp' in violation_data and hasattr(violation_data['timestamp'], 'timestamp'):
                    violation_data['timestamp'] = violation_data['timestamp'].timestamp()
                
                print(f"[VIOLATION DEBUG] Emitting ROI violation: {violation_data}")
                self.violation_detected.emit(violation_data)
                
        except Exception as e:
            print(f"Error handling ROI event: {e}")

    @Slot(str)
    def on_model_switched(self, device):
        """Handle device switch from config panel."""
        try:
            print(f"🔄 Video Controller: Device switch requested to {device}")
            
            # Update our device reference
            self.current_device = device
            print(f"✅ Video Controller: current_device updated to {device}")
            
            # If we have a model manager, the device switch should already be done
            # Just log the current state for verification
            if self.model_manager and hasattr(self.model_manager, 'detector'):
                if hasattr(self.model_manager.detector, 'device'):
                    current_device = self.model_manager.detector.device
                    print(f"✅ Video Controller: Model manager detector now using device: {current_device}")
                else:
                    print(f"✅ Video Controller: Model manager detector updated to {device}")
                    
            print(f"✅ Video Controller: Device switch to {device} completed")
            
        except Exception as e:
            print(f"❌ Video Controller: Error during device switch: {e}")
