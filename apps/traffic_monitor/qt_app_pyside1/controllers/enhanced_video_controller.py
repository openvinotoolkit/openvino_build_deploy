"""
Enhanced video controller with async inference and separated FPS tracking
"""

import sys
import os
import time
import cv2
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from PySide6.QtCore import QObject, Signal, QThread, Qt, QMutex, QWaitCondition, QTimer
from PySide6.QtGui import QImage, QPixmap

# Add parent directory to path for imports
current_dir = Path(__file__).parent.parent.parent
sys.path.append(str(current_dir))

# Import our async detector
try:
    # Try direct import first
    from detection_openvino_async import OpenVINOVehicleDetector
except ImportError:
    # Fall back to import from project root
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from detection_openvino_async import OpenVINOVehicleDetector

# Import traffic light color detection utility
try:
    from utils.traffic_light_utils import detect_traffic_light_color, draw_traffic_light_status
    print("✅ Imported traffic light color detection utilities")
except ImportError:
    # Create simple placeholder functions if imports fail
    def detect_traffic_light_color(frame, bbox):
        return {"color": "unknown", "confidence": 0.0}
    
    def draw_traffic_light_status(frame, bbox, color):
        return frame
    print("⚠️ Failed to import traffic light color detection utilities")

# Import utilities for visualization
try:
    # Try the direct import when running inside the qt_app_pyside directory
    from utils.enhanced_annotation_utils import (
        enhanced_draw_detections,
        draw_performance_overlay,
        enhanced_cv_to_qimage,
        enhanced_cv_to_pixmap
    )
    print("✅ Successfully imported enhanced_annotation_utils from utils package")
except ImportError:
    try:
        # Try fully qualified import path
        from qt_app_pyside.utils.enhanced_annotation_utils import (
            enhanced_draw_detections,
            draw_performance_overlay,
            enhanced_cv_to_qimage,
            enhanced_cv_to_pixmap
        )
        print("✅ Successfully imported enhanced_annotation_utils from qt_app_pyside.utils package")
    except ImportError:
        # Fall back to our minimal implementation
        print("⚠️ Could not import enhanced_annotation_utils, using fallback implementation")
        sys.path.append(str(Path(__file__).parent.parent.parent))
        try:
            from fallback_annotation_utils import (
                enhanced_draw_detections,
                draw_performance_overlay,
                enhanced_cv_to_qimage,
                enhanced_cv_to_pixmap
            )
            print("✅ Using fallback_annotation_utils")
        except ImportError:
            print("❌ CRITICAL: Could not import annotation utilities! UI will be broken.")
            # Define minimal stub functions to prevent crashes
            def enhanced_draw_detections(frame, detections, **kwargs):
                return frame
            def draw_performance_overlay(frame, metrics):
                return frame
            def enhanced_cv_to_qimage(frame):
                return None
            def enhanced_cv_to_pixmap(frame):
                return None

class AsyncVideoProcessingThread(QThread):
    """Thread for async video processing with separate detection and UI threads."""
    
    # Signal for UI update with enhanced metadata
    frame_processed = Signal(np.ndarray, list, dict)  # frame, detections, metrics
    
    # Signal for separate processing metrics
    stats_updated = Signal(dict)  # All performance metrics
    
    def __init__(self, model_manager, parent=None):
        super().__init__(parent)
        self.model_manager = model_manager
        self.running = False
        self.paused = False
        
        # Video source
        self.source = 0
        self.cap = None
        self.source_fps = 0
        self.target_fps = 30  # Target FPS for UI updates
        
        # Performance tracking
        self.detection_fps = 0
        self.ui_fps = 0
        self.frame_count = 0
        self.start_time = 0
        self.detection_times = deque(maxlen=30)  # Last 30 detection times
        self.ui_frame_times = deque(maxlen=30)  # Last 30 UI frame times
        self.last_ui_frame_time = 0
        
        # Mutexes for thread safety
        self.mutex = QMutex()
        self.wait_condition = QWaitCondition()
        
        # FPS limiter to avoid CPU overload
        self.last_frame_time = 0
        self.min_frame_interval = 1.0 / 60  # Max 60 FPS
        
        # Async processing queue with frame IDs
        self.frame_queue = []  # List of (frame_id, frame) tuples
        self.next_frame_id = 0
        self.processed_frames = {}  # frame_id -> (frame, detections, metrics)
        self.last_emitted_frame_id = -1
          # Separate UI thread timer for smooth display
        self.ui_timer = QTimer()
        self.ui_timer.timeout.connect(self._emit_next_frame)
        
    def set_source(self, source):
        """Set video source - camera index or file path."""
        print(f"[AsyncThread] set_source: {source} ({type(source)})")
        if source is None:
            self.source = 0
        elif isinstance(source, str) and os.path.isfile(source):
            self.source = source
        elif isinstance(source, int):
            self.source = source
        else:
            print("[AsyncThread] Invalid source, defaulting to camera")
            self.source = 0
            
    def start_processing(self):
        """Start video processing."""
        self.running = True
        self.start()
        # Start UI timer for smooth frame emission
        self.ui_timer.start(int(1000 / self.target_fps))
        
    def stop_processing(self):
        """Stop video processing."""
        self.running = False
        self.wait_condition.wakeAll()
        self.wait()
        self.ui_timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        
    def pause_processing(self):
        """Pause video processing."""
        self.mutex.lock()
        self.paused = True
        self.mutex.unlock()
        
    def resume_processing(self):
        """Resume video processing."""
        self.mutex.lock()
        self.paused = False
        self.wait_condition.wakeAll()
        self.mutex.unlock()
        
    def run(self):
        """Main thread execution loop."""
        self._initialize_video()
        self.start_time = time.time()
        self.frame_count = 0
        
        while self.running:
            # Check if paused
            self.mutex.lock()
            if self.paused:
                self.wait_condition.wait(self.mutex)
            self.mutex.unlock()
            
            if not self.running:
                break
                
            # Control frame rate
            current_time = time.time()
            time_diff = current_time - self.last_frame_time
            if time_diff < self.min_frame_interval:
                time.sleep(self.min_frame_interval - time_diff)
                
            # Read frame
            ret, frame = self.cap.read()
            self.last_frame_time = time.time()
            
            if not ret or frame is None:
                print("End of video or failed to read frame")
                # Check if we're using a file and should restart
                if isinstance(self.source, str) and os.path.isfile(self.source):
                    self._initialize_video()  # Restart video
                    continue
                else:
                    break
            
            # Process frame asynchronously
            self._process_frame_async(frame)
            
            # Update frame counter
            self.frame_count += 1
            
        # Clean up when thread exits
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def _initialize_video(self):
        """Initialize video source."""
        try:
            if self.cap:
                self.cap.release()

            print(f"[EnhancedVideoController] _initialize_video: self.source = {self.source} (type: {type(self.source)})")
            # Only use camera if source is int or '0', else use file path
            if isinstance(self.source, int):
                self.cap = cv2.VideoCapture(self.source)
            elif isinstance(self.source, str) and os.path.isfile(self.source):
                self.cap = cv2.VideoCapture(self.source)
            else:
                print(f"[EnhancedVideoController] Invalid source: {self.source}, not opening VideoCapture.")
                return False

            if not self.cap.isOpened():
                print(f"Failed to open video source: {self.source}")
                return False

            # Get source FPS
            self.source_fps = self.cap.get(cv2.CAP_PROP_FPS)
            if self.source_fps <= 0:
                self.source_fps = 30  # Default fallback

            print(f"Video source initialized: {self.source}, FPS: {self.source_fps}")
            return True

        except Exception as e:
            print(f"Error initializing video: {e}")
            return False
    
    def _process_frame_async(self, frame):
        """Process a frame with async detection."""
        try:
            # Start detection timer
            detection_start = time.time()
            
            # Assign frame ID
            frame_id = self.next_frame_id
            self.next_frame_id += 1
            
            # Get detector and start async inference
            detector = self.model_manager.detector
            
            # Check if detector supports async API
            if hasattr(detector, 'detect_async_start'):
                # Use async API
                inf_frame_id = detector.detect_async_start(frame)
                
                # Store frame in queue with the right ID
                self.mutex.lock()
                self.frame_queue.append((frame_id, frame, inf_frame_id))
                self.mutex.unlock()
                
                # Try getting results from previous frames
                self._check_async_results()
                
            else:
                # Fallback to synchronous API
                detections = self.model_manager.detect(frame)
                
                # Calculate detection time
                detection_time = time.time() - detection_start
                self.detection_times.append(detection_time)
                
                # Update detection FPS
                elapsed = time.time() - self.start_time
                if elapsed > 0:
                    self.detection_fps = self.frame_count / elapsed
                
                # Calculate detection metrics
                detection_ms = detection_time * 1000
                avg_detection_ms = np.mean(self.detection_times) * 1000
                
                # Store metrics
                metrics = {
                    'detection_fps': self.detection_fps,
                    'detection_ms': detection_ms,
                    'avg_detection_ms': avg_detection_ms,
                    'frame_id': frame_id
                }
                
                # Store processed frame
                self.mutex.lock()
                self.processed_frames[frame_id] = (frame, detections, metrics)
                self.mutex.unlock()
                
                # Emit stats update
                self.stats_updated.emit(metrics)
            
        except Exception as e:
            print(f"Error in frame processing: {e}")
            import traceback
            traceback.print_exc()
    
    def _check_async_results(self):
        """Check for completed async inference requests."""
        try:
            detector = self.model_manager.detector
            if not hasattr(detector, 'detect_async_get_result'):
                return
                
            # Get any frames waiting for results
            self.mutex.lock()
            queue_copy = self.frame_queue.copy()
            self.mutex.unlock()
            
            processed_frames = []
            
            # Check each frame in the queue
            for idx, (frame_id, frame, inf_frame_id) in enumerate(queue_copy):
                # Try to get results without waiting
                detections = detector.detect_async_get_result(inf_frame_id, wait=False)
                
                # If results are ready
                if detections is not None:
                    # Calculate metrics
                    detection_time = time.time() - detector.active_requests[inf_frame_id][2] if inf_frame_id in detector.active_requests else 0
                    self.detection_times.append(detection_time)
                    
                    # Update detection FPS
                    elapsed = time.time() - self.start_time
                    if elapsed > 0:
                        self.detection_fps = self.frame_count / elapsed
                    
                    # Calculate metrics
                    detection_ms = detection_time * 1000
                    avg_detection_ms = np.mean(self.detection_times) * 1000
                    
                    # Store metrics
                    metrics = {
                        'detection_fps': self.detection_fps,
                        'detection_ms': detection_ms,
                        'avg_detection_ms': avg_detection_ms,
                        'frame_id': frame_id
                    }
                    
                    # Store processed frame
                    self.mutex.lock()
                    self.processed_frames[frame_id] = (frame, detections, metrics)
                    processed_frames.append(frame_id)
                    self.mutex.unlock()
                    
                    # Emit stats update
                    self.stats_updated.emit(metrics)
            
            # Remove processed frames from queue
            if processed_frames:
                self.mutex.lock()
                self.frame_queue = [item for item in self.frame_queue 
                                 if item[0] not in processed_frames]
                self.mutex.unlock()
                
        except Exception as e:
            print(f"Error checking async results: {e}")
            import traceback
            traceback.print_exc()
    
    def _emit_next_frame(self):
        """Emit the next processed frame to UI at a controlled rate."""
        try:
            # Update UI FPS calculation
            current_time = time.time()
            if self.last_ui_frame_time > 0:
                ui_frame_time = current_time - self.last_ui_frame_time
                self.ui_frame_times.append(ui_frame_time)
                self.ui_fps = 1.0 / ui_frame_time if ui_frame_time > 0 else 0
            self.last_ui_frame_time = current_time
            
            # Check async results first
            self._check_async_results()
            
            # Find the next frame to emit
            self.mutex.lock()
            available_frames = sorted(self.processed_frames.keys())
            self.mutex.unlock()
            
            if not available_frames:
                return
                
            next_frame_id = available_frames[0]
            
            # Get the frame data
            self.mutex.lock()
            frame, detections, metrics = self.processed_frames.pop(next_frame_id)
            self.mutex.unlock()
            
            # Add UI FPS to metrics
            metrics['ui_fps'] = self.ui_fps
            
            # Apply tracking if available
            if self.model_manager.tracker:
                detections = self.model_manager.update_tracking(detections, frame)
            
            # Emit the frame to the UI
            self.frame_processed.emit(frame, detections, metrics)
            
            # Store as last emitted frame
            self.last_emitted_frame_id = next_frame_id
            
        except Exception as e:
            print(f"Error emitting frame: {e}")
            import traceback
            traceback.print_exc()

class EnhancedVideoController(QObject):
    """
    Enhanced video controller with better file handling and statistics.
    """
    # Define signals
    frame_ready = Signal(QPixmap)  # Frame as QPixmap for direct display
    frame_np_ready = Signal(np.ndarray)  # Frame as NumPy array
    raw_frame_ready = Signal(dict)  # Raw frame data with detections
    stats_ready = Signal(dict)  # All performance stats (dictionary with fps and detection_time)
    
    # Add instance variable to track the most recent traffic light color
    def __init__(self, model_manager=None):
        """Initialize the video controller"""
        super().__init__()
        
        # Input source
        self._source = 0  # Default to camera 0
        self._source_type = "camera"
        self._running = False
        self._last_traffic_light_color = "unknown"
        
        # Regular Controller instance variables
        self.model_manager = model_manager
        self.processing_thread = None
        self.show_annotations = True
        self.show_fps = True
        self.save_video = False
        self.video_writer = None
    
    def set_source(self, source):
        """Set video source - camera index or file path."""
        print(f"[EnhancedVideoController] set_source: {source} ({type(source)})")
        if self.processing_thread:
            self.processing_thread.set_source(source)
        
    def start(self):
        """Start video processing."""
        if self.processing_thread and self.processing_thread.running:
            return
            
        # Create new processing thread
        self.processing_thread = AsyncVideoProcessingThread(self.model_manager)
        
        # Connect signals
        self.processing_thread.frame_processed.connect(self._on_frame_processed)
        self.processing_thread.stats_updated.connect(self._on_stats_updated)
        
        # Start processing
        self.processing_thread.start_processing()
        
    def stop(self):
        """Stop video processing."""
        if self.processing_thread:
            self.processing_thread.stop_processing()
            self.processing_thread = None
            
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            
    def pause(self):
        """Pause video processing."""
        if self.processing_thread:
            self.processing_thread.pause_processing()
            
    def resume(self):
        """Resume video processing."""
        if self.processing_thread:
            self.processing_thread.resume_processing()
            
    def toggle_annotations(self, enabled):
        """Toggle annotations on/off."""
        self.show_annotations = enabled
        
    def toggle_fps_display(self, enabled):
        """Toggle FPS display on/off."""
        self.show_fps = enabled
        
    def start_recording(self, output_path, frame_size=(640, 480), fps=30):
        """Start recording video to file."""
        self.save_video = True
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer = cv2.VideoWriter(
            output_path, fourcc, fps, 
            (frame_size[0], frame_size[1])
        )
        
    def stop_recording(self):
        """Stop recording video."""
        self.save_video = False
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            
    def _on_frame_processed(self, frame, detections, metrics):
        """Handle processed frame from the worker thread."""
        try:
            # Create a copy of the frame for annotation
            display_frame = frame.copy()
            
            # Apply annotations if enabled
            if self.show_annotations and detections:
                display_frame = enhanced_draw_detections(display_frame, detections)                # Detect and annotate traffic light colors
                for detection in detections:
                    # Check for both class_id 9 (COCO) and any other traffic light classes
                    if detection.get('class_id') == 9 or detection.get('class_name') == 'traffic light':
                        bbox = detection.get('bbox')
                        if not bbox:
                            continue
                        
                        # Get traffic light color
                        color = detect_traffic_light_color(frame, bbox)
                        # Store the latest traffic light color
                        self._last_traffic_light_color = color
                        # Draw traffic light status
                        display_frame = draw_traffic_light_status(display_frame, bbox, color)
                        print(f"🚦 Traffic light detected with color: {color}")
                
            # Add FPS counter if enabled
            if self.show_fps:
                # Add both detection and UI FPS
                detection_fps = metrics.get('detection_fps', 0)
                ui_fps = metrics.get('ui_fps', 0)
                detection_ms = metrics.get('avg_detection_ms', 0)
                
                display_frame = draw_performance_overlay(
                    display_frame,
                    {
                        "Detection FPS": f"{detection_fps:.1f}",
                        "UI FPS": f"{ui_fps:.1f}",
                        "Inference": f"{detection_ms:.1f} ms"
                    }
                )
                
            # Save frame if recording
            if self.save_video and self.video_writer:
                self.video_writer.write(display_frame)
                
            # Convert to QPixmap for display
            pixmap = enhanced_cv_to_pixmap(display_frame)
            
            # Emit signals
            self.frame_ready.emit(pixmap, detections, metrics)
            self.raw_frame_ready.emit(frame, detections, metrics)
            # Emit numpy frame for compatibility with existing connections
            self.frame_np_ready.emit(frame)
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            import traceback
            traceback.print_exc()
    def _on_stats_updated(self, stats):
        """Handle updated statistics from the worker thread."""
        try:
            # Create a proper stats dictionary for the LiveTab
            ui_stats = {
                'fps': stats.get('detection_fps', 0.0),
                'detection_time': stats.get('avg_detection_ms', 0.0),
                'traffic_light_color': self._last_traffic_light_color
            }
            print(f"Emitting stats: {ui_stats}")
            # Emit as a dictionary - fixed signal/slot mismatch
            self.stats_ready.emit(ui_stats)
        except Exception as e:
            print(f"Error in stats update: {e}")
            import traceback
            traceback.print_exc()
    
    def _process_frame_for_display(self, frame, detections, metrics=None):
        """Process a frame for display, adding annotations."""
        try:
            # Create a copy for display
            display_frame = frame.copy()
              # Process traffic light detections to identify colors
            for det in detections:
                if det.get('class_name') == 'traffic light':
                    # Get traffic light color
                    bbox = det['bbox']
                    light_color = detect_traffic_light_color(frame, bbox)
                    
                    # Add color information to detection
                    det['traffic_light_color'] = light_color
                    
                    # Store the latest traffic light color
                    self._last_traffic_light_color = light_color
                    
                    # Use specialized drawing for traffic lights
                    display_frame = draw_traffic_light_status(display_frame, bbox, light_color)
                    
                    print(f"🚦 Traffic light detected with color: {light_color}")
                else:
                    # Draw regular detection box
                    bbox = det['bbox']
                    x1, y1, x2, y2 = [int(c) for c in bbox]
                    class_name = det.get('class_name', 'object')
                    confidence = det.get('confidence', 0.0)
                    
                    label = f"{class_name} {confidence:.2f}"
                    color = (0, 255, 0)  # Green for other objects
                    
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Add tracker visualization if tracking is enabled
            if self.tracker and hasattr(self, 'visualization_tracks'):
                # Draw current tracks
                for track_id, track_info in self.visualization_tracks.items():
                    track_box = track_info.get('box')
                    if track_box:
                        x1, y1, x2, y2 = [int(c) for c in track_box]
                        track_class = track_info.get('class_name', 'tracked')
                        
                        # Draw track ID and class
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                        cv2.putText(display_frame, f"{track_class} #{track_id}", 
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                        
                        # Draw trail if available
                        trail = track_info.get('trail', [])
                        if len(trail) > 1:
                            for i in range(1, len(trail)):
                                cv2.line(display_frame, 
                                        (int(trail[i-1][0]), int(trail[i-1][1])),
                                        (int(trail[i][0]), int(trail[i][1])),
                                        (255, 0, 255), 2)
            
            # Add FPS counter if enabled
            if self.show_fps:
                # Add both detection and UI FPS
                detection_fps = metrics.get('detection_fps', 0)
                ui_fps = metrics.get('ui_fps', 0)
                detection_ms = metrics.get('avg_detection_ms', 0)
                
                display_frame = draw_performance_overlay(
                    display_frame,
                    {
                        "Detection FPS": f"{detection_fps:.1f}",
                        "UI FPS": f"{ui_fps:.1f}",
                        "Inference": f"{detection_ms:.1f} ms"
                    }
                )
                
            # Save frame if recording
            if self.save_video and self.video_writer:
                self.video_writer.write(display_frame)
                
            # Convert to QPixmap for display
            pixmap = enhanced_cv_to_pixmap(display_frame)
            
            # Emit signals
            self.frame_ready.emit(pixmap, detections, metrics)
            self.raw_frame_ready.emit(frame, detections, metrics)
            # Emit numpy frame for compatibility with existing connections
            self.frame_np_ready.emit(frame)
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            import traceback
            traceback.print_exc()
