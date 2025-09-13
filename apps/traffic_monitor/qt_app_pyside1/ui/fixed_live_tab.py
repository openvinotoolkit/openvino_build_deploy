from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QComboBox, QGroupBox, QToolButton, QMessageBox
)
from PySide6.QtCore import Qt, Signal, QSize, Slot, QTimer
from PySide6.QtGui import QPixmap, QImage, QIcon
import cv2

# Import our enhanced display widget for better video rendering
from ui.enhanced_simple_live_display import SimpleLiveDisplay
from utils.annotation_utils import convert_cv_to_pixmap

import os
import sys
import time
import numpy as np

class LiveTab(QWidget):
    """Live video processing and detection tab."""
    
    video_dropped = Signal(str)  # Emitted when video is dropped onto display
    source_changed = Signal(object)  # Emitted when video source changes
    snapshot_requested = Signal()  # Emitted when snapshot button is clicked
    run_requested = Signal(bool)  # Emitted when run/stop button is clicked
    
    def __init__(self):
        super().__init__()
        self.current_source = 0  # Default to camera
        self.initUI()
        
    def initUI(self):
        """Initialize UI components"""
        layout = QVBoxLayout(self)
        
        # Video display - use simple label-based display
        self.display = SimpleLiveDisplay()
        layout.addWidget(self.display)
        
        # Connect drag and drop signal from the display
        self.display.video_dropped.connect(self.video_dropped)
        
        # Control panel
        controls = QHBoxLayout()
        
        # Source selection
        self.source_combo = QComboBox()
        self.source_combo.addItem("📹 Camera 0", 0)
        self.source_combo.addItem("📁 Video File", "file")
        self.source_combo.setCurrentIndex(0)
        self.source_combo.currentIndexChanged.connect(self.on_source_changed)
        
        self.file_btn = QPushButton("📂 Browse")
        self.file_btn.setMaximumWidth(100)
        self.file_btn.clicked.connect(self.browse_files)
        
        self.snapshot_btn = QPushButton("📸 Snapshot")
        self.snapshot_btn.clicked.connect(self.snapshot_requested)
        
        # Run/Stop button
        self.run_btn = QPushButton("▶️ Run")
        self.run_btn.setCheckable(True)
        self.run_btn.clicked.connect(self.on_run_clicked)
        self.run_btn.setStyleSheet("QPushButton:checked { background-color: #f44336; color: white; }")
        
        # Performance metrics
        self.fps_label = QLabel("FPS: -- | Inference: -- ms")
        self.fps_label.setObjectName("fpsLabel")
        self.fps_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        
        # Add controls to layout
        src_layout = QHBoxLayout()
        src_layout.addWidget(QLabel("Source:"))
        src_layout.addWidget(self.source_combo)
        src_layout.addWidget(self.file_btn)
        
        controls.addLayout(src_layout)
        controls.addWidget(self.run_btn)
        controls.addWidget(self.snapshot_btn)
        controls.addStretch(1)
        controls.addWidget(self.fps_label)
        
        layout.addLayout(controls)
        
        # Status bar
        status_bar = QHBoxLayout()
        self.status_label = QLabel("Ready")
        status_bar.addWidget(self.status_label)
        layout.addLayout(status_bar)

    @Slot()
    def on_source_changed(self):
        """Handle source selection change"""
        source_data = self.source_combo.currentData()
        print(f"DEBUG: on_source_changed - current data: {source_data} (type: {type(source_data)})")
        if source_data == "file":
            # If "Video File" option is selected, open file dialog
            self.browse_files()
            return  # browse_files will emit the signal
        # For camera or specific file path
        if isinstance(source_data, str) and os.path.isfile(source_data):
            self.current_source = source_data
            print(f"DEBUG: emitting source_changed with file path: {source_data}")
            self.source_changed.emit(source_data)
        elif source_data == 0:
            self.current_source = 0
            print(f"DEBUG: emitting source_changed with camera index 0")
            self.source_changed.emit(0)
        else:
            print(f"WARNING: Unknown source_data: {source_data}")

    @Slot()
    def browse_files(self):
        """Open file dialog to select video file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Video File", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.webm);;All Files (*)"
        )
        if file_path:
            print(f"DEBUG: Selected file: {file_path} (type: {type(file_path)})")
            # Always add or select the file path in the combo box
            existing_idx = self.source_combo.findData(file_path)
            if existing_idx == -1:
                self.source_combo.addItem(os.path.basename(file_path), file_path)
                self.source_combo.setCurrentIndex(self.source_combo.count() - 1)
            else:
                self.source_combo.setCurrentIndex(existing_idx)
            self.current_source = file_path
            print(f"DEBUG: Setting current_source to: {self.current_source}")
            print(f"DEBUG: emitting source_changed with {file_path}")
            self.source_changed.emit(file_path)
        else:
            # If user cancels, revert to previous valid source
            if isinstance(self.current_source, str) and os.path.isfile(self.current_source):
                idx = self.source_combo.findData(self.current_source)
                if idx != -1:
                    self.source_combo.setCurrentIndex(idx)
            else:
                self.source_combo.setCurrentIndex(0)

    @Slot(bool)
    def on_run_clicked(self, checked):
        """Handle run/stop button clicks"""
        if checked:
            self.run_btn.setText("⏹️ Stop")
            print(f"DEBUG: on_run_clicked - current_source: {self.current_source} (type: {type(self.current_source)})")
            if isinstance(self.current_source, str) and os.path.isfile(self.current_source):
                print(f"DEBUG: Re-emitting source_changed with file: {self.current_source}")
                self.source_changed.emit(self.current_source)
                QTimer.singleShot(500, lambda: self.run_requested.emit(True))
            elif self.current_source == 0:
                print(f"DEBUG: Re-emitting source_changed with camera index 0")
                self.source_changed.emit(0)
                QTimer.singleShot(500, lambda: self.run_requested.emit(True))
            else:
                print("ERROR: No valid source selected")
                self.run_btn.setChecked(False)
                self.run_btn.setText("▶️ Run")
                return
            self.status_label.setText(f"Running... (Source: {self.current_source})")
        else:
            self.run_btn.setText("▶️ Run")
            self.run_requested.emit(False)
            self.status_label.setText("Stopped")
    
    @Slot(object, object, dict)
    def update_display(self, pixmap, detections, metrics):
        """Update display with processed frame (detections only)"""
        if pixmap:
            # Print debug info about the pixmap
            print(f"DEBUG: Received pixmap: {pixmap.width()}x{pixmap.height()}, null: {pixmap.isNull()}")
            
            # Ensure pixmap is valid
            if not pixmap.isNull():
                # --- COMMENTED OUT: Draw vehicle info for all detections (ID below bbox) ---
                # for det in detections:
                #     if 'bbox' in det and 'id' in det:
                #         x1, y1, x2, y2 = det['bbox']
                #         vehicle_id = det['id']
                #         class_name = det.get('class_name', 'object')
                #         confidence = det.get('confidence', 0.0)
                #         color = (0, 255, 0)
                #         if class_name == 'traffic light':
                #             color = (0, 0, 255)
                #         label_text = f"{class_name}:{confidence:.2f}"  # Removed vehicle_id from label
                #         label_y = y2 + 20
                #         cv2.putText(frame, label_text, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                # --- END COMMENTED BLOCK ---
                self.display.update_frame(pixmap)
                
                # Update metrics display
                fps = metrics.get('FPS', '--')
                detection_time = metrics.get('Detection (ms)', '--')
                self.fps_label.setText(f"FPS: {fps} | Detection: {detection_time} ms")
                
                # Update status with detection counts and traffic light status
                detection_counts = {}
                traffic_light_statuses = []
                
                for det in detections:
                    class_name = det.get('class_name', 'unknown')
                    detection_counts[class_name] = detection_counts.get(class_name, 0) + 1
                    
                    # Check for traffic light color
                    if class_name == 'traffic light' and 'traffic_light_color' in det:
                        color = det['traffic_light_color']
                        # Handle both dict and string for color
                        if isinstance(color, dict):
                            color_str = color.get('color', 'unknown')
                        else:
                            color_str = str(color)
                        traffic_light_statuses.append(f"Traffic Light: {color_str.upper()}")
                
                # Show traffic light status if available
                if traffic_light_statuses:
                    self.status_label.setText(" | ".join(traffic_light_statuses))
                
                # Otherwise show detection counts
                elif detection_counts:
                    sorted_counts = sorted(
                        detection_counts.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:3]
                    
                    status_text = " | ".join([
                        f"{cls}: {count}" for cls, count in sorted_counts
                    ])
                    
                    self.status_label.setText(status_text)
                else:
                    self.status_label.setText("No detections")
            else:
                print("ERROR: Received null pixmap in update_display")
    @Slot(np.ndarray)
    def update_display_np(self, frame):
        """Update display with direct NumPy frame (optional)"""
        print(f"🟢 Frame received in UI - LiveTab.update_display_np called")
        print(f"🔵 Frame info: type={type(frame)}, shape={getattr(frame, 'shape', 'None')}")
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            print("⚠️ Received None or empty frame in update_display_np")
            return
        # Ensure BGR to RGB conversion for OpenCV frames
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            # Scale pixmap to fit display
            scaled_pixmap = pixmap.scaled(
                self.display.width(), self.display.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            print("📺 Sending scaled pixmap to display widget")
            self.display.update_frame(scaled_pixmap)
        except Exception as e:
            print(f"❌ Error displaying frame: {e}")
            import traceback
            traceback.print_exc()
            self.status_label.setText(f"Error displaying frame: {str(e)[:30]}...")

    def reset_display(self):
        """Reset display to empty state"""
        empty_pixmap = QPixmap(640, 480)
        empty_pixmap.fill(Qt.black)
        self.display.update_frame(empty_pixmap)
        self.fps_label.setText("FPS: -- | Inference: -- ms")
        self.status_label.setText("Ready")
          
    @Slot(dict)
    def update_stats(self, stats):
        """Update performance statistics display"""
        # Extract values from stats dictionary
        fps = stats.get('fps', 0.0)
        detection_time = stats.get('detection_time', 0.0)
        traffic_light_color = stats.get('traffic_light_color', 'unknown')
        
        print(f"🟢 Stats Updated: FPS={fps:.2f}, Inference={detection_time:.2f}ms, Traffic Light={traffic_light_color}")
        self.fps_label.setText(f"FPS: {fps:.1f}")
          # Update status with traffic light information if available
        if traffic_light_color != 'unknown':
            # Create colorful text for traffic light
            # Handle both dictionary and string formats
            if isinstance(traffic_light_color, dict):
                color_text = traffic_light_color.get("color", "unknown").upper()
            else:
                color_text = str(traffic_light_color).upper()
            # Set text with traffic light information prominently displayed
            self.status_label.setText(f"Inference: {detection_time:.1f} ms | 🚦 Traffic Light: {color_text}")
        else:
            self.status_label.setText(f"Inference: {detection_time:.1f} ms")
    
    @Slot(np.ndarray, object, object, str, int)
    def update_display_with_violations(self, frame, detections, violations, traffic_light_state, frame_idx):
        """
        Update display with frame, detections, and violations overlay from controller logic
        """
        # Draw overlay using the new logic (now in controller, not external)
        violation_line_y = None
        if violations and len(violations) > 0:
            violation_line_y = violations[0]['details'].get('violation_line_y', None)
        frame_with_overlay = self._draw_violation_overlay(frame, violations, violation_line_y)
        pixmap = convert_cv_to_pixmap(frame_with_overlay)
        self.display.update_frame(pixmap)
        self.status_label.setText(f"Violations: {len(violations)} | Traffic Light: {traffic_light_state.upper()} | Frame: {frame_idx}")

    def _draw_violation_overlay(self, frame, violations, violation_line_y=None, vehicle_tracks=None):
        frame_copy = frame.copy()
        violation_color = (0, 140, 255)  # Orange
        if violation_line_y is not None:
            cv2.line(frame_copy, (0, violation_line_y), (frame.shape[1], violation_line_y), violation_color, 3)
            cv2.putText(frame_copy, "VIOLATION LINE", (10, violation_line_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, violation_color, 2)
        for violation in violations:
            bbox = violation['details']['bbox']
            confidence = violation['details']['confidence']
            vehicle_type = violation['details']['vehicle_type']
            vehicle_id = violation.get('id', None)
            x1, y1, x2, y2 = bbox
            color = violation_color
            label = f"VIOLATION: {vehicle_type.upper()}"
            print(f"\033[93m[OVERLAY DRAW] Drawing violation overlay: ID={vehicle_id}, BBOX={bbox}, TYPE={vehicle_type}, CONF={confidence:.2f}\033[0m")
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame_copy, label, (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame_copy, f"Confidence: {confidence:.2f}", (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            if vehicle_id is not None:
                cv2.putText(frame_copy, f"ID: {vehicle_id}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if vehicle_tracks is not None:
            for track_id, track in vehicle_tracks.items():
                for pos in track['positions']:
                    cv2.circle(frame_copy, pos, 3, (255, 0, 255), -1)
        return frame_copy
    
    @Slot(np.ndarray, list, list)
    def update_display_np_with_violations(self, frame, detections, violators):
        """
        Display annotated frame and highlight violators in orange, print violations to console.
        Args:
            frame (np.ndarray): Already-annotated frame from controller.
            detections (list): List of all vehicle detections (with id, bbox).
            violators (list): List of violator dicts (with id, bbox, etc.).
        """
        print(f"🟢 Frame received in UI - update_display_np_with_violations called")
        print(f"🔵 Frame info: type={type(frame)}, shape={getattr(frame, 'shape', 'None')}")
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            print("⚠️ Received None or empty frame in update_display_np_with_violations")
            return
        frame_disp = frame.copy()
        # Draw orange boxes for violators
        for v in violators:
            bbox = v.get('bbox')
            vid = v.get('id')
            if bbox is not None and len(bbox) == 4:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame_disp, (x1, y1), (x2, y2), (0,140,255), 4)
                cv2.putText(frame_disp, f"VIOLATION ID:{vid}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,140,255), 2)
                print(f"[VIOLATION] Vehicle {vid} crossed at bbox {bbox}")
        pixmap = convert_cv_to_pixmap(frame_disp)
        print("📺 Sending frame to display widget")
        self.display.update_frame(pixmap)
        print("✅ Frame passed to display widget successfully")
        self.status_label.setText(f"Frame displayed: {frame.shape[1]}x{frame.shape[0]}, Violations: {len(violators)}")
