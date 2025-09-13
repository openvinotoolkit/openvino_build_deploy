from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QComboBox, QGroupBox, QToolButton
)
from PySide6.QtCore import Qt, Signal, QSize, Slot, QTimer
from PySide6.QtGui import QPixmap, QImage, QIcon

# Import our enhanced display widget for better video rendering
from ui.enhanced_simple_live_display import SimpleLiveDisplay

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
        self.current_source = source_data
        print(f"DEBUG: emitting source_changed with {source_data} (type: {type(source_data)})")
        self.source_changed.emit(source_data)
              
    @Slot()
    def browse_files(self):
        """Open file dialog to select video file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Video File", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.webm);;All Files (*)"
        )
        
        if file_path:
            print(f"DEBUG: Selected file: {file_path} (type: {type(file_path)})")
            # First set dropdown to "Video File" option
            file_idx = self.source_combo.findData("file")
            if file_idx >= 0:
                self.source_combo.setCurrentIndex(file_idx)
                
            # Then add the specific file
            existing_idx = self.source_combo.findData(file_path)
            if existing_idx == -1:
                # Add new item
                self.source_combo.addItem(os.path.basename(file_path), file_path)
                self.source_combo.setCurrentIndex(self.source_combo.count() - 1)
            else:
                # Select existing item
                self.source_combo.setCurrentIndex(existing_idx)
                
            # Update current source
            self.current_source = file_path
            print(f"DEBUG: Setting current_source to: {self.current_source}")
            print(f"DEBUG: emitting source_changed with {file_path}")
            self.source_changed.emit(file_path)
    @Slot(bool)
    def on_run_clicked(self, checked):
        """Handle run/stop button clicks"""
        if checked:
            # If run is clicked, ensure we're using the current source
            self.run_btn.setText("⏹️ Stop")
            
            # Print detailed debug info
            print(f"DEBUG: on_run_clicked - current_source: {self.current_source} (type: {type(self.current_source)})")
            
            # First ensure the correct source is set before running
            if self.current_source is not None:
                # Re-emit the source to make sure it's properly set
                print(f"DEBUG: Re-emitting source_changed with: {self.current_source}")
                self.source_changed.emit(self.current_source)
                
                # Use a timer to give the source time to be set
                QTimer.singleShot(500, lambda: self.run_requested.emit(True))
            else:
                print("ERROR: No source selected")
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
                self.display.update_frame(pixmap)
                
                # Update metrics display
                fps = metrics.get('FPS', '--')
                detection_time = metrics.get('Detection (ms)', '--')
                self.fps_label.setText(f"FPS: {fps} | Detection: {detection_time} ms")
                
                # Update status with detection counts
                detection_counts = {}
                for det in detections:
                    class_name = det.get('class_name', 'unknown')
                    detection_counts[class_name] = detection_counts.get(class_name, 0) + 1
                
                # Show top 3 detected classes
                if detection_counts:
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
        print(f"� Frame received in UI - LiveTab.update_display_np called")
        print(f"🔵 Frame info: type={type(frame)}, shape={getattr(frame, 'shape', 'None')}")
        
        if frame is None:
            print("⚠️ Received None frame in update_display_np")
            return
            
        if not isinstance(frame, np.ndarray):
            print(f"⚠️ Received non-numpy frame type: {type(frame)}")
            return
            
        if frame.size == 0 or frame.shape[0] == 0 or frame.shape[1] == 0:
            print(f"⚠️ Received empty frame with shape: {frame.shape}")
            return
        
        try:
            # Make sure we have a fresh copy of the data
            frame_copy = frame.copy()
            # Display the frame through our display widget
            print("📺 Sending frame to display widget")
            self.display.display_frame(frame_copy)
            print("✅ Frame passed to display widget successfully")
        except Exception as e:
            print(f"❌ Error displaying frame: {e}")
            import traceback
            traceback.print_exc()

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
        traffic_light_info = stats.get('traffic_light_color', 'unknown')
        
        # Handle both string and dictionary formats for traffic light color
        if isinstance(traffic_light_info, dict):
            traffic_light_color = traffic_light_info.get('color', 'unknown')
            confidence = traffic_light_info.get('confidence', 0.0)
            confidence_text = f" (Conf: {confidence:.2f})"
        else:
            traffic_light_color = traffic_light_info
            confidence_text = ""
        
        print(f"🟢 Stats Updated: FPS={fps:.2f}, Inference={detection_time:.2f}ms, Traffic Light={traffic_light_color}{confidence_text}")
        self.fps_label.setText(f"FPS: {fps:.1f}")
        
        # Update status with traffic light information if available
        if traffic_light_color != 'unknown':
            # Create colorful text for traffic light
            color_text = str(traffic_light_color).upper()
            
            # Set color-coded style based on traffic light color
            color_style = ""
            if color_text == "RED":
                color_style = "color: red; font-weight: bold;"
            elif color_text == "YELLOW":
                color_style = "color: #FFD700; font-weight: bold;"  # Golden yellow for better visibility
            elif color_text == "GREEN":
                color_style = "color: green; font-weight: bold;"
                
            # Set text with traffic light information prominently displayed
            self.status_label.setText(f"Inference: {detection_time:.1f} ms | 🚦 Traffic Light: <span style='{color_style}'>{color_text}</span>{confidence_text}")
            # Print the status to console too for debugging
            if isinstance(traffic_light_info, dict) and 'confidence' in traffic_light_info:
                print(f"🚦 UI Updated: Traffic Light = {color_text} (Confidence: {confidence:.2f})")
            else:
                print(f"🚦 UI Updated: Traffic Light = {color_text}")
        else:
            self.status_label.setText(f"Inference: {detection_time:.1f} ms")
