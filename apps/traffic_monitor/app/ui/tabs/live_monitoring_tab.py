"""
Live Monitoring Tab - Real-time traffic monitoring with multi-camera grid view
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                               QLabel, QPushButton, QFrame, QSplitter, QScrollArea,
                               QGroupBox, QComboBox, QSpinBox, QCheckBox,
                               QProgressBar, QSlider)
from PySide6.QtCore import Qt, Signal, QTimer, pyqtSignal
from PySide6.QtGui import QFont, QPixmap, QPainter, QColor

from ..widgets.video_display_widget import VideoDisplayWidget
from ..widgets.camera_control_panel import CameraControlPanel
from ..widgets.detection_overlay_widget import DetectionOverlayWidget
from ..widgets.statistics_panel import StatisticsPanel

class LiveMonitoringTab(QWidget):
    """
    Live Monitoring Tab with real-time multi-camera grid view
    
    Features:
    - Multi-camera grid layout (1x1, 2x2, 3x3, 4x4)
    - Real-time video feeds with detection overlays
    - Camera control panels for each feed
    - Live statistics and alerts
    - Recording controls
    - Full-screen mode for individual cameras
    """
    
    # Signals
    camera_status_changed = Signal(str, str)  # camera_id, status
    recording_started = Signal(str)           # camera_id
    recording_stopped = Signal(str)           # camera_id
    fullscreen_requested = Signal(str)        # camera_id
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Initialize properties
        self.camera_feeds = {}
        self.grid_size = (2, 2)
        self.recording_status = {}
        
        self._setup_ui()
        self._setup_connections()
        
        # Timer for real-time updates
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_displays)
        self.update_timer.start(100)  # Update every 100ms for smooth video
        
        print("✅ Live Monitoring Tab initialized")
    
    def _setup_ui(self):
        """Setup the live monitoring UI"""
        
        # Main layout
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Left panel - Camera grid and controls
        left_panel = self._create_left_panel()
        main_layout.addWidget(left_panel, 3)  # 75% width
        
        # Right panel - Statistics and controls
        right_panel = self._create_right_panel()
        main_layout.addWidget(right_panel, 1)  # 25% width
    
    def _create_left_panel(self):
        """Create the left panel with camera grid"""
        panel = QFrame()
        panel.setObjectName("leftPanel")
        layout = QVBoxLayout(panel)
        
        # Header with grid controls
        header = self._create_grid_header()
        layout.addWidget(header)
        
        # Camera grid container
        self.grid_container = QScrollArea()
        self.grid_container.setWidgetResizable(True)
        self.grid_container.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.grid_container.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Create initial grid
        self.camera_grid_widget = QWidget()
        self.camera_grid_layout = QGridLayout(self.camera_grid_widget)
        self.camera_grid_layout.setSpacing(5)
        
        self._setup_camera_grid()
        
        self.grid_container.setWidget(self.camera_grid_widget)
        layout.addWidget(self.grid_container)
        
        return panel
    
    def _create_grid_header(self):
        """Create the grid control header"""
        header = QFrame()
        header.setFixedHeight(50)
        layout = QHBoxLayout(header)
        
        # Grid size selector
        layout.addWidget(QLabel("Grid Layout:"))
        
        self.grid_selector = QComboBox()
        self.grid_selector.addItems(["1×1", "2×2", "3×3", "4×4"])
        self.grid_selector.setCurrentText("2×2")
        self.grid_selector.currentTextChanged.connect(self._change_grid_size)
        layout.addWidget(self.grid_selector)
        
        layout.addStretch()
        
        # Global recording controls
        self.record_all_button = QPushButton("🔴 Record All")
        self.record_all_button.clicked.connect(self._toggle_record_all)
        layout.addWidget(self.record_all_button)
        
        # Snapshot all button
        snapshot_button = QPushButton("📸 Snapshot All")
        snapshot_button.clicked.connect(self._snapshot_all)
        layout.addWidget(snapshot_button)
        
        # Full screen toggle
        fullscreen_button = QPushButton("⛶ Fullscreen")
        fullscreen_button.clicked.connect(self._toggle_fullscreen)
        layout.addWidget(fullscreen_button)
        
        return header
    
    def _create_right_panel(self):
        """Create the right panel with statistics and controls"""
        panel = QFrame()
        panel.setObjectName("rightPanel")
        layout = QVBoxLayout(panel)
        
        # Live statistics
        stats_group = QGroupBox("Live Statistics")
        stats_layout = QVBoxLayout(stats_group)
        
        self.stats_panel = StatisticsPanel()
        stats_layout.addWidget(self.stats_panel)
        
        layout.addWidget(stats_group)
        
        # Camera controls
        controls_group = QGroupBox("Camera Controls")
        controls_layout = QVBoxLayout(controls_group)
        
        # Active camera selector
        controls_layout.addWidget(QLabel("Active Camera:"))
        self.active_camera_combo = QComboBox()
        self.active_camera_combo.currentTextChanged.connect(self._select_active_camera)
        controls_layout.addWidget(self.active_camera_combo)
        
        # Camera control panel
        self.camera_controls = CameraControlPanel()
        controls_layout.addWidget(self.camera_controls)
        
        layout.addWidget(controls_group)
        
        # Detection settings
        detection_group = QGroupBox("Detection Settings")
        detection_layout = QVBoxLayout(detection_group)
        
        # Confidence threshold
        detection_layout.addWidget(QLabel("Confidence Threshold:"))
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(1, 100)
        self.confidence_slider.setValue(50)
        self.confidence_slider.valueChanged.connect(self._update_confidence)
        detection_layout.addWidget(self.confidence_slider)
        
        self.confidence_label = QLabel("50%")
        detection_layout.addWidget(self.confidence_label)
        
        # Detection toggles
        self.show_boxes_cb = QCheckBox("Show Bounding Boxes")
        self.show_boxes_cb.setChecked(True)
        detection_layout.addWidget(self.show_boxes_cb)
        
        self.show_tracks_cb = QCheckBox("Show Tracking IDs")
        self.show_tracks_cb.setChecked(True)
        detection_layout.addWidget(self.show_tracks_cb)
        
        self.show_speed_cb = QCheckBox("Show Speed Estimates")
        self.show_speed_cb.setChecked(False)
        detection_layout.addWidget(self.show_speed_cb)
        
        layout.addWidget(detection_group)
        
        # Alert panel
        alerts_group = QGroupBox("Live Alerts")
        alerts_layout = QVBoxLayout(alerts_group)
        
        self.alerts_scroll = QScrollArea()
        self.alerts_scroll.setMaximumHeight(150)
        self.alerts_widget = QWidget()
        self.alerts_layout = QVBoxLayout(self.alerts_widget)
        self.alerts_scroll.setWidget(self.alerts_widget)
        alerts_layout.addWidget(self.alerts_scroll)
        
        layout.addWidget(alerts_group)
        
        layout.addStretch()
        
        return panel
    
    def _setup_camera_grid(self):
        """Setup the camera grid based on current grid size"""
        
        # Clear existing grid
        for i in reversed(range(self.camera_grid_layout.count())):
            child = self.camera_grid_layout.itemAt(i).widget()
            if child:
                child.setParent(None)
        
        rows, cols = self.grid_size
        
        # Create camera display widgets
        for row in range(rows):
            for col in range(cols):
                camera_id = f"camera_{row}_{col}"
                
                # Create video display widget
                video_widget = VideoDisplayWidget(camera_id)
                video_widget.setMinimumSize(320, 240)
                video_widget.double_clicked.connect(lambda cid=camera_id: self._camera_double_clicked(cid))
                
                self.camera_feeds[camera_id] = video_widget
                self.camera_grid_layout.addWidget(video_widget, row, col)
                
                # Add to active camera selector
                self.active_camera_combo.addItem(f"Camera {row+1}-{col+1}", camera_id)
    
    def _setup_connections(self):
        """Setup signal connections"""
        
        # Detection overlay connections
        if hasattr(self, 'show_boxes_cb'):
            self.show_boxes_cb.toggled.connect(self._update_overlay_settings)
        if hasattr(self, 'show_tracks_cb'):
            self.show_tracks_cb.toggled.connect(self._update_overlay_settings)
        if hasattr(self, 'show_speed_cb'):
            self.show_speed_cb.toggled.connect(self._update_overlay_settings)
    
    def _change_grid_size(self, size_text):
        """Change the camera grid size"""
        size_map = {
            "1×1": (1, 1),
            "2×2": (2, 2),
            "3×3": (3, 3),
            "4×4": (4, 4)
        }
        
        if size_text in size_map:
            self.grid_size = size_map[size_text]
            
            # Clear active camera combo
            self.active_camera_combo.clear()
            
            # Recreate grid
            self._setup_camera_grid()
            
            print(f"📺 Grid size changed to {size_text}")
    
    def _toggle_record_all(self):
        """Toggle recording for all cameras"""
        recording = self.record_all_button.text().startswith("🔴")
        
        if recording:
            # Start recording all
            self.record_all_button.setText("⏹️ Stop All")
            for camera_id in self.camera_feeds:
                self.recording_status[camera_id] = True
                self.recording_started.emit(camera_id)
        else:
            # Stop recording all
            self.record_all_button.setText("🔴 Record All")
            for camera_id in self.camera_feeds:
                self.recording_status[camera_id] = False
                self.recording_stopped.emit(camera_id)
        
        # Update individual camera displays
        self._update_recording_indicators()
    
    def _snapshot_all(self):
        """Take snapshots of all camera feeds"""
        for camera_id, widget in self.camera_feeds.items():
            widget.take_snapshot()
        print("📸 Snapshots taken for all cameras")
    
    def _toggle_fullscreen(self):
        """Toggle fullscreen mode for active camera"""
        current_camera = self.active_camera_combo.currentData()
        if current_camera:
            self.fullscreen_requested.emit(current_camera)
    
    def _select_active_camera(self, camera_text):
        """Select the active camera for controls"""
        camera_id = self.active_camera_combo.currentData()
        if camera_id and camera_id in self.camera_feeds:
            # Update camera controls for selected camera
            self.camera_controls.set_active_camera(camera_id)
            print(f"📹 Active camera: {camera_text}")
    
    def _camera_double_clicked(self, camera_id):
        """Handle camera double-click for fullscreen"""
        self.fullscreen_requested.emit(camera_id)
    
    def _update_confidence(self, value):
        """Update confidence threshold"""
        self.confidence_label.setText(f"{value}%")
        
        # Update all video widgets
        for widget in self.camera_feeds.values():
            widget.set_confidence_threshold(value / 100.0)
    
    def _update_overlay_settings(self):
        """Update detection overlay settings"""
        settings = {
            'show_boxes': self.show_boxes_cb.isChecked(),
            'show_tracks': self.show_tracks_cb.isChecked(),
            'show_speed': self.show_speed_cb.isChecked()
        }
        
        # Apply to all video widgets
        for widget in self.camera_feeds.values():
            widget.update_overlay_settings(settings)
    
    def _update_displays(self):
        """Update all video displays (called by timer)"""
        # This will be connected to actual video streams
        pass
    
    def _update_recording_indicators(self):
        """Update recording indicators on camera widgets"""
        for camera_id, widget in self.camera_feeds.items():
            is_recording = self.recording_status.get(camera_id, False)
            widget.set_recording_indicator(is_recording)
    
    def add_alert(self, message, level="info"):
        """Add a new alert to the alerts panel"""
        from ..widgets.alert_widget import AlertWidget
        
        alert = AlertWidget(message, level)
        self.alerts_layout.addWidget(alert)
        
        # Remove old alerts if too many
        if self.alerts_layout.count() > 10:
            old_alert = self.alerts_layout.itemAt(0).widget()
            if old_alert:
                old_alert.setParent(None)
        
        # Scroll to bottom
        self.alerts_scroll.verticalScrollBar().setValue(
            self.alerts_scroll.verticalScrollBar().maximum()
        )
    
    def update_statistics(self, stats_data):
        """Update the statistics panel"""
        if hasattr(self, 'stats_panel'):
            self.stats_panel.update_data(stats_data)
    
    def set_camera_feed(self, camera_id, frame):
        """Set video frame for a specific camera"""
        if camera_id in self.camera_feeds:
            self.camera_feeds[camera_id].set_frame(frame)
    
    def add_detections(self, camera_id, detections):
        """Add detection results to a camera feed"""
        if camera_id in self.camera_feeds:
            self.camera_feeds[camera_id].add_detections(detections)
