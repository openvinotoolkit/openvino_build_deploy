"""
Camera Control Panel Widget - Individual camera controls
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                               QPushButton, QSlider, QSpinBox, QComboBox, 
                               QCheckBox, QGroupBox, QGridLayout)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont

class CameraControlPanel(QWidget):
    """
    Individual camera control panel with adjustment capabilities
    
    Features:
    - Brightness/Contrast/Saturation controls
    - Zoom and pan controls
    - Recording controls
    - Detection sensitivity
    - ROI (Region of Interest) settings
    """
    
    # Signals
    setting_changed = Signal(str, str, object)  # camera_id, setting_name, value
    recording_toggled = Signal(str, bool)        # camera_id, recording_state
    snapshot_requested = Signal(str)             # camera_id
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.active_camera = None
        self.camera_settings = {}
        
        self._setup_ui()
        
        print("🎛️ Camera Control Panel initialized")
    
    def _setup_ui(self):
        """Setup the camera control panel UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(8)
        
        # Camera selection info
        info_section = self._create_info_section()
        layout.addWidget(info_section)
        
        # Image adjustment controls
        image_section = self._create_image_section()
        layout.addWidget(image_section)
        
        # Position controls
        position_section = self._create_position_section()
        layout.addWidget(position_section)
        
        # Recording controls
        recording_section = self._create_recording_section()
        layout.addWidget(recording_section)
        
        layout.addStretch()
    
    def _create_info_section(self):
        """Create camera info section"""
        section = QGroupBox("Camera Info")
        layout = QVBoxLayout(section)
        
        # Active camera display
        self.camera_info_label = QLabel("No camera selected")
        self.camera_info_label.setFont(QFont("Segoe UI", 9, QFont.Bold))
        layout.addWidget(self.camera_info_label)
        
        # Status
        self.status_info_label = QLabel("Status: Offline")
        layout.addWidget(self.status_info_label)
        
        # Resolution
        self.resolution_label = QLabel("Resolution: --")
        layout.addWidget(self.resolution_label)
        
        return section
    
    def _create_image_section(self):
        """Create image adjustment section"""
        section = QGroupBox("Image Adjustment")
        layout = QGridLayout(section)
        
        # Brightness
        layout.addWidget(QLabel("Brightness:"), 0, 0)
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(0)
        self.brightness_slider.valueChanged.connect(
            lambda v: self._setting_changed("brightness", v)
        )
        layout.addWidget(self.brightness_slider, 0, 1)
        
        self.brightness_label = QLabel("0")
        layout.addWidget(self.brightness_label, 0, 2)
        
        # Contrast
        layout.addWidget(QLabel("Contrast:"), 1, 0)
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(-100, 100)
        self.contrast_slider.setValue(0)
        self.contrast_slider.valueChanged.connect(
            lambda v: self._setting_changed("contrast", v)
        )
        layout.addWidget(self.contrast_slider, 1, 1)
        
        self.contrast_label = QLabel("0")
        layout.addWidget(self.contrast_label, 1, 2)
        
        # Saturation
        layout.addWidget(QLabel("Saturation:"), 2, 0)
        self.saturation_slider = QSlider(Qt.Horizontal)
        self.saturation_slider.setRange(-100, 100)
        self.saturation_slider.setValue(0)
        self.saturation_slider.valueChanged.connect(
            lambda v: self._setting_changed("saturation", v)
        )
        layout.addWidget(self.saturation_slider, 2, 1)
        
        self.saturation_label = QLabel("0")
        layout.addWidget(self.saturation_label, 2, 2)
        
        # Auto adjust button
        auto_btn = QPushButton("Auto Adjust")
        auto_btn.clicked.connect(self._auto_adjust)
        layout.addWidget(auto_btn, 3, 0, 1, 3)
        
        # Connect slider value updates to labels
        self.brightness_slider.valueChanged.connect(
            lambda v: self.brightness_label.setText(str(v))
        )
        self.contrast_slider.valueChanged.connect(
            lambda v: self.contrast_label.setText(str(v))
        )
        self.saturation_slider.valueChanged.connect(
            lambda v: self.saturation_label.setText(str(v))
        )
        
        return section
    
    def _create_position_section(self):
        """Create position and zoom controls"""
        section = QGroupBox("Position & Zoom")
        layout = QGridLayout(section)
        
        # Zoom control
        layout.addWidget(QLabel("Zoom:"), 0, 0)
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(100, 500)  # 100% to 500%
        self.zoom_slider.setValue(100)
        self.zoom_slider.valueChanged.connect(
            lambda v: self._setting_changed("zoom", v)
        )
        layout.addWidget(self.zoom_slider, 0, 1)
        
        self.zoom_label = QLabel("100%")
        layout.addWidget(self.zoom_label, 0, 2)
        
        # Pan controls
        pan_layout = QHBoxLayout()
        
        # Pan buttons
        pan_up_btn = QPushButton("↑")
        pan_up_btn.setFixedSize(30, 30)
        pan_up_btn.clicked.connect(lambda: self._pan_camera("up"))
        
        pan_down_btn = QPushButton("↓")
        pan_down_btn.setFixedSize(30, 30)
        pan_down_btn.clicked.connect(lambda: self._pan_camera("down"))
        
        pan_left_btn = QPushButton("←")
        pan_left_btn.setFixedSize(30, 30)
        pan_left_btn.clicked.connect(lambda: self._pan_camera("left"))
        
        pan_right_btn = QPushButton("→")
        pan_right_btn.setFixedSize(30, 30)
        pan_right_btn.clicked.connect(lambda: self._pan_camera("right"))
        
        # Arrange pan buttons in cross pattern
        pan_grid = QGridLayout()
        pan_grid.addWidget(pan_up_btn, 0, 1)
        pan_grid.addWidget(pan_left_btn, 1, 0)
        pan_grid.addWidget(pan_right_btn, 1, 2)
        pan_grid.addWidget(pan_down_btn, 2, 1)
        
        # Reset button in center
        reset_btn = QPushButton("⌂")
        reset_btn.setFixedSize(30, 30)
        reset_btn.setToolTip("Reset to center")
        reset_btn.clicked.connect(self._reset_position)
        pan_grid.addWidget(reset_btn, 1, 1)
        
        layout.addLayout(pan_grid, 1, 0, 1, 3)
        
        # Connect zoom slider to label
        self.zoom_slider.valueChanged.connect(
            lambda v: self.zoom_label.setText(f"{v}%")
        )
        
        return section
    
    def _create_recording_section(self):
        """Create recording controls"""
        section = QGroupBox("Recording Controls")
        layout = QVBoxLayout(section)
        
        # Recording toggle
        self.record_btn = QPushButton("🔴 Start Recording")
        self.record_btn.clicked.connect(self._toggle_recording)
        layout.addWidget(self.record_btn)
        
        # Snapshot button
        snapshot_btn = QPushButton("📸 Take Snapshot")
        snapshot_btn.clicked.connect(self._take_snapshot)
        layout.addWidget(snapshot_btn)
        
        # Recording settings
        settings_layout = QGridLayout()
        
        # Quality setting
        settings_layout.addWidget(QLabel("Quality:"), 0, 0)
        self.quality_combo = QComboBox()
        self.quality_combo.addItems(["Low", "Medium", "High", "Ultra"])
        self.quality_combo.setCurrentText("High")
        self.quality_combo.currentTextChanged.connect(
            lambda v: self._setting_changed("quality", v)
        )
        settings_layout.addWidget(self.quality_combo, 0, 1)
        
        # Frame rate
        settings_layout.addWidget(QLabel("FPS:"), 1, 0)
        self.fps_spinbox = QSpinBox()
        self.fps_spinbox.setRange(1, 60)
        self.fps_spinbox.setValue(30)
        self.fps_spinbox.valueChanged.connect(
            lambda v: self._setting_changed("fps", v)
        )
        settings_layout.addWidget(self.fps_spinbox, 1, 1)
        
        layout.addLayout(settings_layout)
        
        # Recording status
        self.recording_status_label = QLabel("Not recording")
        self.recording_status_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(self.recording_status_label)
        
        return section
    
    def set_active_camera(self, camera_id):
        """Set the active camera for controls"""
        self.active_camera = camera_id
        
        # Update display
        display_name = camera_id.replace('_', ' ').title()
        self.camera_info_label.setText(f"Camera: {display_name}")
        
        # Load camera settings if available
        if camera_id in self.camera_settings:
            self._load_camera_settings(camera_id)
        else:
            self._reset_all_settings()
        
        print(f"🎛️ Active camera set to: {camera_id}")
    
    def _setting_changed(self, setting_name, value):
        """Handle setting change"""
        if self.active_camera:
            # Store setting
            if self.active_camera not in self.camera_settings:
                self.camera_settings[self.active_camera] = {}
            
            self.camera_settings[self.active_camera][setting_name] = value
            
            # Emit signal
            self.setting_changed.emit(self.active_camera, setting_name, value)
    
    def _toggle_recording(self):
        """Toggle recording for active camera"""
        if not self.active_camera:
            return
        
        is_recording = self.record_btn.text().startswith("🔴")
        
        if is_recording:
            # Start recording
            self.record_btn.setText("⏹️ Stop Recording")
            self.recording_status_label.setText("Recording...")
            self.recording_status_label.setStyleSheet("color: red; font-weight: bold;")
            self.recording_toggled.emit(self.active_camera, True)
        else:
            # Stop recording
            self.record_btn.setText("🔴 Start Recording")
            self.recording_status_label.setText("Not recording")
            self.recording_status_label.setStyleSheet("color: gray; font-style: italic;")
            self.recording_toggled.emit(self.active_camera, False)
    
    def _take_snapshot(self):
        """Take snapshot of active camera"""
        if self.active_camera:
            self.snapshot_requested.emit(self.active_camera)
    
    def _auto_adjust(self):
        """Auto-adjust image settings"""
        if self.active_camera:
            # Reset to defaults with slight improvements
            self.brightness_slider.setValue(10)
            self.contrast_slider.setValue(15)
            self.saturation_slider.setValue(5)
            print(f"🎛️ Auto-adjusted settings for {self.active_camera}")
    
    def _pan_camera(self, direction):
        """Pan camera in specified direction"""
        if self.active_camera:
            self._setting_changed(f"pan_{direction}", True)
            print(f"🎛️ Panning camera {direction}")
    
    def _reset_position(self):
        """Reset camera position to center"""
        if self.active_camera:
            self.zoom_slider.setValue(100)
            self._setting_changed("reset_position", True)
            print(f"🎛️ Reset position for {self.active_camera}")
    
    def _load_camera_settings(self, camera_id):
        """Load stored settings for camera"""
        settings = self.camera_settings.get(camera_id, {})
        
        # Apply settings to controls
        self.brightness_slider.setValue(settings.get("brightness", 0))
        self.contrast_slider.setValue(settings.get("contrast", 0))
        self.saturation_slider.setValue(settings.get("saturation", 0))
        self.zoom_slider.setValue(settings.get("zoom", 100))
        self.quality_combo.setCurrentText(settings.get("quality", "High"))
        self.fps_spinbox.setValue(settings.get("fps", 30))
    
    def _reset_all_settings(self):
        """Reset all controls to default values"""
        self.brightness_slider.setValue(0)
        self.contrast_slider.setValue(0)
        self.saturation_slider.setValue(0)
        self.zoom_slider.setValue(100)
        self.quality_combo.setCurrentText("High")
        self.fps_spinbox.setValue(30)
    
    def update_camera_status(self, camera_id, status, resolution=None):
        """Update camera status information"""
        if camera_id == self.active_camera:
            self.status_info_label.setText(f"Status: {status.title()}")
            
            if resolution:
                self.resolution_label.setText(f"Resolution: {resolution}")
            
            # Update status color
            if status == "online":
                self.status_info_label.setStyleSheet("color: green;")
            elif status == "offline":
                self.status_info_label.setStyleSheet("color: red;")
            else:
                self.status_info_label.setStyleSheet("color: orange;")
