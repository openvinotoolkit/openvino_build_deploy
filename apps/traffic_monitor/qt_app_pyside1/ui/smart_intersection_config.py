"""
Smart Intersection Configuration Panel
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox, QPushButton,
    QComboBox, QLabel, QTextEdit, QTabWidget, QMessageBox,
    QFileDialog, QSlider, QFrame
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QFont

import json
import os
from pathlib import Path

class SmartIntersectionConfigPanel(QWidget):
    """Configuration panel for Smart Intersection features"""
    
    # Signals
    config_changed = Signal(dict)
    apply_requested = Signal(dict)
    
    def __init__(self):
        super().__init__()
        self.config_path = Path(__file__).parent.parent / "config" / "smart-intersection"
        self.current_config = self.load_default_config()
        self.setup_ui()
        self.load_config()
    
    def setup_ui(self):
        """Setup the user interface"""
        layout = QVBoxLayout(self)
        
        # Header
        header_label = QLabel("🚦 Smart Intersection Configuration")
        header_font = QFont()
        header_font.setPointSize(12)
        header_font.setBold(True)
        header_label.setFont(header_font)
        layout.addWidget(header_label)
        
        # Tabs for different config sections
        self.tabs = QTabWidget()
        
        # Analytics Settings Tab
        self.analytics_tab = self.create_analytics_tab()
        self.tabs.addTab(self.analytics_tab, "Analytics")
        
        # Camera Settings Tab
        self.camera_tab = self.create_camera_tab()
        self.tabs.addTab(self.camera_tab, "Cameras")
        
        # Performance Settings Tab
        self.performance_tab = self.create_performance_tab()
        self.tabs.addTab(self.performance_tab, "Performance")
        
        # ROI Settings Tab
        self.roi_tab = self.create_roi_tab()
        self.tabs.addTab(self.roi_tab, "ROI & Zones")
        
        layout.addWidget(self.tabs)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.load_btn = QPushButton("📁 Load Config")
        self.load_btn.clicked.connect(self.load_config_file)
        
        self.save_btn = QPushButton("💾 Save Config")
        self.save_btn.clicked.connect(self.save_config_file)
        
        self.apply_btn = QPushButton("✅ Apply Settings")
        self.apply_btn.clicked.connect(self.apply_settings)
        self.apply_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        
        self.reset_btn = QPushButton("🔄 Reset to Defaults")
        self.reset_btn.clicked.connect(self.reset_to_defaults)
        
        button_layout.addWidget(self.load_btn)
        button_layout.addWidget(self.save_btn)
        button_layout.addStretch()
        button_layout.addWidget(self.reset_btn)
        button_layout.addWidget(self.apply_btn)
        
        layout.addLayout(button_layout)
        
        # Status
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)
    
    def create_analytics_tab(self) -> QWidget:
        """Create analytics settings tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Scene Analytics Group
        scene_group = QGroupBox("Scene Analytics")
        scene_layout = QFormLayout(scene_group)
        
        self.enable_multi_camera = QCheckBox()
        self.enable_multi_camera.setChecked(True)
        scene_layout.addRow("Enable Multi-Camera Tracking:", self.enable_multi_camera)
        
        self.enable_roi_analytics = QCheckBox()
        self.enable_roi_analytics.setChecked(True)
        scene_layout.addRow("Enable ROI Analytics:", self.enable_roi_analytics)
        
        self.enable_vlm_integration = QCheckBox()
        self.enable_vlm_integration.setChecked(True)
        scene_layout.addRow("Enable VLM Integration:", self.enable_vlm_integration)
        
        layout.addWidget(scene_group)
        
        # Object Analytics Group
        object_group = QGroupBox("Object Analytics")
        object_layout = QFormLayout(object_group)
        
        self.object_tracking = QCheckBox()
        self.object_tracking.setChecked(True)
        object_layout.addRow("Object Tracking:", self.object_tracking)
        
        self.speed_estimation = QCheckBox()
        self.speed_estimation.setChecked(True)
        object_layout.addRow("Speed Estimation:", self.speed_estimation)
        
        self.direction_analysis = QCheckBox()
        self.direction_analysis.setChecked(True)
        object_layout.addRow("Direction Analysis:", self.direction_analysis)
        
        self.dwell_time_analysis = QCheckBox()
        self.dwell_time_analysis.setChecked(True)
        object_layout.addRow("Dwell Time Analysis:", self.dwell_time_analysis)
        
        self.safety_monitoring = QCheckBox()
        self.safety_monitoring.setChecked(True)
        object_layout.addRow("Safety Monitoring:", self.safety_monitoring)
        
        layout.addWidget(object_group)
        
        # Tracker Parameters Group
        tracker_group = QGroupBox("Tracker Parameters")
        tracker_layout = QFormLayout(tracker_group)
        
        self.max_unreliable_frames = QSpinBox()
        self.max_unreliable_frames.setRange(1, 50)
        self.max_unreliable_frames.setValue(10)
        tracker_layout.addRow("Max Unreliable Frames:", self.max_unreliable_frames)
        
        self.non_measurement_dynamic = QSpinBox()
        self.non_measurement_dynamic.setRange(1, 50)
        self.non_measurement_dynamic.setValue(8)
        tracker_layout.addRow("Non-measurement Frames (Dynamic):", self.non_measurement_dynamic)
        
        self.non_measurement_static = QSpinBox()
        self.non_measurement_static.setRange(1, 50)
        self.non_measurement_static.setValue(16)
        tracker_layout.addRow("Non-measurement Frames (Static):", self.non_measurement_static)
        
        self.baseline_frame_rate = QSpinBox()
        self.baseline_frame_rate.setRange(1, 120)
        self.baseline_frame_rate.setValue(30)
        tracker_layout.addRow("Baseline Frame Rate:", self.baseline_frame_rate)
        
        layout.addWidget(tracker_group)
        
        layout.addStretch()
        return widget
    
    def create_camera_tab(self) -> QWidget:
        """Create camera settings tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Camera Settings Group
        camera_group = QGroupBox("Camera Configuration")
        camera_layout = QFormLayout(camera_group)
        
        self.default_resolution = QComboBox()
        self.default_resolution.addItems(["640x480", "1280x720", "1920x1080", "3840x2160"])
        self.default_resolution.setCurrentText("1920x1080")
        camera_layout.addRow("Default Resolution:", self.default_resolution)
        
        self.default_fps = QSpinBox()
        self.default_fps.setRange(1, 120)
        self.default_fps.setValue(30)
        camera_layout.addRow("Default FPS:", self.default_fps)
        
        self.max_cameras = QSpinBox()
        self.max_cameras.setRange(1, 16)
        self.max_cameras.setValue(4)
        camera_layout.addRow("Maximum Cameras:", self.max_cameras)
        
        self.auto_calibration = QCheckBox()
        self.auto_calibration.setChecked(False)
        camera_layout.addRow("Auto Calibration:", self.auto_calibration)
        
        layout.addWidget(camera_group)
        
        # Camera List (placeholder for future multi-camera UI)
        camera_list_group = QGroupBox("Camera Sources")
        camera_list_layout = QVBoxLayout(camera_list_group)
        
        self.camera_list_text = QTextEdit()
        self.camera_list_text.setPlainText("Camera 1: Default\n# Add more camera sources here")
        self.camera_list_text.setMaximumHeight(100)
        camera_list_layout.addWidget(self.camera_list_text)
        
        layout.addWidget(camera_list_group)
        
        layout.addStretch()
        return widget
    
    def create_performance_tab(self) -> QWidget:
        """Create performance settings tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # GPU Settings Group
        gpu_group = QGroupBox("GPU & Performance")
        gpu_layout = QFormLayout(gpu_group)
        
        self.gpu_device = QComboBox()
        self.gpu_device.addItems(["AUTO", "GPU", "CPU"])
        self.gpu_device.setCurrentText("AUTO")
        gpu_layout.addRow("GPU Device:", self.gpu_device)
        
        self.inference_threads = QSpinBox()
        self.inference_threads.setRange(1, 16)
        self.inference_threads.setValue(4)
        gpu_layout.addRow("Inference Threads:", self.inference_threads)
        
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 8)
        self.batch_size.setValue(1)
        gpu_layout.addRow("Batch Size:", self.batch_size)
        
        self.memory_optimization = QCheckBox()
        self.memory_optimization.setChecked(True)
        gpu_layout.addRow("Memory Optimization:", self.memory_optimization)
        
        layout.addWidget(gpu_group)
        
        # Performance Limits Group
        limits_group = QGroupBox("Performance Limits")
        limits_layout = QFormLayout(limits_group)
        
        self.max_concurrent_cameras = QSpinBox()
        self.max_concurrent_cameras.setRange(1, 8)
        self.max_concurrent_cameras.setValue(4)
        limits_layout.addRow("Max Concurrent Cameras:", self.max_concurrent_cameras)
        
        self.target_fps = QSpinBox()
        self.target_fps.setRange(1, 120)
        self.target_fps.setValue(30)
        limits_layout.addRow("Target FPS:", self.target_fps)
        
        self.gpu_memory_limit = QSpinBox()
        self.gpu_memory_limit.setRange(512, 16384)
        self.gpu_memory_limit.setValue(2048)
        self.gpu_memory_limit.setSuffix(" MB")
        limits_layout.addRow("GPU Memory Limit:", self.gpu_memory_limit)
        
        layout.addWidget(limits_group)
        
        layout.addStretch()
        return widget
    
    def create_roi_tab(self) -> QWidget:
        """Create ROI settings tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # UI Settings Group
        ui_group = QGroupBox("Display Settings")
        ui_layout = QFormLayout(ui_group)
        
        self.show_roi_overlay = QCheckBox()
        self.show_roi_overlay.setChecked(True)
        ui_layout.addRow("Show ROI Overlay:", self.show_roi_overlay)
        
        self.show_tracking_lines = QCheckBox()
        self.show_tracking_lines.setChecked(True)
        ui_layout.addRow("Show Tracking Lines:", self.show_tracking_lines)
        
        self.show_analytics_overlay = QCheckBox()
        self.show_analytics_overlay.setChecked(True)
        ui_layout.addRow("Show Analytics Overlay:", self.show_analytics_overlay)
        
        self.auto_refresh_rate = QSpinBox()
        self.auto_refresh_rate.setRange(1, 120)
        self.auto_refresh_rate.setValue(30)
        ui_layout.addRow("Auto Refresh Rate:", self.auto_refresh_rate)
        
        layout.addWidget(ui_group)
        
        # Alert Settings Group
        alert_group = QGroupBox("Alert Settings")
        alert_layout = QFormLayout(alert_group)
        
        self.enable_alerts = QCheckBox()
        self.enable_alerts.setChecked(True)
        alert_layout.addRow("Enable Alerts:", self.enable_alerts)
        
        self.pedestrian_safety_alerts = QCheckBox()
        self.pedestrian_safety_alerts.setChecked(True)
        alert_layout.addRow("Pedestrian Safety Alerts:", self.pedestrian_safety_alerts)
        
        self.traffic_anomaly_alerts = QCheckBox()
        self.traffic_anomaly_alerts.setChecked(True)
        alert_layout.addRow("Traffic Anomaly Alerts:", self.traffic_anomaly_alerts)
        
        self.violation_alerts = QCheckBox()
        self.violation_alerts.setChecked(True)
        alert_layout.addRow("Violation Alerts:", self.violation_alerts)
        
        layout.addWidget(alert_group)
        
        # ROI Definition (placeholder)
        roi_group = QGroupBox("Regions of Interest")
        roi_layout = QVBoxLayout(roi_group)
        
        roi_info = QLabel("ROI definition interface will be available in the Analytics tab.\nHere you can view current ROI settings.")
        roi_info.setWordWrap(True)
        roi_layout.addWidget(roi_info)
        
        self.roi_summary = QTextEdit()
        self.roi_summary.setPlainText("No ROI regions defined yet.")
        self.roi_summary.setMaximumHeight(80)
        roi_layout.addWidget(self.roi_summary)
        
        layout.addWidget(roi_group)
        
        layout.addStretch()
        return widget
    
    def load_default_config(self) -> dict:
        """Load default configuration"""
        return {
            "desktop_app_config": {
                "scene_analytics": {
                    "enable_multi_camera": True,
                    "enable_roi_analytics": True,
                    "enable_vlm_integration": True
                },
                "camera_settings": {
                    "default_resolution": "1920x1080",
                    "default_fps": 30,
                    "max_cameras": 4,
                    "auto_calibration": False
                },
                "analytics_settings": {
                    "object_tracking": True,
                    "speed_estimation": True,
                    "direction_analysis": True,
                    "dwell_time_analysis": True,
                    "safety_monitoring": True
                },
                "performance_settings": {
                    "gpu_device": "AUTO",
                    "inference_threads": 4,
                    "batch_size": 1,
                    "memory_optimization": True
                },
                "ui_settings": {
                    "show_roi_overlay": True,
                    "show_tracking_lines": True,
                    "show_analytics_overlay": True,
                    "auto_refresh_rate": 30
                }
            },
            "tracker_config": {
                "max_unreliable_frames": 10,
                "non_measurement_frames_dynamic": 8,
                "non_measurement_frames_static": 16,
                "baseline_frame_rate": 30
            },
            "alert_settings": {
                "enable_alerts": True,
                "pedestrian_safety_alerts": True,
                "traffic_anomaly_alerts": True,
                "violation_alerts": True
            }
        }
    
    def load_config(self):
        """Load configuration from file"""
        try:
            desktop_config_file = self.config_path / "desktop-config.json"
            tracker_config_file = self.config_path / "tracker-config.json"
            
            if desktop_config_file.exists():
                with open(desktop_config_file, 'r') as f:
                    config = json.load(f)
                self.current_config.update(config)
            
            if tracker_config_file.exists():
                with open(tracker_config_file, 'r') as f:
                    tracker_config = json.load(f)
                self.current_config["tracker_config"] = tracker_config
            
            self.update_ui_from_config()
            self.status_label.setText("Configuration loaded successfully")
            
        except Exception as e:
            self.status_label.setText(f"Error loading config: {e}")
    
    def update_ui_from_config(self):
        """Update UI controls from current configuration"""
        try:
            config = self.current_config
            
            # Scene analytics
            scene = config.get("desktop_app_config", {}).get("scene_analytics", {})
            self.enable_multi_camera.setChecked(scene.get("enable_multi_camera", True))
            self.enable_roi_analytics.setChecked(scene.get("enable_roi_analytics", True))
            self.enable_vlm_integration.setChecked(scene.get("enable_vlm_integration", True))
            
            # Analytics settings
            analytics = config.get("desktop_app_config", {}).get("analytics_settings", {})
            self.object_tracking.setChecked(analytics.get("object_tracking", True))
            self.speed_estimation.setChecked(analytics.get("speed_estimation", True))
            self.direction_analysis.setChecked(analytics.get("direction_analysis", True))
            self.dwell_time_analysis.setChecked(analytics.get("dwell_time_analysis", True))
            self.safety_monitoring.setChecked(analytics.get("safety_monitoring", True))
            
            # Tracker parameters
            tracker = config.get("tracker_config", {})
            self.max_unreliable_frames.setValue(tracker.get("max_unreliable_frames", 10))
            self.non_measurement_dynamic.setValue(tracker.get("non_measurement_frames_dynamic", 8))
            self.non_measurement_static.setValue(tracker.get("non_measurement_frames_static", 16))
            self.baseline_frame_rate.setValue(tracker.get("baseline_frame_rate", 30))
            
            # Camera settings
            camera = config.get("desktop_app_config", {}).get("camera_settings", {})
            self.default_resolution.setCurrentText(camera.get("default_resolution", "1920x1080"))
            self.default_fps.setValue(camera.get("default_fps", 30))
            self.max_cameras.setValue(camera.get("max_cameras", 4))
            self.auto_calibration.setChecked(camera.get("auto_calibration", False))
            
            # Performance settings
            performance = config.get("desktop_app_config", {}).get("performance_settings", {})
            self.gpu_device.setCurrentText(performance.get("gpu_device", "AUTO"))
            self.inference_threads.setValue(performance.get("inference_threads", 4))
            self.batch_size.setValue(performance.get("batch_size", 1))
            self.memory_optimization.setChecked(performance.get("memory_optimization", True))
            
            # UI settings
            ui = config.get("desktop_app_config", {}).get("ui_settings", {})
            self.show_roi_overlay.setChecked(ui.get("show_roi_overlay", True))
            self.show_tracking_lines.setChecked(ui.get("show_tracking_lines", True))
            self.show_analytics_overlay.setChecked(ui.get("show_analytics_overlay", True))
            self.auto_refresh_rate.setValue(ui.get("auto_refresh_rate", 30))
            
            # Alert settings
            alerts = config.get("alert_settings", {})
            self.enable_alerts.setChecked(alerts.get("enable_alerts", True))
            self.pedestrian_safety_alerts.setChecked(alerts.get("pedestrian_safety_alerts", True))
            self.traffic_anomaly_alerts.setChecked(alerts.get("traffic_anomaly_alerts", True))
            self.violation_alerts.setChecked(alerts.get("violation_alerts", True))
            
        except Exception as e:
            print(f"Error updating UI from config: {e}")
    
    def get_config_from_ui(self) -> dict:
        """Get configuration from UI controls"""
        config = {
            "desktop_app_config": {
                "scene_analytics": {
                    "enable_multi_camera": self.enable_multi_camera.isChecked(),
                    "enable_roi_analytics": self.enable_roi_analytics.isChecked(),
                    "enable_vlm_integration": self.enable_vlm_integration.isChecked()
                },
                "camera_settings": {
                    "default_resolution": self.default_resolution.currentText(),
                    "default_fps": self.default_fps.value(),
                    "max_cameras": self.max_cameras.value(),
                    "auto_calibration": self.auto_calibration.isChecked()
                },
                "analytics_settings": {
                    "object_tracking": self.object_tracking.isChecked(),
                    "speed_estimation": self.speed_estimation.isChecked(),
                    "direction_analysis": self.direction_analysis.isChecked(),
                    "dwell_time_analysis": self.dwell_time_analysis.isChecked(),
                    "safety_monitoring": self.safety_monitoring.isChecked()
                },
                "performance_settings": {
                    "gpu_device": self.gpu_device.currentText(),
                    "inference_threads": self.inference_threads.value(),
                    "batch_size": self.batch_size.value(),
                    "memory_optimization": self.memory_optimization.isChecked()
                },
                "ui_settings": {
                    "show_roi_overlay": self.show_roi_overlay.isChecked(),
                    "show_tracking_lines": self.show_tracking_lines.isChecked(),
                    "show_analytics_overlay": self.show_analytics_overlay.isChecked(),
                    "auto_refresh_rate": self.auto_refresh_rate.value()
                }
            },
            "tracker_config": {
                "max_unreliable_frames": self.max_unreliable_frames.value(),
                "non_measurement_frames_dynamic": self.non_measurement_dynamic.value(),
                "non_measurement_frames_static": self.non_measurement_static.value(),
                "baseline_frame_rate": self.baseline_frame_rate.value()
            },
            "alert_settings": {
                "enable_alerts": self.enable_alerts.isChecked(),
                "pedestrian_safety_alerts": self.pedestrian_safety_alerts.isChecked(),
                "traffic_anomaly_alerts": self.traffic_anomaly_alerts.isChecked(),
                "violation_alerts": self.violation_alerts.isChecked()
            }
        }
        return config
    
    def apply_settings(self):
        """Apply current settings"""
        try:
            config = self.get_config_from_ui()
            self.current_config = config
            self.apply_requested.emit(config)
            self.status_label.setText("Settings applied successfully")
            
        except Exception as e:
            self.status_label.setText(f"Error applying settings: {e}")
            QMessageBox.warning(self, "Apply Error", f"Error applying settings: {e}")
    
    def save_config_file(self):
        """Save configuration to file"""
        try:
            config = self.get_config_from_ui()
            
            # Ensure directory exists
            self.config_path.mkdir(parents=True, exist_ok=True)
            
            # Save desktop config
            desktop_config_file = self.config_path / "desktop-config.json"
            with open(desktop_config_file, 'w') as f:
                json.dump({
                    "desktop_app_config": config["desktop_app_config"],
                    "alert_settings": config["alert_settings"]
                }, f, indent=2)
            
            # Save tracker config
            tracker_config_file = self.config_path / "tracker-config.json"
            with open(tracker_config_file, 'w') as f:
                json.dump(config["tracker_config"], f, indent=2)
            
            self.status_label.setText("Configuration saved successfully")
            
        except Exception as e:
            self.status_label.setText(f"Error saving config: {e}")
            QMessageBox.warning(self, "Save Error", f"Error saving configuration: {e}")
    
    def load_config_file(self):
        """Load configuration from file dialog"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Load Configuration", str(self.config_path), "JSON Files (*.json)"
            )
            
            if file_path:
                with open(file_path, 'r') as f:
                    config = json.load(f)
                
                self.current_config.update(config)
                self.update_ui_from_config()
                self.status_label.setText(f"Loaded configuration from {os.path.basename(file_path)}")
                
        except Exception as e:
            self.status_label.setText(f"Error loading config: {e}")
            QMessageBox.warning(self, "Load Error", f"Error loading configuration: {e}")
    
    def reset_to_defaults(self):
        """Reset to default configuration"""
        reply = QMessageBox.question(
            self, "Reset Configuration", 
            "Are you sure you want to reset all settings to defaults?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.current_config = self.load_default_config()
            self.update_ui_from_config()
            self.status_label.setText("Reset to default configuration")
