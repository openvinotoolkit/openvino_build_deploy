from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, 
    QSlider, QCheckBox, QPushButton, QGroupBox, QFormLayout,
    QSpinBox, QDoubleSpinBox, QTabWidget, QLineEdit, QFileDialog,
    QSpacerItem, QSizePolicy, QScrollArea
)
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QFont

# Import VLM insights widget
from ui.vlm_insights_widget import VLMInsightsWidget

class ConfigPanel(QWidget):
    """Side panel for application configuration."""
    
    config_changed = Signal(dict)  # Emitted when configuration changes are applied
    theme_toggled = Signal(bool)   # Emitted when theme toggle button is clicked (True = dark)
    device_switch_requested = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.setObjectName("ConfigPanel")
        self.setStyleSheet(self._panel_qss())
        self.initUI()
        self.dark_theme = True  # Start with dark theme
        
    def _panel_qss(self):
        return """
        #ConfigPanel {
            background: #181C20;
            border-top-left-radius: 18px;
            border-bottom-left-radius: 18px;
            border: none;
        }
        QTabWidget::pane {
            border-radius: 12px;
            background: #232323;
        }
        QTabBar::tab {
            background: #232323;
            color: #bbb;
            border-radius: 10px 10px 0 0;
            padding: 8px 18px;
            font-size: 15px;
        }
        QTabBar::tab:selected {
            background: #03DAC5;
            color: #181C20;
        }
        QGroupBox {
            border: 1px solid #30343A;
            border-radius: 12px;
            margin-top: 16px;
            background: #232323;
            font-weight: bold;
            color: #fff;
            font-size: 15px;
        }
        QGroupBox:title {
            subcontrol-origin: margin;
            left: 12px;
            top: 8px;
            padding: 0 4px;
            background: transparent;
        }
        QLabel, QCheckBox, QRadioButton {
            color: #eee;
            font-size: 14px;
        }
        QLineEdit, QSpinBox, QDoubleSpinBox {
            background: #181C20;
            border: 1.5px solid #30343A;
            border-radius: 8px;
            color: #fff;
            padding: 6px 10px;
            font-size: 14px;
        }
        QSlider::groove:horizontal {
            height: 8px;
            background: #30343A;
            border-radius: 4px;
        }
        QSlider::handle:horizontal {
            background: #03DAC5;
            border-radius: 10px;
            width: 20px;
        }
        QPushButton {
            background: #03DAC5;
            color: #181C20;
            border-radius: 10px;
            font-size: 15px;
            font-weight: 600;
            padding: 8px 18px;
            border: none;
        }
        QPushButton:hover {
            background: #018786;
            color: #fff;
        }
        QPushButton:pressed {
            background: #03DAC5;
            color: #232323;
        }
        QCheckBox::indicator {
            border-radius: 6px;
            width: 18px;
            height: 18px;
        }
        QCheckBox::indicator:checked {
            background: #03DAC5;
            border: 1.5px solid #018786;
        }
        QCheckBox::indicator:unchecked {
            background: #232323;
            border: 1.5px solid #30343A;
        }
        """

    def initUI(self):
        """Initialize UI components"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(10)
        
        # Create tab widget for better organization
        tabs = QTabWidget()
        tabs.setStyleSheet("")  # Use panel QSS
        
        # Detection tab
        detection_tab = QWidget()
        detection_layout = QVBoxLayout(detection_tab)
        
        # Device selection
        device_group = QGroupBox("Inference Device")
        device_layout = QVBoxLayout(device_group)
        
        self.device_combo = QComboBox()
        self.device_combo.addItems(["AUTO", "CPU", "GPU", "MYRIAD", "VPU"])
        device_layout.addWidget(self.device_combo)
        
        detection_layout.addWidget(device_group)
        
        # Detection settings
        detection_group = QGroupBox("Detection Settings")
        detection_form = QFormLayout(detection_group)
        
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(10, 100)
        self.conf_slider.setValue(50)
        self.conf_slider.setTracking(True)
        self.conf_slider.valueChanged.connect(self.update_conf_label)
        
        self.conf_label = QLabel("50%")
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(self.conf_slider)
        conf_layout.addWidget(self.conf_label)
        
        self.tracking_checkbox = QCheckBox("Enable")
        self.tracking_checkbox.setChecked(True)
        
        model_layout = QHBoxLayout()
        self.model_path = QLineEdit()
        self.model_path.setReadOnly(True)
        self.model_path.setPlaceholderText("Auto-detected")
        
        self.browse_btn = QPushButton("...")
        self.browse_btn.setMaximumWidth(30)
        self.browse_btn.clicked.connect(self.browse_model)
        
        model_layout.addWidget(self.model_path)
        model_layout.addWidget(self.browse_btn)
        
        detection_form.addRow("Confidence Threshold:", conf_layout)
        detection_form.addRow("Object Tracking:", self.tracking_checkbox)
        detection_form.addRow("Model Path:", model_layout)
        detection_layout.addWidget(detection_group)
        # Add quick switch buttons for YOLO11n/YOLO11x
        quick_switch_layout = QHBoxLayout()
        self.cpu_switch_btn = QPushButton("Switch to CPU (YOLO11n)")
        self.gpu_switch_btn = QPushButton("Switch to GPU (YOLO11x)")
        self.cpu_switch_btn.clicked.connect(lambda: self.quick_switch_device("CPU"))
        self.gpu_switch_btn.clicked.connect(lambda: self.quick_switch_device("GPU"))
        quick_switch_layout.addWidget(self.cpu_switch_btn)
        quick_switch_layout.addWidget(self.gpu_switch_btn)
        detection_layout.addLayout(quick_switch_layout)
        # --- Current Model Info Section (PREMIUM FORMAT) ---
        model_info_group = QGroupBox()
        model_info_group.setTitle("")
        model_info_group.setStyleSheet("""
            QGroupBox {
                border: 1.5px solid #03DAC5;
                border-radius: 12px;
                margin-top: 16px;
                background: #181C20;
                font-weight: bold;
                color: #03DAC5;
                font-size: 16px;
            }
        """)
        model_info_layout = QVBoxLayout(model_info_group)
        model_info_layout.setContentsMargins(16, 10, 16, 10)
        # Title
        title = QLabel("Current Model")
        title.setStyleSheet("font-size: 17px; font-weight: bold; color: #03DAC5; margin-bottom: 8px;")
        model_info_layout.addWidget(title)
        # Info rows
        row_style = "font-size: 15px; color: #fff; font-family: 'Consolas', 'SF Mono', 'monospace'; padding: 2px 0;"
        row_widget = QWidget()
        row_layout = QVBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(2)
        # Model
        model_row = QHBoxLayout()
        model_label = QLabel("Model:")
        model_label.setStyleSheet(row_style + "font-weight: 600; color: #80cbc4;")
        self.current_model_label = QLabel("-")
        self.current_model_label.setStyleSheet(row_style)
        model_row.addWidget(model_label)
        model_row.addWidget(self.current_model_label, 1)
        row_layout.addLayout(model_row)
        # Device
        device_row = QHBoxLayout()
        device_label = QLabel("Device:")
        device_label.setStyleSheet(row_style + "font-weight: 600; color: #80cbc4;")
        self.current_device_label = QLabel("-")
        self.current_device_label.setStyleSheet(row_style)
        device_row.addWidget(device_label)
        device_row.addWidget(self.current_device_label, 1)
        row_layout.addLayout(device_row)
        # Recommended For
        rec_row = QHBoxLayout()
        rec_label = QLabel("Recommended For:")
        rec_label.setStyleSheet(row_style + "font-weight: 600; color: #80cbc4;")
        self.model_recommendation_label = QLabel("")
        self.model_recommendation_label.setStyleSheet(row_style)
        rec_row.addWidget(rec_label)
        rec_row.addWidget(self.model_recommendation_label, 1)
        row_layout.addLayout(rec_row)
        model_info_layout.addWidget(row_widget)
        model_info_layout.addStretch(1)
        detection_layout.addWidget(model_info_group)
        
        # --- OpenVINO Devices Info Section ---
        devices_info_group = QGroupBox()
        devices_info_group.setTitle("")
        devices_info_group.setStyleSheet("""
            QGroupBox {
                border: 1.5px solid #80cbc4;
                border-radius: 12px;
                margin-top: 16px;
                background: #181C20;
                font-weight: bold;
                color: #80cbc4;
                font-size: 16px;
            }
        """)
        devices_info_layout = QVBoxLayout(devices_info_group)
        devices_info_layout.setContentsMargins(16, 10, 16, 10)
        devices_title = QLabel("Available OpenVINO Devices")
        devices_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #80cbc4; margin-bottom: 8px;")
        devices_info_layout.addWidget(devices_title)
        self.devices_info_text = QLabel("Yolov11n and Yolov11x models are optimized for CPU and GPU respectively.<br>")
        self.devices_info_text.setStyleSheet("font-size: 14px; color: #fff; font-family: 'Consolas', 'SF Mono', 'monospace';")
        self.devices_info_text.setWordWrap(True)
        self.devices_info_text.setTextFormat(Qt.RichText)
        self.devices_info_text.setObjectName("devices_info_text")
        devices_info_layout.addWidget(self.devices_info_text)
        devices_info_layout.addStretch(1)
        detection_layout.addWidget(devices_info_group)

        display_tab = QWidget()
        display_layout = QVBoxLayout(display_tab)
        
        # Display options
        display_group = QGroupBox("Display Options")
        display_form = QFormLayout(display_group)
        
        self.labels_checkbox = QCheckBox()
        self.labels_checkbox.setChecked(True)
        
        self.confidence_checkbox = QCheckBox()
        self.confidence_checkbox.setChecked(True)
        
        self.perf_checkbox = QCheckBox()
        self.perf_checkbox.setChecked(True)
        
        self.max_width = QSpinBox()
        self.max_width.setRange(320, 4096)
        self.max_width.setValue(800)
        self.max_width.setSingleStep(10)
        self.max_width.setSuffix(" px")
        
        display_form.addRow("Show Labels:", self.labels_checkbox)
        display_form.addRow("Show Confidence:", self.confidence_checkbox)
        display_form.addRow("Show Performance:", self.perf_checkbox)
        display_form.addRow("Max Display Width:", self.max_width)
        
        display_layout.addWidget(display_group)
        
        # Analytics Group
        analytics_group = QGroupBox("Analytics Settings")
        analytics_form = QFormLayout(analytics_group)
        
        self.charts_checkbox = QCheckBox()
        self.charts_checkbox.setChecked(True)
        
        self.history_spinbox = QSpinBox()
        self.history_spinbox.setRange(10, 10000)
        self.history_spinbox.setValue(1000)
        self.history_spinbox.setSingleStep(100)
        self.history_spinbox.setSuffix(" frames")
        
        analytics_form.addRow("Enable Live Charts:", self.charts_checkbox)
        analytics_form.addRow("History Length:", self.history_spinbox)
        
        display_layout.addWidget(analytics_group)
        
        # Violation tab
        violation_tab = QWidget()
        violation_layout = QVBoxLayout(violation_tab)
        
        # Violation settings
        violation_group = QGroupBox("Violation Detection")
        violation_form = QFormLayout(violation_group)
        
        self.red_light_grace = QDoubleSpinBox()
        self.red_light_grace.setRange(0.1, 5.0)
        self.red_light_grace.setValue(2.0)
        self.red_light_grace.setSingleStep(0.1)
        self.red_light_grace.setSuffix(" sec")
        
        self.stop_sign_duration = QDoubleSpinBox()
        self.stop_sign_duration.setRange(0.5, 5.0)
        self.stop_sign_duration.setValue(2.0)
        self.stop_sign_duration.setSingleStep(0.1)
        self.stop_sign_duration.setSuffix(" sec")
        
        self.speed_tolerance = QSpinBox()
        self.speed_tolerance.setRange(0, 20)
        self.speed_tolerance.setValue(5)
        self.speed_tolerance.setSingleStep(1)
        self.speed_tolerance.setSuffix(" km/h")
        
        violation_form.addRow("Red Light Grace:", self.red_light_grace)
        violation_form.addRow("Stop Sign Duration:", self.stop_sign_duration)
        violation_form.addRow("Speed Tolerance:", self.speed_tolerance)
        
        self.enable_red_light = QCheckBox("Enabled")
        self.enable_red_light.setChecked(True)
        
        self.enable_stop_sign = QCheckBox("Enabled")
        self.enable_stop_sign.setChecked(True)
        
        self.enable_speed = QCheckBox("Enabled")
        self.enable_speed.setChecked(True)
        
        self.enable_lane = QCheckBox("Enabled")
        self.enable_lane.setChecked(True)
        
        violation_form.addRow("Red Light Detection:", self.enable_red_light)
        violation_form.addRow("Stop Sign Detection:", self.enable_stop_sign)
        violation_form.addRow("Speed Detection:", self.enable_speed)
        violation_form.addRow("Lane Detection:", self.enable_lane)
        
        violation_layout.addWidget(violation_group)
        
        # === VLM Insights Tab ===
        vlm_tab = QWidget()
        vlm_layout = QVBoxLayout(vlm_tab)
        
        # Create scroll area for VLM insights
        vlm_scroll = QScrollArea()
        vlm_scroll.setWidgetResizable(True)
        vlm_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        vlm_scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background: transparent;
            }
        """)
        
        # Add VLM insights widget
        print("[CONFIG PANEL DEBUG] Creating VLM insights widget...")
        self.vlm_insights = VLMInsightsWidget()
        print("[CONFIG PANEL DEBUG] VLM insights widget created successfully")
        vlm_scroll.setWidget(self.vlm_insights)
        vlm_layout.addWidget(vlm_scroll)
        
        # Smart Intersection Tab - Scene Analytics
        smart_intersection_tab = QWidget()
        si_layout = QVBoxLayout(smart_intersection_tab)
        
        # Smart Intersection config widget
        si_scroll = QScrollArea()
        si_scroll.setWidgetResizable(True)
        si_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        si_scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background: transparent;
            }
        """)
        
        try:
            from ui.smart_intersection_config import SmartIntersectionConfigPanel
            self.smart_intersection_config = SmartIntersectionConfigPanel()
            si_scroll.setWidget(self.smart_intersection_config)
            print("[CONFIG PANEL DEBUG] Smart Intersection config panel created successfully")
        except Exception as e:
            print(f"[CONFIG PANEL DEBUG] Error creating Smart Intersection config: {e}")
            self.smart_intersection_config = None
            si_scroll.setWidget(QLabel(f"Smart Intersection config unavailable: {e}"))
        
        si_layout.addWidget(si_scroll)
        
        # Add all tabs
        tabs.addTab(detection_tab, "Detection")
        tabs.addTab(display_tab, "Display")
        tabs.addTab(violation_tab, "Violations")
        tabs.addTab(vlm_tab, "🤖 AI Insights")  # Add VLM insights tab
        tabs.addTab(smart_intersection_tab, "🚦 Smart Intersection")  # Add Smart Intersection tab
        print("[CONFIG PANEL DEBUG] Added AI Insights and Smart Intersection tabs to config panel")
        
        layout.addWidget(tabs)
        
        # Theme toggle
        self.theme_toggle = QPushButton("🌙 Dark Theme")
        self.theme_toggle.setFixedHeight(36)
        self.theme_toggle.setStyleSheet("margin-top: 8px;")
        self.theme_toggle.clicked.connect(self.toggle_theme)
        layout.addWidget(self.theme_toggle)
        
        # Spacer to push buttons to bottom
        layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        
        # Control buttons (fixed at bottom)
        btns = QHBoxLayout()
        self.apply_btn = QPushButton("Apply")
        self.apply_btn.setFixedHeight(32)
        self.apply_btn.clicked.connect(self.apply_config)
        
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.setFixedHeight(32)
        self.reset_btn.clicked.connect(self.reset_config)
        
        btns.addWidget(self.apply_btn)
        btns.addWidget(self.reset_btn)
        layout.addLayout(btns)
        
        layout.addStretch(1)  # Push everything to the top
        
        # Set tooltips for major controls
        self.device_combo.setToolTip("Select inference device (CPU, GPU, etc.)")
        self.cpu_switch_btn.setToolTip("Switch to CPU-optimized YOLO11n model")
        self.gpu_switch_btn.setToolTip("Switch to GPU-optimized YOLO11x model")
        self.conf_slider.setToolTip("Set detection confidence threshold")
        self.tracking_checkbox.setToolTip("Enable or disable object tracking")
        self.model_path.setToolTip("Path to the detection model")
        self.browse_btn.setToolTip("Browse for a model file")
        self.labels_checkbox.setToolTip("Show/hide detection labels on video")
        self.confidence_checkbox.setToolTip("Show/hide confidence scores on video")
        self.perf_checkbox.setToolTip("Show/hide performance overlay")
        self.max_width.setToolTip("Maximum display width for video")
        self.charts_checkbox.setToolTip("Enable/disable live analytics charts")
        self.history_spinbox.setToolTip("Number of frames to keep in analytics history")
        self.red_light_grace.setToolTip("Grace period for red light violation (seconds)")
        self.stop_sign_duration.setToolTip("Stop sign violation duration (seconds)")
        self.speed_tolerance.setToolTip("Speed tolerance for speed violation (km/h)")
        self.enable_red_light.setToolTip("Enable/disable red light violation detection")
        self.enable_stop_sign.setToolTip("Enable/disable stop sign violation detection")
        self.enable_speed.setToolTip("Enable/disable speed violation detection")
        self.enable_lane.setToolTip("Enable/disable lane violation detection")
        self.theme_toggle.setToolTip("Toggle between dark and light theme")
        self.apply_btn.setToolTip("Apply all changes")
        self.reset_btn.setToolTip("Reset all settings to default")
    
    @Slot(int)
    def update_conf_label(self, value):
        """Update confidence threshold label"""
        self.conf_label.setText(f"{value}%")
        
    @Slot()
    def browse_model(self):
        """Browse for model file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Model File",
            "",
            "OpenVINO Models (*.xml);;PyTorch Models (*.pt);;All Files (*)"
        )
        
        if file_path:
            self.model_path.setText(file_path)
    
    @Slot()
    def toggle_theme(self):
        """Toggle between light and dark theme"""
        self.dark_theme = not self.dark_theme
        
        if self.dark_theme:
            self.theme_toggle.setText("🌙 Dark Theme")
        else:
            self.theme_toggle.setText("☀️ Light Theme")
            
        self.theme_toggled.emit(self.dark_theme)
    
    @Slot()
    def apply_config(self):
        """Apply configuration changes"""
        config = self.get_config()
        self.config_changed.emit(config)
    
    @Slot()
    def reset_config(self):
        """Reset configuration to defaults"""
        self.device_combo.setCurrentText("AUTO")
        self.conf_slider.setValue(50)
        self.tracking_checkbox.setChecked(True)
        self.labels_checkbox.setChecked(True)
        self.confidence_checkbox.setChecked(True)
        self.perf_checkbox.setChecked(True)
        self.max_width.setValue(800)
        self.red_light_grace.setValue(2.0)
        self.stop_sign_duration.setValue(2.0)
        self.speed_tolerance.setValue(5)
        self.enable_red_light.setChecked(True)
        self.enable_stop_sign.setChecked(True)
        self.enable_speed.setChecked(True)
        self.enable_lane.setChecked(True)
        self.model_path.setText("")
        
        self.apply_config()
    
    def quick_switch_device(self, device: str):
        index = self.device_combo.findText(device)
        if index >= 0:
            self.device_combo.setCurrentIndex(index)
            self.device_switch_requested.emit(device)
            self.apply_config()
    
    def update_model_info(self, model_info: dict):
        if not model_info:
            self.current_model_label.setText("No model loaded")
            self.current_device_label.setText("None")
            self.model_recommendation_label.setText("None")
            return
        model_name = model_info.get("model_name", "Unknown")
        device = model_info.get("device", "Unknown")
        recommended_for = model_info.get("recommended_for", "Unknown")
        self.current_model_label.setText(model_name)
        self.current_device_label.setText(device)
        self.model_recommendation_label.setText(recommended_for)
        if device == "CPU":
            self.cpu_switch_btn.setEnabled(False)
            self.cpu_switch_btn.setText("✓ CPU Active (YOLO11n)")
            self.gpu_switch_btn.setEnabled(True)
            self.gpu_switch_btn.setText("Switch to GPU (YOLO11x)")
        elif device == "GPU":
            self.cpu_switch_btn.setEnabled(True)
            self.cpu_switch_btn.setText("Switch to CPU (YOLO11n)")
            self.gpu_switch_btn.setEnabled(False)
            self.gpu_switch_btn.setText("✓ GPU Active (YOLO11x)")
        else:
            self.cpu_switch_btn.setEnabled(True)
            self.cpu_switch_btn.setText("Switch to CPU (YOLO11n)")
            self.gpu_switch_btn.setEnabled(True)
            self.gpu_switch_btn.setText("Switch to GPU (YOLO11x)")
    
    @Slot(object, object)
    def update_live_stats(self, fps, inference_time):
        """Update FPS and inference time labels in the settings panel."""
        if fps is not None:
            self.fps_label.setText(f"FPS: {fps:.1f}")
        else:
            self.fps_label.setText("FPS: --")
        if inference_time is not None:
            self.infer_label.setText(f"Inference: {inference_time:.1f} ms")
        else:
            self.infer_label.setText("Inference: -- ms")
    
    @Slot(object, object)
    def set_video_stats(self, stats):
        """Update FPS and inference time labels in the settings panel from stats dict."""
        fps = stats.get('fps', None)
        inference_time = None
        if 'detection_time_ms' in stats:
            inference_time = float(stats['detection_time_ms'])
        elif 'detection_time' in stats:
            inference_time = float(stats['detection_time'])
        self.update_live_stats(fps, inference_time)
    
    def get_config(self):
        """
        Get current configuration from UI.
        
        Returns:
            Configuration dictionary
        """
        return {
            'detection': {
                'device': self.device_combo.currentText(),
                'confidence_threshold': self.conf_slider.value() / 100.0,
                'enable_tracking': self.tracking_checkbox.isChecked(),
                'model_path': self.model_path.text() if self.model_path.text() else None
            },
            'display': {
                'show_labels': self.labels_checkbox.isChecked(),
                'show_confidence': self.confidence_checkbox.isChecked(),
                'show_performance': self.perf_checkbox.isChecked(),
                'max_display_width': self.max_width.value()
            },
            'violations': {
                'red_light_grace_period': self.red_light_grace.value(),
                'stop_sign_duration': self.stop_sign_duration.value(),
                'speed_tolerance': self.speed_tolerance.value(),
                'enable_red_light': self.enable_red_light.isChecked(),
                'enable_stop_sign': self.enable_stop_sign.isChecked(),
                'enable_speed': self.enable_speed.isChecked(),
                'enable_lane': self.enable_lane.isChecked()
            },
            'analytics': {
                'enable_charts': self.charts_checkbox.isChecked(),
                'history_length': self.history_spinbox.value()
            }
        }
    
    def set_config(self, config):
        """
        Set configuration in UI.
        
        Args:
            config: Configuration dictionary
        """
        if not config:
            return
            
        # Detection settings
        detection = config.get('detection', {})
        if 'device' in detection:
            index = self.device_combo.findText(detection['device'])
            if index >= 0:
                self.device_combo.setCurrentIndex(index)
                
        if 'confidence_threshold' in detection:
            self.conf_slider.setValue(int(detection['confidence_threshold'] * 100))
            
        if 'enable_tracking' in detection:
            self.tracking_checkbox.setChecked(detection['enable_tracking'])
            
        if 'model_path' in detection and detection['model_path']:
            self.model_path.setText(detection['model_path'])
        
        # Display settings
        display = config.get('display', {})
        if 'show_labels' in display:
            self.labels_checkbox.setChecked(display['show_labels'])
            
        if 'show_confidence' in display:
            self.confidence_checkbox.setChecked(display['show_confidence'])
            
        if 'show_performance' in display:
            self.perf_checkbox.setChecked(display['show_performance'])
            
        if 'max_display_width' in display:
            self.max_width.setValue(display['max_display_width'])
        
        # Violation settings
        violations = config.get('violations', {})
        if 'red_light_grace_period' in violations:
            self.red_light_grace.setValue(violations['red_light_grace_period'])
            
        if 'stop_sign_duration' in violations:
            self.stop_sign_duration.setValue(violations['stop_sign_duration'])
            
        if 'speed_tolerance' in violations:
            self.speed_tolerance.setValue(violations['speed_tolerance'])
            
        if 'enable_red_light' in violations:
            self.enable_red_light.setChecked(violations['enable_red_light'])
            
        if 'enable_stop_sign' in violations:
            self.enable_stop_sign.setChecked(violations['enable_stop_sign'])
            
        if 'enable_speed' in violations:
            self.enable_speed.setChecked(violations['enable_speed'])
            
        if 'enable_lane' in violations:
            self.enable_lane.setChecked(violations['enable_lane'])
        
        # Analytics settings
        analytics = config.get('analytics', {})
        if 'enable_charts' in analytics:
            self.charts_checkbox.setChecked(analytics['enable_charts'])
            
        if 'history_length' in analytics:
            self.history_spinbox.setValue(analytics['history_length'])

    @Slot(object)
    def update_devices_info(self, device_info: dict):
        """
        Update the OpenVINO devices info section with the given device info dict.
        """
        print(f"[UI] update_devices_info called with: {device_info}", flush=True)  # DEBUG
        if not device_info:
            self.devices_info_text.setText("<span style='color:#ffb300;'>No OpenVINO device info received.<br>Check if OpenVINO is installed and the backend emits device_info_ready.</span>")
            return
        if 'error' in device_info:
            self.devices_info_text.setText(f"<span style='color:#ff5370;'>Error: {device_info['error']}</span>")
            return
        text = ""
        for device, props in device_info.items():
            text += f"<b>{device}</b><br>"
            if isinstance(props, dict) and props:
                for k, v in props.items():
                    text += f"&nbsp;&nbsp;<span style='color:#b2dfdb;'>{k}</span>: <span style='color:#fff'>{v}</span><br>"
            else:
                text += "&nbsp;&nbsp;<span style='color:#888'>No properties</span><br>"
            text += "<br>"
        self.devices_info_text.setText(f"<div style='font-size:13px;'>{text}</div>")
        self.devices_info_text.repaint()  # Force repaint in case of async update
