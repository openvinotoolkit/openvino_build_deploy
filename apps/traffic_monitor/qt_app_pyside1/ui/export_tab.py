from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, 
    QPlainTextEdit, QGroupBox, QLabel, QComboBox, QCheckBox, 
    QTableWidget, QTableWidgetItem, QFormLayout, QLineEdit,
    QDateTimeEdit, QSpinBox, QTabWidget, QStyle
)
from PySide6.QtCore import Qt, Slot, QDateTime
from PySide6.QtGui import QFont

class ConfigSection(QGroupBox):
    """Configuration editor section"""
    
    def __init__(self, title):
        super().__init__(title)
        self.layout = QVBoxLayout(self)
    
class ExportTab(QWidget):
    """Tab for exporting data and managing configuration."""
    
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        """Initialize UI components"""
        main_layout = QVBoxLayout(self)
        
        # Create tab widget for organizing export and config sections
        tab_widget = QTabWidget()
        
        # Tab 1: Export Data
        export_tab = QWidget()
        export_layout = QVBoxLayout(export_tab)
        
        # Export options
        export_options = QGroupBox("Export Options")
        options_layout = QFormLayout(export_options)
        
        self.export_format_combo = QComboBox()
        self.export_format_combo.addItems(["CSV", "JSON", "Excel", "PDF Report"])
        
        self.export_data_combo = QComboBox()
        self.export_data_combo.addItems([
            "All Data", 
            "Detections Only", 
            "Violations Only",
            "Analytics Summary"
        ])
        
        # Time range
        time_layout = QHBoxLayout()
        self.export_range_check = QCheckBox("Time Range:")
        self.export_range_check.setChecked(False)
        
        self.export_start_time = QDateTimeEdit(QDateTime.currentDateTime().addDays(-1))
        self.export_start_time.setEnabled(False)
        self.export_end_time = QDateTimeEdit(QDateTime.currentDateTime())
        self.export_end_time.setEnabled(False)
        
        self.export_range_check.toggled.connect(self.export_start_time.setEnabled)
        self.export_range_check.toggled.connect(self.export_end_time.setEnabled)
        
        time_layout.addWidget(self.export_range_check)
        time_layout.addWidget(self.export_start_time)
        time_layout.addWidget(QLabel("to"))
        time_layout.addWidget(self.export_end_time)
        
        options_layout.addRow("Export Format:", self.export_format_combo)
        options_layout.addRow("Data to Export:", self.export_data_combo)
        options_layout.addRow(time_layout)
        
        # Include options
        include_layout = QHBoxLayout()
        self.include_images_check = QCheckBox("Include Images")
        self.include_images_check.setChecked(True)
        self.include_analytics_check = QCheckBox("Include Analytics")
        self.include_analytics_check.setChecked(True)
        
        include_layout.addWidget(self.include_images_check)
        include_layout.addWidget(self.include_analytics_check)
        options_layout.addRow("Include:", include_layout)
        
        export_layout.addWidget(export_options)
          # Export preview
        preview_box = QGroupBox("Export Preview")
        preview_layout = QVBoxLayout(preview_box)
        self.export_preview = QTableWidget(5, 3)
        self.export_preview.setHorizontalHeaderLabels(["Type", "Count", "Details"])
        self.export_preview.setAlternatingRowColors(True)
        self.export_preview.setEditTriggers(QTableWidget.NoEditTriggers)
        
        # Initialize table items with default values
        self.export_preview.setItem(0, 0, QTableWidgetItem("Vehicles"))
        self.export_preview.setItem(0, 1, QTableWidgetItem("0"))
        self.export_preview.setItem(0, 2, QTableWidgetItem("Cars, trucks, buses"))
        
        self.export_preview.setItem(1, 0, QTableWidgetItem("Pedestrians"))
        self.export_preview.setItem(1, 1, QTableWidgetItem("0"))
        self.export_preview.setItem(1, 2, QTableWidgetItem("People detected"))
        
        self.export_preview.setItem(2, 0, QTableWidgetItem("Red Light Violations"))
        self.export_preview.setItem(2, 1, QTableWidgetItem("0"))
        self.export_preview.setItem(2, 2, QTableWidgetItem("Vehicles running red lights"))
        
        self.export_preview.setItem(3, 0, QTableWidgetItem("Stop Sign Violations"))
        self.export_preview.setItem(3, 1, QTableWidgetItem("0"))
        self.export_preview.setItem(3, 2, QTableWidgetItem("Vehicles ignoring stop signs"))
        
        self.export_preview.setItem(4, 0, QTableWidgetItem("Speed Violations"))
        self.export_preview.setItem(4, 1, QTableWidgetItem("0"))
        self.export_preview.setItem(4, 2, QTableWidgetItem("Vehicles exceeding speed limits"))
        
        preview_layout.addWidget(self.export_preview)
        export_layout.addWidget(preview_box)
        
        # Export buttons
        export_buttons = QHBoxLayout()
        self.export_btn = QPushButton("Export Data")
        self.export_btn.setIcon(self.style().standardIcon(QStyle.SP_DialogSaveButton))
        self.clear_export_btn = QPushButton("Clear Data")
        export_buttons.addWidget(self.export_btn)
        export_buttons.addWidget(self.clear_export_btn)
        export_layout.addLayout(export_buttons)
        
        tab_widget.addTab(export_tab, "Export Data")
        
        # Tab 2: Configuration
        config_tab = QWidget()
        config_layout = QVBoxLayout(config_tab)
        
        # Detection configuration
        detection_config = ConfigSection("Detection Configuration")
        detection_form = QFormLayout()
        
        self.conf_threshold = QSpinBox()
        self.conf_threshold.setRange(1, 100)
        self.conf_threshold.setValue(50)
        self.conf_threshold.setSuffix("%")
        
        self.enable_tracking = QCheckBox()
        self.enable_tracking.setChecked(True)
        
        self.model_path = QLineEdit()
        self.model_path.setPlaceholderText("Path to model file")
        self.browse_model_btn = QPushButton("Browse...")
        model_layout = QHBoxLayout()
        model_layout.addWidget(self.model_path)
        model_layout.addWidget(self.browse_model_btn)
        
        detection_form.addRow("Confidence Threshold:", self.conf_threshold)
        detection_form.addRow("Enable Tracking:", self.enable_tracking)
        detection_form.addRow("Model Path:", model_layout)
        
        detection_config.layout.addLayout(detection_form)
        
        # Violation configuration
        violation_config = ConfigSection("Violation Configuration")
        violation_form = QFormLayout()
        
        self.red_light_grace = QSpinBox()
        self.red_light_grace.setRange(0, 10)
        self.red_light_grace.setValue(2)
        self.red_light_grace.setSuffix(" sec")
        
        self.stop_sign_duration = QSpinBox()
        self.stop_sign_duration.setRange(0, 10)
        self.stop_sign_duration.setValue(2)
        self.stop_sign_duration.setSuffix(" sec")
        
        self.speed_tolerance = QSpinBox()
        self.speed_tolerance.setRange(0, 20)
        self.speed_tolerance.setValue(5)
        self.speed_tolerance.setSuffix(" km/h")
        
        violation_form.addRow("Red Light Grace Period:", self.red_light_grace)
        violation_form.addRow("Stop Sign Duration:", self.stop_sign_duration)
        violation_form.addRow("Speed Tolerance:", self.speed_tolerance)
        
        violation_config.layout.addLayout(violation_form)
        
        # Display configuration
        display_config = ConfigSection("Display Configuration")
        display_form = QFormLayout()
        
        self.show_labels = QCheckBox()
        self.show_labels.setChecked(True)
        
        self.show_confidence = QCheckBox()
        self.show_confidence.setChecked(True)
        
        self.max_display_width = QSpinBox()
        self.max_display_width.setRange(320, 4096)
        self.max_display_width.setValue(800)
        self.max_display_width.setSingleStep(10)
        self.max_display_width.setSuffix(" px")
        
        display_form.addRow("Show Labels:", self.show_labels)
        display_form.addRow("Show Confidence:", self.show_confidence)
        display_form.addRow("Max Display Width:", self.max_display_width)
        
        display_config.layout.addLayout(display_form)
        
        # Add config sections
        config_layout.addWidget(detection_config)
        config_layout.addWidget(violation_config)
        config_layout.addWidget(display_config)
        
        # Config buttons
        config_buttons = QHBoxLayout()
        self.save_config_btn = QPushButton("Save Configuration")
        self.save_config_btn.setIcon(self.style().standardIcon(QStyle.SP_DialogSaveButton))
        self.reload_config_btn = QPushButton("Reload Configuration")
        self.reload_config_btn.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        
        self.reset_btn = QPushButton("Reset Defaults")
        self.reset_btn.setIcon(self.style().standardIcon(QStyle.SP_DialogResetButton))
        
        config_buttons.addWidget(self.save_config_btn)
        config_buttons.addWidget(self.reload_config_btn)
        config_buttons.addWidget(self.reset_btn)
        config_layout.addLayout(config_buttons)
        
        # Raw config editor
        raw_config = QGroupBox("Raw Configuration (JSON)")
        raw_layout = QVBoxLayout(raw_config)
        
        self.config_editor = QPlainTextEdit()
        self.config_editor.setFont(QFont("Consolas", 10))
        raw_layout.addWidget(self.config_editor)
        
        config_layout.addWidget(raw_config)
        
        tab_widget.addTab(config_tab, "Configuration")
        
        main_layout.addWidget(tab_widget)
    
    @Slot()
    def browse_model_path(self):
        """Browse for model file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Model File",
            "",
            "Model Files (*.xml *.bin *.pt *.pth);;All Files (*)"
        )
        
        if file_path:
            self.model_path.setText(file_path)
    
    @Slot(dict)
    def update_export_preview(self, analytics):
        """
        Update export preview with analytics data.
        
        Args:
            analytics: Dictionary of analytics data
        """
        if not analytics:
            return
        
        # Update detection counts
        detection_counts = analytics.get('detection_counts', {})
        vehicle_count = sum([
            detection_counts.get('car', 0),
            detection_counts.get('truck', 0),
            detection_counts.get('bus', 0),
            detection_counts.get('motorcycle', 0)
        ])
        pedestrian_count = detection_counts.get('person', 0)
        
        # Update violation counts
        violation_counts = analytics.get('violation_counts', {})
        red_light_count = violation_counts.get('red_light_violation', 0)
        stop_sign_count = violation_counts.get('stop_sign_violation', 0)
        speed_count = violation_counts.get('speed_violation', 0)
          # Update table - create items if they don't exist
        item_data = [
            (0, "Vehicles", vehicle_count, "Cars, trucks, buses"),
            (1, "Pedestrians", pedestrian_count, "People detected"),
            (2, "Red Light Violations", red_light_count, "Vehicles running red lights"),
            (3, "Stop Sign Violations", stop_sign_count, "Vehicles ignoring stop signs"),
            (4, "Speed Violations", speed_count, "Vehicles exceeding speed limits")
        ]
        
        for row, label, count, details in item_data:
            # Check and create Type column item
            if self.export_preview.item(row, 0) is None:
                self.export_preview.setItem(row, 0, QTableWidgetItem(label))
                
            # Check and create or update Count column item
            if self.export_preview.item(row, 1) is None:
                self.export_preview.setItem(row, 1, QTableWidgetItem(str(count)))
            else:
                self.export_preview.item(row, 1).setText(str(count))
                
            # Check and create Details column item
            if self.export_preview.item(row, 2) is None:
                self.export_preview.setItem(row, 2, QTableWidgetItem(details))
    
    @Slot(dict)
    def update_config_display(self, config):
        """
        Update configuration display.
        
        Args:
            config: Configuration dictionary
        """
        if not config:
            return
            
        # Convert to JSON for display
        import json
        self.config_editor.setPlainText(
            json.dumps(config, indent=2)
        )
        
        # Update form fields
        detection_config = config.get('detection', {})
        self.conf_threshold.setValue(int(detection_config.get('confidence_threshold', 0.5) * 100))
        self.enable_tracking.setChecked(detection_config.get('enable_tracking', True))
        
        if detection_config.get('model_path'):
            self.model_path.setText(detection_config.get('model_path'))
            
        violation_config = config.get('violations', {})
        self.red_light_grace.setValue(violation_config.get('red_light_grace_period', 2))
        self.stop_sign_duration.setValue(violation_config.get('stop_sign_duration', 2))
        self.speed_tolerance.setValue(violation_config.get('speed_tolerance', 5))
        
        display_config = config.get('display', {})
        self.show_labels.setChecked(display_config.get('show_labels', True))
        self.show_confidence.setChecked(display_config.get('show_confidence', True))
        self.max_display_width.setValue(display_config.get('max_display_width', 800))
    
    def get_config_from_ui(self):
        """
        Get configuration from UI fields.
        
        Returns:
            Configuration dictionary
        """
        config = {
            'detection': {
                'confidence_threshold': self.conf_threshold.value() / 100.0,
                'enable_tracking': self.enable_tracking.isChecked(),
                'model_path': self.model_path.text() if self.model_path.text() else None
            },
            'violations': {
                'red_light_grace_period': self.red_light_grace.value(),
                'stop_sign_duration': self.stop_sign_duration.value(),
                'speed_tolerance': self.speed_tolerance.value()
            },
            'display': {
                'max_display_width': self.max_display_width.value(),
                'show_confidence': self.show_confidence.isChecked(),
                'show_labels': self.show_labels.isChecked()
            }
        }
        
        return config
