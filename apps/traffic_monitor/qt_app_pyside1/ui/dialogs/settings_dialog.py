"""
Settings Dialog - Application settings and configuration
"""

from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QTabWidget,
                               QGroupBox, QLabel, QPushButton, QCheckBox,
                               QComboBox, QSpinBox, QSlider, QLineEdit,
                               QTextEdit, QFileDialog, QColorDialog, QFrame,
                               QGridLayout, QFormLayout, QListWidget,
                               QListWidgetItem, QSplitter, QScrollArea, QSpacerItem,
                               QSizePolicy, QWidget, QDialogButtonBox)
from PySide6.QtCore import Qt, Signal, QSettings, QSize
from PySide6.QtGui import QFont, QColor, QIcon, QPalette
import json
import os

class SettingsDialog(QDialog):
    """
    Comprehensive settings dialog for the Smart Intersection Monitoring System
    
    Sections:
    - General: Application preferences
    - Display: UI themes and display options
    - Detection: AI model and detection settings
    - IoT: Device connections and protocols
    - Performance: System optimization settings
    - Security: Authentication and access control
    """
    
    # Signals
    settings_changed = Signal(dict)
    theme_changed = Signal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.settings = QSettings("SmartIntersection", "MonitoringSystem")
        self.pending_changes = {}
        
        self.setWindowTitle("Settings - Smart Intersection Monitoring")
        self.setModal(True)
        self.resize(800, 600)
        
        self._setup_ui()
        self._load_settings()
    
    def _setup_ui(self):
        """Setup settings dialog UI"""
        layout = QVBoxLayout(self)
        
        # Header
        header = self._create_header()
        layout.addWidget(header)
        
        # Main content
        content_splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(content_splitter)
        
        # Categories list
        categories_list = self._create_categories_list()
        content_splitter.addWidget(categories_list)
        
        # Settings panels
        self.settings_stack = QTabWidget()
        self.settings_stack.setTabPosition(QTabWidget.North)
        content_splitter.addWidget(self.settings_stack)
        
        # Create settings tabs
        self._create_general_tab()
        self._create_display_tab()
        self._create_detection_tab()
        self._create_iot_tab()
        self._create_performance_tab()
        self._create_security_tab()
        
        # Set splitter proportions
        content_splitter.setSizes([200, 600])
        
        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Apply)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        button_box.button(QDialogButtonBox.Apply).clicked.connect(self._apply_settings)
        layout.addWidget(button_box)
    
    def _create_header(self):
        """Create settings dialog header"""
        header = QFrame()
        header.setFixedHeight(60)
        header.setStyleSheet("""
            QFrame {
                background-color: #2c3e50;
                border-bottom: 1px solid #34495e;
            }
        """)
        
        layout = QHBoxLayout(header)
        layout.setContentsMargins(20, 10, 20, 10)
        
        # Title
        title = QLabel("⚙️ System Settings")
        title.setFont(QFont("Segoe UI", 14, QFont.Bold))
        title.setStyleSheet("color: white;")
        layout.addWidget(title)
        
        # Subtitle
        subtitle = QLabel("Configure your Smart Intersection Monitoring System")
        subtitle.setFont(QFont("Segoe UI", 9))
        subtitle.setStyleSheet("color: #ecf0f1;")
        layout.addWidget(subtitle)
        
        layout.addStretch()
        
        return header
    
    def _create_categories_list(self):
        """Create settings categories list"""
        categories = QListWidget()
        categories.setFixedWidth(180)
        categories.setStyleSheet("""
            QListWidget {
                background-color: #34495e;
                border: none;
                outline: none;
                font-size: 10pt;
            }
            QListWidget::item {
                color: white;
                padding: 12px;
                border-bottom: 1px solid #2c3e50;
            }
            QListWidget::item:selected {
                background-color: #3498db;
            }
            QListWidget::item:hover {
                background-color: #495860;
            }
        """)
        
        category_items = [
            ("⚙️ General", 0),
            ("🎨 Display", 1),
            ("🤖 Detection", 2),
            ("📡 IoT Devices", 3),
            ("⚡ Performance", 4),
            ("🔒 Security", 5),
        ]
        
        for text, tab_index in category_items:
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, tab_index)
            categories.addItem(item)
        
        categories.currentItemChanged.connect(self._on_category_changed)
        categories.setCurrentRow(0)
        
        return categories
    
    def _on_category_changed(self, current, previous):
        """Handle category selection change"""
        if current:
            tab_index = current.data(Qt.UserRole)
            self.settings_stack.setCurrentIndex(tab_index)
    
    def _create_general_tab(self):
        """Create general settings tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Application settings
        app_group = QGroupBox("Application Settings")
        app_layout = QFormLayout(app_group)
        
        # Auto-start
        self.auto_start_cb = QCheckBox("Start with system")
        app_layout.addRow("Startup:", self.auto_start_cb)
        
        # Language
        self.language_combo = QComboBox()
        self.language_combo.addItems(["English", "Spanish", "French", "German"])
        app_layout.addRow("Language:", self.language_combo)
        
        # Update check
        self.auto_update_cb = QCheckBox("Check for updates automatically")
        app_layout.addRow("Updates:", self.auto_update_cb)
        
        # Data directory
        data_layout = QHBoxLayout()
        self.data_dir_edit = QLineEdit()
        data_browse_btn = QPushButton("Browse")
        data_browse_btn.clicked.connect(self._browse_data_directory)
        data_layout.addWidget(self.data_dir_edit)
        data_layout.addWidget(data_browse_btn)
        app_layout.addRow("Data Directory:", data_layout)
        
        layout.addWidget(app_group)
        
        # Logging settings
        log_group = QGroupBox("Logging Settings")
        log_layout = QFormLayout(log_group)
        
        # Log level
        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        log_layout.addRow("Log Level:", self.log_level_combo)
        
        # Log retention
        self.log_retention_spin = QSpinBox()
        self.log_retention_spin.setRange(1, 365)
        self.log_retention_spin.setSuffix(" days")
        log_layout.addRow("Log Retention:", self.log_retention_spin)
        
        layout.addWidget(log_group)
        
        layout.addStretch()
        self.settings_stack.addTab(tab, "General")
    
    def _create_display_tab(self):
        """Create display settings tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Theme settings
        theme_group = QGroupBox("Theme Settings")
        theme_layout = QFormLayout(theme_group)
        
        # Theme selection
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Light", "Dark", "Auto (System)"])
        self.theme_combo.currentTextChanged.connect(self._on_theme_changed)
        theme_layout.addRow("Theme:", self.theme_combo)
        
        # Color scheme
        self.color_scheme_combo = QComboBox()
        self.color_scheme_combo.addItems(["Default", "Blue", "Green", "Purple", "Orange"])
        theme_layout.addRow("Color Scheme:", self.color_scheme_combo)
        
        layout.addWidget(theme_group)
        
        # Display options
        display_group = QGroupBox("Display Options")
        display_layout = QFormLayout(display_group)
        
        # UI scaling
        self.ui_scale_slider = QSlider(Qt.Horizontal)
        self.ui_scale_slider.setRange(80, 150)
        self.ui_scale_slider.setValue(100)
        scale_layout = QHBoxLayout()
        scale_layout.addWidget(self.ui_scale_slider)
        self.scale_label = QLabel("100%")
        scale_layout.addWidget(self.scale_label)
        self.ui_scale_slider.valueChanged.connect(
            lambda v: self.scale_label.setText(f"{v}%")
        )
        display_layout.addRow("UI Scale:", scale_layout)
        
        # Font size
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(8, 20)
        self.font_size_spin.setSuffix(" pt")
        display_layout.addRow("Font Size:", self.font_size_spin)
        
        # Show tooltips
        self.tooltips_cb = QCheckBox("Show tooltips")
        display_layout.addRow("Tooltips:", self.tooltips_cb)
        
        # Animations
        self.animations_cb = QCheckBox("Enable animations")
        display_layout.addRow("Animations:", self.animations_cb)
        
        layout.addWidget(display_group)
        
        layout.addStretch()
        self.settings_stack.addTab(tab, "Display")
    
    def _create_detection_tab(self):
        """Create detection settings tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Model settings
        model_group = QGroupBox("AI Model Settings")
        model_layout = QFormLayout(model_group)
        
        # Detection model
        self.detection_model_combo = QComboBox()
        self.detection_model_combo.addItems([
            "YOLO11n (Fast)", "YOLO11s (Balanced)", "YOLO11m (Accurate)", "YOLO11x (Best)"
        ])
        model_layout.addRow("Detection Model:", self.detection_model_combo)
        
        # Device
        self.device_combo = QComboBox()
        self.device_combo.addItems(["AUTO", "CPU", "GPU"])
        model_layout.addRow("Compute Device:", self.device_combo)
        
        # Confidence threshold
        conf_layout = QHBoxLayout()
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(10, 95)
        self.conf_slider.setValue(50)
        conf_layout.addWidget(self.conf_slider)
        self.conf_label = QLabel("0.50")
        conf_layout.addWidget(self.conf_label)
        self.conf_slider.valueChanged.connect(
            lambda v: self.conf_label.setText(f"{v/100:.2f}")
        )
        model_layout.addRow("Confidence Threshold:", conf_layout)
        
        # NMS threshold
        nms_layout = QHBoxLayout()
        self.nms_slider = QSlider(Qt.Horizontal)
        self.nms_slider.setRange(10, 80)
        self.nms_slider.setValue(45)
        nms_layout.addWidget(self.nms_slider)
        self.nms_label = QLabel("0.45")
        nms_layout.addWidget(self.nms_label)
        self.nms_slider.valueChanged.connect(
            lambda v: self.nms_label.setText(f"{v/100:.2f}")
        )
        model_layout.addRow("NMS Threshold:", nms_layout)
        
        layout.addWidget(model_group)
        
        # Detection classes
        classes_group = QGroupBox("Detection Classes")
        classes_layout = QVBoxLayout(classes_group)
        
        # Class selection list
        self.classes_list = QListWidget()
        classes = [
            "person", "bicycle", "car", "motorcycle", "bus", "truck",
            "traffic light", "stop sign", "parking meter", "bench"
        ]
        
        for class_name in classes:
            item = QListWidgetItem(class_name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.classes_list.addItem(item)
        
        classes_layout.addWidget(self.classes_list)
        
        # Class buttons
        class_buttons = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self._select_all_classes)
        class_buttons.addWidget(select_all_btn)
        
        deselect_all_btn = QPushButton("Deselect All")
        deselect_all_btn.clicked.connect(self._deselect_all_classes)
        class_buttons.addWidget(deselect_all_btn)
        
        class_buttons.addStretch()
        classes_layout.addLayout(class_buttons)
        
        layout.addWidget(classes_group)
        
        layout.addStretch()
        self.settings_stack.addTab(tab, "Detection")
    
    def _create_iot_tab(self):
        """Create IoT settings tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # MQTT settings
        mqtt_group = QGroupBox("MQTT Configuration")
        mqtt_layout = QFormLayout(mqtt_group)
        
        # Broker settings
        self.mqtt_host_edit = QLineEdit()
        mqtt_layout.addRow("MQTT Broker:", self.mqtt_host_edit)
        
        self.mqtt_port_spin = QSpinBox()
        self.mqtt_port_spin.setRange(1, 65535)
        self.mqtt_port_spin.setValue(1883)
        mqtt_layout.addRow("Port:", self.mqtt_port_spin)
        
        self.mqtt_username_edit = QLineEdit()
        mqtt_layout.addRow("Username:", self.mqtt_username_edit)
        
        self.mqtt_password_edit = QLineEdit()
        self.mqtt_password_edit.setEchoMode(QLineEdit.Password)
        mqtt_layout.addRow("Password:", self.mqtt_password_edit)
        
        # Topics
        self.mqtt_topic_edit = QLineEdit()
        mqtt_layout.addRow("Base Topic:", self.mqtt_topic_edit)
        
        layout.addWidget(mqtt_group)
        
        # InfluxDB settings
        influx_group = QGroupBox("InfluxDB Configuration")
        influx_layout = QFormLayout(influx_group)
        
        self.influx_url_edit = QLineEdit()
        influx_layout.addRow("InfluxDB URL:", self.influx_url_edit)
        
        self.influx_token_edit = QLineEdit()
        self.influx_token_edit.setEchoMode(QLineEdit.Password)
        influx_layout.addRow("Token:", self.influx_token_edit)
        
        self.influx_org_edit = QLineEdit()
        influx_layout.addRow("Organization:", self.influx_org_edit)
        
        self.influx_bucket_edit = QLineEdit()
        influx_layout.addRow("Bucket:", self.influx_bucket_edit)
        
        layout.addWidget(influx_group)
        
        # Device settings
        device_group = QGroupBox("Device Settings")
        device_layout = QFormLayout(device_group)
        
        # Discovery
        self.device_discovery_cb = QCheckBox("Enable device auto-discovery")
        device_layout.addRow("Discovery:", self.device_discovery_cb)
        
        # Heartbeat interval
        self.heartbeat_spin = QSpinBox()
        self.heartbeat_spin.setRange(5, 300)
        self.heartbeat_spin.setSuffix(" seconds")
        device_layout.addRow("Heartbeat Interval:", self.heartbeat_spin)
        
        layout.addWidget(device_group)
        
        layout.addStretch()
        self.settings_stack.addTab(tab, "IoT Devices")
    
    def _create_performance_tab(self):
        """Create performance settings tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Processing settings
        processing_group = QGroupBox("Processing Settings")
        processing_layout = QFormLayout(processing_group)
        
        # Frame rate
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setSuffix(" FPS")
        processing_layout.addRow("Target Frame Rate:", self.fps_spin)
        
        # Buffer size
        self.buffer_spin = QSpinBox()
        self.buffer_spin.setRange(1, 100)
        self.buffer_spin.setSuffix(" frames")
        processing_layout.addRow("Frame Buffer Size:", self.buffer_spin)
        
        # Thread count
        self.threads_spin = QSpinBox()
        self.threads_spin.setRange(1, 16)
        processing_layout.addRow("Processing Threads:", self.threads_spin)
        
        layout.addWidget(processing_group)
        
        # Memory settings
        memory_group = QGroupBox("Memory Settings")
        memory_layout = QFormLayout(memory_group)
        
        # Cache size
        self.cache_spin = QSpinBox()
        self.cache_spin.setRange(100, 2000)
        self.cache_spin.setSuffix(" MB")
        memory_layout.addRow("Cache Size:", self.cache_spin)
        
        # Cleanup interval
        self.cleanup_spin = QSpinBox()
        self.cleanup_spin.setRange(1, 60)
        self.cleanup_spin.setSuffix(" minutes")
        memory_layout.addRow("Cleanup Interval:", self.cleanup_spin)
        
        layout.addWidget(memory_group)
        
        # GPU settings
        gpu_group = QGroupBox("GPU Settings")
        gpu_layout = QFormLayout(gpu_group)
        
        # GPU acceleration
        self.gpu_acceleration_cb = QCheckBox("Enable GPU acceleration")
        gpu_layout.addRow("Acceleration:", self.gpu_acceleration_cb)
        
        # Memory fraction
        gpu_mem_layout = QHBoxLayout()
        self.gpu_memory_slider = QSlider(Qt.Horizontal)
        self.gpu_memory_slider.setRange(10, 90)
        self.gpu_memory_slider.setValue(70)
        gpu_mem_layout.addWidget(self.gpu_memory_slider)
        self.gpu_memory_label = QLabel("70%")
        gpu_mem_layout.addWidget(self.gpu_memory_label)
        self.gpu_memory_slider.valueChanged.connect(
            lambda v: self.gpu_memory_label.setText(f"{v}%")
        )
        gpu_layout.addRow("GPU Memory Usage:", gpu_mem_layout)
        
        layout.addWidget(gpu_group)
        
        layout.addStretch()
        self.settings_stack.addTab(tab, "Performance")
    
    def _create_security_tab(self):
        """Create security settings tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Authentication
        auth_group = QGroupBox("Authentication")
        auth_layout = QFormLayout(auth_group)
        
        # Enable authentication
        self.auth_enabled_cb = QCheckBox("Enable user authentication")
        auth_layout.addRow("Authentication:", self.auth_enabled_cb)
        
        # Session timeout
        self.session_timeout_spin = QSpinBox()
        self.session_timeout_spin.setRange(5, 480)
        self.session_timeout_spin.setSuffix(" minutes")
        auth_layout.addRow("Session Timeout:", self.session_timeout_spin)
        
        layout.addWidget(auth_group)
        
        # Encryption
        encryption_group = QGroupBox("Encryption")
        encryption_layout = QFormLayout(encryption_group)
        
        # Enable encryption
        self.encryption_cb = QCheckBox("Enable data encryption")
        encryption_layout.addRow("Encryption:", self.encryption_cb)
        
        # SSL/TLS
        self.ssl_cb = QCheckBox("Use SSL/TLS connections")
        encryption_layout.addRow("SSL/TLS:", self.ssl_cb)
        
        layout.addWidget(encryption_group)
        
        # Audit logging
        audit_group = QGroupBox("Audit & Compliance")
        audit_layout = QFormLayout(audit_group)
        
        # Enable audit log
        self.audit_log_cb = QCheckBox("Enable audit logging")
        audit_layout.addRow("Audit Logging:", self.audit_log_cb)
        
        # Compliance mode
        self.compliance_combo = QComboBox()
        self.compliance_combo.addItems(["None", "GDPR", "HIPAA", "SOX"])
        audit_layout.addRow("Compliance Mode:", self.compliance_combo)
        
        layout.addWidget(audit_group)
        
        layout.addStretch()
        self.settings_stack.addTab(tab, "Security")
    
    def _browse_data_directory(self):
        """Browse for data directory"""
        directory = QFileDialog.getExistingDirectory(self, "Select Data Directory")
        if directory:
            self.data_dir_edit.setText(directory)
    
    def _on_theme_changed(self, theme):
        """Handle theme change"""
        self.theme_changed.emit(theme.lower())
    
    def _select_all_classes(self):
        """Select all detection classes"""
        for i in range(self.classes_list.count()):
            item = self.classes_list.item(i)
            item.setCheckState(Qt.Checked)
    
    def _deselect_all_classes(self):
        """Deselect all detection classes"""
        for i in range(self.classes_list.count()):
            item = self.classes_list.item(i)
            item.setCheckState(Qt.Unchecked)
    
    def _load_settings(self):
        """Load settings from QSettings"""
        # General settings
        self.auto_start_cb.setChecked(self.settings.value("general/auto_start", False, type=bool))
        self.language_combo.setCurrentText(self.settings.value("general/language", "English"))
        self.auto_update_cb.setChecked(self.settings.value("general/auto_update", True, type=bool))
        self.data_dir_edit.setText(self.settings.value("general/data_dir", "./data"))
        
        # Logging
        self.log_level_combo.setCurrentText(self.settings.value("logging/level", "INFO"))
        self.log_retention_spin.setValue(self.settings.value("logging/retention", 30, type=int))
        
        # Display
        self.theme_combo.setCurrentText(self.settings.value("display/theme", "Light"))
        self.color_scheme_combo.setCurrentText(self.settings.value("display/color_scheme", "Default"))
        self.ui_scale_slider.setValue(self.settings.value("display/ui_scale", 100, type=int))
        self.font_size_spin.setValue(self.settings.value("display/font_size", 9, type=int))
        self.tooltips_cb.setChecked(self.settings.value("display/tooltips", True, type=bool))
        self.animations_cb.setChecked(self.settings.value("display/animations", True, type=bool))
        
        # Detection
        self.detection_model_combo.setCurrentText(self.settings.value("detection/model", "YOLO11n (Fast)"))
        self.device_combo.setCurrentText(self.settings.value("detection/device", "AUTO"))
        self.conf_slider.setValue(int(self.settings.value("detection/confidence", 0.5, type=float) * 100))
        self.nms_slider.setValue(int(self.settings.value("detection/nms", 0.45, type=float) * 100))
        
        # IoT
        self.mqtt_host_edit.setText(self.settings.value("iot/mqtt_host", "localhost"))
        self.mqtt_port_spin.setValue(self.settings.value("iot/mqtt_port", 1883, type=int))
        self.mqtt_username_edit.setText(self.settings.value("iot/mqtt_username", ""))
        self.mqtt_topic_edit.setText(self.settings.value("iot/mqtt_topic", "smart_intersection"))
        
        # Performance
        self.fps_spin.setValue(self.settings.value("performance/fps", 30, type=int))
        self.buffer_spin.setValue(self.settings.value("performance/buffer_size", 10, type=int))
        self.threads_spin.setValue(self.settings.value("performance/threads", 4, type=int))
        self.cache_spin.setValue(self.settings.value("performance/cache_size", 500, type=int))
        self.cleanup_spin.setValue(self.settings.value("performance/cleanup_interval", 5, type=int))
        self.gpu_acceleration_cb.setChecked(self.settings.value("performance/gpu_acceleration", True, type=bool))
        self.gpu_memory_slider.setValue(self.settings.value("performance/gpu_memory", 70, type=int))
        
        # Security
        self.auth_enabled_cb.setChecked(self.settings.value("security/auth_enabled", False, type=bool))
        self.session_timeout_spin.setValue(self.settings.value("security/session_timeout", 60, type=int))
        self.encryption_cb.setChecked(self.settings.value("security/encryption", True, type=bool))
        self.ssl_cb.setChecked(self.settings.value("security/ssl", True, type=bool))
        self.audit_log_cb.setChecked(self.settings.value("security/audit_log", True, type=bool))
        self.compliance_combo.setCurrentText(self.settings.value("security/compliance", "None"))
    
    def _apply_settings(self):
        """Apply settings without closing dialog"""
        self._save_settings()
        self.settings_changed.emit(self.pending_changes)
        self.pending_changes.clear()
    
    def _save_settings(self):
        """Save settings to QSettings"""
        # General settings
        self.settings.setValue("general/auto_start", self.auto_start_cb.isChecked())
        self.settings.setValue("general/language", self.language_combo.currentText())
        self.settings.setValue("general/auto_update", self.auto_update_cb.isChecked())
        self.settings.setValue("general/data_dir", self.data_dir_edit.text())
        
        # Logging
        self.settings.setValue("logging/level", self.log_level_combo.currentText())
        self.settings.setValue("logging/retention", self.log_retention_spin.value())
        
        # Display
        self.settings.setValue("display/theme", self.theme_combo.currentText())
        self.settings.setValue("display/color_scheme", self.color_scheme_combo.currentText())
        self.settings.setValue("display/ui_scale", self.ui_scale_slider.value())
        self.settings.setValue("display/font_size", self.font_size_spin.value())
        self.settings.setValue("display/tooltips", self.tooltips_cb.isChecked())
        self.settings.setValue("display/animations", self.animations_cb.isChecked())
        
        # Detection
        self.settings.setValue("detection/model", self.detection_model_combo.currentText())
        self.settings.setValue("detection/device", self.device_combo.currentText())
        self.settings.setValue("detection/confidence", self.conf_slider.value() / 100.0)
        self.settings.setValue("detection/nms", self.nms_slider.value() / 100.0)
        
        # IoT
        self.settings.setValue("iot/mqtt_host", self.mqtt_host_edit.text())
        self.settings.setValue("iot/mqtt_port", self.mqtt_port_spin.value())
        self.settings.setValue("iot/mqtt_username", self.mqtt_username_edit.text())
        self.settings.setValue("iot/mqtt_topic", self.mqtt_topic_edit.text())
        
        # Performance
        self.settings.setValue("performance/fps", self.fps_spin.value())
        self.settings.setValue("performance/buffer_size", self.buffer_spin.value())
        self.settings.setValue("performance/threads", self.threads_spin.value())
        self.settings.setValue("performance/cache_size", self.cache_spin.value())
        self.settings.setValue("performance/cleanup_interval", self.cleanup_spin.value())
        self.settings.setValue("performance/gpu_acceleration", self.gpu_acceleration_cb.isChecked())
        self.settings.setValue("performance/gpu_memory", self.gpu_memory_slider.value())
        
        # Security
        self.settings.setValue("security/auth_enabled", self.auth_enabled_cb.isChecked())
        self.settings.setValue("security/session_timeout", self.session_timeout_spin.value())
        self.settings.setValue("security/encryption", self.encryption_cb.isChecked())
        self.settings.setValue("security/ssl", self.ssl_cb.isChecked())
        self.settings.setValue("security/audit_log", self.audit_log_cb.isChecked())
        self.settings.setValue("security/compliance", self.compliance_combo.currentText())
        
        print("⚙️ Settings saved successfully")
    
    def accept(self):
        """Accept dialog and save settings"""
        self._save_settings()
        self.settings_changed.emit(self.pending_changes)
        super().accept()
    
    def get_current_settings(self):
        """Get current settings as dictionary"""
        return {
            'general': {
                'auto_start': self.auto_start_cb.isChecked(),
                'language': self.language_combo.currentText(),
                'auto_update': self.auto_update_cb.isChecked(),
                'data_dir': self.data_dir_edit.text()
            },
            'display': {
                'theme': self.theme_combo.currentText(),
                'ui_scale': self.ui_scale_slider.value(),
                'font_size': self.font_size_spin.value()
            },
            'detection': {
                'model': self.detection_model_combo.currentText(),
                'device': self.device_combo.currentText(),
                'confidence': self.conf_slider.value() / 100.0,
                'nms': self.nms_slider.value() / 100.0
            }
        }
