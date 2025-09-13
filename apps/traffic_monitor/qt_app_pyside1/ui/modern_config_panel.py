from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, 
    QSlider, QCheckBox, QPushButton, QGroupBox, QFormLayout,
    QSpinBox, QDoubleSpinBox, QTabWidget, QLineEdit, QFileDialog,
    QSpacerItem, QSizePolicy, QScrollArea, QFrame
)
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QFont

# Import VLM insights widget
from ui.vlm_insights_widget import VLMInsightsWidget

class ModernConfigPanel(QWidget):
    def get_vlm_widget(self):
        """Return the VLMInsightsWidget instance for controller integration."""
        return getattr(self, 'vlm_insights_widget', None)

    def vlm_set_current_frame(self, frame):
        if hasattr(self, 'vlm_insights_widget'):
            self.vlm_insights_widget.set_current_frame(frame)

    def vlm_set_detection_data(self, detection_data):
        if hasattr(self, 'vlm_insights_widget'):
            self.vlm_insights_widget.set_detection_data(detection_data)

    def vlm_on_video_paused(self, is_paused):
        if hasattr(self, 'vlm_insights_widget'):
            self.vlm_insights_widget.on_video_paused(is_paused)

    def vlm_on_analysis_result(self, result):
        if hasattr(self, 'vlm_insights_widget'):
            self.vlm_insights_widget.on_analysis_result(result)
    """Enhanced side panel with modern dark theme and pill-style tabs."""
    
    config_changed = Signal(dict)  # Emitted when configuration changes are applied
    theme_toggled = Signal(bool)   # Emitted when theme toggle button is clicked (True = dark)
    device_switch_requested = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.setObjectName("ModernConfigPanel")
        self.setStyleSheet(self._get_modern_style())
        
        # Set minimum and preferred size for the panel
        self.setMinimumSize(380, 600)
        self.setMaximumWidth(500)
        
        # Flag to track automatic model changes (to avoid showing "Manual Override")
        self._updating_automatically = False
        
        self.initUI()
        self.dark_theme = True  # Start with dark theme
        self._updating_automatically = False  # Flag to track automatic updates
        
    def _get_modern_style(self):
        """Modern dark theme with compact layout"""
        return """
        #ModernConfigPanel {
            background: #121212;
            border: none;
            min-width: 320px;
            max-width: 400px;
        }
        
        /* Tab Widget Styling */
        QTabWidget::pane {
            background: #1E1E1E;
            border: 1px solid #2C2C2C;
            border-radius: 8px;
            padding: 8px;
        }
        
        QTabBar::tab {
            background: transparent;
            color: #B0B0B0;
            border-radius: 12px;
            padding: 8px 16px;
            margin: 1px;
            font-size: 12px;
            font-weight: 500;
            min-width: 70px;
        }
        
        QTabBar::tab:selected {
            background: #007BFF;
            color: #FFFFFF;
        }
        
        QTabBar::tab:hover:!selected {
            background: #2C2C2C;
            color: #FFFFFF;
        }
        
        /* Section Headers */
        QLabel.section-header {
            font-weight: bold;
            color: #FFFFFF;
            border-bottom: 1px solid #2C2C2C;
            margin-bottom: 8px;
            padding-bottom: 4px;
            font-size: 14px;
        }
        
        /* Regular Labels */
        QLabel {
            color: #FFFFFF;
            font-size: 12px;
            background: transparent;
        }
        
        QLabel.secondary {
            color: #B0B0B0;
        }
        
        /* Buttons */
        QPushButton {
            background: #007BFF;
            color: #FFFFFF;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 600;
            padding: 10px 16px;
            border: none;
        }
        
        QPushButton:hover {
            background: #3399FF;
        }
        
        QPushButton.secondary {
            background: #2ECC71;
        }
        
        QPushButton.secondary:hover {
            background: #48D187;
        }
        
        QPushButton.warning {
            background: #E74C3C;
        }
        
        QPushButton.warning:hover {
            background: #FF6B5A;
        }
        
        /* Sliders */
        QSlider::groove:horizontal {
            background: #1E1E1E;
            border: 1px solid #00E6E6;
            height: 6px;
            border-radius: 3px;
        }
        
        QSlider::handle:horizontal {
            background: #00E6E6;
            border-radius: 8px;
            width: 16px;
            height: 16px;
            margin: -5px 0;
        }
        
        QSlider::handle:horizontal:hover {
            background: #00FFFF;
        }
        
        /* Combo Boxes */
        QComboBox {
            background: #1E1E1E;
            border: 1px solid #00E6E6;
            border-radius: 6px;
            padding: 6px 12px;
            color: #FFFFFF;
            font-size: 12px;
            min-height: 20px;
        }
        
        QComboBox::drop-down {
            border: none;
            width: 20px;
        }
        
        QComboBox::down-arrow {
            image: none;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 5px solid #00E6E6;
            margin-right: 5px;
        }
        
        QComboBox QAbstractItemView {
            background: #1E1E1E;
            border: 1px solid #2C2C2C;
            color: #FFFFFF;
            selection-background-color: #007BFF;
        }
        
        /* Checkboxes */
        QCheckBox {
            color: #FFFFFF;
            font-size: 12px;
            spacing: 8px;
        }
        
        QCheckBox::indicator {
            width: 16px;
            height: 16px;
            border-radius: 3px;
            border: 1px solid #2C2C2C;
            background: #1E1E1E;
        }
        
        QCheckBox::indicator:checked {
            background: #007BFF;
            border: 1px solid #007BFF;
        }
        
        QCheckBox::indicator:checked:hover {
            background: #3399FF;
        }
        
        /* Spin Boxes */
        QSpinBox, QDoubleSpinBox {
            background: #1E1E1E;
            border: 1px solid #2C2C2C;
            border-radius: 6px;
            color: #FFFFFF;
            padding: 6px 8px;
            font-size: 12px;
        }
        
        QSpinBox::up-button, QDoubleSpinBox::up-button {
            background: #2C2C2C;
            border-radius: 3px;
        }
        
        QSpinBox::down-button, QDoubleSpinBox::down-button {
            background: #2C2C2C;
            border-radius: 3px;
        }
        
        /* Group Boxes */
        QGroupBox {
            border: 1px solid #2C2C2C;
            border-radius: 8px;
            margin-top: 16px;
            background: transparent;
            font-weight: bold;
            color: #FFFFFF;
            font-size: 13px;
            padding-top: 10px;
        }
        
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 12px;
            top: 0px;
            padding: 0 8px;
            background: #121212;
        }
        
        /* Scroll Area */
        QScrollArea {
            background: transparent;
            border: none;
        }
        
        QScrollArea QWidget {
            background: transparent;
        }
        
        /* Scroll Bars */
        QScrollBar:vertical {
            background: #1E1E1E;
            width: 8px;
            border-radius: 4px;
        }
        
        QScrollBar::handle:vertical {
            background: #2C2C2C;
            border-radius: 4px;
        }
        
        QScrollBar::handle:vertical:hover {
            background: #007BFF;
        }
        """
        
    def initUI(self):
        """Initialize the modern UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(0)
        
        # Create tab widget with pill-style tabs
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.North)
        
        # Create tabs
        self._create_detection_tab()
        self._create_ai_insights_tab()
        self._create_display_tab()
        self._create_violations_tab()
        
        
        layout.addWidget(self.tabs)
        
    def _create_detection_tab(self):
        """Create advanced detection settings tab with dynamic model selection"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(32)  # Increased spacing between major sections
        layout.setContentsMargins(12, 20, 12, 20)  # Better margins
        
        # System Info Section
        system_group = QGroupBox("System Info")
        system_layout = QVBoxLayout(system_group)
        system_layout.setSpacing(12)  # Better spacing within section
        
        # Auto-detected device info
        device_info_layout = QHBoxLayout()
        device_info_layout.setSpacing(8)  # Better spacing between elements
        device_info_label = QLabel("Current Device:")
        device_info_label.setStyleSheet("color: #B0B0B0; font-size: 11px;")
        
        self.current_device_label = QLabel("AUTO")
        self.current_device_label.setStyleSheet("""
            QLabel {
                color: #00E6E6;
                font-size: 11px;
                font-weight: bold;
                background: #1E1E1E;
                border: 1px solid #00E6E6;
                border-radius: 3px;
                padding: 2px 6px;
            }
        """)
        
        device_info_layout.addWidget(device_info_label)
        device_info_layout.addWidget(self.current_device_label)
        device_info_layout.addStretch()
        
        # Device selector
        device_selector_layout = QHBoxLayout()
        device_selector_layout.setSpacing(8)  # Better spacing
        device_selector_label = QLabel("Override Device:")
        device_selector_label.setStyleSheet("color: #FFFFFF; font-size: 12px;")
        
        self.device_combo = QComboBox()
        self.device_combo.addItems(["AUTO", "CPU", "GPU", "NPU"])
        self.device_combo.setCurrentText("AUTO")
        self.device_combo.currentTextChanged.connect(self._on_device_changed)
        
        device_selector_layout.addWidget(device_selector_label)
        device_selector_layout.addWidget(self.device_combo)
        device_selector_layout.addStretch()
        
        system_layout.addLayout(device_info_layout)
        system_layout.addLayout(device_selector_layout)
        
        # Model Settings Section (Enhanced)
        model_group = QGroupBox("Model Settings (Dynamic Selection)")
        model_layout = QVBoxLayout(model_group)
        model_layout.setSpacing(16)  # Better spacing within section
        
        # Auto-selection explanation
        auto_info = QLabel("📊 Auto-select based on device:")
        auto_info.setStyleSheet("color: #03DAC5; font-weight: bold; font-size: 12px;")
        model_layout.addWidget(auto_info)
        
        # Device-model mapping info
        mapping_layout = QVBoxLayout()
        mapping_layout.setSpacing(4)  # Better spacing for readability
        cpu_mapping = QLabel("• CPU → YOLOv11n (lightweight)")
        gpu_mapping = QLabel("• GPU → YOLOv11x (heavyweight)")
        npu_mapping = QLabel("• NPU → YOLOv11n (optimized)")
        
        for label in [cpu_mapping, gpu_mapping, npu_mapping]:
            label.setStyleSheet("color: #B0B0B0; font-size: 10px; margin-left: 12px;")
            mapping_layout.addWidget(label)
        model_layout.addLayout(mapping_layout)
        
        # Auto-selected model info
        model_info_layout = QHBoxLayout()
        model_info_layout.setSpacing(8)  # Better spacing
        model_info_label = QLabel("Auto-Selected Model:")
        model_info_label.setStyleSheet("color: #B0B0B0; font-size: 12px;")
        
        self.auto_model_label = QLabel("YOLOv11n (CPU Optimized)")
        self.auto_model_label.setStyleSheet("""
            QLabel {
                color: #FFD700;
                font-size: 12px;
                font-weight: bold;
                background: #1E1E1E;
                border: 1px solid #FFD700;
                border-radius: 4px;
                padding: 4px 8px;
            }
        """)
        
        model_info_layout.addWidget(model_info_label)
        model_info_layout.addWidget(self.auto_model_label)
        model_info_layout.addStretch()
        
        # Manual model selector dropdown
        manual_model_layout = QHBoxLayout()
        manual_model_layout.setSpacing(8)  # Better spacing
        manual_model_label = QLabel("Manual Override:")
        manual_model_label.setStyleSheet("color: #FFFFFF; font-size: 12px;")
        
        self.model_combo = QComboBox()
        self.model_combo.addItems(["AUTO", "YOLOv11n", "YOLOv11s", "YOLOv11m", "YOLOv11l", "YOLOv11x"])
        self.model_combo.setCurrentText("YOLOv11x")  # Default to YOLOv11x instead of AUTO
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        
        # Add input size combo
        input_size_label = QLabel("Input Size:")
        input_size_label.setStyleSheet("color: #FFFFFF; font-size: 12px;")
        
        self.input_size_combo = QComboBox()
        self.input_size_combo.addItems(["320", "416", "640", "832"])
        self.input_size_combo.setCurrentText("640")  # Default input size
        
        manual_model_layout.addWidget(manual_model_label)
        manual_model_layout.addWidget(self.model_combo)
        manual_model_layout.addWidget(input_size_label)
        manual_model_layout.addWidget(self.input_size_combo)
        manual_model_layout.addStretch()
        
        # Quick-switch buttons with glow highlight
        quick_switch_label = QLabel("🚀 Quick-switch buttons:")
        quick_switch_label.setStyleSheet("color: #03DAC5; font-weight: bold; font-size: 11px; margin-top: 8px;")
        
        quick_switch_layout = QHBoxLayout()
        quick_switch_layout.setSpacing(12)  # Better spacing between buttons
        
        self.lightweight_btn = QPushButton("Lightweight\n(YOLOv11n)")
        self.lightweight_btn.setObjectName("quickSwitchLight")
        self.lightweight_btn.clicked.connect(lambda: self._quick_switch_model("YOLOv11n"))
        
        self.heavyweight_btn = QPushButton("High-Accuracy\n(YOLOv11x)")
        self.heavyweight_btn.setObjectName("quickSwitchHeavy")
        self.heavyweight_btn.clicked.connect(lambda: self._quick_switch_model("YOLOv11x"))
        
        quick_switch_layout.addWidget(self.lightweight_btn)
        quick_switch_layout.addWidget(self.heavyweight_btn)
        
        model_layout.addLayout(model_info_layout)
        model_layout.addLayout(manual_model_layout)
        model_layout.addWidget(quick_switch_label)
        model_layout.addLayout(quick_switch_layout)

        layout.addWidget(system_group)
        layout.addWidget(model_group)
        layout.addStretch()        # Add custom styling for enhanced elements
        self._add_detection_tab_styles()
        
        self.tabs.addTab(widget, "System Info")
        
        # Auto-detect system at startup
        self._auto_detect_system()
        
    def _add_detection_tab_styles(self):
        """Add custom styles for enhanced detection tab elements"""
        additional_style = """
            QPushButton[objectName="quickSwitchLight"] {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2196F3, stop:1 #1976D2);
                color: #FFFFFF;
                border-radius: 8px;
                font-size: 10px;
                font-weight: 600;
                padding: 6px 8px;
                border: 1px solid transparent;
                text-align: center;
                max-height: 40px;
            }
            
            QPushButton[objectName="quickSwitchLight"]:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #42A5F5, stop:1 #1E88E5);
                border: 1px solid #03DAC5;
            }
            
            QPushButton[objectName="quickSwitchLight"]:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #1565C0, stop:1 #0D47A1);
            }
            
            QPushButton[objectName="quickSwitchHeavy"] {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #4CAF50, stop:1 #388E3C);
                color: #FFFFFF;
                border-radius: 8px;
                font-size: 10px;
                font-weight: 600;
                padding: 6px 8px;
                border: 1px solid transparent;
                text-align: center;
                max-height: 40px;
            }
            
            QPushButton[objectName="quickSwitchHeavy"]:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #66BB6A, stop:1 #43A047);
                border: 1px solid #03DAC5;
            }
            
            QPushButton[objectName="quickSwitchHeavy"]:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2E7D32, stop:1 #1B5E20);
            }
            
            QPushButton[objectName="activeModelLight"] {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #03DAC5, stop:1 #018786);
                color: #121212;
                border: 3px solid #00FFFF;
            }
            
            QPushButton[objectName="activeModelHeavy"] {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #03DAC5, stop:1 #018786);
                color: #121212;
                border: 3px solid #00FFFF;
            }
            
            QComboBox {
                background: #232323;
                color: #FFFFFF;
                border: 1px solid #424242;
                border-radius: 6px;
                padding: 6px 12px;
                font-size: 12px;
            }
            
            QComboBox:hover {
                border: 1px solid #03DAC5;
            }
            
            QComboBox::drop-down {
                border: none;
                background: #232323;
                width: 20px;
                border-radius: 6px;
            }
            
            QComboBox::down-arrow {
                image: none;
                border: 2px solid #FFFFFF;
                border-top: none;
                border-right: none;
                width: 6px;
                height: 6px;
                margin-right: 8px;
            }
            
            QComboBox QAbstractItemView {
                background: #232323;
                border: 1px solid #424242;
                border-radius: 6px;
                selection-background-color: #03DAC5;
                color: #FFFFFF;
            }
        """
        current_style = self.styleSheet()
        self.setStyleSheet(current_style + additional_style)
        
    def _auto_detect_system(self):
        """Auto-detect system capabilities and set defaults"""
        try:
            # Try to detect GPU using OpenVINO
            try:
                import openvino as ov
                core = ov.Core()
                available_devices = core.available_devices
                
                if 'GPU' in available_devices:
                    detected_device = "GPU"
                    recommended_model = "YOLOv11x (GPU Optimized)"
                    self.auto_model_label.setText(recommended_model)
                    self.auto_model_label.setStyleSheet("""
                        QLabel {
                            color: #00FF00;
                            font-size: 12px;
                            font-weight: bold;
                            background: #1E1E1E;
                            border: 1px solid #00FF00;
                            border-radius: 4px;
                            padding: 4px 8px;
                        }
                    """)
                    print(f"[CONFIG PANEL] GPU detected via OpenVINO: {available_devices}")
                else:
                    detected_device = "CPU"
                    recommended_model = "YOLOv11n (CPU Optimized)"
                    print(f"[CONFIG PANEL] Only CPU available: {available_devices}")
            except ImportError:
                # Fallback: Try nvidia-smi for NVIDIA GPUs
                try:
                    import subprocess
                    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, shell=True)
                    if result.returncode == 0:
                        detected_device = "GPU"
                        recommended_model = "YOLOv11n (GPU Optimized)"
                        self.auto_model_label.setText(recommended_model)
                        self.auto_model_label.setStyleSheet("""
                            QLabel {
                                color: #00FF00;
                                font-size: 12px;
                                font-weight: bold;
                                background: #1E1E1E;
                                border: 1px solid #00FF00;
                                border-radius: 4px;
                                padding: 4px 8px;
                            }
                        """)
                        print("[CONFIG PANEL] GPU detected via nvidia-smi")
                    else:
                        detected_device = "CPU"
                        recommended_model = "YOLOv11n (CPU Optimized)"
                        print("[CONFIG PANEL] nvidia-smi failed, defaulting to CPU")
                except Exception as e:
                    detected_device = "CPU"
                    recommended_model = "YOLOv11n (CPU Optimized)"
                    print(f"[CONFIG PANEL] nvidia-smi exception: {e}")
                    
        except Exception as e:
            detected_device = "CPU"
            recommended_model = "YOLOv11X (GPU Optimized)"
            print(f"[CONFIG PANEL] Detection failed: {e}")
            
        self.current_device_label.setText(detected_device)
        self.auto_model_label.setText(recommended_model)
        
        # Update device combo to show detected device and sync model
        if detected_device in ["CPU", "GPU"]:
            # Set automatic flag
            self._updating_automatically = True
            
            # Set device combo
            self.device_combo.setCurrentText(detected_device)
            # Set model combo to match device - block signals to prevent loops
            self.model_combo.blockSignals(True)
            if detected_device == "GPU":
                self.model_combo.setCurrentText("YOLOv11x")
            else:
                self.model_combo.setCurrentText("YOLOv11n")
            self.model_combo.blockSignals(False)
            
            # Clear automatic flag
            self._updating_automatically = False
        
        # Highlight the appropriate quick switch button and update model preview
        if detected_device == "GPU":
            self._highlight_active_model("YOLOv11x")
        else:
            self._highlight_active_model("YOLOv11n")
        
        print(f"[CONFIG PANEL] ✅ Auto-detection complete: {detected_device} -> {recommended_model}")
        
    def _on_device_changed(self, device):
        """Handle device selection change - automatically switch to appropriate model"""
        print(f"[CONFIG PANEL] Device changed to: {device}")
        
        if device == "AUTO":
            self._auto_detect_system()
        else:
            self.current_device_label.setText(device)
            
            # Set flag to indicate automatic update
            self._updating_automatically = True
            
            if device == "GPU":
                # Automatically set model to YOLOv11x for GPU - block signals to prevent loops
                self.model_combo.blockSignals(True)
                self.model_combo.setCurrentText("YOLOv11x")
                self.model_combo.blockSignals(False)
                
                self.auto_model_label.setText("YOLOv11x (GPU Optimized)")
                self.auto_model_label.setStyleSheet("""
                    QLabel {
                        color: #00FF00;
                        font-size: 12px;
                        font-weight: bold;
                        background: #1E1E1E;
                        border: 1px solid #00FF00;
                        border-radius: 4px;
                        padding: 4px 8px;
                    }
                """)
                self._highlight_active_model("YOLOv11x")
                print(f"[CONFIG PANEL] GPU selected - switched to YOLOv11x")
            else:  # CPU
                # Automatically set model to YOLOv11n for CPU - block signals to prevent loops
                self.model_combo.blockSignals(True)
                self.model_combo.setCurrentText("YOLOv11n")
                self.model_combo.blockSignals(False)
                
                self.auto_model_label.setText("YOLOv11n (CPU Optimized)")
                self.auto_model_label.setStyleSheet("""
                    QLabel {
                        color: #FFD700;
                        font-size: 12px;
                        font-weight: bold;
                        background: #1E1E1E;
                        border: 1px solid #FFD700;
                        border-radius: 4px;
                        padding: 4px 8px;
                    }
                """)
                self._highlight_active_model("YOLOv11n")
                print(f"[CONFIG PANEL] CPU selected - switched to YOLOv11n")
            
            # Clear the automatic update flag
            self._updating_automatically = False
        
        # Emit device switch signal
        self.device_switch_requested.emit(device)
        
        # Apply configuration immediately (with error handling)
        try:
            self.apply_config()
        except AttributeError as e:
            print(f"[CONFIG PANEL] Warning: Attribute error during config apply: {e}")
            # Continue initialization
        
    def _on_model_changed(self, model):
        """Handle manual model selection change"""
        # Check if this is an automatic update to avoid showing "Manual Override"
        if hasattr(self, '_updating_automatically') and self._updating_automatically:
            return  # Skip processing for automatic updates
            
        if model != "AUTO":
            self.auto_model_label.setText(f"{model} (Manual Override)")
            self.auto_model_label.setStyleSheet("""
                QLabel {
                    color: #FF6B5A;
                    font-size: 12px;
                    font-weight: bold;
                    background: #1E1E1E;
                    border: 1px solid #FF6B5A;
                    border-radius: 4px;
                    padding: 4px 8px;
                }
            """)
        else:
            self.auto_model_label.setText("Auto-Select Model")
            self.auto_model_label.setStyleSheet("""
                QLabel {
                    color: #00E6E6;
                    font-size: 12px;
                    font-weight: normal;
                    background: transparent;
                    border: none;
                    padding: 0px;
                }
            """)
        
        # Apply configuration immediately when model changes
        print(f"🔧 Model changed to: {model}, applying config...")
        print(f"🔧 Current model combo text: {self.model_combo.currentText()}")
        print(f"🔧 Current device combo text: {self.device_combo.currentText()}")
        self.apply_config()
        self._highlight_active_model(model)
        
    def _quick_switch_model(self, model):
        """Handle quick switch button clicks"""
        self.model_combo.setCurrentText(model)
        self._on_model_changed(model)
        
    def _highlight_active_model(self, model):
        """Highlight the currently active model button with glow effect"""
        # Reset both buttons
        self.lightweight_btn.setObjectName("quickSwitchLight")
        self.heavyweight_btn.setObjectName("quickSwitchHeavy")
        
        # Highlight active button with special glow styling
        if model in ["YOLOv11n", "AUTO"] and "YOLOv11n" in self.auto_model_label.text():
            self.lightweight_btn.setObjectName("activeModelLight")
        elif model in ["YOLOv11x"] or ("YOLOv11x" in self.auto_model_label.text() and model == "AUTO"):
            self.heavyweight_btn.setObjectName("activeModelHeavy")
            
        # Refresh styles
        self._add_detection_tab_styles()
        
        
    def _create_display_tab(self):
        """Create display settings tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(20)
        layout.setContentsMargins(0, 16, 0, 16)
        
        # Display Options
        display_group = QGroupBox("Display Options")
        display_layout = QVBoxLayout(display_group)
        display_layout.setSpacing(12)
        
        self.show_boxes = QCheckBox("Show Bounding Boxes")
        self.show_boxes.setChecked(True)
        
        self.show_labels = QCheckBox("Show Class Labels")
        self.show_labels.setChecked(True)
        
        self.show_confidence = QCheckBox("Show Confidence Scores")
        self.show_confidence.setChecked(True)
        
        self.show_fps = QCheckBox("Show FPS Counter")
        self.show_fps.setChecked(True)
        
        display_layout.addWidget(self.show_boxes)
        display_layout.addWidget(self.show_labels)
        display_layout.addWidget(self.show_confidence)
        display_layout.addWidget(self.show_fps)
        
        # Visual Settings
        visual_group = QGroupBox("Visual Settings")
        visual_layout = QVBoxLayout(visual_group)
        visual_layout.setSpacing(16)
        
        # Box thickness
        thickness_label = QLabel("Bounding Box Thickness")
        self.thickness_slider = QSlider(Qt.Horizontal)
        self.thickness_slider.setRange(1, 5)
        self.thickness_slider.setValue(2)
        self.thickness_value = QLabel("2")
        self.thickness_value.setObjectName("secondary")
        self.thickness_slider.valueChanged.connect(
            lambda v: self.thickness_value.setText(str(v))
        )
        
        thickness_layout = QHBoxLayout()
        thickness_layout.addWidget(thickness_label)
        thickness_layout.addStretch()
        thickness_layout.addWidget(self.thickness_value)
        
        # Font size
        font_label = QLabel("Label Font Size")
        self.font_slider = QSlider(Qt.Horizontal)
        self.font_slider.setRange(10, 24)
        self.font_slider.setValue(14)
        self.font_value = QLabel("14")
        self.font_value.setObjectName("secondary")
        self.font_slider.valueChanged.connect(
            lambda v: self.font_value.setText(str(v))
        )
        
        font_layout = QHBoxLayout()
        font_layout.addWidget(font_label)
        font_layout.addStretch()
        font_layout.addWidget(self.font_value)
        
        visual_layout.addLayout(thickness_layout)
        visual_layout.addWidget(self.thickness_slider)
        visual_layout.addLayout(font_layout)
        visual_layout.addWidget(self.font_slider)
        
        layout.addWidget(display_group)
        layout.addWidget(visual_group)
        layout.addStretch()
        
        self.tabs.addTab(widget, "Display")
        
    def _create_violations_tab(self):
        """Create violations settings tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(20)
        layout.setContentsMargins(0, 16, 0, 16)
        
        # Violation Detection
        violations_group = QGroupBox("Violation Detection")
        violations_layout = QVBoxLayout(violations_group)
        violations_layout.setSpacing(12)
        
        self.red_light_detection = QCheckBox("Red Light Violations")
        self.red_light_detection.setChecked(True)
        
        self.speed_violations = QCheckBox("Speed Violations")
        self.speed_violations.setChecked(False)
        
        self.wrong_way_detection = QCheckBox("Wrong Way Detection")
        self.wrong_way_detection.setChecked(False)
        
        self.crosswalk_violations = QCheckBox("Crosswalk Violations")
        self.crosswalk_violations.setChecked(False)
        
        violations_layout.addWidget(self.red_light_detection)
        violations_layout.addWidget(self.speed_violations)
        violations_layout.addWidget(self.wrong_way_detection)
        violations_layout.addWidget(self.crosswalk_violations)
        
        # Alert Settings
        alerts_group = QGroupBox("Alert Settings")
        alerts_layout = QVBoxLayout(alerts_group)
        alerts_layout.setSpacing(12)
        
        self.sound_alerts = QCheckBox("Sound Alerts")
        self.sound_alerts.setChecked(True)
        
        self.email_notifications = QCheckBox("Email Notifications")
        self.email_notifications.setChecked(False)
        
        self.auto_screenshot = QCheckBox("Auto Screenshot on Violation")
        self.auto_screenshot.setChecked(True)
        
        self.enable_vlm = QCheckBox("Enable VLM Analysis")
        self.enable_vlm.setChecked(False)
        
        self.traffic_analysis = QCheckBox("Traffic Flow Analysis")
        self.traffic_analysis.setChecked(True)
        
        self.anomaly_detection = QCheckBox("Anomaly Detection")
        self.anomaly_detection.setChecked(False)
        
        self.crowd_analysis = QCheckBox("Crowd Analysis")
        self.crowd_analysis.setChecked(False)
        
        alerts_layout.addWidget(self.sound_alerts)
        alerts_layout.addWidget(self.email_notifications)
        alerts_layout.addWidget(self.auto_screenshot)
        alerts_layout.addWidget(self.enable_vlm)
        alerts_layout.addWidget(self.traffic_analysis)
        alerts_layout.addWidget(self.anomaly_detection)
        alerts_layout.addWidget(self.crowd_analysis)
        
        # Performance settings group
        performance_group = QGroupBox("Performance Settings")
        performance_layout = QVBoxLayout(performance_group)
        performance_layout.setSpacing(12)
        
        # Frame skip slider
        frame_skip_layout = QHBoxLayout()
        frame_skip_layout.addWidget(QLabel("Frame Skip:"))
        self.frame_skip_slider = QSlider(Qt.Horizontal)
        self.frame_skip_slider.setMinimum(0)
        self.frame_skip_slider.setMaximum(10)
        self.frame_skip_slider.setValue(0)
        self.frame_skip_label = QLabel("0")
        self.frame_skip_slider.valueChanged.connect(lambda v: self.frame_skip_label.setText(str(v)))
        frame_skip_layout.addWidget(self.frame_skip_slider)
        frame_skip_layout.addWidget(self.frame_skip_label)
        performance_layout.addLayout(frame_skip_layout)
        
        # Batch size spinner
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("Batch Size:"))
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setMinimum(1)
        self.batch_size_spin.setMaximum(16)
        self.batch_size_spin.setValue(1)
        batch_layout.addWidget(self.batch_size_spin)
        batch_layout.addStretch()
        performance_layout.addLayout(batch_layout)
        
        layout.addWidget(violations_group)
        layout.addWidget(alerts_group)
        layout.addWidget(performance_group)
        layout.addStretch()
        
        self.tabs.addTab(widget, "Violations")
        
    def _create_ai_insights_tab(self):
        """Create AI insights tab with VLMInsightsWidget"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(20)
        layout.setContentsMargins(0, 16, 0, 16)

        # Add the VLM Insights Widget
        self.vlm_insights_widget = VLMInsightsWidget()
        layout.addWidget(self.vlm_insights_widget)
        layout.addStretch()
        self.tabs.addTab(widget, "Insights")
        
    def get_config(self):
        """Get current configuration as dictionary"""
        return {
            'device': self.device_combo.currentText(),
            'confidence_threshold': 0.5,  # Default value
            'iou_threshold': 0.45,  # Default value
            'model': self.model_combo.currentText(),
            'input_size': int(self.input_size_combo.currentText()),
            'show_boxes': getattr(self, 'show_boxes', None) and self.show_boxes.isChecked() if hasattr(self, 'show_boxes') else True,
            'show_labels': getattr(self, 'show_labels', None) and self.show_labels.isChecked() if hasattr(self, 'show_labels') else True,
            'show_confidence': getattr(self, 'show_confidence', None) and self.show_confidence.isChecked() if hasattr(self, 'show_confidence') else True,
            'show_fps': getattr(self, 'show_fps', None) and self.show_fps.isChecked() if hasattr(self, 'show_fps') else True,
            'box_thickness': getattr(self, 'thickness_slider', None) and self.thickness_slider.value() if hasattr(self, 'thickness_slider') else 2,
            'font_size': getattr(self, 'font_slider', None) and self.font_slider.value() if hasattr(self, 'font_slider') else 14,
            'red_light_detection': getattr(self, 'red_light_detection', None) and self.red_light_detection.isChecked() if hasattr(self, 'red_light_detection') else True,
            'speed_violations': getattr(self, 'speed_violations', None) and self.speed_violations.isChecked() if hasattr(self, 'speed_violations') else False,
            'wrong_way_detection': getattr(self, 'wrong_way_detection', None) and self.wrong_way_detection.isChecked() if hasattr(self, 'wrong_way_detection') else False,
            'crosswalk_violations': getattr(self, 'crosswalk_violations', None) and self.crosswalk_violations.isChecked() if hasattr(self, 'crosswalk_violations') else False,
            'sound_alerts': getattr(self, 'sound_alerts', None) and self.sound_alerts.isChecked() if hasattr(self, 'sound_alerts') else True,
            'email_notifications': getattr(self, 'email_notifications', None) and self.email_notifications.isChecked() if hasattr(self, 'email_notifications') else False,
            'auto_screenshot': getattr(self, 'auto_screenshot', None) and self.auto_screenshot.isChecked() if hasattr(self, 'auto_screenshot') else True,
            'enable_vlm': getattr(self, 'enable_vlm', None) and self.enable_vlm.isChecked() if hasattr(self, 'enable_vlm') else False,
            'traffic_analysis': getattr(self, 'traffic_analysis', None) and self.traffic_analysis.isChecked() if hasattr(self, 'traffic_analysis') else True,
            'anomaly_detection': getattr(self, 'anomaly_detection', None) and self.anomaly_detection.isChecked() if hasattr(self, 'anomaly_detection') else False,
            'crowd_analysis': getattr(self, 'crowd_analysis', None) and self.crowd_analysis.isChecked() if hasattr(self, 'crowd_analysis') else False,
            'frame_skip': getattr(self, 'frame_skip_slider', None) and self.frame_skip_slider.value() if hasattr(self, 'frame_skip_slider') else 0,
            'batch_size': getattr(self, 'batch_size_spin', None) and self.batch_size_spin.value() if hasattr(self, 'batch_size_spin') else 1
        }
        
    def set_config(self, config):
        """Set configuration from dictionary"""
        try:
            # Handle nested config structure
            detection_config = config.get('detection', {})
            display_config = config.get('display', {})
            violations_config = config.get('violations', {})
            
            # Detection settings
            if 'device' in detection_config:
                device = detection_config['device']
                self.device_combo.setCurrentText(device)
                print(f"🔧 Config Panel: Set device to {device}")
                
            if 'model' in detection_config:
                model = detection_config['model']
                # Convert model format if needed (yolo11n -> YOLOv11n)
                if model and model.lower() != 'auto':
                    if 'yolo11' in model.lower():
                        if '11n' in model.lower():
                            model = 'YOLOv11n'
                        elif '11x' in model.lower():
                            model = 'YOLOv11x'
                        elif '11s' in model.lower():
                            model = 'YOLOv11s'
                        elif '11m' in model.lower():
                            model = 'YOLOv11m'
                        elif '11l' in model.lower():
                            model = 'YOLOv11l'
                    # Try to find and set the model in combo box
                    index = self.model_combo.findText(model)
                    if index >= 0:
                        self.model_combo.setCurrentIndex(index)
                        print(f"🔧 Config Panel: Set model to {model}")
                    else:
                        print(f"⚠️ Config Panel: Model {model} not found in combo box")
                        
            # Skip confidence and IOU threshold settings (removed from UI)
                
            # Display settings
            if 'show_confidence' in display_config:
                self.show_confidence.setChecked(display_config['show_confidence'])
            if 'show_labels' in display_config:
                self.show_labels.setChecked(display_config['show_labels'])
            if 'show_performance' in display_config:
                self.show_fps.setChecked(display_config['show_performance'])
            if 'show_boxes' in display_config:
                self.show_boxes.setChecked(display_config['show_boxes'])
                
            # Handle flat config structure for violations
            if 'red_light_detection' in config:
                self.red_light_detection.setChecked(config['red_light_detection'])
            if 'speed_violations' in config:
                self.speed_violations.setChecked(config['speed_violations'])
            if 'wrong_way_detection' in config:
                self.wrong_way_detection.setChecked(config['wrong_way_detection'])
            if 'crosswalk_violations' in config:
                self.crosswalk_violations.setChecked(config['crosswalk_violations'])
            if 'sound_alerts' in config:
                self.sound_alerts.setChecked(config['sound_alerts'])
            if 'email_notifications' in config:
                self.email_notifications.setChecked(config['email_notifications'])
            if 'auto_screenshot' in config:
                self.auto_screenshot.setChecked(config['auto_screenshot'])
            if 'enable_vlm' in config:
                self.enable_vlm.setChecked(config['enable_vlm'])
            if 'traffic_analysis' in config:
                self.traffic_analysis.setChecked(config['traffic_analysis'])
            if 'anomaly_detection' in config:
                self.anomaly_detection.setChecked(config['anomaly_detection'])
            if 'crowd_analysis' in config:
                self.crowd_analysis.setChecked(config['crowd_analysis'])
                
            # Performance settings
            if 'frame_skip' in config:
                self.frame_skip_slider.setValue(config['frame_skip'])
            if 'batch_size' in config:
                self.batch_size_spin.setValue(config['batch_size'])
                
            # Violations settings (nested structure)
            if 'enable_red_light' in violations_config:
                self.red_light_detection.setChecked(violations_config['enable_red_light'])
                
            print("✅ Config Panel: Configuration loaded successfully")
        except Exception as e:
            print(f"❌ Error setting config in panel: {e}")
            import traceback
            traceback.print_exc()
            
    @Slot()
    def apply_config(self):
        """Apply current configuration"""
        config = self.get_config()
        print(f"🔧 Config Panel: Applying config: {config}")
        self.config_changed.emit(config)
        
    @Slot()
    def reset_config(self):
        """Reset configuration to defaults"""
        try:
            # Reset to default values
            self.device_combo.setCurrentText("CPU")
            # Skip confidence and IOU sliders (removed from UI)
            self.model_combo.setCurrentIndex(0)
            self.input_size_combo.setCurrentText("640")
            
            # Display settings
            self.show_boxes.setChecked(True)
            self.show_labels.setChecked(True)
            self.show_confidence.setChecked(True)
            self.show_fps.setChecked(True)
            self.thickness_slider.setValue(2)
            self.font_slider.setValue(14)
            
            # Violations settings
            self.red_light_detection.setChecked(True)
            self.speed_violations.setChecked(False)
            self.wrong_way_detection.setChecked(False)
            self.crosswalk_violations.setChecked(False)
            
            # Alert settings
            self.sound_alerts.setChecked(True)
            self.email_notifications.setChecked(False)
            self.auto_screenshot.setChecked(True)
            
            # AI settings
            self.enable_vlm.setChecked(False)
            self.traffic_analysis.setChecked(True)
            self.anomaly_detection.setChecked(False)
            self.crowd_analysis.setChecked(False)
            self.frame_skip_slider.setValue(0)
            self.batch_size_spin.setValue(1)
            
            print("Configuration reset to defaults")
        except Exception as e:
            print(f"Error resetting config: {e}")
            
    @Slot(dict)
    def update_devices_info(self, device_info):
        """Update device information in the config panel"""
        try:
            # Update device combo with available devices
            available_devices = device_info.get('available_devices', ['CPU'])
            current_device = self.device_combo.currentText()
            
            # Clear and repopulate device combo
            self.device_combo.clear()
            self.device_combo.addItems(available_devices)
            
            # Restore previous selection if available
            if current_device in available_devices:
                self.device_combo.setCurrentText(current_device)
            else:
                # Default to first available device
                if available_devices:
                    self.device_combo.setCurrentText(available_devices[0])
                    
            print(f"[CONFIG PANEL] Updated available devices: {available_devices}")
        except Exception as e:
            print(f"[CONFIG PANEL] Error updating device info: {e}")
            
    @Slot(str)
    def update_status(self, status_message):
        """Update status message (placeholder for compatibility)"""
        print(f"[CONFIG PANEL] Status: {status_message}")
        
    @Slot(dict)
    def update_model_info(self, model_info):
        """Update model information in the config panel"""
        try:
            # Update model combo if models are provided
            if 'available_models' in model_info:
                current_model = self.model_combo.currentText()
                self.model_combo.clear()
                self.model_combo.addItems(model_info['available_models'])
                
                # Restore previous selection if available
                if current_model in model_info['available_models']:
                    self.model_combo.setCurrentText(current_model)
                    
            print(f"[CONFIG PANEL] Updated model info: {model_info}")
        except Exception as e:
            print(f"[CONFIG PANEL] Error updating model info: {e}")
