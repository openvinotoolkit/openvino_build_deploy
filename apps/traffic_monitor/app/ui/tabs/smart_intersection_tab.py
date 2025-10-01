"""
Smart Intersection Tab - IoT integration and traffic control management
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                               QGroupBox, QLabel, QPushButton, QComboBox,
                               QSlider, QProgressBar, QFrame, QSplitter,
                               QScrollArea, QCheckBox, QSpinBox, QTabWidget,
                               QTableWidget, QTableWidgetItem, QTextEdit,
                               QDateTimeEdit, QHeaderView)
from PySide6.QtCore import Qt, Signal, QTimer, QDateTime
from PySide6.QtGui import QFont, QColor, QPainter, QPen, QBrush
import json

class TrafficLightController(QWidget):
    """Traffic light control widget"""
    
    light_changed = Signal(str, str)  # intersection_id, new_state
    
    def __init__(self, intersection_id="intersection_1", parent=None):
        super().__init__(parent)
        
        self.intersection_id = intersection_id
        self.current_state = "auto"
        self.current_phase = "green_ns"  # north-south green
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup traffic light control UI"""
        layout = QVBoxLayout(self)
        
        # Header
        header = QLabel(f"🚦 {self.intersection_id.replace('_', ' ').title()}")
        header.setFont(QFont("Segoe UI", 10, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)
        
        # Traffic light visualization
        lights_frame = QFrame()
        lights_frame.setFixedSize(120, 200)
        lights_frame.setStyleSheet("""
            QFrame {
                background-color: #2c3e50;
                border: 2px solid #34495e;
                border-radius: 8px;
            }
        """)
        
        lights_layout = QVBoxLayout(lights_frame)
        lights_layout.setSpacing(5)
        lights_layout.setContentsMargins(10, 10, 10, 10)
        
        # North-South lights
        ns_label = QLabel("N-S")
        ns_label.setStyleSheet("color: white; font-size: 8pt;")
        ns_label.setAlignment(Qt.AlignCenter)
        lights_layout.addWidget(ns_label)
        
        self.ns_red = self._create_light_indicator("red", False)
        self.ns_yellow = self._create_light_indicator("yellow", False)
        self.ns_green = self._create_light_indicator("green", True)
        
        lights_layout.addWidget(self.ns_red)
        lights_layout.addWidget(self.ns_yellow)
        lights_layout.addWidget(self.ns_green)
        
        # East-West lights
        ew_label = QLabel("E-W")
        ew_label.setStyleSheet("color: white; font-size: 8pt;")
        ew_label.setAlignment(Qt.AlignCenter)
        lights_layout.addWidget(ew_label)
        
        self.ew_red = self._create_light_indicator("red", True)
        self.ew_yellow = self._create_light_indicator("yellow", False)
        self.ew_green = self._create_light_indicator("green", False)
        
        lights_layout.addWidget(self.ew_red)
        lights_layout.addWidget(self.ew_yellow)
        lights_layout.addWidget(self.ew_green)
        
        layout.addWidget(lights_frame, 0, Qt.AlignCenter)
        
        # Control buttons
        controls_layout = QVBoxLayout()
        
        self.auto_btn = QPushButton("🤖 Auto")
        self.auto_btn.setCheckable(True)
        self.auto_btn.setChecked(True)
        self.auto_btn.clicked.connect(lambda: self._set_mode("auto"))
        controls_layout.addWidget(self.auto_btn)
        
        self.manual_btn = QPushButton("👤 Manual")
        self.manual_btn.setCheckable(True)
        self.manual_btn.clicked.connect(lambda: self._set_mode("manual"))
        controls_layout.addWidget(self.manual_btn)
        
        self.emergency_btn = QPushButton("🚨 Emergency")
        self.emergency_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        self.emergency_btn.clicked.connect(lambda: self._set_mode("emergency"))
        controls_layout.addWidget(self.emergency_btn)
        
        layout.addLayout(controls_layout)
        
        # Status
        self.status_label = QLabel("Status: Auto Mode")
        self.status_label.setStyleSheet("font-size: 8pt; color: #27ae60;")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
    
    def _create_light_indicator(self, color, active=False):
        """Create a traffic light indicator"""
        light = QLabel()
        light.setFixedSize(20, 20)
        light.setStyleSheet(f"""
            QLabel {{
                background-color: {'#' + color if active else '#2c3e50'};
                border: 1px solid #34495e;
                border-radius: 10px;
                opacity: {'1.0' if active else '0.3'};
            }}
        """)
        return light
    
    def _set_mode(self, mode):
        """Set traffic light control mode"""
        self.current_state = mode
        
        # Update button states
        self.auto_btn.setChecked(mode == "auto")
        self.manual_btn.setChecked(mode == "manual")
        
        # Update status
        status_colors = {
            "auto": "#27ae60",
            "manual": "#f39c12", 
            "emergency": "#e74c3c"
        }
        
        self.status_label.setText(f"Status: {mode.title()} Mode")
        self.status_label.setStyleSheet(f"font-size: 8pt; color: {status_colors.get(mode, '#95a5a6')};")
        
        # Emit signal
        self.light_changed.emit(self.intersection_id, mode)
        
        print(f"🚦 {self.intersection_id} set to {mode} mode")

class IoTDeviceWidget(QFrame):
    """IoT device status widget"""
    
    def __init__(self, device_id, device_type, status="online", parent=None):
        super().__init__(parent)
        
        self.device_id = device_id
        self.device_type = device_type
        self.status = status
        
        self.setFixedSize(150, 80)
        self._setup_ui()
        self._apply_style()
    
    def _setup_ui(self):
        """Setup IoT device widget UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        
        # Device header
        header_layout = QHBoxLayout()
        
        # Device icon
        icons = {
            "camera": "📷",
            "sensor": "📡",
            "controller": "🎛️",
            "display": "📺",
            "gateway": "🌐"
        }
        
        icon_label = QLabel(icons.get(self.device_type, "📟"))
        icon_label.setFont(QFont("Arial", 14))
        header_layout.addWidget(icon_label)
        
        # Device name
        name_label = QLabel(self.device_id.replace('_', ' ').title())
        name_label.setFont(QFont("Segoe UI", 8, QFont.Bold))
        header_layout.addWidget(name_label)
        
        header_layout.addStretch()
        
        # Status indicator
        self.status_indicator = QLabel("●")
        self.status_indicator.setFont(QFont("Arial", 10))
        header_layout.addWidget(self.status_indicator)
        
        layout.addLayout(header_layout)
        
        # Device info
        info_layout = QVBoxLayout()
        
        type_label = QLabel(f"Type: {self.device_type.title()}")
        type_label.setFont(QFont("Segoe UI", 7))
        info_layout.addWidget(type_label)
        
        self.status_label = QLabel(f"Status: {self.status.title()}")
        self.status_label.setFont(QFont("Segoe UI", 7))
        info_layout.addWidget(self.status_label)
        
        layout.addLayout(info_layout)
        
        self._update_status_display()
    
    def _apply_style(self):
        """Apply device widget styling"""
        self.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 1px solid #e1e8ed;
                border-radius: 6px;
                margin: 2px;
            }
            QFrame:hover {
                border-color: #3498db;
            }
        """)
    
    def _update_status_display(self):
        """Update status indicator colors"""
        colors = {
            "online": "#27ae60",
            "offline": "#e74c3c",
            "warning": "#f39c12",
            "error": "#c0392b"
        }
        
        color = colors.get(self.status, "#95a5a6")
        self.status_indicator.setStyleSheet(f"color: {color};")
        self.status_label.setText(f"Status: {self.status.title()}")
    
    def set_status(self, status):
        """Update device status"""
        self.status = status
        self._update_status_display()

class SmartIntersectionTab(QWidget):
    """
    Smart Intersection Tab for IoT integration and traffic control
    
    Features:
    - Traffic light control and monitoring
    - IoT device management
    - Real-time traffic flow optimization
    - Emergency response coordination
    - Data analytics and reporting
    - System integration dashboard
    """
    
    # Signals
    traffic_control_changed = Signal(dict)
    iot_device_updated = Signal(str, str)
    emergency_activated = Signal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.iot_devices = {}
        self.traffic_controllers = {}
        self.intersection_data = {}
        
        self._setup_ui()
        
        # Timer for real-time updates
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_intersection_data)
        self.update_timer.start(2000)  # Update every 2 seconds
        
        print("🌉 Smart Intersection Tab initialized")
    
    def _setup_ui(self):
        """Setup the smart intersection UI"""
        layout = QVBoxLayout(self)
        
        # Header with system overview
        header = self._create_header()
        layout.addWidget(header)
        
        # Main content splitter
        main_splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(main_splitter)
        
        # Left panel - Traffic control
        left_panel = self._create_traffic_control_panel()
        main_splitter.addWidget(left_panel)
        
        # Right panel - IoT devices and analytics
        right_panel = self._create_iot_panel()
        main_splitter.addWidget(right_panel)
        
        # Set splitter proportions
        main_splitter.setSizes([500, 500])
    
    def _create_header(self):
        """Create header with intersection overview"""
        header = QFrame()
        header.setFixedHeight(80)
        header.setStyleSheet("""
            QFrame {
                background-color: #16a085;
                border-radius: 8px;
                margin-bottom: 10px;
            }
        """)
        
        layout = QHBoxLayout(header)
        layout.setContentsMargins(20, 10, 20, 10)
        
        # Title section
        title_layout = QVBoxLayout()
        
        title = QLabel("🌉 Smart Intersection Control Center")
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        title.setStyleSheet("color: white;")
        title_layout.addWidget(title)
        
        subtitle = QLabel("Real-time traffic optimization and IoT device management")
        subtitle.setFont(QFont("Segoe UI", 9))
        subtitle.setStyleSheet("color: #ecf0f1;")
        title_layout.addWidget(subtitle)
        
        layout.addLayout(title_layout)
        
        layout.addStretch()
        
        # Statistics cards
        stats_layout = QHBoxLayout()
        
        # Active intersections
        intersections_card = self._create_header_card("Active Intersections", "4", "#27ae60")
        stats_layout.addWidget(intersections_card)
        
        # IoT devices
        devices_card = self._create_header_card("IoT Devices", "12", "#3498db")
        stats_layout.addWidget(devices_card)
        
        # Traffic efficiency
        efficiency_card = self._create_header_card("Traffic Efficiency", "87%", "#f39c12")
        stats_layout.addWidget(efficiency_card)
        
        layout.addLayout(stats_layout)
        
        return header
    
    def _create_header_card(self, title, value, color):
        """Create a header statistics card"""
        card = QFrame()
        card.setFixedSize(120, 50)
        card.setStyleSheet(f"""
            QFrame {{
                background-color: {color};
                border-radius: 6px;
                margin: 2px;
            }}
        """)
        
        layout = QVBoxLayout(card)
        layout.setContentsMargins(8, 4, 8, 4)
        
        value_label = QLabel(value)
        value_label.setFont(QFont("Segoe UI", 14, QFont.Bold))
        value_label.setStyleSheet("color: white;")
        value_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(value_label)
        
        title_label = QLabel(title)
        title_label.setFont(QFont("Segoe UI", 7))
        title_label.setStyleSheet("color: white;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        return card
    
    def _create_traffic_control_panel(self):
        """Create traffic control panel"""
        panel = QFrame()
        layout = QVBoxLayout(panel)
        
        # Traffic control section
        control_group = QGroupBox("Traffic Light Control")
        control_layout = QVBoxLayout(control_group)
        
        # Intersection controllers grid
        controllers_scroll = QScrollArea()
        controllers_widget = QWidget()
        controllers_layout = QGridLayout(controllers_widget)
        
        # Create traffic light controllers
        intersections = [
            "main_oak", "5th_pine", "broadway_2nd", "market_1st"
        ]
        
        for i, intersection_id in enumerate(intersections):
            controller = TrafficLightController(intersection_id)
            controller.light_changed.connect(self._on_traffic_control_changed)
            self.traffic_controllers[intersection_id] = controller
            controllers_layout.addWidget(controller, i // 2, i % 2)
        
        controllers_scroll.setWidget(controllers_widget)
        controllers_scroll.setMaximumHeight(450)
        control_layout.addWidget(controllers_scroll)
        
        layout.addWidget(control_group)
        
        # Global controls
        global_controls = self._create_global_controls()
        layout.addWidget(global_controls)
        
        return panel
    
    def _create_global_controls(self):
        """Create global traffic control section"""
        section = QGroupBox("Global Controls")
        layout = QVBoxLayout(section)
        
        # Emergency controls
        emergency_layout = QHBoxLayout()
        
        emergency_all_btn = QPushButton("🚨 Emergency All")
        emergency_all_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                font-weight: bold;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        emergency_all_btn.clicked.connect(self._emergency_all_intersections)
        emergency_layout.addWidget(emergency_all_btn)
        
        clear_emergency_btn = QPushButton("✅ Clear Emergency")
        clear_emergency_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                font-weight: bold;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #229954;
            }
        """)
        clear_emergency_btn.clicked.connect(self._clear_emergency_all)
        emergency_layout.addWidget(clear_emergency_btn)
        
        layout.addLayout(emergency_layout)
        
        # Traffic optimization
        optimization_layout = QHBoxLayout()
        
        self.adaptive_cb = QCheckBox("Adaptive Traffic Control")
        self.adaptive_cb.setChecked(True)
        self.adaptive_cb.toggled.connect(self._toggle_adaptive_control)
        optimization_layout.addWidget(self.adaptive_cb)
        
        optimize_btn = QPushButton("🔄 Optimize Flow")
        optimize_btn.clicked.connect(self._optimize_traffic_flow)
        optimization_layout.addWidget(optimize_btn)
        
        layout.addLayout(optimization_layout)
        
        return section
    
    def _create_iot_panel(self):
        """Create IoT devices and analytics panel"""
        panel = QFrame()
        layout = QVBoxLayout(panel)
        
        # IoT devices section
        iot_group = QGroupBox("IoT Device Network")
        iot_layout = QVBoxLayout(iot_group)
        
        # Device grid
        devices_scroll = QScrollArea()
        devices_widget = QWidget()
        self.devices_layout = QGridLayout(devices_widget)
        
        # Create sample IoT devices
        devices_data = [
            ("camera_001", "camera", "online"),
            ("sensor_002", "sensor", "online"),
            ("controller_003", "controller", "online"),
            ("display_004", "display", "warning"),
            ("gateway_005", "gateway", "online"),
            ("sensor_006", "sensor", "offline"),
            ("camera_007", "camera", "online"),
            ("controller_008", "controller", "error"),
        ]
        
        for i, (device_id, device_type, status) in enumerate(devices_data):
            device_widget = IoTDeviceWidget(device_id, device_type, status)
            self.iot_devices[device_id] = device_widget
            self.devices_layout.addWidget(device_widget, i // 3, i % 3)
        
        devices_scroll.setWidget(devices_widget)
        devices_scroll.setMaximumHeight(200)
        iot_layout.addWidget(devices_scroll)
        
        # Device controls
        device_controls = QHBoxLayout()
        
        refresh_devices_btn = QPushButton("🔄 Refresh")
        refresh_devices_btn.clicked.connect(self._refresh_devices)
        device_controls.addWidget(refresh_devices_btn)
        
        add_device_btn = QPushButton("➕ Add Device")
        add_device_btn.clicked.connect(self._add_device)
        device_controls.addWidget(add_device_btn)
        
        device_controls.addStretch()
        
        iot_layout.addLayout(device_controls)
        
        layout.addWidget(iot_group)
        
        # Analytics section
        analytics_group = self._create_analytics_section()
        layout.addWidget(analytics_group)
        
        return panel
    
    def _create_analytics_section(self):
        """Create analytics and reporting section"""
        section = QGroupBox("Traffic Analytics")
        layout = QVBoxLayout(section)
        
        # Analytics tabs
        analytics_tabs = QTabWidget()
        
        # Real-time tab
        realtime_tab = QWidget()
        realtime_layout = QVBoxLayout(realtime_tab)
        
        # Real-time metrics
        metrics_layout = QGridLayout()
        
        metrics_layout.addWidget(QLabel("Current Traffic Volume:"), 0, 0)
        self.volume_label = QLabel("Medium")
        metrics_layout.addWidget(self.volume_label, 0, 1)
        
        metrics_layout.addWidget(QLabel("Average Wait Time:"), 1, 0)
        self.wait_time_label = QLabel("45 seconds")
        metrics_layout.addWidget(self.wait_time_label, 1, 1)
        
        metrics_layout.addWidget(QLabel("Traffic Efficiency:"), 2, 0)
        self.efficiency_label = QLabel("87%")
        metrics_layout.addWidget(self.efficiency_label, 2, 1)
        
        metrics_layout.addWidget(QLabel("Active Violations:"), 3, 0)
        self.violations_label = QLabel("3")
        metrics_layout.addWidget(self.violations_label, 3, 1)
        
        realtime_layout.addLayout(metrics_layout)
        
        # Traffic flow visualization (placeholder)
        flow_frame = QFrame()
        flow_frame.setFixedHeight(100)
        flow_frame.setStyleSheet("""
            QFrame {
                background-color: #ecf0f1;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
            }
        """)
        
        flow_layout = QVBoxLayout(flow_frame)
        flow_label = QLabel("Traffic Flow Visualization\n(Real-time heatmap)")
        flow_label.setAlignment(Qt.AlignCenter)
        flow_label.setStyleSheet("color: #7f8c8d;")
        flow_layout.addWidget(flow_label)
        
        realtime_layout.addWidget(flow_frame)
        
        analytics_tabs.addTab(realtime_tab, "📊 Real-time")
        
        # Historical tab
        historical_tab = QWidget()
        historical_layout = QVBoxLayout(historical_tab)
        
        # Time range selector
        range_layout = QHBoxLayout()
        range_layout.addWidget(QLabel("Time Range:"))
        
        time_range_combo = QComboBox()
        time_range_combo.addItems(["Last Hour", "Last 24 Hours", "Last Week", "Last Month"])
        range_layout.addWidget(time_range_combo)
        
        range_layout.addStretch()
        
        export_btn = QPushButton("📤 Export Report")
        export_btn.clicked.connect(self._export_analytics)
        range_layout.addWidget(export_btn)
        
        historical_layout.addLayout(range_layout)
        
        # Historical data display (placeholder)
        historical_frame = QFrame()
        historical_frame.setFixedHeight(120)
        historical_frame.setStyleSheet("""
            QFrame {
                background-color: #ecf0f1;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
            }
        """)
        
        historical_layout_frame = QVBoxLayout(historical_frame)
        historical_label = QLabel("Historical Traffic Patterns\n(Charts and trends)")
        historical_label.setAlignment(Qt.AlignCenter)
        historical_label.setStyleSheet("color: #7f8c8d;")
        historical_layout_frame.addWidget(historical_label)
        
        historical_layout.addWidget(historical_frame)
        
        analytics_tabs.addTab(historical_tab, "📈 Historical")
        
        layout.addWidget(analytics_tabs)
        
        return section
    
    def _on_traffic_control_changed(self, intersection_id, mode):
        """Handle traffic control changes"""
        control_data = {
            'intersection': intersection_id,
            'mode': mode,
            'timestamp': QDateTime.currentDateTime().toString()
        }
        
        self.traffic_control_changed.emit(control_data)
        print(f"🌉 Traffic control changed: {intersection_id} -> {mode}")
    
    def _emergency_all_intersections(self):
        """Activate emergency mode for all intersections"""
        for controller in self.traffic_controllers.values():
            controller._set_mode("emergency")
        
        self.emergency_activated.emit("all_intersections")
        print("🌉 Emergency mode activated for all intersections")
    
    def _clear_emergency_all(self):
        """Clear emergency mode for all intersections"""
        for controller in self.traffic_controllers.values():
            controller._set_mode("auto")
        
        print("🌉 Emergency mode cleared for all intersections")
    
    def _toggle_adaptive_control(self, enabled):
        """Toggle adaptive traffic control"""
        status = "enabled" if enabled else "disabled"
        print(f"🌉 Adaptive traffic control {status}")
    
    def _optimize_traffic_flow(self):
        """Optimize traffic flow patterns"""
        print("🌉 Optimizing traffic flow patterns")
        
        # Simulate optimization results
        self.efficiency_label.setText("92%")
        self.wait_time_label.setText("38 seconds")
    
    def _refresh_devices(self):
        """Refresh IoT device status"""
        import random
        
        statuses = ["online", "offline", "warning", "error"]
        
        for device in self.iot_devices.values():
            # Randomly update some device statuses
            if random.random() < 0.3:  # 30% chance to change status
                new_status = random.choice(statuses)
                device.set_status(new_status)
        
        print("🌉 IoT devices refreshed")
    
    def _add_device(self):
        """Add new IoT device"""
        # Simulate adding a new device
        device_count = len(self.iot_devices) + 1
        device_id = f"device_{device_count:03d}"
        device_type = "sensor"
        
        device_widget = IoTDeviceWidget(device_id, device_type, "online")
        self.iot_devices[device_id] = device_widget
        
        # Add to grid
        row = (len(self.iot_devices) - 1) // 3
        col = (len(self.iot_devices) - 1) % 3
        self.devices_layout.addWidget(device_widget, row, col)
        
        print(f"🌉 Added new IoT device: {device_id}")
    
    def _export_analytics(self):
        """Export traffic analytics"""
        print("🌉 Exporting traffic analytics")
    
    def _update_intersection_data(self):
        """Update real-time intersection data"""
        import random
        
        # Simulate real-time data updates
        volumes = ["Low", "Medium", "High"]
        self.volume_label.setText(random.choice(volumes))
        
        wait_time = random.randint(30, 60)
        self.wait_time_label.setText(f"{wait_time} seconds")
        
        efficiency = random.randint(80, 95)
        self.efficiency_label.setText(f"{efficiency}%")
        
        violations = random.randint(0, 5)
        self.violations_label.setText(str(violations))
    
    def set_intersection_mode(self, intersection_id, mode):
        """Set mode for specific intersection"""
        if intersection_id in self.traffic_controllers:
            self.traffic_controllers[intersection_id]._set_mode(mode)
    
    def get_system_status(self):
        """Get current system status"""
        online_devices = sum(1 for device in self.iot_devices.values() 
                           if device.status == "online")
        total_devices = len(self.iot_devices)
        
        return {
            'intersections': len(self.traffic_controllers),
            'devices_online': online_devices,
            'devices_total': total_devices,
            'traffic_efficiency': self.efficiency_label.text(),
            'adaptive_control': self.adaptive_cb.isChecked()
        }
