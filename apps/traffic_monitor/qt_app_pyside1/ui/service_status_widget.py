"""
Service Status Widget - Monitor MQTT + InfluxDB + Grafana status
Shows real-time connection status and provides quick access to services
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame,
    QProgressBar, QTextEdit, QTabWidget, QGroupBox, QScrollArea,
    QGridLayout, QSizePolicy
)
from PySide6.QtCore import Signal, QTimer, Qt, QUrl
from PySide6.QtGui import QIcon, QFont, QColor, QPalette
import json
import webbrowser
from datetime import datetime
from pathlib import Path


class ServiceStatusIndicator(QFrame):
    """Individual service status indicator"""
    
    def __init__(self, service_name: str, service_url: str = "", parent=None):
        super().__init__(parent)
        self.service_name = service_name
        self.service_url = service_url
        self.is_connected = False
        
        self.setFixedSize(200, 80)
        self.setFrameStyle(QFrame.Box)
        self.setLineWidth(2)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        
        # Service name label
        self.name_label = QLabel(service_name)
        font = QFont()
        font.setBold(True)
        font.setPointSize(9)
        self.name_label.setFont(font)
        self.name_label.setAlignment(Qt.AlignCenter)
        
        # Status indicator
        self.status_label = QLabel("🔴 Disconnected")
        self.status_label.setAlignment(Qt.AlignCenter)
        
        # Access button (if URL provided)
        if service_url:
            self.access_button = QPushButton("Open")
            self.access_button.setMaximumHeight(25)
            self.access_button.clicked.connect(self._open_service)
            layout.addWidget(self.access_button)
        
        layout.addWidget(self.name_label)
        layout.addWidget(self.status_label)
        
        self._update_style()
    
    def set_connected(self, connected: bool):
        """Update connection status"""
        self.is_connected = connected
        
        if connected:
            self.status_label.setText("🟢 Connected")
        else:
            self.status_label.setText("🔴 Disconnected")
        
        self._update_style()
    
    def _update_style(self):
        """Update widget styling based on connection status"""
        if self.is_connected:
            self.setStyleSheet("""
                QFrame {
                    border: 2px solid #4CAF50;
                    border-radius: 8px;
                    background-color: #E8F5E8;
                }
                QLabel {
                    color: #2E7D32;
                }
            """)
        else:
            self.setStyleSheet("""
                QFrame {
                    border: 2px solid #F44336;
                    border-radius: 8px;
                    background-color: #FFEBEE;
                }
                QLabel {
                    color: #C62828;
                }
            """)
    
    def _open_service(self):
        """Open service URL in browser"""
        if self.service_url:
            webbrowser.open(self.service_url)


class ServiceMetricsWidget(QWidget):
    """Widget showing service metrics and statistics"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.metrics_data = {}
    
    def setup_ui(self):
        """Setup the metrics UI"""
        layout = QVBoxLayout(self)
        
        # Metrics display
        self.metrics_group = QGroupBox("Service Metrics")
        metrics_layout = QGridLayout(self.metrics_group)
        
        # MQTT metrics
        mqtt_group = QGroupBox("MQTT Broker")
        mqtt_layout = QVBoxLayout(mqtt_group)
        
        self.mqtt_messages_label = QLabel("Messages Published: 0")
        self.mqtt_topics_label = QLabel("Active Topics: 0")
        self.mqtt_uptime_label = QLabel("Uptime: 0 min")
        
        mqtt_layout.addWidget(self.mqtt_messages_label)
        mqtt_layout.addWidget(self.mqtt_topics_label)
        mqtt_layout.addWidget(self.mqtt_uptime_label)
        
        # InfluxDB metrics
        influx_group = QGroupBox("InfluxDB")
        influx_layout = QVBoxLayout(influx_group)
        
        self.influx_points_label = QLabel("Data Points: 0")
        self.influx_series_label = QLabel("Series: 0")
        self.influx_size_label = QLabel("Database Size: 0 MB")
        
        influx_layout.addWidget(self.influx_points_label)
        influx_layout.addWidget(self.influx_series_label)
        influx_layout.addWidget(self.influx_size_label)
        
        # Add to grid
        metrics_layout.addWidget(mqtt_group, 0, 0)
        metrics_layout.addWidget(influx_group, 0, 1)
        
        layout.addWidget(self.metrics_group)
    
    def update_metrics(self, metrics: dict):
        """Update displayed metrics"""
        self.metrics_data = metrics
        
        # Update MQTT metrics
        mqtt_data = metrics.get("mqtt", {})
        self.mqtt_messages_label.setText(f"Messages Published: {mqtt_data.get('messages_published', 0):,}")
        self.mqtt_topics_label.setText(f"Active Topics: {mqtt_data.get('active_topics', 0)}")
        self.mqtt_uptime_label.setText(f"Uptime: {mqtt_data.get('uptime_minutes', 0)} min")
        
        # Update InfluxDB metrics
        influx_data = metrics.get("influxdb", {})
        self.influx_points_label.setText(f"Data Points: {influx_data.get('total_points', 0):,}")
        self.influx_series_label.setText(f"Series: {influx_data.get('series_count', 0):,}")
        self.influx_size_label.setText(f"Database Size: {influx_data.get('db_size_mb', 0):.1f} MB")


class ServiceLogViewer(QWidget):
    """Widget for viewing service logs"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        """Setup log viewer UI"""
        layout = QVBoxLayout(self)
        
        # Log display
        self.log_display = QTextEdit()
        self.log_display.setMaximumHeight(200)
        self.log_display.setFont(QFont("Consolas", 9))
        self.log_display.setStyleSheet("""
            QTextEdit {
                background-color: #2b2b2b;
                color: #ffffff;
                border: 1px solid #555;
            }
        """)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        self.clear_button = QPushButton("Clear Logs")
        self.clear_button.clicked.connect(self.clear_logs)
        
        self.auto_scroll_button = QPushButton("Auto Scroll: ON")
        self.auto_scroll_button.setCheckable(True)
        self.auto_scroll_button.setChecked(True)
        self.auto_scroll_button.clicked.connect(self._toggle_auto_scroll)
        
        controls_layout.addWidget(self.clear_button)
        controls_layout.addWidget(self.auto_scroll_button)
        controls_layout.addStretch()
        
        layout.addWidget(self.log_display)
        layout.addLayout(controls_layout)
    
    def add_log_entry(self, timestamp: str, service: str, level: str, message: str):
        """Add a log entry"""
        color_map = {
            "INFO": "#00FF00",
            "WARNING": "#FFFF00", 
            "ERROR": "#FF0000",
            "DEBUG": "#CCCCCC"
        }
        
        color = color_map.get(level, "#FFFFFF")
        
        log_line = f'<span style="color: #888">[{timestamp}]</span> ' \
                  f'<span style="color: #0088FF">[{service}]</span> ' \
                  f'<span style="color: {color}">[{level}]</span> ' \
                  f'<span style="color: #FFFFFF">{message}</span>'
        
        self.log_display.append(log_line)
        
        # Auto scroll if enabled
        if self.auto_scroll_button.isChecked():
            scrollbar = self.log_display.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
    
    def clear_logs(self):
        """Clear log display"""
        self.log_display.clear()
    
    def _toggle_auto_scroll(self, checked: bool):
        """Toggle auto scroll"""
        self.auto_scroll_button.setText(f"Auto Scroll: {'ON' if checked else 'OFF'}")


class ServiceStatusWidget(QWidget):
    """Complete service status monitoring widget"""
    
    # Signals
    open_grafana_requested = Signal()
    restart_services_requested = Signal()
    configure_services_requested = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.service_indicators = {}
        self.setup_ui()
        
        # Status update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_status)
        self.update_timer.start(5000)  # Update every 5 seconds
    
    def setup_ui(self):
        """Setup the status widget UI"""
        layout = QVBoxLayout(self)
        
        # Header
        header_layout = QHBoxLayout()
        
        header_label = QLabel("🔧 Services Status")
        header_font = QFont()
        header_font.setBold(True)
        header_font.setPointSize(12)
        header_label.setFont(header_font)
        
        # Control buttons
        self.grafana_button = QPushButton("📊 Open Grafana")
        self.grafana_button.clicked.connect(self.open_grafana_requested.emit)
        
        self.restart_button = QPushButton("🔄 Restart Services")
        self.restart_button.clicked.connect(self.restart_services_requested.emit)
        
        self.config_button = QPushButton("⚙️ Configure")
        self.config_button.clicked.connect(self.configure_services_requested.emit)
        
        header_layout.addWidget(header_label)
        header_layout.addStretch()
        header_layout.addWidget(self.grafana_button)
        header_layout.addWidget(self.restart_button)
        header_layout.addWidget(self.config_button)
        
        layout.addLayout(header_layout)
        
        # Service indicators
        indicators_group = QGroupBox("Service Status")
        indicators_layout = QHBoxLayout(indicators_group)
        
        # Create service indicators
        services = [
            ("MQTT Broker", ""),
            ("InfluxDB", "http://localhost:8086"),
            ("Grafana", "http://localhost:3000")
        ]
        
        for service_name, service_url in services:
            indicator = ServiceStatusIndicator(service_name, service_url)
            self.service_indicators[service_name.lower().replace(" ", "_")] = indicator
            indicators_layout.addWidget(indicator)
        
        indicators_layout.addStretch()
        layout.addWidget(indicators_group)
        
        # Tabs for detailed information
        self.tabs = QTabWidget()
        
        # Metrics tab
        self.metrics_widget = ServiceMetricsWidget()
        self.tabs.addTab(self.metrics_widget, "📊 Metrics")
        
        # Logs tab
        self.log_viewer = ServiceLogViewer()
        self.tabs.addTab(self.log_viewer, "📋 Logs")
        
        layout.addWidget(self.tabs)
        
        # Overall status
        self.overall_status = QLabel("🔄 Checking services...")
        self.overall_status.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.overall_status)
    
    def update_service_status(self, service_name: str, connected: bool):
        """Update individual service status"""
        indicator_key = service_name.lower().replace(" ", "_")
        if indicator_key in self.service_indicators:
            self.service_indicators[indicator_key].set_connected(connected)
        
        # Update overall status
        self._update_overall_status()
        
        # Add log entry
        status_text = "Connected" if connected else "Disconnected"
        level = "INFO" if connected else "ERROR"
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_viewer.add_log_entry(timestamp, service_name, level, f"Status: {status_text}")
    
    def update_metrics(self, metrics: dict):
        """Update service metrics"""
        self.metrics_widget.update_metrics(metrics)
    
    def _update_overall_status(self):
        """Update overall status based on all services"""
        connected_services = sum(1 for indicator in self.service_indicators.values() 
                               if indicator.is_connected)
        total_services = len(self.service_indicators)
        
        if connected_services == total_services:
            self.overall_status.setText("🟢 All services operational")
            self.overall_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
        elif connected_services > 0:
            self.overall_status.setText(f"🟡 {connected_services}/{total_services} services operational")
            self.overall_status.setStyleSheet("color: #FF9800; font-weight: bold;")
        else:
            self.overall_status.setText("🔴 No services connected")
            self.overall_status.setStyleSheet("color: #F44336; font-weight: bold;")
    
    def _update_status(self):
        """Periodic status update"""
        # This would be called by the enhanced controller
        # For now, just update timestamp in logs
        pass
    
    def add_service_log(self, service: str, level: str, message: str):
        """Add a service log entry"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_viewer.add_log_entry(timestamp, service, level, message)
