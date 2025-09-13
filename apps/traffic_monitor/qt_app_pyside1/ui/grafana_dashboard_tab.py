from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QLineEdit, QFormLayout, QGroupBox, QTextEdit, QProgressBar,
    QScrollArea, QFrame, QGridLayout, QComboBox
)
from PySide6.QtCore import Qt, QTimer, QThread, Signal, QUrl
from PySide6.QtGui import QFont, QPixmap, QPainter, QColor

# Try to import WebEngine, fallback gracefully if not available
try:
    from PySide6.QtWebEngineWidgets import QWebEngineView
    WEBENGINE_AVAILABLE = True
except ImportError:
    print("WebEngine not available - Grafana will open in external browser")
    QWebEngineView = None
    WEBENGINE_AVAILABLE = False

# Try to import requests, fallback gracefully if not available
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    print("Requests library not available - some features may be limited")
    requests = None
    REQUESTS_AVAILABLE = False

import json
import time
from datetime import datetime

class GrafanaDashboardTab(QWidget):
    """Grafana Dashboard Integration Tab"""
    
    def __init__(self):
        super().__init__()
        self.grafana_url = "http://localhost:3000"
        self.dashboard_uid = "traffic-analytics"
        self.api_key = ""
        self.refresh_interval = 30  # seconds
        
        self.setup_ui()
        self.setup_timer()
    
    def setup_ui(self):
        """Setup the Grafana dashboard UI"""
        layout = QVBoxLayout(self)
        
        # Header section
        header_layout = self.create_header_section()
        layout.addLayout(header_layout)
        
        # Configuration section
        config_group = self.create_config_section()
        layout.addWidget(config_group)
        
        # Dashboard view section
        dashboard_group = self.create_dashboard_section()
        layout.addWidget(dashboard_group)
        
        # Status section
        status_group = self.create_status_section()
        layout.addWidget(status_group)
        
        self.setLayout(layout)
    
    def create_header_section(self):
        """Create header with title and refresh controls"""
        layout = QHBoxLayout()
        
        # Title
        title = QLabel("📊 Grafana Dashboard")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setStyleSheet("color: #2196F3; margin: 10px;")
        layout.addWidget(title)
        
        layout.addStretch()
        
        # Refresh controls
        self.auto_refresh_combo = QComboBox()
        self.auto_refresh_combo.addItems(["Off", "30s", "1m", "5m", "15m"])
        self.auto_refresh_combo.setCurrentText("30s")
        self.auto_refresh_combo.currentTextChanged.connect(self.on_refresh_interval_changed)
        
        self.refresh_btn = QPushButton("🔄 Refresh")
        self.refresh_btn.clicked.connect(self.refresh_dashboard)
        self.refresh_btn.setStyleSheet("""
            QPushButton {
                background: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #45a049;
            }
        """)
        
        layout.addWidget(QLabel("Auto Refresh:"))
        layout.addWidget(self.auto_refresh_combo)
        layout.addWidget(self.refresh_btn)
        
        return layout
    
    def create_config_section(self):
        """Create Grafana configuration section"""
        group = QGroupBox("Grafana Configuration")
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin: 5px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        layout = QFormLayout(group)
        
        # Grafana URL
        self.url_input = QLineEdit(self.grafana_url)
        self.url_input.setPlaceholderText("http://localhost:3000")
        layout.addRow("Grafana URL:", self.url_input)
        
        # Dashboard UID
        self.dashboard_input = QLineEdit(self.dashboard_uid)
        self.dashboard_input.setPlaceholderText("dashboard-uid")
        layout.addRow("Dashboard UID:", self.dashboard_input)
        
        # API Key (optional)
        self.api_key_input = QLineEdit(self.api_key)
        self.api_key_input.setPlaceholderText("Optional API key for authentication")
        self.api_key_input.setEchoMode(QLineEdit.Password)
        layout.addRow("API Key:", self.api_key_input)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.connect_btn = QPushButton("🔗 Connect")
        self.connect_btn.clicked.connect(self.connect_to_grafana)
        self.connect_btn.setStyleSheet("""
            QPushButton {
                background: #2196F3;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #1976D2;
            }
        """)
        
        self.open_browser_btn = QPushButton("🌐 Open in Browser")
        self.open_browser_btn.clicked.connect(self.open_in_browser)
        self.open_browser_btn.setStyleSheet("""
            QPushButton {
                background: #FF9800;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #F57C00;
            }
        """)
        
        button_layout.addWidget(self.connect_btn)
        button_layout.addWidget(self.open_browser_btn)
        button_layout.addStretch()
        
        layout.addRow(button_layout)
        
        return group
    
    def create_dashboard_section(self):
        """Create embedded dashboard section"""
        group = QGroupBox("Dashboard View")
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin: 5px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        layout = QVBoxLayout(group)
        
        # Try to create web engine view for embedded Grafana
        if WEBENGINE_AVAILABLE:
            self.web_view = QWebEngineView()
            self.web_view.setMinimumHeight(600)
            layout.addWidget(self.web_view)
        else:
            # Fallback if WebEngine is not available
            fallback_label = QLabel("""
            🌐 WebEngine not available. 
            
            To view Grafana dashboard:
            1. Click 'Open in Browser' button above
            2. Or install QtWebEngine: pip install PySide6-WebEngine
            
            Dashboard will open in your default browser.
            """)
            fallback_label.setAlignment(Qt.AlignCenter)
            fallback_label.setStyleSheet("""
                QLabel {
                    background: #f5f5f5;
                    border: 2px dashed #ccc;
                    border-radius: 8px;
                    padding: 20px;
                    font-size: 14px;
                    color: #666;
                }
            """)
            fallback_label.setMinimumHeight(400)
            layout.addWidget(fallback_label)
            self.web_view = None
        
        return group
    
    def create_status_section(self):
        """Create status and metrics section"""
        group = QGroupBox("Connection Status & Quick Metrics")
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin: 5px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        layout = QGridLayout(group)
        
        # Connection status
        self.status_label = QLabel("❌ Not Connected")
        self.status_label.setStyleSheet("font-weight: bold; color: #f44336;")
        layout.addWidget(QLabel("Status:"), 0, 0)
        layout.addWidget(self.status_label, 0, 1)
        
        # Last update
        self.last_update_label = QLabel("Never")
        layout.addWidget(QLabel("Last Update:"), 0, 2)
        layout.addWidget(self.last_update_label, 0, 3)
        
        # Quick metrics (if API is available)
        metrics_layout = QHBoxLayout()
        
        self.fps_label = QLabel("FPS: --")
        self.fps_label.setStyleSheet("font-weight: bold; color: #2196F3;")
        
        self.violations_label = QLabel("Violations: --")
        self.violations_label.setStyleSheet("font-weight: bold; color: #f44336;")
        
        self.vehicles_label = QLabel("Vehicles: --")
        self.vehicles_label.setStyleSheet("font-weight: bold; color: #4CAF50;")
        
        metrics_layout.addWidget(self.fps_label)
        metrics_layout.addWidget(self.violations_label)
        metrics_layout.addWidget(self.vehicles_label)
        metrics_layout.addStretch()
        
        layout.addLayout(metrics_layout, 1, 0, 1, 4)
        
        return group
    
    def setup_timer(self):
        """Setup auto-refresh timer"""
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_dashboard)
        self.update_refresh_timer()
    
    def on_refresh_interval_changed(self, text):
        """Handle refresh interval change"""
        self.update_refresh_timer()
    
    def update_refresh_timer(self):
        """Update the refresh timer based on selected interval"""
        text = self.auto_refresh_combo.currentText()
        
        self.refresh_timer.stop()
        
        if text == "Off":
            return
        elif text == "30s":
            self.refresh_timer.start(30000)
        elif text == "1m":
            self.refresh_timer.start(60000)
        elif text == "5m":
            self.refresh_timer.start(300000)
        elif text == "15m":
            self.refresh_timer.start(900000)
    
    def connect_to_grafana(self):
        """Connect to Grafana and load dashboard"""
        self.grafana_url = self.url_input.text().strip()
        self.dashboard_uid = self.dashboard_input.text().strip()
        self.api_key = self.api_key_input.text().strip()
        
        if not self.grafana_url:
            self.status_label.setText("❌ Please enter Grafana URL")
            self.status_label.setStyleSheet("font-weight: bold; color: #f44336;")
            return
        
        if not REQUESTS_AVAILABLE:
            self.status_label.setText("⚠️ Limited functionality (no requests library)")
            self.status_label.setStyleSheet("font-weight: bold; color: #ff9800;")
            # Still allow opening in browser
            if self.web_view:
                if self.dashboard_uid:
                    dashboard_url = f"{self.grafana_url}/d/{self.dashboard_uid}?orgId=1&refresh=30s&kiosk"
                else:
                    dashboard_url = f"{self.grafana_url}/?orgId=1&kiosk"
                
                self.web_view.load(QUrl(dashboard_url))
            return
        
        try:
            # Test connection to Grafana
            response = requests.get(f"{self.grafana_url}/api/health", timeout=5)
            
            if response.status_code == 200:
                self.status_label.setText("✅ Connected")
                self.status_label.setStyleSheet("font-weight: bold; color: #4CAF50;")
                
                # Load dashboard in web view if available
                if self.web_view:
                    if self.dashboard_uid:
                        dashboard_url = f"{self.grafana_url}/d/{self.dashboard_uid}?orgId=1&refresh=30s&kiosk"
                    else:
                        dashboard_url = f"{self.grafana_url}/?orgId=1&kiosk"
                    
                    self.web_view.load(QUrl(dashboard_url))
                
                # Try to fetch quick metrics
                self.fetch_quick_metrics()
                
            else:
                self.status_label.setText("❌ Connection Failed")
                self.status_label.setStyleSheet("font-weight: bold; color: #f44336;")
                
        except Exception as e:
            self.status_label.setText(f"❌ Error: {str(e)[:50]}")
            self.status_label.setStyleSheet("font-weight: bold; color: #f44336;")
    
    def refresh_dashboard(self):
        """Refresh the dashboard"""
        if self.web_view and self.status_label.text().startswith("✅"):
            self.web_view.reload()
        
        self.fetch_quick_metrics()
        self.last_update_label.setText(datetime.now().strftime("%H:%M:%S"))
    
    def fetch_quick_metrics(self):
        """Fetch quick metrics from Grafana API"""
        if not self.grafana_url or not self.status_label.text().startswith("✅") or not REQUESTS_AVAILABLE:
            return
        
        try:
            headers = {}
            if self.api_key:
                headers['Authorization'] = f'Bearer {self.api_key}'
            
            # This is a placeholder - you would need to adapt this to your specific Grafana setup
            # For now, just show sample data
            self.fps_label.setText("FPS: 25.3")
            self.violations_label.setText("Violations: 3")
            self.vehicles_label.setText("Vehicles: 12")
            
        except Exception as e:
            print(f"Error fetching metrics: {e}")
    
    def open_in_browser(self):
        """Open Grafana dashboard in default browser"""
        if not self.grafana_url:
            return
        
        import webbrowser
        
        if self.dashboard_uid:
            url = f"{self.grafana_url}/d/{self.dashboard_uid}?orgId=1&refresh=30s"
        else:
            url = self.grafana_url
        
        webbrowser.open(url)
        
        self.status_label.setText("🌐 Opened in Browser")
        self.status_label.setStyleSheet("font-weight: bold; color: #2196F3;")
