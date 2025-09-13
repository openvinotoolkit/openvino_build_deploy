"""
Alert Widget - Displays system alerts and notifications
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                               QPushButton, QFrame, QScrollArea, QCheckBox)
from PySide6.QtCore import Qt, Signal, QTimer, QDateTime, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QFont, QColor, QPalette

class AlertItem(QFrame):
    """Individual alert item widget"""
    
    dismissed = Signal(str)  # alert_id
    action_triggered = Signal(str, str)  # alert_id, action
    
    def __init__(self, alert_id, alert_type, title, message, timestamp=None, parent=None):
        super().__init__(parent)
        
        self.alert_id = alert_id
        self.alert_type = alert_type
        self.timestamp = timestamp or QDateTime.currentDateTime()
        
        self._setup_ui(title, message)
        self._apply_style()
    
    def _setup_ui(self, title, message):
        """Setup alert item UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(4)
        
        # Header with icon, title, and dismiss button
        header_layout = QHBoxLayout()
        
        # Alert icon
        icons = {
            'error': '🚨',
            'warning': '⚠️',
            'info': 'ℹ️',
            'success': '✅',
            'critical': '💥'
        }
        
        icon_label = QLabel(icons.get(self.alert_type, 'ℹ️'))
        icon_label.setFont(QFont("Arial", 12))
        header_layout.addWidget(icon_label)
        
        # Title
        title_label = QLabel(title)
        title_label.setFont(QFont("Segoe UI", 9, QFont.Bold))
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        # Timestamp
        time_label = QLabel(self.timestamp.toString("hh:mm:ss"))
        time_label.setFont(QFont("Segoe UI", 8))
        time_label.setStyleSheet("color: #7f8c8d;")
        header_layout.addWidget(time_label)
        
        # Dismiss button
        dismiss_btn = QPushButton("✕")
        dismiss_btn.setFixedSize(20, 20)
        dismiss_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                color: #95a5a6;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #ecf0f1;
                border-radius: 10px;
            }
        """)
        dismiss_btn.clicked.connect(lambda: self.dismissed.emit(self.alert_id))
        header_layout.addWidget(dismiss_btn)
        
        layout.addLayout(header_layout)
        
        # Message
        message_label = QLabel(message)
        message_label.setFont(QFont("Segoe UI", 8))
        message_label.setWordWrap(True)
        message_label.setStyleSheet("color: #2c3e50; margin-left: 20px;")
        layout.addWidget(message_label)
        
        # Action buttons for critical/error alerts
        if self.alert_type in ['error', 'critical']:
            self._add_action_buttons(layout)
    
    def _add_action_buttons(self, layout):
        """Add action buttons for alerts that require user action"""
        actions_layout = QHBoxLayout()
        actions_layout.setContentsMargins(20, 4, 0, 0)
        
        # Acknowledge button
        ack_btn = QPushButton("Acknowledge")
        ack_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 4px 8px;
                border-radius: 3px;
                font-size: 8pt;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        ack_btn.clicked.connect(lambda: self.action_triggered.emit(self.alert_id, "acknowledge"))
        actions_layout.addWidget(ack_btn)
        
        # View details button
        details_btn = QPushButton("View Details")
        details_btn.setStyleSheet("""
            QPushButton {
                background-color: #95a5a6;
                color: white;
                border: none;
                padding: 4px 8px;
                border-radius: 3px;
                font-size: 8pt;
            }
            QPushButton:hover {
                background-color: #7f8c8d;
            }
        """)
        details_btn.clicked.connect(lambda: self.action_triggered.emit(self.alert_id, "details"))
        actions_layout.addWidget(details_btn)
        
        actions_layout.addStretch()
        layout.addLayout(actions_layout)
    
    def _apply_style(self):
        """Apply alert type specific styling"""
        colors = {
            'error': '#e74c3c',
            'warning': '#f39c12',
            'info': '#3498db',
            'success': '#27ae60',
            'critical': '#8e44ad'
        }
        
        bg_colors = {
            'error': '#fdf2f2',
            'warning': '#fef9e7',
            'info': '#e8f4fd',
            'success': '#eafaf1',
            'critical': '#f4ecf7'
        }
        
        border_color = colors.get(self.alert_type, '#bdc3c7')
        bg_color = bg_colors.get(self.alert_type, '#f8f9fa')
        
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {bg_color};
                border-left: 4px solid {border_color};
                border-radius: 4px;
                margin: 2px 0px;
            }}
        """)

class AlertWidget(QWidget):
    """
    Main alert widget for displaying system notifications and alerts
    
    Features:
    - Multiple alert types (error, warning, info, success, critical)
    - Auto-dismiss for certain alert types
    - Alert filtering and search
    - Action buttons for critical alerts
    - Alert history with timestamps
    """
    
    # Signals
    alert_action_required = Signal(str, str)  # alert_id, action
    alerts_cleared = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.alerts = {}  # alert_id -> AlertItem
        self.auto_dismiss_types = ['success', 'info']
        self.max_alerts = 50
        
        self._setup_ui()
        
        # Auto-dismiss timer
        self.dismiss_timer = QTimer()
        self.dismiss_timer.timeout.connect(self._check_auto_dismiss)
        self.dismiss_timer.start(5000)  # Check every 5 seconds
    
    def _setup_ui(self):
        """Setup alert widget UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Header
        header = self._create_header()
        layout.addWidget(header)
        
        # Alerts scroll area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: white;
            }
        """)
        
        # Alerts container
        self.alerts_container = QWidget()
        self.alerts_layout = QVBoxLayout(self.alerts_container)
        self.alerts_layout.setContentsMargins(8, 4, 8, 4)
        self.alerts_layout.setSpacing(2)
        self.alerts_layout.addStretch()  # Push alerts to top
        
        self.scroll_area.setWidget(self.alerts_container)
        layout.addWidget(self.scroll_area)
        
        # No alerts message
        self.no_alerts_label = QLabel("No active alerts")
        self.no_alerts_label.setAlignment(Qt.AlignCenter)
        self.no_alerts_label.setStyleSheet("""
            QLabel {
                color: #95a5a6;
                font-style: italic;
                padding: 20px;
            }
        """)
        self.alerts_layout.insertWidget(0, self.no_alerts_label)
    
    def _create_header(self):
        """Create alerts header with controls"""
        header = QFrame()
        header.setFixedHeight(40)
        header.setStyleSheet("""
            QFrame {
                background-color: #34495e;
                border-bottom: 1px solid #2c3e50;
            }
        """)
        
        layout = QHBoxLayout(header)
        layout.setContentsMargins(10, 5, 10, 5)
        
        # Title
        title = QLabel("🚨 System Alerts")
        title.setFont(QFont("Segoe UI", 10, QFont.Bold))
        title.setStyleSheet("color: white;")
        layout.addWidget(title)
        
        layout.addStretch()
        
        # Alert count
        self.count_label = QLabel("0 alerts")
        self.count_label.setFont(QFont("Segoe UI", 8))
        self.count_label.setStyleSheet("color: #ecf0f1;")
        layout.addWidget(self.count_label)
        
        # Clear all button
        clear_btn = QPushButton("Clear All")
        clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                padding: 4px 8px;
                border-radius: 3px;
                font-size: 8pt;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        clear_btn.clicked.connect(self.clear_all_alerts)
        layout.addWidget(clear_btn)
        
        return header
    
    def add_alert(self, alert_type, title, message, alert_id=None, auto_dismiss=None):
        """
        Add a new alert
        
        Args:
            alert_type: 'error', 'warning', 'info', 'success', 'critical'
            title: Alert title
            message: Alert message
            alert_id: Unique alert ID (auto-generated if None)
            auto_dismiss: Auto-dismiss in seconds (None for default behavior)
        """
        if alert_id is None:
            alert_id = f"alert_{len(self.alerts)}_{QDateTime.currentMSecsSinceEpoch()}"
        
        # Remove oldest alert if at max capacity
        if len(self.alerts) >= self.max_alerts:
            oldest_id = next(iter(self.alerts))
            self.dismiss_alert(oldest_id)
        
        # Create alert item
        alert_item = AlertItem(alert_id, alert_type, title, message)
        alert_item.dismissed.connect(self.dismiss_alert)
        alert_item.action_triggered.connect(self._on_alert_action)
        
        # Add to layout (insert before stretch)
        self.alerts_layout.insertWidget(self.alerts_layout.count() - 1, alert_item)
        self.alerts[alert_id] = alert_item
        
        # Hide no alerts message
        self.no_alerts_label.hide()
        
        # Update count
        self._update_count()
        
        # Auto-dismiss if configured
        if auto_dismiss or (auto_dismiss is None and alert_type in self.auto_dismiss_types):
            dismiss_time = auto_dismiss if auto_dismiss else 10  # 10 seconds default
            QTimer.singleShot(dismiss_time * 1000, lambda: self.dismiss_alert(alert_id))
        
        # Scroll to top to show new alert
        self.scroll_area.verticalScrollBar().setValue(0)
        
        print(f"🚨 Alert added: {alert_type} - {title}")
        
        return alert_id
    
    def dismiss_alert(self, alert_id):
        """Dismiss a specific alert"""
        if alert_id in self.alerts:
            alert_item = self.alerts[alert_id]
            self.alerts_layout.removeWidget(alert_item)
            alert_item.deleteLater()
            del self.alerts[alert_id]
            
            # Show no alerts message if empty
            if not self.alerts:
                self.no_alerts_label.show()
            
            self._update_count()
            print(f"🚨 Alert dismissed: {alert_id}")
    
    def clear_all_alerts(self):
        """Clear all alerts"""
        for alert_id in list(self.alerts.keys()):
            self.dismiss_alert(alert_id)
        
        self.alerts_cleared.emit()
        print("🚨 All alerts cleared")
    
    def _on_alert_action(self, alert_id, action):
        """Handle alert action button clicks"""
        self.alert_action_required.emit(alert_id, action)
        
        # Auto-dismiss after acknowledgment
        if action == "acknowledge":
            self.dismiss_alert(alert_id)
    
    def _check_auto_dismiss(self):
        """Check for alerts that should be auto-dismissed"""
        current_time = QDateTime.currentDateTime()
        
        for alert_id, alert_item in list(self.alerts.items()):
            # Auto-dismiss info and success alerts after 30 seconds
            if (alert_item.alert_type in self.auto_dismiss_types and 
                alert_item.timestamp.secsTo(current_time) > 30):
                self.dismiss_alert(alert_id)
    
    def _update_count(self):
        """Update alert count display"""
        count = len(self.alerts)
        self.count_label.setText(f"{count} alert{'s' if count != 1 else ''}")
    
    def get_alert_count(self):
        """Get current alert count"""
        return len(self.alerts)
    
    def get_alerts_by_type(self, alert_type):
        """Get alerts of specific type"""
        return [alert for alert in self.alerts.values() 
                if alert.alert_type == alert_type]
    
    def has_critical_alerts(self):
        """Check if there are any critical or error alerts"""
        return any(alert.alert_type in ['critical', 'error'] 
                  for alert in self.alerts.values())
