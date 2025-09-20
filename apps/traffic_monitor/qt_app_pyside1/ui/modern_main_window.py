"""
Modern Smart Intersection Monitoring System - Main Window
Production-grade PySide6 UI with dark/light theme support and intuitive tabbed navigation
"""

from PySide6.QtWidgets import (QMainWindow, QTabWidget, QVBoxLayout, QHBoxLayout, 
                               QWidget, QPushButton, QLabel, QFrame, QSizePolicy,
                               QStatusBar, QToolBar, QMessageBox, QSplitter)
from PySide6.QtCore import Qt, Signal, QTimer, QSettings, pyqtSignal
from PySide6.QtGui import QIcon, QFont, QPixmap, QPalette, QAction

from .tabs.live_monitoring_tab import LiveMonitoringTab
from .tabs.video_analysis_tab import VideoAnalysisTab
from .tabs.vlm_insights_tab import VLMInsightsTab
from .tabs.violations_tab import ViolationsTab
from .tabs.system_performance_tab import SystemPerformanceTab
from .tabs.smart_intersection_tab import SmartIntersectionTab
from .theme_manager import ThemeManager
from .widgets.status_indicator import StatusIndicator
from .widgets.notification_center import NotificationCenter

class ModernMainWindow(QMainWindow):
    """
    Modern Smart Intersection Monitoring System Main Window
    
    Features:
    - 6 specialized tabs with modern UI design
    - Dark/Light theme switching with smooth transitions
    - WCAG AAA compliant color schemes
    - Real-time status indicators
    - Notification center
    - Production-grade navigation and controls
    """
    
    # Signals for system communication
    theme_changed = Signal(bool)  # True for dark theme, False for light
    tab_changed = Signal(int)     # Current tab index
    system_notification = Signal(str, str)  # message, level (info/warning/error)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Initialize settings
        self.settings = QSettings("SmartIntersection", "MonitoringSystem")
        
        # Initialize theme manager
        self.theme_manager = ThemeManager()
        
        # Setup window properties
        self.setWindowTitle("Smart Intersection Monitoring System")
        self.setMinimumSize(1400, 900)
        self.resize(1920, 1080)
        
        # Initialize UI components
        self._setup_ui()
        self._setup_toolbar()
        self._setup_status_bar()
        self._setup_connections()
        
        # Apply saved theme
        self._restore_settings()
        
        # Setup update timer for real-time data
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_real_time_data)
        self.update_timer.start(1000)  # Update every second
        
        print("✅ Modern Main Window initialized successfully")
    
    def _setup_ui(self):
        """Setup the main UI layout and components"""
        
        # Central widget with modern layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Header section with title and controls
        header_widget = self._create_header()
        main_layout.addWidget(header_widget)
        
        # Content splitter (main tabs + notification panel)
        content_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(content_splitter)
        
        # Main tab widget
        self.tab_widget = self._create_tab_widget()
        content_splitter.addWidget(self.tab_widget)
        
        # Notification center (collapsible)
        self.notification_center = NotificationCenter()
        content_splitter.addWidget(self.notification_center)
        
        # Set splitter proportions (80% main content, 20% notifications)
        content_splitter.setSizes([1536, 384])  # Based on 1920px width
        
    def _create_header(self):
        """Create the modern header with title and theme controls"""
        header = QFrame()
        header.setObjectName("headerFrame")
        header.setFixedHeight(60)
        
        layout = QHBoxLayout(header)
        layout.setContentsMargins(20, 10, 20, 10)
        
        # Application title with icon
        title_layout = QHBoxLayout()
        
        # System status indicator
        self.system_status = StatusIndicator()
        title_layout.addWidget(self.system_status)
        
        # Title label
        title_label = QLabel("Smart Intersection Monitoring System")
        title_label.setObjectName("titleLabel")
        title_font = QFont("Segoe UI", 18, QFont.Bold)
        title_label.setFont(title_font)
        title_layout.addWidget(title_label)
        
        title_layout.addStretch()
        layout.addLayout(title_layout)
        
        # Control buttons
        controls_layout = QHBoxLayout()
        
        # Theme toggle button
        self.theme_button = QPushButton()
        self.theme_button.setObjectName("themeToggleButton")
        self.theme_button.setFixedSize(40, 40)
        self.theme_button.setToolTip("Toggle Dark/Light Theme")
        self.theme_button.clicked.connect(self._toggle_theme)
        controls_layout.addWidget(self.theme_button)
        
        # Notification toggle
        self.notification_button = QPushButton()
        self.notification_button.setObjectName("notificationButton")
        self.notification_button.setFixedSize(40, 40)
        self.notification_button.setToolTip("Toggle Notifications Panel")
        self.notification_button.clicked.connect(self._toggle_notifications)
        controls_layout.addWidget(self.notification_button)
        
        # Settings button
        settings_button = QPushButton()
        settings_button.setObjectName("settingsButton")
        settings_button.setFixedSize(40, 40)
        settings_button.setToolTip("System Settings")
        settings_button.clicked.connect(self._show_settings)
        controls_layout.addWidget(settings_button)
        
        layout.addLayout(controls_layout)
        
        return header
    
    def _create_tab_widget(self):
        """Create the main tab widget with 6 specialized tabs"""
        tab_widget = QTabWidget()
        tab_widget.setObjectName("mainTabWidget")
        tab_widget.setTabPosition(QTabWidget.North)
        tab_widget.setMovable(False)
        tab_widget.setTabsClosable(False)
        
        # Tab 1: Live Monitoring
        self.live_monitoring_tab = LiveMonitoringTab()
        tab_widget.addTab(self.live_monitoring_tab, "📺 Live Monitoring")
        
        # Tab 2: Video Analysis
        self.video_analysis_tab = VideoAnalysisTab()
        tab_widget.addTab(self.video_analysis_tab, "🎬 Video Analysis")
        
        # Tab 3: VLM AI Insights
        self.vlm_insights_tab = VLMInsightsTab()
        tab_widget.addTab(self.vlm_insights_tab, "🤖 VLM AI Insights")
        
        # Tab 4: Violations
        self.violations_tab = ViolationsTab()
        tab_widget.addTab(self.violations_tab, "🚨 Violations")
        
        # Tab 5: System Performance
        self.system_performance_tab = SystemPerformanceTab()
        tab_widget.addTab(self.system_performance_tab, "🔥 System Performance")
        
        # Tab 6: Smart Intersection
        self.smart_intersection_tab = SmartIntersectionTab()
        tab_widget.addTab(self.smart_intersection_tab, "🌉 Smart Intersection")
        
        # Connect tab change signal
        tab_widget.currentChanged.connect(self._on_tab_changed)
        
        return tab_widget
    
    def _setup_toolbar(self):
        """Setup the modern toolbar with essential actions"""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setObjectName("mainToolbar")
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        self.addToolBar(toolbar)
        
        # Start/Stop monitoring
        self.start_action = QAction("▶️ Start Monitoring", self)
        self.start_action.setToolTip("Start traffic monitoring")
        self.start_action.triggered.connect(self._start_monitoring)
        toolbar.addAction(self.start_action)
        
        self.stop_action = QAction("⏹️ Stop Monitoring", self)
        self.stop_action.setToolTip("Stop traffic monitoring")
        self.stop_action.triggered.connect(self._stop_monitoring)
        self.stop_action.setEnabled(False)
        toolbar.addAction(self.stop_action)
        
        toolbar.addSeparator()
        
        # Recording controls
        self.record_action = QAction("🔴 Record", self)
        self.record_action.setToolTip("Start/Stop recording")
        self.record_action.triggered.connect(self._toggle_recording)
        toolbar.addAction(self.record_action)
        
        toolbar.addSeparator()
        
        # Emergency controls
        emergency_action = QAction("🚨 Emergency", self)
        emergency_action.setToolTip("Emergency traffic control")
        emergency_action.triggered.connect(self._emergency_mode)
        toolbar.addAction(emergency_action)
        
        toolbar.addSeparator()
        
        # Export data
        export_action = QAction("📊 Export Data", self)
        export_action.setToolTip("Export monitoring data")
        export_action.triggered.connect(self._export_data)
        toolbar.addAction(export_action)
    
    def _setup_status_bar(self):
        """Setup the modern status bar with real-time information"""
        status_bar = QStatusBar()
        self.setStatusBar(status_bar)
        
        # System status
        self.status_label = QLabel("System Ready")
        status_bar.addWidget(self.status_label)
        
        # Performance indicators
        self.fps_label = QLabel("FPS: --")
        self.fps_label.setMinimumWidth(80)
        status_bar.addPermanentWidget(self.fps_label)
        
        self.cpu_label = QLabel("CPU: --%")
        self.cpu_label.setMinimumWidth(80)
        status_bar.addPermanentWidget(self.cpu_label)
        
        self.memory_label = QLabel("Memory: --%")
        self.memory_label.setMinimumWidth(100)
        status_bar.addPermanentWidget(self.memory_label)
        
        # Connection status
        self.connection_label = QLabel("🔴 Disconnected")
        status_bar.addPermanentWidget(self.connection_label)
    
    def _setup_connections(self):
        """Setup signal connections between components"""
        
        # Theme manager connections
        self.theme_manager.theme_changed.connect(self._apply_theme)
        
        # Notification center connections
        self.system_notification.connect(self.notification_center.add_notification)
        
        # Tab-specific connections
        self._setup_tab_connections()
    
    def _setup_tab_connections(self):
        """Setup connections specific to each tab"""
        
        # Live Monitoring Tab
        if hasattr(self.live_monitoring_tab, 'camera_status_changed'):
            self.live_monitoring_tab.camera_status_changed.connect(self._update_camera_status)
        
        # Video Analysis Tab
        if hasattr(self.video_analysis_tab, 'roi_changed'):
            self.video_analysis_tab.roi_changed.connect(self._update_roi_settings)
        
        # VLM Insights Tab
        if hasattr(self.vlm_insights_tab, 'insight_generated'):
            self.vlm_insights_tab.insight_generated.connect(self._log_vlm_insight)
        
        # Violations Tab
        if hasattr(self.violations_tab, 'violation_acknowledged'):
            self.violations_tab.violation_acknowledged.connect(self._acknowledge_violation)
        
        # System Performance Tab
        if hasattr(self.system_performance_tab, 'performance_alert'):
            self.system_performance_tab.performance_alert.connect(self._handle_performance_alert)
        
        # Smart Intersection Tab
        if hasattr(self.smart_intersection_tab, 'traffic_control_changed'):
            self.smart_intersection_tab.traffic_control_changed.connect(self._update_traffic_control)
    
    def _toggle_theme(self):
        """Toggle between dark and light themes"""
        self.theme_manager.toggle_theme()
    
    def _apply_theme(self, is_dark):
        """Apply the selected theme to the entire application"""
        self.theme_changed.emit(is_dark)
        
        # Update theme button icon
        if is_dark:
            self.theme_button.setText("☀️")
            self.theme_button.setToolTip("Switch to Light Theme")
        else:
            self.theme_button.setText("🌙")
            self.theme_button.setToolTip("Switch to Dark Theme")
    
    def _toggle_notifications(self):
        """Toggle the notification panel visibility"""
        self.notification_center.setVisible(not self.notification_center.isVisible())
        
        # Update button state
        if self.notification_center.isVisible():
            self.notification_button.setText("🔕")
            self.notification_button.setToolTip("Hide Notifications")
        else:
            self.notification_button.setText("🔔")
            self.notification_button.setToolTip("Show Notifications")
    
    def _show_settings(self):
        """Show system settings dialog"""
        from .dialogs.settings_dialog import SettingsDialog
        settings_dialog = SettingsDialog(self)
        settings_dialog.exec()
    
    def _on_tab_changed(self, index):
        """Handle tab change events"""
        self.tab_changed.emit(index)
        
        # Update status based on current tab
        tab_names = [
            "Live Monitoring Active",
            "Video Analysis Mode",
            "VLM AI Insights Active",
            "Violations Dashboard",
            "Performance Monitoring",
            "Smart Intersection Control"
        ]
        
        if 0 <= index < len(tab_names):
            self.status_label.setText(tab_names[index])
    
    def _update_real_time_data(self):
        """Update real-time data displays"""
        # This will be connected to actual data sources
        pass
    
    def _start_monitoring(self):
        """Start traffic monitoring"""
        self.start_action.setEnabled(False)
        self.stop_action.setEnabled(True)
        self.system_status.set_status("running")
        self.status_label.setText("Monitoring Started")
        self.connection_label.setText("🟢 Connected")
        
        # Emit notification
        self.system_notification.emit("Traffic monitoring started successfully", "info")
    
    def _stop_monitoring(self):
        """Stop traffic monitoring"""
        self.start_action.setEnabled(True)
        self.stop_action.setEnabled(False)
        self.system_status.set_status("stopped")
        self.status_label.setText("Monitoring Stopped")
        self.connection_label.setText("🔴 Disconnected")
        
        # Emit notification
        self.system_notification.emit("Traffic monitoring stopped", "warning")
    
    def _toggle_recording(self):
        """Toggle video recording"""
        # Implementation for recording toggle
        pass
    
    def _emergency_mode(self):
        """Activate emergency traffic control mode"""
        reply = QMessageBox.question(
            self, 
            "Emergency Mode",
            "Activate emergency traffic control mode?\n\nThis will override normal traffic patterns.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.system_notification.emit("Emergency mode activated", "error")
    
    def _export_data(self):
        """Export monitoring data"""
        # Implementation for data export
        pass
    
    def _update_camera_status(self, camera_id, status):
        """Update camera status"""
        self.system_notification.emit(f"Camera {camera_id}: {status}", "info")
    
    def _update_roi_settings(self, roi_data):
        """Update ROI settings"""
        self.system_notification.emit("ROI settings updated", "info")
    
    def _log_vlm_insight(self, insight):
        """Log VLM insight"""
        self.system_notification.emit(f"VLM Insight: {insight[:50]}...", "info")
    
    def _acknowledge_violation(self, violation_id):
        """Acknowledge a violation"""
        self.system_notification.emit(f"Violation {violation_id} acknowledged", "info")
    
    def _handle_performance_alert(self, alert_type, message):
        """Handle performance alerts"""
        self.system_notification.emit(f"Performance Alert: {message}", "warning")
    
    def _update_traffic_control(self, control_data):
        """Update traffic control settings"""
        self.system_notification.emit("Traffic control updated", "info")
    
    def _restore_settings(self):
        """Restore saved settings"""
        # Restore theme
        is_dark = self.settings.value("theme/dark_mode", True, type=bool)
        self.theme_manager.set_dark_theme(is_dark)
        
        # Restore window geometry
        geometry = self.settings.value("window/geometry")
        if geometry:
            self.restoreGeometry(geometry)
        
        # Restore window state
        state = self.settings.value("window/state")
        if state:
            self.restoreState(state)
    
    def closeEvent(self, event):
        """Handle application close event"""
        # Save settings
        self.settings.setValue("theme/dark_mode", self.theme_manager.is_dark_theme())
        self.settings.setValue("window/geometry", self.saveGeometry())
        self.settings.setValue("window/state", self.saveState())
        
        # Stop monitoring if active
        if self.stop_action.isEnabled():
            self._stop_monitoring()
        
        # Accept close event
        event.accept()
        
        print("✅ Main window closed and settings saved")
