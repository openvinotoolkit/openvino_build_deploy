from PySide6.QtWidgets import (
    QMainWindow, QTabWidget, QDockWidget, QMessageBox,
    QApplication, QFileDialog, QSplashScreen, QVBoxLayout, QWidget, QLabel
)
from PySide6.QtCore import Qt, QTimer, QSettings, QSize, Slot
from PySide6.QtGui import QIcon, QPixmap, QAction, QFont

import os
import sys
import json
import time
import traceback
from pathlib import Path

# Custom exception handler for Qt
def qt_message_handler(mode, context, message):
    print(f"Qt Message: {message} (Mode: {mode})")

# Install custom handler for Qt messages
if hasattr(Qt, 'qInstallMessageHandler'):
    Qt.qInstallMessageHandler(qt_message_handler)

# Import UI components
from ui.analytics_tab import AnalyticsTab
from ui.violations_tab import ViolationsTab
from ui.export_tab import ExportTab
from ui.modern_config_panel import ModernConfigPanel
from ui.modern_live_detection_tab import ModernLiveDetectionTab
# from ui.video_analysis_tab import VideoAnalysisTab
# from ui.video_detection_tab import VideoDetectionTab  # Commented out - split into two separate tabs
from ui.video_detection_only_tab import VideoDetectionOnlyTab
from ui.smart_intersection_tab import SmartIntersectionTab
from ui.global_status_panel import GlobalStatusPanel
from ui.vlm_insights_widget import VLMInsightsWidget  # Import the new VLM Insights Widget
# Grafana dashboard removed from main window

# Import controllers
from controllers.video_controller_new import VideoController
from controllers.analytics_controller import AnalyticsController
from controllers.performance_overlay import PerformanceOverlay
from controllers.model_manager import ModelManager
# VLM Controller removed - functionality moved to insights widget

# Import utilities
from utils.helpers import load_configuration, save_configuration, save_snapshot
from utils.data_publisher import DataPublisher

class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        
        # Initialize settings and configuration
        self.settings = QSettings("OpenVINO", "TrafficMonitoring")
        self.config_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")
        self.config = load_configuration(self.config_file)
        
        # Set up UI
        self.setupUI()
        
        # Initialize controllers
        self.setupControllers()
        
        # Connect signals and slots
        self.connectSignals()
        
        # Initialize config panel with current configuration
        self.config_panel.set_config(self.config)
        
        # Restore settings
        self.restoreSettings()
        
        # Apply theme
        self.applyTheme(True)  # Start with dark theme
        
        # Show ready message
        self.statusBar().showMessage("Ready")
        
    def setupUI(self):
        """Set up the user interface"""
        # Window properties
        self.setWindowTitle("Traffic Intersection Monitoring System")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)
        
        # Set up central widget with tabs
        self.tabs = QTabWidget()
        
        # Style the tabs
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #444;
                background-color: #2b2b2b;
            }
            QTabBar::tab {
                background-color: #3c3c3c;
                color: white;
                padding: 8px 16px;
                margin: 2px;
                border: 1px solid #555;
                border-bottom: none;
                border-radius: 4px 4px 0px 0px;
                min-width: 120px;
            }
            QTabBar::tab:selected {
                background-color: #0078d4;
                border-color: #0078d4;
            }
            QTabBar::tab:hover {
                background-color: #4a4a4a;
            }
            QTabBar::tab:!selected {
                margin-top: 2px;
            }
        """)
        
        # Create tabs
        self.live_tab = ModernLiveDetectionTab()
        # self.video_analysis_tab = VideoAnalysisTab()
        # self.video_detection_tab = VideoDetectionTab()  # Commented out - split into two separate tabs
        self.video_detection_only_tab = VideoDetectionOnlyTab()
        self.smart_intersection_tab = SmartIntersectionTab()
        self.analytics_tab = AnalyticsTab()
        self.violations_tab = ViolationsTab()
        self.export_tab = ExportTab()
        # Remove VLM tab - VLM functionality moved to settings panel
        # self.vlm_tab = VLMTab()  # Create the VLM tab
        from ui.performance_graphs import PerformanceGraphsWidget
        self.performance_tab = PerformanceGraphsWidget()
        
        # Add User Guide tab
        try:
            from ui.user_guide_tab import UserGuideTab
            self.user_guide_tab = UserGuideTab()
        except Exception as e:
            print(f"Warning: Could not create User Guide tab: {e}")
            self.user_guide_tab = None
        
        # Add tabs to tab widget
        self.tabs.addTab(self.live_tab, "Live Detection")
        # self.tabs.addTab(self.video_analysis_tab, "Video Analysis")
        # self.tabs.addTab(self.video_detection_tab, "Smart Intersection")  # Commented out - split into two tabs
        self.tabs.addTab(self.video_detection_only_tab, "Video Detection")
        # self.tabs.addTab(self.smart_intersection_tab, "Smart Intersection")  # Temporarily hidden
        # Grafana dashboard removed from main window
        self.tabs.addTab(self.analytics_tab, "Analytics")
        self.tabs.addTab(self.performance_tab, "Performance Graphs")
        
        self.tabs.addTab(self.violations_tab, "Violations")
        # VLM functionality moved to settings panel
        # self.tabs.addTab(self.vlm_tab, "🔍 Vision AI")  # Add VLM tab with icon
        self.tabs.addTab(self.export_tab, "Export & Config")
        
        # Add User Guide tab if available
        if self.user_guide_tab:
            self.tabs.addTab(self.user_guide_tab, "Help")
        
        # Create config panel in dock widget
        self.config_panel = ModernConfigPanel()
        dock = QDockWidget("Settings", self)
        dock.setObjectName("SettingsDock")  # Set object name to avoid warning
        dock.setWidget(self.config_panel)
        dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetClosable)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        
        # Set minimum and preferred size for the dock widget
        dock.setMinimumWidth(400)
        dock.resize(450, 800)  # Set preferred width and height
        
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        
        # Create status bar
        self.statusBar().showMessage("Initializing...")
        
        # Create main layout with header
        main_layout = QVBoxLayout()
        
        # Add header title above tabs
        header_label = QLabel("Traffic Intersection Monitoring System")
        header_label.setAlignment(Qt.AlignCenter)
        header_font = QFont()
        header_font.setPointSize(14)
        header_font.setBold(True)
        header_label.setFont(header_font)
        header_label.setStyleSheet("""
            QLabel {
                color: #ffffff;
                background-color: #2b2b2b;
                padding: 10px;
                border-bottom: 2px solid #0078d4;
                margin-bottom: 5px;
            }
        """)
        main_layout.addWidget(header_label)
        
        main_layout.addWidget(self.tabs)
        central = QWidget()
        central.setLayout(main_layout)
        self.setCentralWidget(central)
        
        # Create menu bar - commented out for cleaner interface
        # self.setupMenus()
        
        # Create performance overlay
        self.performance_overlay = PerformanceOverlay()
        
    def setupControllers(self):
        """Set up controllers and models"""
        try:
            # Initialize model manager
            self.model_manager = ModelManager(self.config_file)

            # Create video controller for live tab
            self.video_controller = VideoController(self.model_manager)

            # Create video controller for video detection tab
            self.video_file_controller = VideoController(self.model_manager)

            # Create analytics controller
            self.analytics_controller = AnalyticsController()
            
            # Initialize data publisher for InfluxDB
            print("[MAIN WINDOW DEBUG] Initializing Data Publisher...")
            self.data_publisher = DataPublisher(self.config_file)
            print("[MAIN WINDOW DEBUG] Data Publisher initialized successfully")
            
            # VLM controller - using only local VLM folder, no backend
            print("[MAIN WINDOW DEBUG] Initializing VLM Controller with local VLM folder...")
            from controllers.vlm_controller_new import VLMController
            self.vlm_controller = VLMController()  # No backend URL needed
            print("[MAIN WINDOW DEBUG] VLM Controller initialized successfully")

            # Setup update timer for performance overlay
            self.perf_timer = QTimer()
            self.perf_timer.timeout.connect(self.performance_overlay.update_stats)
            self.perf_timer.start(1000)  # Update every second

            # Connect video_file_controller outputs to video_detection_tab
            # Connect video file controller signals to both video tabs
            self.video_file_controller.frame_ready.connect(self.video_detection_only_tab.update_display, Qt.QueuedConnection)
            self.video_file_controller.stats_ready.connect(self.video_detection_only_tab.update_stats, Qt.QueuedConnection)
            self.video_file_controller.progress_ready.connect(lambda value, max_value, timestamp: self.video_detection_only_tab.update_progress(value, max_value, timestamp), Qt.QueuedConnection)
            
            self.video_file_controller.frame_ready.connect(self.smart_intersection_tab.update_display, Qt.QueuedConnection)
            self.video_file_controller.stats_ready.connect(self.smart_intersection_tab.update_stats, Qt.QueuedConnection)
            self.video_file_controller.progress_ready.connect(lambda value, max_value, timestamp: self.smart_intersection_tab.update_progress(value, max_value, timestamp), Qt.QueuedConnection)
            
            # Connect video frames to VLM insights for analysis
            if hasattr(self.video_file_controller, 'raw_frame_ready'):
                print("[MAIN WINDOW DEBUG] Connecting raw_frame_ready signal to VLM insights")
                self.video_file_controller.raw_frame_ready.connect(
                    self._forward_frame_to_vlm, Qt.QueuedConnection
                )
                print("[MAIN WINDOW DEBUG] raw_frame_ready signal connected to VLM insights")
                
                # Also connect to analytics tab
                print("[MAIN WINDOW DEBUG] Connecting raw_frame_ready signal to analytics tab")
                self.video_file_controller.raw_frame_ready.connect(
                    self._forward_frame_to_analytics, Qt.QueuedConnection
                )
                print("[MAIN WINDOW DEBUG] raw_frame_ready signal connected to analytics tab")
            else:
                print("[MAIN WINDOW DEBUG] raw_frame_ready signal not found in video_file_controller")
            # Connect auto model/device selection signal
            # Connect video tab auto-select signals
            self.video_detection_only_tab.auto_select_model_device.connect(self.video_file_controller.auto_select_model_device, Qt.QueuedConnection)
            self.smart_intersection_tab.auto_select_model_device.connect(self.video_file_controller.auto_select_model_device, Qt.QueuedConnection)
            
            # Connect VLM insights analysis requests to a simple mock handler (since optimum is disabled)
            print("[MAIN WINDOW DEBUG] Checking for VLM insights widget...")
            if hasattr(self.config_panel, 'vlm_insights_widget'):
                print("[MAIN WINDOW DEBUG] VLM insights widget found, connecting signals...")
                self.config_panel.vlm_insights_widget.analyze_frame_requested.connect(self._handle_vlm_analysis, Qt.QueuedConnection)
                print("[MAIN WINDOW DEBUG] VLM insights analysis signal connected")
                
                # Connect pause state signal from video file controller to VLM insights
                if hasattr(self.video_file_controller, 'pause_state_changed'):
                    self.video_file_controller.pause_state_changed.connect(self.config_panel.vlm_insights_widget.on_video_paused, Qt.QueuedConnection)
                    print("[MAIN WINDOW DEBUG] VLM insights pause state signal connected")
                else:
                    print("[MAIN WINDOW DEBUG] pause_state_changed signal not found in video_file_controller")
            else:
                print("[MAIN WINDOW DEBUG] VLM insights widget NOT found in config panel")
            
            # Old VLM tab connections removed - functionality moved to insights widget
            # self.vlm_tab.process_image_requested.connect(self.vlm_controller.process_image, Qt.QueuedConnection)
            # self.video_controller.frame_np_ready.connect(self.vlm_tab.set_frame, Qt.QueuedConnection)
            # self.video_file_controller.frame_np_ready.connect(self.vlm_tab.set_frame, Qt.QueuedConnection)
        except Exception as e:
            QMessageBox.critical(
                self,
                "Initialization Error",
                f"Error initializing controllers: {str(e)}"
            )
            print(f"Error details: {e}")
            traceback.print_exc()

    
    def connectSignals(self):
        """Connect signals and slots between components"""
        print("🔌 Connecting video controller signals...")
        try:
            self.video_controller.frame_ready.connect(self.live_tab.update_display, Qt.QueuedConnection)
            print("✅ Connected frame_ready signal")
            try:
                self.video_controller.frame_np_ready.connect(self.live_tab.update_display_np, Qt.QueuedConnection)
                print("✅ Connected frame_np_ready signal")
                print("🔌 frame_np_ready connection should be established")
            except Exception as e:
                print(f"❌ Error connecting frame_np_ready signal: {e}")
                import traceback
                traceback.print_exc()
            self.video_controller.stats_ready.connect(self.live_tab.update_stats, Qt.QueuedConnection)
            self.video_controller.stats_ready.connect(self.update_traffic_light_status, Qt.QueuedConnection)
            print("✅ Connected stats_ready signals")
            # Only connect analytics_controller if it exists
            if hasattr(self, 'analytics_controller'):
                self.video_controller.raw_frame_ready.connect(self.analytics_controller.process_frame_data)
                print("✅ Connected raw_frame_ready signal")
            else:
                print("❌ analytics_controller not found, skipping analytics signal connection")
            self.video_controller.stats_ready.connect(self.update_traffic_light_status, Qt.QueuedConnection)
            print("✅ Connected stats_ready signal to update_traffic_light_status")
            
            # Connect violation detection signal for LIVE controller
            try:
                print(f"[MAIN WINDOW DEBUG] About to connect violation_detected signal for LIVE controller")
                print(f"[MAIN WINDOW DEBUG] VideoController object: {self.video_controller}")
                print(f"[MAIN WINDOW DEBUG] VideoController violation_detected signal: {self.video_controller.violation_detected}")
                self.video_controller.violation_detected.connect(self.handle_violation_detected, Qt.QueuedConnection)
                print("✅ Connected violation_detected signal successfully for LIVE controller")
            except Exception as e:
                print(f"⚠️ Could not connect violation signal for LIVE controller: {e}")
                import traceback
                traceback.print_exc()
                
            # Connect violation detection signal for VIDEO FILE controller  
            try:
                print(f"[MAIN WINDOW DEBUG] About to connect violation_detected signal for VIDEO FILE controller")
                print(f"[MAIN WINDOW DEBUG] VideoFileController object: {self.video_file_controller}")
                print(f"[MAIN WINDOW DEBUG] VideoFileController violation_detected signal: {self.video_file_controller.violation_detected}")
                self.video_file_controller.violation_detected.connect(self.handle_violation_detected, Qt.QueuedConnection)
                print("✅ Connected violation_detected signal successfully for VIDEO FILE controller")
            except Exception as e:
                print(f"⚠️ Could not connect violation signal for VIDEO FILE controller: {e}")
                import traceback
                traceback.print_exc()
        except Exception as e:
            print(f"❌ Error connecting signals: {e}")
            import traceback
            traceback.print_exc()
        
        # Live tab connections
        self.live_tab.source_changed.connect(self.video_controller.set_source)
        self.live_tab.video_dropped.connect(self.video_controller.set_source)
        self.live_tab.snapshot_requested.connect(self.take_snapshot)
        self.live_tab.run_requested.connect(self.toggle_video_processing)
        
        # Config panel connections
        self.config_panel.config_changed.connect(self.apply_config)
        self.config_panel.theme_toggled.connect(self.applyTheme)
        # Connect device switch signal for robust model switching
        self.config_panel.device_switch_requested.connect(self.handle_device_switch)
        
        # Analytics controller connections
        self.analytics_controller.analytics_updated.connect(self.analytics_tab.update_analytics)
        self.analytics_controller.analytics_updated.connect(self.export_tab.update_export_preview)
        
        # Tab-specific connections
        self.violations_tab.clear_btn.clicked.connect(self.analytics_controller.clear_statistics)
        self.export_tab.reset_btn.clicked.connect(self.config_panel.reset_config)
        self.export_tab.save_config_btn.clicked.connect(self.save_config)
        self.export_tab.reload_config_btn.clicked.connect(self.load_config)
        self.export_tab.export_btn.clicked.connect(self.export_data)
        
        # Video Detection tab connections (standard tab)
        self.video_detection_only_tab.file_selected.connect(self._handle_video_file_selected)
        self.video_detection_only_tab.play_clicked.connect(self._handle_video_play)
        self.video_detection_only_tab.pause_clicked.connect(self._handle_video_pause)
        self.video_detection_only_tab.stop_clicked.connect(self._handle_video_stop)
        self.video_detection_only_tab.detection_toggled.connect(self._handle_video_detection_toggle)
        self.video_detection_only_tab.screenshot_clicked.connect(self._handle_video_screenshot)
        self.video_detection_only_tab.seek_changed.connect(self._handle_video_seek)
        
        # Smart Intersection tab connections
        self.smart_intersection_tab.file_selected.connect(self._handle_video_file_selected)
        self.smart_intersection_tab.play_clicked.connect(self._handle_video_play)
        self.smart_intersection_tab.pause_clicked.connect(self._handle_video_pause)
        self.smart_intersection_tab.stop_clicked.connect(self._handle_video_stop)
        self.smart_intersection_tab.detection_toggled.connect(self._handle_video_detection_toggle)
        self.smart_intersection_tab.screenshot_clicked.connect(self._handle_video_screenshot)
        self.smart_intersection_tab.seek_changed.connect(self._handle_video_seek)
        
        # Smart Intersection specific connections
        self.smart_intersection_tab.smart_intersection_enabled.connect(self._handle_smart_intersection_enabled)
        self.smart_intersection_tab.multi_camera_mode_enabled.connect(self._handle_multi_camera_mode)
        self.smart_intersection_tab.roi_configuration_changed.connect(self._handle_roi_configuration_changed)
        self.smart_intersection_tab.scene_analytics_toggled.connect(self._handle_scene_analytics_toggle)
        
        # Connect smart intersection controller if available
        try:
            from controllers.smart_intersection_controller import SmartIntersectionController
            self.smart_intersection_controller = SmartIntersectionController()
            
            # Connect scene analytics signals
            self.video_file_controller.frame_np_ready.connect(
                self.smart_intersection_controller.process_frame, Qt.QueuedConnection
            )
            self.smart_intersection_controller.scene_analytics_ready.connect(
                self._handle_scene_analytics_update, Qt.QueuedConnection
            )
            print("✅ Smart Intersection Controller connected")
        except Exception as e:
            print(f"⚠️ Smart Intersection Controller not available: {e}")
            self.smart_intersection_controller = None
        
        # Connect OpenVINO device info signal to config panel from BOTH controllers
        self.video_controller.device_info_ready.connect(self.config_panel.update_devices_info, Qt.QueuedConnection)
        self.video_file_controller.device_info_ready.connect(self.config_panel.update_devices_info, Qt.QueuedConnection)

        # After connecting video_file_controller and video_detection_tab, trigger auto model/device update
        QTimer.singleShot(0, self.video_file_controller.auto_select_model_device.emit)
        
        # Connect performance statistics from both controllers
        self.video_controller.performance_stats_ready.connect(self.update_performance_graphs)
        self.video_file_controller.performance_stats_ready.connect(self.update_performance_graphs)
        
        # Connect enhanced performance tab signals
        if hasattr(self, 'performance_tab'):
            try:
                # Connect performance tab signals for better integration
                self.performance_tab.spike_detected.connect(self.handle_performance_spike)
                self.performance_tab.device_switched.connect(self.handle_device_switch_notification)
                self.performance_tab.performance_data_updated.connect(self.handle_performance_data_update)
                print("✅ Performance tab signals connected successfully")
            except Exception as e:
                print(f"⚠️ Could not connect performance tab signals: {e}")
    
    @Slot(dict)
    def handle_performance_spike(self, spike_data):
        """Handle performance spike detection"""
        try:
            latency = spike_data.get('latency', 0)
            device = spike_data.get('device', 'Unknown')
            print(f"🚨 Performance spike detected: {latency:.1f}ms on {device}")
            
            # Optionally show notification or log to analytics
            if hasattr(self, 'analytics_tab'):
                # Could add spike to analytics if needed
                pass
                
        except Exception as e:
            print(f"❌ Error handling performance spike: {e}")
    
    @Slot(str)
    def handle_device_switch_notification(self, device):
        """Handle device switch notification"""
        try:
            print(f"🔄 Device switched to: {device}")
            # Could update UI elements or show notification
        except Exception as e:
            print(f"❌ Error handling device switch notification: {e}")
    
    @Slot(dict)
    def handle_performance_data_update(self, performance_data):
        """Handle performance data updates for other components"""
        try:
            # Could forward to other tabs or components that need performance data
            if hasattr(self, 'analytics_tab'):
                # Forward performance data to analytics if needed
                pass
        except Exception as e:
            print(f"❌ Error handling performance data update: {e}")
    def setupMenus(self):
        """Set up application menus"""
        # File menu
        file_menu = self.menuBar().addMenu("&File")
        
        open_action = QAction("&Open Video...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_video_file)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        snapshot_action = QAction("Take &Snapshot", self)
        snapshot_action.setShortcut("Ctrl+S")
        snapshot_action.triggered.connect(self.take_snapshot)
        file_menu.addAction(snapshot_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Alt+F4")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = self.menuBar().addMenu("&View")
        
        toggle_config_action = QAction("Show/Hide &Settings Panel", self)
        toggle_config_action.setShortcut("F4")
        toggle_config_action.triggered.connect(self.toggle_config_panel)
        view_menu.addAction(toggle_config_action)
        
        toggle_perf_action = QAction("Show/Hide &Performance Overlay", self)
        toggle_perf_action.setShortcut("F5")
        toggle_perf_action.triggered.connect(self.toggle_performance_overlay)
        view_menu.addAction(toggle_perf_action)
        
        # Add separator and Grafana dashboard option
        view_menu.addSeparator()
        grafana_action = QAction("Open &Grafana Dashboard", self)
        grafana_action.setShortcut("Ctrl+G")
        grafana_action.triggered.connect(self.open_grafana_dashboard)
        view_menu.addAction(grafana_action)
        
        # Help menu
        help_menu = self.menuBar().addMenu("&Help")
        
        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)
        
    @Slot(dict)
    def apply_config(self, config):
        """
        Apply configuration changes.
        
        Args:
            config: Configuration dictionary
        """
        # Update configuration
        if not config:
            return
            
        # Convert flat config to nested structure for model manager
        nested_config = {
            "detection": {}
        }
        
        # Map config panel values to model manager format
        if 'device' in config:
            nested_config["detection"]["device"] = config['device']
        if 'model' in config:
            # Convert YOLOv11x format to yolo11x format for model manager
            model_name = config['model'].lower()
            if 'yolov11' in model_name:
                model_name = model_name.replace('yolov11', 'yolo11')
            elif model_name == 'auto':
                model_name = 'auto'
            nested_config["detection"]["model"] = model_name
        if 'confidence_threshold' in config:
            nested_config["detection"]["confidence_threshold"] = config['confidence_threshold']
        if 'iou_threshold' in config:
            nested_config["detection"]["iou_threshold"] = config['iou_threshold']
        
        print(f"🔧 Main Window: Applying config to model manager: {nested_config}")
        print(f"🔧 Main Window: Received config from panel: {config}")
        
        # Update config
        for section in nested_config:
            if section in self.config:
                self.config[section].update(nested_config[section])
            else:
                self.config[section] = nested_config[section]
        
        # Update model manager with nested config
        if self.model_manager:
            self.model_manager.update_config(nested_config)
        
        # Refresh model information in video controllers
        if hasattr(self, 'video_controller') and self.video_controller:
            self.video_controller.refresh_model_info()
        if hasattr(self, 'video_file_controller') and self.video_file_controller:
            self.video_file_controller.refresh_model_info()
        
        # Save config to file
        save_configuration(self.config, self.config_file)
        
        # Update export tab
        self.export_tab.update_config_display(self.config)
        
        # Update status
        device = config.get('device', 'Unknown')
        model = config.get('model', 'Unknown')
        self.statusBar().showMessage(f"Configuration applied - Device: {device}, Model: {model}", 3000)
    
    @Slot()
    def load_config(self):
        """Load configuration from file"""
        # Ask for confirmation if needed
        if self.video_controller and self.video_controller._running:
            reply = QMessageBox.question(
                self,
                "Reload Configuration",
                "Reloading configuration will stop current processing. Continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.No:
                return
                
            # Stop processing
            self.video_controller.stop()
        
        # Load config
        self.config = load_configuration(self.config_file)
        
        # Update UI
        self.config_panel.set_config(self.config)
        self.export_tab.update_config_display(self.config)
        
        # Update model manager
        if self.model_manager:
            self.model_manager.update_config(self.config)
        
        # Update status
        self.statusBar().showMessage("Configuration loaded", 2000)
    
    @Slot()
    def save_config(self):
        """Save configuration to file"""
        # Get config from UI
        ui_config = self.export_tab.get_config_from_ui()
        
        # Update config
        for section in ui_config:
            if section in self.config:
                self.config[section].update(ui_config[section])
            else:
                self.config[section] = ui_config[section]
        
        # Save to file
        if save_configuration(self.config, self.config_file):
            self.statusBar().showMessage("Configuration saved", 2000)
        else:
            self.statusBar().showMessage("Error saving configuration", 2000)
            
        # Update model manager
        if self.model_manager:
            self.model_manager.update_config(self.config)
    
    @Slot()
    def open_video_file(self):
        """Open video file dialog"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Video File",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.webm);;All Files (*)"
        )
        
        if file_path:
            # Update live tab
            self.live_tab.source_changed.emit(file_path)
            
            # Update status
            self.statusBar().showMessage(f"Loaded video: {os.path.basename(file_path)}")
    
    @Slot()
    def take_snapshot(self):
        """Take snapshot of current frame"""
        if self.video_controller:
            # Get current frame
            frame = self.video_controller.capture_snapshot()
            
            if frame is not None:
                # Save frame to file
                save_dir = self.settings.value("snapshot_dir", ".")
                file_path = os.path.join(save_dir, "snapshot_" + 
                                        str(int(time.time())) + ".jpg")
                
                saved_path = save_snapshot(frame, file_path)
                
                if saved_path:
                    self.statusBar().showMessage(f"Snapshot saved: {saved_path}", 3000)
                else:
                    self.statusBar().showMessage("Error saving snapshot", 3000)
            else:
                self.statusBar().showMessage("No frame to capture", 3000)
    
    @Slot()
    def toggle_config_panel(self):
        """Toggle configuration panel visibility"""
        dock_widgets = self.findChildren(QDockWidget)
        for dock in dock_widgets:
            dock.setVisible(not dock.isVisible())
    
    @Slot()
    def toggle_performance_overlay(self):
        """Toggle performance overlay visibility"""
        if self.performance_overlay.isVisible():
            self.performance_overlay.hide()
        else:
            # Position in the corner
            self.performance_overlay.move(self.pos().x() + 10, self.pos().y() + 30)
            self.performance_overlay.show()
    
    @Slot()
    def open_grafana_dashboard(self):
        """Open Grafana dashboard in default web browser"""
        import webbrowser
        
        # Default Grafana URL - you can configure this
        grafana_url = "http://localhost:3000"
        
        # Try to read Grafana URL from config if available
        try:
            grafana_config = self.config.get("grafana", {})
            grafana_url = grafana_config.get("url", grafana_url)
        except:
            pass
        
        try:
            webbrowser.open(grafana_url)
            self.statusBar().showMessage(f"Opening Grafana dashboard: {grafana_url}", 3000)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not open Grafana dashboard:\n{str(e)}")
    
    @Slot(bool)
    def applyTheme(self, dark_theme):
        """
        Apply light or dark theme.
        
        Args:
            dark_theme: True for dark theme, False for light theme
        """
        if dark_theme:
            # Load dark theme stylesheet
            theme_file = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "resources", "themes", "dark.qss"
            )
        else:
            # Load light theme stylesheet
            theme_file = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "resources", "themes", "light.qss"
            )
            
        # Apply theme if file exists
        if os.path.exists(theme_file):
            with open(theme_file, "r") as f:
                self.setStyleSheet(f.read())
        else:
            # Fallback to built-in style
            self.setStyleSheet("")
    
    @Slot()
    def export_data(self):
        """Export data to file"""
        export_format = self.export_tab.export_format_combo.currentText()
        export_data = self.export_tab.export_data_combo.currentText()
        
        # Get file type filter based on format
        if export_format == "CSV":
            file_filter = "CSV Files (*.csv)"
            default_ext = ".csv"
        elif export_format == "JSON":
            file_filter = "JSON Files (*.json)"
            default_ext = ".json"
        elif export_format == "Excel":
            file_filter = "Excel Files (*.xlsx)"
            default_ext = ".xlsx"
        elif export_format == "PDF Report":
            file_filter = "PDF Files (*.pdf)"
            default_ext = ".pdf"
        else:
            file_filter = "All Files (*)"
            default_ext = ".txt"
        
        # Get save path
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Data",
            f"traffic_data{default_ext}",
            file_filter
        )
        
        if not file_path:
            return
            
        try:
            # Get analytics data
            analytics = self.analytics_controller.get_analytics()
            
            # Export based on format
            if export_format == "CSV":
                from utils.helpers import create_export_csv
                result = create_export_csv(analytics['detection_counts'], file_path)
            elif export_format == "JSON":
                from utils.helpers import create_export_json
                result = create_export_json(analytics, file_path)
            elif export_format == "Excel":
                # Requires openpyxl
                try:
                    import pandas as pd
                    df = pd.DataFrame({
                        'Class': list(analytics['detection_counts'].keys()),
                        'Count': list(analytics['detection_counts'].values())
                    })
                    df.to_excel(file_path, index=False)
                    result = True
                except Exception as e:
                    print(f"Excel export error: {e}")
                    result = False
            else:
                # Not implemented
                QMessageBox.information(
                    self,
                    "Not Implemented",
                    f"Export to {export_format} is not yet implemented."
                )
                return
                
            if result:
                self.statusBar().showMessage(f"Data exported to {file_path}", 3000)
            else:
                self.statusBar().showMessage("Error exporting data", 3000)
                
        except Exception as e:
            QMessageBox.critical(
                self,
                "Export Error",
                f"Error exporting data: {str(e)}"
            )
    
    @Slot()
    def show_about_dialog(self):
        """Show about dialog"""
        QMessageBox.about(
            self,
            "About Traffic Monitoring System",
            "<h3>Traffic Monitoring System</h3>"
            "<p>Based on OpenVINO™ and PySide6</p>"
            "<p>Version 1.0.0</p>"
            "<p>© 2025 GSOC Project</p>"
        )
    @Slot(bool)
    def toggle_video_processing(self, start):
        """
        Start or stop video processing.
        
        Args:
            start: True to start processing, False to stop
        """
        if self.video_controller:
            if start:
                try:
                    # Make sure the source is correctly set to what the LiveTab has
                    current_source = self.live_tab.current_source
                    print(f"DEBUG: MainWindow toggle_processing with source: {current_source} (type: {type(current_source)})")
                    
                    # Validate source
                    if current_source is None:
                        self.statusBar().showMessage("Error: No valid source selected")
                        return
                        
                    # For file sources, verify file exists
                    if isinstance(current_source, str) and not current_source.isdigit():
                        if not os.path.exists(current_source):
                            self.statusBar().showMessage(f"Error: File not found: {current_source}")
                            return
                    
                    # Ensure the source is set before starting
                    print(f"🎥 Setting video controller source to: {current_source}")
                    self.video_controller.set_source(current_source)
                    
                    # Now start processing after a short delay to ensure source is set
                    print("⏱️ Scheduling video processing start after 200ms delay...")
                    QTimer.singleShot(200, lambda: self._start_video_processing())
                    
                    source_desc = f"file: {os.path.basename(current_source)}" if isinstance(current_source, str) and os.path.exists(current_source) else f"camera: {current_source}"
                    self.statusBar().showMessage(f"Video processing started with {source_desc}")
                except Exception as e:
                    print(f"❌ Error starting video: {e}")
                    traceback.print_exc()
                    self.statusBar().showMessage(f"Error: {str(e)}")
            else:
                try:
                    print("🛑 Stopping video processing...")
                    self.video_controller.stop()
                    print("✅ Video controller stopped")
                    self.statusBar().showMessage("Video processing stopped")
                except Exception as e:
                    print(f"❌ Error stopping video: {e}")
                    traceback.print_exc()
                    
    def _start_video_processing(self):
        """Actual video processing start with extra error handling"""
        try:
            print("🚀 Starting video controller...")
            self.video_controller.start()
            print("✅ Video controller started successfully")
        except Exception as e:
            print(f"❌ Error in video processing start: {e}")
            traceback.print_exc()
            self.statusBar().showMessage(f"Video processing error: {str(e)}")
                
    def closeEvent(self, event):
        """Handle window close event"""
        # Stop processing
        if self.video_controller and self.video_controller._running:
            self.video_controller.stop()
            
        # Save settings
        self.saveSettings()
        
        # Accept close event
        event.accept()
    
    def restoreSettings(self):
        """Restore application settings"""
        # Restore window geometry
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        
        # Restore window state
        state = self.settings.value("windowState")
        if state:
            self.restoreState(state)
    
    def saveSettings(self):
        """Save application settings"""
        # Save window geometry
        self.settings.setValue("geometry", self.saveGeometry())
        
        # Save window state
        self.settings.setValue("windowState", self.saveState())
        
        # Save current directory as snapshot directory
        self.settings.setValue("snapshot_dir", os.getcwd())
    @Slot(dict)
    def update_traffic_light_status(self, stats):
        """Update status bar with traffic light information if detected"""
        traffic_light_info = stats.get('traffic_light_color', 'unknown')
        
        # Handle both string and dictionary return formats
        if isinstance(traffic_light_info, dict):
            traffic_light_color = traffic_light_info.get('color', 'unknown')
            confidence = traffic_light_info.get('confidence', 0.0)
            confidence_str = f" (Confidence: {confidence:.2f})" if confidence > 0 else ""
        else:
            traffic_light_color = traffic_light_info
            confidence = 1.0
            confidence_str = ""
            
        if traffic_light_color != 'unknown':
            current_message = self.statusBar().currentMessage()
            if not current_message or "Traffic Light" not in current_message:
                # Handle both dictionary and string formats
                if isinstance(traffic_light_color, dict):
                    color_text = traffic_light_color.get("color", "unknown").upper()
                else:
                    color_text = str(traffic_light_color).upper()
                self.statusBar().showMessage(f"Traffic Light: {color_text}{confidence_str}")
                
            # Publish traffic light status to InfluxDB
            if hasattr(self, 'data_publisher') and self.data_publisher:
                try:
                    color_for_publishing = traffic_light_color
                    if isinstance(traffic_light_color, dict):
                        color_for_publishing = traffic_light_color.get("color", "unknown")
                    self.data_publisher.publish_traffic_light_status(color_for_publishing, confidence)
                except Exception as e:
                    print(f"❌ Error publishing traffic light status: {e}")
    @Slot(dict)
    def handle_violation_detected(self, violation):
        """Handle a detected traffic violation"""
        try:
            print(f"🚨 [MAIN WINDOW] ⚠️ VIOLATION RECEIVED: {violation}")
            print(f"🚨 [MAIN WINDOW] ⚠️ VIOLATION TYPE: {type(violation)}")
            print(f"🚨 [MAIN WINDOW] ⚠️ VIOLATION KEYS: {list(violation.keys()) if isinstance(violation, dict) else 'Not a dict'}")
            
            # Flash red status message
            track_id = violation.get('track_id', 'Unknown')
            violation_type = violation.get('violation_type', violation.get('violation', 'Unknown'))
            self.statusBar().showMessage(f"🚨 VIOLATION DETECTED - Vehicle ID: {track_id}, Type: {violation_type}", 5000)
            
            # Add to violations tab
            try:
                self.violations_tab.add_violation(violation)
                print(f"🚨 [MAIN WINDOW] ✅ Added violation to violations tab")
            except Exception as e:
                print(f"🚨 [MAIN WINDOW] ❌ Error adding violation to violations tab: {e}")
            
            # Update analytics tab with violation data
            if hasattr(self.analytics_tab, 'update_violation_data'):
                print(f"🚨 [MAIN WINDOW] 🔄 Forwarding violation to analytics tab...")
                print(f"🚨 [MAIN WINDOW] 🔄 Analytics tab type: {type(self.analytics_tab)}")
                try:
                    self.analytics_tab.update_violation_data(violation)
                    print(f"🚨 [MAIN WINDOW] ✅ Successfully forwarded violation to analytics tab")
                except Exception as e:
                    print(f"🚨 [MAIN WINDOW] ❌ Error forwarding violation to analytics tab: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"🚨 [MAIN WINDOW] ❌ Analytics tab does not have update_violation_data method")
                print(f"🚨 [MAIN WINDOW] ❌ Analytics tab available methods: {[m for m in dir(self.analytics_tab) if not m.startswith('_')]}")
            
            # Update analytics
            if self.analytics_controller:
                try:
                    self.analytics_controller.register_violation(violation)
                    print(f"🚨 [MAIN WINDOW] ✅ Registered violation with analytics controller")
                except Exception as e:
                    print(f"🚨 [MAIN WINDOW] ❌ Error registering violation with analytics controller: {e}")
            else:
                print(f"🚨 [MAIN WINDOW] ❌ No analytics controller available")
            
            # Publish violation to InfluxDB
            if hasattr(self, 'data_publisher') and self.data_publisher:
                try:
                    violation_type = violation.get('type', 'red_light_violation')
                    vehicle_id = violation.get('track_id', 'unknown')
                    details = {
                        'timestamp': violation.get('timestamp', ''),
                        'confidence': violation.get('confidence', 1.0),
                        'location': violation.get('location', 'crosswalk')
                    }
                    self.data_publisher.publish_violation_event(violation_type, vehicle_id, details)
                except Exception as e:
                    print(f"❌ Error publishing violation event: {e}")
                
            print(f"🚨 Violation processed: {violation['id']} at {violation['timestamp']}")
        except Exception as e:
            print(f"❌ Error handling violation: {e}")
            import traceback
            traceback.print_exc()
    
    def _handle_video_file_selected(self, file_path):
        print(f"[VideoDetection] File selected: {file_path}")
        self.video_file_controller.set_source(file_path)
    def _handle_video_play(self):
        print("[VideoDetection] Play clicked")
        # Check if video is paused, if so resume, otherwise start
        if hasattr(self.video_file_controller, '_paused') and self.video_file_controller._paused:
            self.video_file_controller.resume()
        else:
            self.video_file_controller.play()
        # Notify VLM insights that video is playing (not paused)
        print("[MAIN WINDOW DEBUG] Notifying VLM insights: video playing")
        if hasattr(self, 'config_panel') and hasattr(self.config_panel, 'vlm_insights_widget'):
            self.config_panel.vlm_insights_widget.on_video_paused(False)
            print("[MAIN WINDOW DEBUG] VLM insights notified: not paused")
        else:
            print("[MAIN WINDOW DEBUG] VLM insights not found for play notification")
    
    def _handle_video_pause(self):
        print("[VideoDetection] Pause clicked")
        self.video_file_controller.pause()
        # Notify VLM insights that video is paused
        print("[MAIN WINDOW DEBUG] Notifying VLM insights: video paused")
        if hasattr(self, 'config_panel') and hasattr(self.config_panel, 'vlm_insights_widget'):
            self.config_panel.vlm_insights_widget.on_video_paused(True)
            print("[MAIN WINDOW DEBUG] VLM insights notified: paused")
        else:
            print("[MAIN WINDOW DEBUG] VLM insights not found for pause notification")
    def _handle_video_stop(self):
        print("[VideoDetection] Stop clicked")
        self.video_file_controller.stop()
    def _handle_video_detection_toggle(self, enabled):
        print(f"[VideoDetection] Detection toggled: {enabled}")
        self.video_file_controller.set_detection_enabled(enabled)
    def _handle_video_screenshot(self):
        print("[VideoDetection] Screenshot clicked")
        frame = self.video_file_controller.capture_snapshot()
        if frame is not None:
            save_dir = self.settings.value("snapshot_dir", ".")
            file_path = os.path.join(save_dir, "video_snapshot_" + str(int(time.time())) + ".jpg")
            saved_path = save_snapshot(frame, file_path)
            if saved_path:
                self.statusBar().showMessage(f"Video snapshot saved: {saved_path}", 3000)
            else:
                self.statusBar().showMessage("Error saving video snapshot", 3000)
        else:
            self.statusBar().showMessage("No frame to capture", 3000)
    def _handle_video_seek(self, value):
        print(f"[VideoDetection] Seek changed: {value}")
        self.video_file_controller.seek(value)
    @Slot(str)
    def handle_device_switch(self, device):
        """Handle device switch request from config panel."""
        try:
            print(f"[MAIN WINDOW] Device switch requested: {device}")
            
            # Switch device/model using ModelManager
            if hasattr(self.model_manager, 'switch_device'):
                self.model_manager.switch_device(device)
            else:
                print("[MAIN WINDOW] ModelManager.switch_device method not found")
            
            # Update video controllers
            if hasattr(self.video_controller, "on_model_switched"):
                self.video_controller.on_model_switched(device)
            if hasattr(self.video_file_controller, "on_model_switched"):
                self.video_file_controller.on_model_switched(device)
            
            # Update status
            current_model = getattr(self.model_manager, 'current_model_name', 'Unknown')
            self.statusBar().showMessage(f"Switched to {device} - {current_model}", 3000)
            print(f"[MAIN WINDOW] Device switch completed: {device} - {current_model}")
            
        except Exception as e:
            print(f"[MAIN WINDOW] Error switching device: {e}")
            import traceback
            traceback.print_exc()
            self.statusBar().showMessage(f"Error switching device: {e}", 3000)
    @Slot(dict)
    def update_performance_graphs(self, stats):
        """Update the performance graphs using the enhanced widget logic."""
        print(f"[MAIN WINDOW PERF DEBUG] update_performance_graphs ENTRY - has performance_tab: {hasattr(self, 'performance_tab')}")
        print(f"[MAIN WINDOW PERF DEBUG] update_performance_graphs called with: {stats}")
        if not hasattr(self, 'performance_tab'):
            print("[MAIN WINDOW PERF DEBUG] No performance_tab attribute, returning")
            return
        if not self.performance_tab:
            print("[MAIN WINDOW PERF DEBUG] performance_tab is None, returning")
            return
        print(f"[MAIN WINDOW PERF DEBUG] About to call performance_tab.update_performance_data")
        
        # Publish performance data to InfluxDB
        if hasattr(self, 'data_publisher') and self.data_publisher:
            try:
                fps = stats.get('fps', 0)
                inference_time = stats.get('inference_time', 0)
                cpu_usage = stats.get('cpu_usage', None)
                gpu_usage = stats.get('gpu_usage', None)
                
                self.data_publisher.publish_performance_data(fps, inference_time, cpu_usage, gpu_usage)
                
                # Publish device info periodically (every 10th frame)
                if hasattr(self, '_device_info_counter'):
                    self._device_info_counter += 1
                else:
                    self._device_info_counter = 1
                    
                if self._device_info_counter % 10 == 0:
                    self.data_publisher.publish_device_info()
            except Exception as e:
                print(f"❌ Error publishing performance data: {e}")
        
        # Enhanced analytics data with proper structure
        current_time = time.time()
        analytics_data = {
            'real_time_data': {
                'timestamps': [current_time],
                'inference_latency': [stats.get('inference_time', 0)],
                'fps': [stats.get('fps', 0)],
                'device_usage': [1 if stats.get('device', 'CPU') == 'GPU' else 0],
                'resolution_width': [int(stats.get('resolution', '640x360').split('x')[0]) if 'x' in stats.get('resolution', '') else 640],
                'resolution_height': [int(stats.get('resolution', '640x360').split('x')[1]) if 'x' in stats.get('resolution', '') else 360],
            },
            'latency_statistics': {
                'avg': stats.get('avg_inference_time', 0),
                'max': stats.get('max_inference_time', 0),
                'min': stats.get('min_inference_time', 0),
                'spike_count': stats.get('spike_count', 0)
            },
            'current_metrics': {
                'device': stats.get('device', 'CPU'),
                'resolution': stats.get('resolution', 'Unknown'),
                'model': stats.get('model_name', stats.get('model', 'Unknown')),  # Try model_name first, then model
                'fps': stats.get('fps', 0),
                'inference_time': stats.get('inference_time', 0)
            },
            'system_metrics': {
                'cpu_usage': stats.get('cpu_usage', 0),
                'gpu_usage': stats.get('gpu_usage', 0),
                'memory_usage': stats.get('memory_usage', 0)
            }
        }
        
        print(f"[PERF DEBUG] Enhanced analytics_data: {analytics_data}")
        
        # Update performance graphs with enhanced data
        self.performance_tab.update_performance_data(analytics_data)

    def _handle_vlm_analysis(self, frame, prompt):
        """Handle VLM analysis requests."""
        print(f"[MAIN WINDOW DEBUG] _handle_vlm_analysis called")
        print(f"[MAIN WINDOW DEBUG] Frame type: {type(frame)}, shape: {frame.shape if hasattr(frame, 'shape') else 'N/A'}")
        print(f"[MAIN WINDOW DEBUG] Prompt: '{prompt}'")
        
        try:
            # Check if VLM controller is available
            if hasattr(self, 'vlm_controller') and self.vlm_controller:
                print(f"[MAIN WINDOW DEBUG] Using VLM controller for analysis")
                
                # Connect VLM result to insights widget if not already connected
                if not hasattr(self, '_vlm_connected'):
                    print(f"[MAIN WINDOW DEBUG] Connecting VLM controller results to insights widget")
                    self.vlm_controller.result_ready.connect(
                        lambda result: self._handle_vlm_result(result), 
                        Qt.QueuedConnection
                    )
                    self._vlm_connected = True
                
                # Process image with VLM controller
                self.vlm_controller.process_image(frame, prompt)
                print(f"[MAIN WINDOW DEBUG] VLM controller processing started")
                
            else:
                print(f"[MAIN WINDOW DEBUG] VLM controller not available, using mock analysis")
                # Fallback to mock analysis
                import cv2
                import numpy as np
                result = self._generate_mock_analysis(frame, prompt)
                print(f"[MAIN WINDOW DEBUG] Mock analysis generated: {len(result)} characters")
                
                # Send result back to VLM insights widget
                if hasattr(self.config_panel, 'vlm_insights_widget'):
                    print(f"[MAIN WINDOW DEBUG] Sending mock result to VLM insights widget")
                    self.config_panel.vlm_insights_widget.on_analysis_result(result)
                    print(f"[MAIN WINDOW DEBUG] Mock result sent successfully")
                else:
                    print(f"[MAIN WINDOW DEBUG] VLM insights widget not found")
                
        except Exception as e:
            print(f"[VLM ERROR] Error in analysis: {e}")
            if hasattr(self.config_panel, 'vlm_insights_widget'):
                self.config_panel.vlm_insights_widget.on_analysis_result(f"Analysis error: {str(e)}")

    def _handle_vlm_result(self, result):
        """Handle VLM controller results."""
        print(f"[MAIN WINDOW DEBUG] _handle_vlm_result called")
        print(f"[MAIN WINDOW DEBUG] Result type: {type(result)}")
        
        try:
            # Extract answer from result dict
            if isinstance(result, dict):
                if 'response' in result:
                    answer = result['response']
                    print(f"[MAIN WINDOW DEBUG] Extracted response: {len(str(answer))} characters")
                elif 'answer' in result:
                    answer = result['answer']
                    print(f"[MAIN WINDOW DEBUG] Extracted answer: {len(str(answer))} characters")
                else:
                    answer = str(result)
                    print(f"[MAIN WINDOW DEBUG] Using result as string: {len(answer)} characters")
            else:
                answer = str(result)
                print(f"[MAIN WINDOW DEBUG] Using result as string: {len(answer)} characters")
            
            # Send result to VLM insights widget
            if hasattr(self.config_panel, 'vlm_insights_widget'):
                print(f"[MAIN WINDOW DEBUG] Sending VLM result to insights widget")
                self.config_panel.vlm_insights_widget.on_analysis_result(answer)
                print(f"[MAIN WINDOW DEBUG] VLM result sent successfully")
            else:
                print(f"[MAIN WINDOW DEBUG] VLM insights widget not found")
                
        except Exception as e:
            print(f"[VLM ERROR] Error handling VLM result: {e}")

    def _forward_frame_to_vlm(self, frame, detections, fps):
        """Forward frame to VLM insights widget."""
        print(f"[MAIN WINDOW DEBUG] _forward_frame_to_vlm called")
        print(f"[MAIN WINDOW DEBUG] Frame type: {type(frame)}, shape: {frame.shape if hasattr(frame, 'shape') else 'N/A'}")
        print(f"[MAIN WINDOW DEBUG] Detections count: {len(detections) if detections else 0}")
        print(f"[MAIN WINDOW DEBUG] FPS: {fps}")
        
        # Publish detection events to InfluxDB
        if hasattr(self, 'data_publisher') and self.data_publisher and detections:
            try:
                # Count vehicles and pedestrians
                vehicle_count = 0
                pedestrian_count = 0
                
                for detection in detections:
                    label = ""
                    if isinstance(detection, dict):
                        label = detection.get('label', '').lower()
                    elif hasattr(detection, 'label'):
                        label = getattr(detection, 'label', '').lower()
                    elif hasattr(detection, 'class_name'):
                        label = getattr(detection, 'class_name', '').lower()
                    elif hasattr(detection, 'cls'):
                        label = str(getattr(detection, 'cls', '')).lower()
                    
                    # Debug the label detection
                    if label and label != 'traffic light':
                        print(f"[PUBLISHER DEBUG] Detected object: {label}")
                    
                    if label in ['car', 'truck', 'bus', 'motorcycle', 'vehicle']:
                        vehicle_count += 1
                    elif label in ['person', 'pedestrian']:
                        pedestrian_count += 1
                
                # Also try to get vehicle count from tracked vehicles if available
                if vehicle_count == 0 and hasattr(self, 'video_file_controller'):
                    try:
                        # Try to get vehicle count from current analysis data
                        analysis_data = getattr(self.video_file_controller, 'get_current_analysis_data', lambda: {})()
                        if isinstance(analysis_data, dict):
                            tracked_vehicles = analysis_data.get('tracked_vehicles', [])
                            if tracked_vehicles:
                                vehicle_count = len(tracked_vehicles)
                                print(f"[PUBLISHER DEBUG] Using tracked vehicle count: {vehicle_count}")
                    except:
                        pass
                
                self.data_publisher.publish_detection_events(vehicle_count, pedestrian_count)
            except Exception as e:
                print(f"❌ Error publishing detection events: {e}")
        
        try:
            if hasattr(self.config_panel, 'vlm_insights_widget'):
                print(f"[MAIN WINDOW DEBUG] Forwarding frame to VLM insights widget")
                self.config_panel.vlm_insights_widget.set_current_frame(frame)
                
                # Store detection data for VLM analysis
                if hasattr(self.config_panel.vlm_insights_widget, 'set_detection_data'):
                    print(f"[MAIN WINDOW DEBUG] Setting detection data for VLM")
                    detection_data = {
                        'detections': detections,
                        'fps': fps,
                        'timestamp': time.time()
                    }
                    # Get additional data from video controller if available
                    if hasattr(self.video_file_controller, 'get_current_analysis_data'):
                        analysis_data = self.video_file_controller.get_current_analysis_data()
                        detection_data.update(analysis_data)
                    
                    self.config_panel.vlm_insights_widget.set_detection_data(detection_data)
                    print(f"[MAIN WINDOW DEBUG] Detection data set successfully")
                
                print(f"[MAIN WINDOW DEBUG] Frame forwarded successfully")
            else:
                print(f"[MAIN WINDOW DEBUG] VLM insights widget not found for frame forwarding")
        except Exception as e:
            print(f"[MAIN WINDOW DEBUG] Error forwarding frame to VLM: {e}")
    
    def _forward_frame_to_analytics(self, frame, detections, fps):
        """Forward frame data to analytics tab for real-time updates."""
        try:
            print(f"[ANALYTICS DEBUG] Forwarding frame data to analytics tab")
            print(f"[ANALYTICS DEBUG] Detections count: {len(detections) if detections else 0}")
            
            # Prepare detection data for analytics
            detection_data = {
                'detections': detections,
                'fps': fps,
                'timestamp': time.time(),
                'frame_shape': frame.shape if hasattr(frame, 'shape') else None
            }
            
            # Get additional analysis data from video controller
            if hasattr(self.video_file_controller, 'get_current_analysis_data'):
                analysis_data = self.video_file_controller.get_current_analysis_data()
                if analysis_data:
                    detection_data.update(analysis_data)
                    print(f"[ANALYTICS DEBUG] Updated with analysis data: {list(analysis_data.keys())}")
            
            # Forward to analytics tab
            if hasattr(self.analytics_tab, 'update_detection_data'):
                self.analytics_tab.update_detection_data(detection_data)
                print(f"[ANALYTICS DEBUG] Detection data forwarded to analytics tab successfully")
            else:
                print(f"[ANALYTICS DEBUG] Analytics tab update_detection_data method not found")
                
        except Exception as e:
            print(f"[ANALYTICS DEBUG] Error forwarding frame to analytics: {e}")
            import traceback
            traceback.print_exc()
    
    def _generate_mock_analysis(self, frame, prompt):
        """Generate a mock analysis response based on frame content and prompt."""
        try:
            import cv2
            import numpy as np
            
            # Analyze frame properties
            h, w = frame.shape[:2] if frame is not None else (0, 0)
            
            # Basic image analysis
            analysis_parts = []
            
            if "traffic" in prompt.lower():
                analysis_parts.append("🚦 Traffic Analysis:")
                analysis_parts.append(f"• Frame resolution: {w}x{h}")
                analysis_parts.append("• Detected scene: Urban traffic intersection")
                analysis_parts.append("• Visible elements: Road, potential vehicles")
                analysis_parts.append("• Traffic flow appears to be moderate")
                
            elif "safety" in prompt.lower():
                analysis_parts.append("⚠️ Safety Assessment:")
                analysis_parts.append("• Monitoring for traffic violations")
                analysis_parts.append("• Checking lane discipline")
                analysis_parts.append("• Observing traffic light compliance")
                analysis_parts.append("• Overall safety level: Monitoring required")
                
            else:
                analysis_parts.append("🔍 General Analysis:")
                analysis_parts.append(f"• Image dimensions: {w}x{h} pixels")
                analysis_parts.append("• Scene type: Traffic monitoring view")
                analysis_parts.append("• Quality: Processing frame for analysis")
                analysis_parts.append(f"• Prompt: {prompt[:100]}...")
            
            # Add timestamp and disclaimer
            from datetime import datetime
            timestamp = datetime.now().strftime("%H:%M:%S")
            analysis_parts.append(f"\n📝 Analysis completed at {timestamp}")
            analysis_parts.append("ℹ️ Note: This is a mock analysis. Full AI analysis requires compatible OpenVINO setup.")
            
            return "\n".join(analysis_parts)
            
        except Exception as e:
            return f"Unable to analyze frame: {str(e)}"

    # Smart Intersection Signal Handlers
    @Slot(bool)
    def _handle_smart_intersection_enabled(self, enabled):
        """Handle smart intersection mode toggle"""
        print(f"🚦 Smart Intersection mode {'enabled' if enabled else 'disabled'}")
        
        if self.smart_intersection_controller:
            self.smart_intersection_controller.set_enabled(enabled)
        
        # Update status
        if enabled:
            self.statusBar().showMessage("Smart Intersection mode activated")
        else:
            self.statusBar().showMessage("Standard detection mode")
    
    @Slot(bool)
    def _handle_multi_camera_mode(self, enabled):
        """Handle multi-camera mode toggle"""
        print(f"📹 Multi-camera mode {'enabled' if enabled else 'disabled'}")
        
        if self.smart_intersection_controller:
            self.smart_intersection_controller.set_multi_camera_mode(enabled)
    
    @Slot(dict)
    def _handle_roi_configuration_changed(self, roi_config):
        """Handle ROI configuration changes"""
        print(f"🎯 ROI configuration updated: {len(roi_config.get('rois', []))} regions")
        
        if self.smart_intersection_controller:
            self.smart_intersection_controller.update_roi_config(roi_config)
    
    @Slot(bool)
    def _handle_scene_analytics_toggle(self, enabled):
        """Handle scene analytics toggle"""
        print(f"📊 Scene analytics {'enabled' if enabled else 'disabled'}")
        
        if self.smart_intersection_controller:
            self.smart_intersection_controller.set_scene_analytics(enabled)
    
    @Slot(dict)
    def _handle_scene_analytics_update(self, analytics_data):
        """Handle scene analytics data updates"""
        try:
            # Update video detection tab with smart intersection data
            smart_stats = {
                'total_objects': analytics_data.get('total_objects', 0),
                'active_tracks': analytics_data.get('active_tracks', 0),
                'roi_events': analytics_data.get('roi_events', 0),
                'crosswalk_events': analytics_data.get('crosswalk_events', 0),
                'lane_events': analytics_data.get('lane_events', 0),
                'safety_events': analytics_data.get('safety_events', 0),
                'north_objects': analytics_data.get('camera_stats', {}).get('north', 0),
                'east_objects': analytics_data.get('camera_stats', {}).get('east', 0),
                'south_objects': analytics_data.get('camera_stats', {}).get('south', 0),
                'west_objects': analytics_data.get('camera_stats', {}).get('west', 0),
                'fps': analytics_data.get('fps', 0),
                'processing_time': analytics_data.get('processing_time_ms', 0),
                'gpu_usage': analytics_data.get('gpu_usage', 0),
                'memory_usage': analytics_data.get('memory_usage', 0)
            }
            
            # Update both video tabs with stats
            self.video_detection_only_tab.update_stats(smart_stats)
            self.smart_intersection_tab.update_stats(smart_stats)
            
            # Update analytics tab if it has smart intersection support
            if hasattr(self.analytics_tab, 'update_smart_intersection_analytics'):
                self.analytics_tab.update_smart_intersection_analytics(analytics_data)
                
        except Exception as e:
            print(f"Error handling scene analytics update: {e}")
