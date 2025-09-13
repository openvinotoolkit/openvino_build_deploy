# from PySide6.QtWidgets import (
#     QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSlider, QCheckBox, 
#     QFileDialog, QSizePolicy, QGridLayout, QFrame, QSpacerItem, QTabWidget,
#     QComboBox, QSpinBox, QDoubleSpinBox, QGroupBox, QScrollArea, QTextEdit,
#     QProgressBar, QSplitter, QListWidget, QListWidgetItem
# )
# from PySide6.QtCore import Signal, Qt, QTimer, QThread
# from PySide6.QtGui import QPixmap, QIcon, QFont, QPainter, QPen, QBrush, QColor
# import json
# import os
# from pathlib import Path

# class SmartIntersectionOverlay(QFrame):
#     """Advanced overlay for Smart Intersection analytics."""
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.setStyleSheet("""
#             background: rgba(0,20,40,0.85);
#             border: 2px solid #03DAC5;
#             border-radius: 12px;
#             color: #fff;
#             font-family: 'Consolas', 'SF Mono', 'monospace';
#             font-size: 12px;
#         """)
#         self.setFixedHeight(140)
#         self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
#         self.setAttribute(Qt.WA_TransparentForMouseEvents)
        
#         layout = QVBoxLayout(self)
#         layout.setContentsMargins(16, 12, 16, 12)
#         layout.setSpacing(4)
        
#         # Title
#         title = QLabel("🚦 Smart Intersection Analytics")
#         title.setStyleSheet("color: #03DAC5; font-weight: bold; font-size: 14px;")
#         layout.addWidget(title)
        
#         # Scene data
#         self.scene_label = QLabel("Scene: Multi-Camera Fusion")
#         self.tracking_label = QLabel("Active Tracks: 0")
#         self.roi_label = QLabel("ROI Events: 0")
        
#         # Camera data
#         self.camera_label = QLabel("Cameras: North(0) East(0) South(0) West(0)")
        
#         # Analytics data
#         self.analytics_label = QLabel("Analytics: Crosswalk(0) Lane(0) Safety(0)")
        
#         for w in [self.scene_label, self.tracking_label, self.roi_label, 
#                   self.camera_label, self.analytics_label]:
#             w.setStyleSheet("color: #fff;")
#             layout.addWidget(w)

#     def update_smart_intersection(self, scene_data):
#         """Update smart intersection specific data"""
#         if not scene_data:
#             return
            
#         # Update tracking info
#         active_tracks = scene_data.get('active_tracks', 0)
#         self.tracking_label.setText(f"Active Tracks: {active_tracks}")
        
#         # Update ROI events
#         roi_events = scene_data.get('roi_events', 0)
#         self.roi_label.setText(f"ROI Events: {roi_events}")
        
#         # Update camera data
#         cameras = scene_data.get('cameras', {})
#         north = cameras.get('north', 0)
#         east = cameras.get('east', 0)
#         south = cameras.get('south', 0)
#         west = cameras.get('west', 0)
#         self.camera_label.setText(f"Cameras: North({north}) East({east}) South({south}) West({west})")
        
#         # Update analytics
#         analytics = scene_data.get('analytics', {})
#         crosswalk = analytics.get('crosswalk_events', 0)
#         lane = analytics.get('lane_events', 0)
#         safety = analytics.get('safety_events', 0)
#         self.analytics_label.setText(f"Analytics: Crosswalk({crosswalk}) Lane({lane}) Safety({safety})")


# class IntersectionROIWidget(QFrame):
#     """Widget for defining and managing ROI regions for smart intersection"""
#     roi_updated = Signal(dict)
    
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.setStyleSheet("""
#             QFrame {
#                 background: #1a1a1a;
#                 border: 1px solid #424242;
#                 border-radius: 8px;
#             }
#         """)
#         self.setFixedWidth(300)
        
#         layout = QVBoxLayout(self)
#         layout.setContentsMargins(16, 16, 16, 16)
        
#         # Title
#         title = QLabel("🎯 Region of Interest (ROI)")
#         title.setStyleSheet("color: #03DAC5; font-weight: bold; font-size: 14px;")
#         layout.addWidget(title)
        
#         # ROI Type selection
#         type_layout = QHBoxLayout()
#         type_layout.addWidget(QLabel("Type:"))
#         self.roi_type = QComboBox()
#         self.roi_type.addItems(["Crosswalk", "Traffic Lane", "Safety Zone", "Intersection Center"])
#         type_layout.addWidget(self.roi_type)
#         layout.addLayout(type_layout)
        
#         # ROI List
#         self.roi_list = QListWidget()
#         self.roi_list.setMaximumHeight(120)
#         layout.addWidget(self.roi_list)
        
#         # ROI Controls
#         roi_controls = QHBoxLayout()
#         self.add_roi_btn = QPushButton("Add ROI")
#         self.delete_roi_btn = QPushButton("Delete")
#         self.add_roi_btn.setStyleSheet("background: #27ae60; color: white; border-radius: 4px; padding: 6px;")
#         self.delete_roi_btn.setStyleSheet("background: #e74c3c; color: white; border-radius: 4px; padding: 6px;")
#         roi_controls.addWidget(self.add_roi_btn)
#         roi_controls.addWidget(self.delete_roi_btn)
#         layout.addLayout(roi_controls)
        
#         # Analytics settings
#         analytics_group = QGroupBox("Analytics Settings")
#         analytics_layout = QVBoxLayout(analytics_group)
        
#         self.enable_tracking = QCheckBox("Multi-Object Tracking")
#         self.enable_speed = QCheckBox("Speed Estimation")
#         self.enable_direction = QCheckBox("Direction Analysis")
#         self.enable_safety = QCheckBox("Safety Monitoring")
        
#         for cb in [self.enable_tracking, self.enable_speed, self.enable_direction, self.enable_safety]:
#             cb.setChecked(True)
#             cb.setStyleSheet("color: white;")
#             analytics_layout.addWidget(cb)
        
#         layout.addWidget(analytics_group)
        
#         # Connect signals
#         self.add_roi_btn.clicked.connect(self._add_roi)
#         self.delete_roi_btn.clicked.connect(self._delete_roi)
        
#         # Initialize with default ROIs
#         self._init_default_rois()
    
#     def _init_default_rois(self):
#         """Initialize with default intersection ROIs"""
#         default_rois = [
#             "North Crosswalk",
#             "South Crosswalk", 
#             "East Crosswalk",
#             "West Crosswalk",
#             "Center Intersection",
#             "North Lane",
#             "South Lane",
#             "East Lane", 
#             "West Lane"
#         ]
        
#         for roi in default_rois:
#             item = QListWidgetItem(roi)
#             item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
#             item.setCheckState(Qt.Checked)
#             self.roi_list.addItem(item)
    
#     def _add_roi(self):
#         """Add new ROI"""
#         roi_type = self.roi_type.currentText()
#         roi_name = f"{roi_type}_{self.roi_list.count() + 1}"
        
#         item = QListWidgetItem(roi_name)
#         item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
#         item.setCheckState(Qt.Checked)
#         self.roi_list.addItem(item)
        
#         self._emit_roi_update()
    
#     def _delete_roi(self):
#         """Delete selected ROI"""
#         current_row = self.roi_list.currentRow()
#         if current_row >= 0:
#             self.roi_list.takeItem(current_row)
#             self._emit_roi_update()
    
#     def _emit_roi_update(self):
#         """Emit ROI configuration update"""
#         roi_config = {
#             'rois': [],
#             'analytics': {
#                 'tracking': self.enable_tracking.isChecked(),
#                 'speed': self.enable_speed.isChecked(),
#                 'direction': self.enable_direction.isChecked(),
#                 'safety': self.enable_safety.isChecked()
#             }
#         }
        
#         for i in range(self.roi_list.count()):
#             item = self.roi_list.item(i)
#             roi_config['rois'].append({
#                 'name': item.text(),
#                 'enabled': item.checkState() == Qt.Checked
#             })
        
#         self.roi_updated.emit(roi_config)


# class MultiCameraView(QFrame):
#     """Multi-camera view for smart intersection"""
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.setStyleSheet("""
#             QFrame {
#                 background: #0a0a0a;
#                 border: 2px solid #424242;
#                 border-radius: 8px;
#             }
#         """)
        
#         layout = QGridLayout(self)
#         layout.setContentsMargins(8, 8, 8, 8)
#         layout.setSpacing(4)
        
#         # Create camera views
#         self.camera_views = {}
#         positions = [('North', 0, 1), ('West', 1, 0), ('East', 1, 2), ('South', 2, 1)]
        
#         for pos_name, row, col in positions:
#             view = self._create_camera_view(pos_name)
#             self.camera_views[pos_name.lower()] = view
#             layout.addWidget(view, row, col)
        
#         # Center intersection view
#         center_view = self._create_intersection_center()
#         layout.addWidget(center_view, 1, 1)
    
#     def _create_camera_view(self, position):
#         """Create individual camera view"""
#         view = QFrame()
#         view.setStyleSheet("""
#             background: #1a1a1a;
#             border: 1px solid #555;
#             border-radius: 4px;
#         """)
#         view.setMinimumSize(160, 120)
#         view.setMaximumSize(200, 150)
        
#         layout = QVBoxLayout(view)
#         layout.setContentsMargins(4, 4, 4, 4)
        
#         # Title
#         title = QLabel(f"📹 {position}")
#         title.setStyleSheet("color: #03DAC5; font-weight: bold; font-size: 10px;")
#         title.setAlignment(Qt.AlignCenter)
#         layout.addWidget(title)
        
#         # Video area
#         video_area = QLabel("No feed")
#         video_area.setStyleSheet("background: #000; color: #666; border: 1px dashed #333;")
#         video_area.setAlignment(Qt.AlignCenter)
#         video_area.setMinimumHeight(80)
#         layout.addWidget(video_area)
        
#         # Stats
#         stats = QLabel("Objects: 0")
#         stats.setStyleSheet("color: #aaa; font-size: 9px;")
#         stats.setAlignment(Qt.AlignCenter)
#         layout.addWidget(stats)
        
#         return view
    
#     def _create_intersection_center(self):
#         """Create center intersection overview"""
#         view = QFrame()
#         view.setStyleSheet("""
#             background: #2a1a1a;
#             border: 2px solid #03DAC5;
#             border-radius: 8px;
#         """)
#         view.setMinimumSize(160, 120)
#         view.setMaximumSize(200, 150)
        
#         layout = QVBoxLayout(view)
#         layout.setContentsMargins(8, 8, 8, 8)
        
#         title = QLabel("🚦 Intersection")
#         title.setStyleSheet("color: #03DAC5; font-weight: bold; font-size: 12px;")
#         title.setAlignment(Qt.AlignCenter)
#         layout.addWidget(title)
        
#         # Intersection map
#         map_area = QLabel("Scene Map")
#         map_area.setStyleSheet("background: #000; color: #03DAC5; border: 1px solid #03DAC5;")
#         map_area.setAlignment(Qt.AlignCenter)
#         map_area.setMinimumHeight(80)
#         layout.addWidget(map_area)
        
#         # Total stats
#         total_stats = QLabel("Total Objects: 0")
#         total_stats.setStyleSheet("color: #03DAC5; font-size: 10px; font-weight: bold;")
#         total_stats.setAlignment(Qt.AlignCenter)
#         layout.addWidget(total_stats)
        
#         return view
    
#     def update_camera_feed(self, camera_position, pixmap, object_count=0):
#         """Update specific camera feed"""
#         if camera_position.lower() in self.camera_views:
#             view = self.camera_views[camera_position.lower()]
#             video_label = view.findChild(QLabel)
#             if video_label and pixmap:
#                 scaled = pixmap.scaled(video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
#                 video_label.setPixmap(scaled)
                
#             # Update stats
#             stats_labels = view.findChildren(QLabel)
#             if len(stats_labels) >= 3:  # title, video, stats
#                 stats_labels[2].setText(f"Objects: {object_count}")


# class DiagnosticOverlay(QFrame):
#     """Semi-transparent overlay for diagnostics."""
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.setStyleSheet("""
#             background: rgba(0,0,0,0.5);
#             border-radius: 8px;
#             color: #fff;
#             font-family: 'Consolas', 'SF Mono', 'monospace';
#             font-size: 13px;
#         """)
#         # self.setFixedWidth(260)  # Remove fixed width
#         self.setFixedHeight(90)
#         self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)  # Allow horizontal stretch
#         self.setAttribute(Qt.WA_TransparentForMouseEvents)
#         layout = QVBoxLayout(self)
#         layout.setContentsMargins(12, 8, 12, 8)
#         self.model_label = QLabel("Model: -")
#         self.device_label = QLabel("Device: -")
#         self.stats_label = QLabel("Cars: 0 | Trucks: 0 | Ped: 0 | TLights: 0 | Moto: 0")
#         for w in [self.model_label, self.device_label, self.stats_label]:
#             w.setStyleSheet("color: #fff;")
#             layout.addWidget(w)
#         layout.addStretch(1)

#     def update_overlay(self, model, device, cars, trucks, peds, tlights, motorcycles):
#         self.model_label.setText(f"Model: {model}")
#         self.device_label.setText(f"Device: {device}")
#         self.stats_label.setText(f"Cars: {cars} | Trucks: {trucks} | Ped: {peds} | TLights: {tlights} | Moto: {motorcycles}")

# class VideoDetectionTab(QWidget):
#     file_selected = Signal(str)
#     play_clicked = Signal()
#     pause_clicked = Signal()
#     stop_clicked = Signal()
#     detection_toggled = Signal(bool)
#     screenshot_clicked = Signal()
#     seek_changed = Signal(int)
#     auto_select_model_device = Signal()
    
#     # Smart Intersection signals
#     smart_intersection_enabled = Signal(bool)
#     multi_camera_mode_enabled = Signal(bool)
#     roi_configuration_changed = Signal(dict)
#     scene_analytics_toggled = Signal(bool)

#     def __init__(self):
#         super().__init__()
#         self.video_loaded = False
#         self.smart_intersection_mode = False
#         self.multi_camera_mode = False
        
#         # Load smart intersection config
#         self.load_smart_intersection_config()
        
#         # Main layout
#         main_layout = QHBoxLayout(self)
#         main_layout.setContentsMargins(16, 16, 16, 16)
#         main_layout.setSpacing(16)
        
#         # Left panel - video and controls
#         left_panel = self._create_left_panel()
#         main_layout.addWidget(left_panel, 3)  # 3/4 of the space
        
#         # Right panel - smart intersection controls
#         right_panel = self._create_right_panel()
#         main_layout.addWidget(right_panel, 1)  # 1/4 of the space
        
#     def load_smart_intersection_config(self):
#         """Load smart intersection configuration"""
#         config_path = Path(__file__).parent.parent / "config" / "smart-intersection" / "desktop-config.json"
#         try:
#             if config_path.exists():
#                 with open(config_path, 'r') as f:
#                     self.smart_config = json.load(f)
#             else:
#                 self.smart_config = self._get_default_config()
#         except Exception as e:
#             print(f"Error loading smart intersection config: {e}")
#             self.smart_config = self._get_default_config()
    
#     def _get_default_config(self):
#         """Get default smart intersection configuration"""
#         return {
#             "desktop_app_config": {
#                 "scene_analytics": {
#                     "enable_multi_camera": True,
#                     "enable_roi_analytics": True,
#                     "enable_vlm_integration": True
#                 },
#                 "camera_settings": {
#                     "max_cameras": 4,
#                     "default_fps": 30
#                 },
#                 "analytics_settings": {
#                     "object_tracking": True,
#                     "speed_estimation": True,
#                     "direction_analysis": True,
#                     "safety_monitoring": True
#                 }
#             }
#         }
    
#     def _create_left_panel(self):
#         """Create main video panel"""
#         panel = QWidget()
#         layout = QVBoxLayout(panel)
#         layout.setContentsMargins(0, 0, 0, 0)
#         layout.setSpacing(16)
        
#         # Smart Intersection Mode Toggle
#         mode_bar = self._create_mode_bar()
#         layout.addWidget(mode_bar)
        
#         # File select bar
#         file_bar = self._create_file_bar()
#         layout.addWidget(file_bar)
        
#         # Video display area (with tabs for different modes)
#         self.video_tabs = QTabWidget()
#         self.video_tabs.setStyleSheet("""
#             QTabWidget::pane {
#                 border: 1px solid #424242;
#                 background: #121212;
#                 border-radius: 8px;
#             }
#             QTabBar::tab {
#                 background: #232323;
#                 color: #fff;
#                 padding: 8px 16px;
#                 margin-right: 2px;
#                 border-top-left-radius: 8px;
#                 border-top-right-radius: 8px;
#             }
#             QTabBar::tab:selected {
#                 background: #03DAC5;
#                 color: #000;
#             }
#         """)
        
#         # Single camera tab
#         self.single_cam_widget = self._create_single_camera_view()
#         self.video_tabs.addTab(self.single_cam_widget, "📹 Single Camera")
        
#         # Multi-camera tab  
#         self.multi_cam_widget = MultiCameraView()
#         self.video_tabs.addTab(self.multi_cam_widget, "🚦 Multi-Camera Intersection")
        
#         layout.addWidget(self.video_tabs)
        
#         # Analytics overlay
#         self.analytics_overlay = self._create_analytics_overlay()
#         layout.addWidget(self.analytics_overlay)
        
#         # Control bar
#         control_bar = self._create_control_bar()
#         layout.addWidget(control_bar)
        
#         return panel
    
#     def _create_mode_bar(self):
#         """Create smart intersection mode toggle bar"""
#         bar = QFrame()
#         bar.setStyleSheet("""
#             QFrame {
#                 background: #1a2332;
#                 border: 2px solid #03DAC5;
#                 border-radius: 12px;
#                 padding: 8px;
#             }
#         """)
#         bar.setFixedHeight(60)
        
#         layout = QHBoxLayout(bar)
#         layout.setContentsMargins(16, 8, 16, 8)
        
#         # Smart Intersection Toggle
#         self.smart_intersection_toggle = QCheckBox("🚦 Smart Intersection Mode")
#         self.smart_intersection_toggle.setStyleSheet("""
#             QCheckBox {
#                 color: #03DAC5;
#                 font-weight: bold;
#                 font-size: 14px;
#             }
#             QCheckBox::indicator {
#                 width: 20px;
#                 height: 20px;
#             }
#             QCheckBox::indicator:checked {
#                 background: #03DAC5;
#                 border: 2px solid #03DAC5;
#                 border-radius: 4px;
#             }
#         """)
#         self.smart_intersection_toggle.toggled.connect(self._toggle_smart_intersection)
#         layout.addWidget(self.smart_intersection_toggle)
        
#         layout.addSpacing(32)
        
#         # Multi-camera Toggle
#         self.multi_camera_toggle = QCheckBox("📹 Multi-Camera Fusion")
#         self.multi_camera_toggle.setStyleSheet("""
#             QCheckBox {
#                 color: #e67e22;
#                 font-weight: bold;
#                 font-size: 14px;
#             }
#             QCheckBox::indicator {
#                 width: 20px;
#                 height: 20px;
#             }
#             QCheckBox::indicator:checked {
#                 background: #e67e22;
#                 border: 2px solid #e67e22;
#                 border-radius: 4px;
#             }
#         """)
#         self.multi_camera_toggle.toggled.connect(self._toggle_multi_camera)
#         layout.addWidget(self.multi_camera_toggle)
        
#         layout.addStretch()
        
#         # Status indicator
#         self.mode_status = QLabel("Standard Detection Mode")
#         self.mode_status.setStyleSheet("color: #bbb; font-size: 12px;")
#         layout.addWidget(self.mode_status)
        
#         return bar
    
#     def _create_file_bar(self):
#         """Create file selection bar"""
#         widget = QWidget()
#         bar = QHBoxLayout(widget)
        
#         self.file_btn = QPushButton()
#         self.file_btn.setIcon(QIcon.fromTheme("folder-video"))
#         self.file_btn.setText("Select Video")
#         self.file_btn.setStyleSheet("padding: 8px 18px; border-radius: 8px; background: #232323; color: #fff;")
#         self.file_label = QLabel("No file selected")
#         self.file_label.setStyleSheet("color: #bbb; font-size: 13px;")
#         self.file_btn.clicked.connect(self._select_file)
        
#         bar.addWidget(self.file_btn)
#         bar.addWidget(self.file_label)
#         bar.addStretch()
        
#         return widget
    
#     def _create_single_camera_view(self):
#         """Create single camera view widget"""
#         widget = QWidget()
#         layout = QVBoxLayout(widget)
#         layout.setContentsMargins(0, 0, 0, 0)
        
#         # Video frame
#         video_frame = QFrame()
#         video_frame.setStyleSheet("""
#             background: #121212;
#             border: 1px solid #424242;
#             border-radius: 8px;
#         """)
#         video_frame.setMinimumSize(640, 360)
#         video_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
#         video_layout = QVBoxLayout(video_frame)
#         video_layout.setContentsMargins(0, 0, 0, 0)
#         video_layout.setAlignment(Qt.AlignCenter)
        
#         self.video_label = QLabel()
#         self.video_label.setAlignment(Qt.AlignCenter)
#         self.video_label.setStyleSheet("background: transparent; color: #888; font-size: 18px;")
#         self.video_label.setText("No video loaded. Please select a file.")
#         self.video_label.setMinimumSize(640, 360)
#         self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
#         video_layout.addWidget(self.video_label)
        
#         layout.addWidget(video_frame)
#         return widget
    
#     def _create_analytics_overlay(self):
#         """Create analytics overlay that switches based on mode"""
#         container = QWidget()
#         self.overlay_layout = QVBoxLayout(container)
#         self.overlay_layout.setContentsMargins(0, 0, 0, 0)
        
#         # Standard overlay
#         self.standard_overlay = DiagnosticOverlay()
#         self.standard_overlay.setStyleSheet(self.standard_overlay.styleSheet() + "border: 1px solid #03DAC5;")
        
#         # Smart intersection overlay
#         self.smart_overlay = SmartIntersectionOverlay()
        
#         # Badge bar
#         self.badge_bar = QHBoxLayout()
#         self.badge_bar.setContentsMargins(0, 8, 0, 8)
        
#         self.fps_badge = QLabel("FPS: --")
#         self.fps_badge.setStyleSheet("background: #27ae60; color: #fff; border-radius: 12px; padding: 4px 24px; font-weight: bold; font-size: 15px;")
#         self.fps_badge.setAlignment(Qt.AlignCenter)
        
#         self.inference_badge = QLabel("Inference: -- ms")
#         self.inference_badge.setStyleSheet("background: #2980b9; color: #fff; border-radius: 12px; padding: 4px 24px; font-weight: bold; font-size: 15px;")
#         self.inference_badge.setAlignment(Qt.AlignCenter)
        
#         self.badge_bar.addWidget(self.fps_badge)
#         self.badge_bar.addSpacing(12)
#         self.badge_bar.addWidget(self.inference_badge)
#         self.badge_bar.addSpacing(18)
        
#         # Add current overlay (start with standard)
#         self.current_overlay = self.standard_overlay
#         self.badge_bar.addWidget(self.current_overlay)
#         self.badge_bar.addStretch()
        
#         self.overlay_layout.addLayout(self.badge_bar)
#         return container
    
#     def _create_control_bar(self):
#         """Create control bar"""
#         widget = QWidget()
#         control_bar = QHBoxLayout(widget)
#         control_bar.setContentsMargins(0, 16, 0, 0)
        
#         # Playback controls
#         self.play_btn = QPushButton()
#         self.play_btn.setIcon(QIcon.fromTheme("media-playback-start"))
#         self.play_btn.setToolTip("Play")
#         self.play_btn.setFixedSize(48, 48)
#         self.play_btn.setEnabled(False)
#         self.play_btn.setStyleSheet(self._button_style())
        
#         self.pause_btn = QPushButton()
#         self.pause_btn.setIcon(QIcon.fromTheme("media-playback-pause"))
#         self.pause_btn.setToolTip("Pause")
#         self.pause_btn.setFixedSize(48, 48)
#         self.pause_btn.setEnabled(False)
#         self.pause_btn.setStyleSheet(self._button_style())
        
#         self.stop_btn = QPushButton()
#         self.stop_btn.setIcon(QIcon.fromTheme("media-playback-stop"))
#         self.stop_btn.setToolTip("Stop")
#         self.stop_btn.setFixedSize(48, 48)
#         self.stop_btn.setEnabled(False)
#         self.stop_btn.setStyleSheet(self._button_style())
        
#         for btn, sig in zip([self.play_btn, self.pause_btn, self.stop_btn], 
#                            [self.play_clicked.emit, self.pause_clicked.emit, self.stop_clicked.emit]):
#             btn.clicked.connect(sig)
        
#         control_bar.addWidget(self.play_btn)
#         control_bar.addWidget(self.pause_btn)
#         control_bar.addWidget(self.stop_btn)
#         control_bar.addSpacing(16)
        
#         # Progress bar
#         self.progress = QSlider(Qt.Horizontal)
#         self.progress.setStyleSheet("QSlider::groove:horizontal { height: 6px; background: #232323; border-radius: 3px; } QSlider::handle:horizontal { background: #03DAC5; border-radius: 8px; width: 18px; }")
#         self.progress.setMinimumWidth(240)
#         self.progress.setEnabled(False)
#         self.progress.valueChanged.connect(self.seek_changed.emit)
#         control_bar.addWidget(self.progress, 2)
        
#         self.timestamp = QLabel("00:00 / 00:00")
#         self.timestamp.setStyleSheet("color: #bbb; font-size: 13px;")
#         control_bar.addWidget(self.timestamp)
#         control_bar.addSpacing(16)
        
#         # Detection toggle & screenshot
#         self.detection_toggle = QCheckBox("Enable Detection")
#         self.detection_toggle.setChecked(True)
#         self.detection_toggle.setStyleSheet("color: #fff; font-size: 14px;")
#         self.detection_toggle.setEnabled(False)
#         self.detection_toggle.toggled.connect(self.detection_toggled.emit)
#         control_bar.addWidget(self.detection_toggle)
        
#         self.screenshot_btn = QPushButton()
#         self.screenshot_btn.setIcon(QIcon.fromTheme("camera-photo"))
#         self.screenshot_btn.setText("Screenshot")
#         self.screenshot_btn.setToolTip("Save current frame as image")
#         self.screenshot_btn.setEnabled(False)
#         self.screenshot_btn.setStyleSheet(self._button_style())
#         self.screenshot_btn.clicked.connect(self.screenshot_clicked.emit)
#         control_bar.addWidget(self.screenshot_btn)
#         control_bar.addStretch()
        
#         return widget
    
#     def _create_right_panel(self):
#         """Create right panel for smart intersection controls"""
#         panel = QScrollArea()
#         panel.setWidgetResizable(True)
#         panel.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
#         panel.setStyleSheet("""
#             QScrollArea {
#                 background: #1a1a1a;
#                 border: 1px solid #424242;
#                 border-radius: 8px;
#             }
#         """)
        
#         content = QWidget()
#         layout = QVBoxLayout(content)
#         layout.setContentsMargins(16, 16, 16, 16)
#         layout.setSpacing(16)
        
#         # Smart Intersection Controls
#         intersection_group = QGroupBox("🚦 Smart Intersection")
#         intersection_group.setStyleSheet("""
#             QGroupBox {
#                 color: #03DAC5;
#                 font-weight: bold;
#                 font-size: 14px;
#                 border: 2px solid #03DAC5;
#                 border-radius: 8px;
#                 margin-top: 12px;
#                 padding-top: 8px;
#             }
#             QGroupBox::title {
#                 subcontrol-origin: margin;
#                 left: 16px;
#                 padding: 0 8px 0 8px;
#             }
#         """)
        
#         intersection_layout = QVBoxLayout(intersection_group)
        
#         # Scene Analytics Toggle
#         self.scene_analytics_toggle = QCheckBox("Scene Analytics")
#         self.scene_analytics_toggle.setChecked(True)
#         self.scene_analytics_toggle.setStyleSheet("color: white; font-size: 12px;")
#         self.scene_analytics_toggle.toggled.connect(self.scene_analytics_toggled.emit)
#         intersection_layout.addWidget(self.scene_analytics_toggle)
        
#         # Multi-object tracking
#         self.multi_tracking_toggle = QCheckBox("Multi-Object Tracking")
#         self.multi_tracking_toggle.setChecked(True)
#         self.multi_tracking_toggle.setStyleSheet("color: white; font-size: 12px;")
#         intersection_layout.addWidget(self.multi_tracking_toggle)
        
#         # Speed estimation
#         self.speed_estimation_toggle = QCheckBox("Speed Estimation")
#         self.speed_estimation_toggle.setChecked(True)
#         self.speed_estimation_toggle.setStyleSheet("color: white; font-size: 12px;")
#         intersection_layout.addWidget(self.speed_estimation_toggle)
        
#         layout.addWidget(intersection_group)
        
#         # ROI Management
#         self.roi_widget = IntersectionROIWidget()
#         self.roi_widget.roi_updated.connect(self.roi_configuration_changed.emit)
#         layout.addWidget(self.roi_widget)
        
#         # Analytics Summary
#         analytics_group = QGroupBox("📊 Analytics Summary")
#         analytics_group.setStyleSheet(intersection_group.styleSheet().replace("#03DAC5", "#e67e22"))
#         analytics_layout = QVBoxLayout(analytics_group)
        
#         self.total_objects_label = QLabel("Total Objects: 0")
#         self.crosswalk_events_label = QLabel("Crosswalk Events: 0")
#         self.lane_events_label = QLabel("Lane Violations: 0")
#         self.safety_alerts_label = QLabel("Safety Alerts: 0")
        
#         for label in [self.total_objects_label, self.crosswalk_events_label, 
#                      self.lane_events_label, self.safety_alerts_label]:
#             label.setStyleSheet("color: white; font-size: 12px;")
#             analytics_layout.addWidget(label)
        
#         layout.addWidget(analytics_group)
        
#         # Performance Monitoring
#         perf_group = QGroupBox("⚡ Performance")
#         perf_group.setStyleSheet(intersection_group.styleSheet().replace("#03DAC5", "#9b59b6"))
#         perf_layout = QVBoxLayout(perf_group)
        
#         self.gpu_usage_label = QLabel("GPU Usage: -%")
#         self.memory_usage_label = QLabel("Memory: - MB")
#         self.processing_time_label = QLabel("Processing: - ms")
        
#         for label in [self.gpu_usage_label, self.memory_usage_label, self.processing_time_label]:
#             label.setStyleSheet("color: white; font-size: 12px;")
#             perf_layout.addWidget(label)
        
#         layout.addWidget(perf_group)
        
#         layout.addStretch()
        
#         panel.setWidget(content)
#         return panel
    
#     def _toggle_smart_intersection(self, enabled):
#         """Toggle smart intersection mode"""
#         self.smart_intersection_mode = enabled
#         self.smart_intersection_enabled.emit(enabled)
        
#         # Switch overlay
#         if enabled:
#             self._switch_to_smart_overlay()
#             self.mode_status.setText("🚦 Smart Intersection Active")
#             self.mode_status.setStyleSheet("color: #03DAC5; font-weight: bold; font-size: 12px;")
#         else:
#             self._switch_to_standard_overlay()
#             self.mode_status.setText("Standard Detection Mode")
#             self.mode_status.setStyleSheet("color: #bbb; font-size: 12px;")
        
#         # Enable/disable multi-camera toggle
#         self.multi_camera_toggle.setEnabled(enabled)
#         if not enabled:
#             self.multi_camera_toggle.setChecked(False)
    
#     def _toggle_multi_camera(self, enabled):
#         """Toggle multi-camera mode"""
#         self.multi_camera_mode = enabled
#         self.multi_camera_mode_enabled.emit(enabled)
        
#         if enabled:
#             self.video_tabs.setCurrentIndex(1)  # Switch to multi-camera tab
#             self.mode_status.setText("🚦 Multi-Camera Intersection Active")
#         else:
#             self.video_tabs.setCurrentIndex(0)  # Switch to single camera tab
#             if self.smart_intersection_mode:
#                 self.mode_status.setText("🚦 Smart Intersection Active")
    
#     def _switch_to_smart_overlay(self):
#         """Switch to smart intersection overlay"""
#         self.badge_bar.removeWidget(self.current_overlay)
#         self.current_overlay.setParent(None)
#         self.current_overlay = self.smart_overlay
#         self.badge_bar.addWidget(self.current_overlay)
    
#     def _switch_to_standard_overlay(self):
#         """Switch to standard overlay"""
#         self.badge_bar.removeWidget(self.current_overlay)
#         self.current_overlay.setParent(None)
#         self.current_overlay = self.standard_overlay
#         self.badge_bar.addWidget(self.current_overlay)
    
#     def _button_style(self):
#         return """
#             QPushButton {
#                 background: #232323;
#                 border-radius: 24px;
#                 color: #fff;
#                 font-size: 15px;
#                 border: none;
#             }
#             QPushButton:hover {
#                 background: #03DAC5;
#                 color: #222;
#             }
#             QPushButton:pressed {
#                 background: #018786;
#             }
#         """

#     def _select_file(self):
#         file_path, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov *.mkv *.webm);;All Files (*)")
#         if file_path:
#             self.file_label.setText(file_path)
#             self.file_selected.emit(file_path)
#             self.video_loaded = True
#             self._enable_controls(True)
#             self.video_label.setText("")
#             self.auto_select_model_device.emit()

#     def _enable_controls(self, enabled):
#         self.play_btn.setEnabled(enabled)
#         self.pause_btn.setEnabled(enabled)
#         self.stop_btn.setEnabled(enabled)
#         self.progress.setEnabled(enabled)
#         self.detection_toggle.setEnabled(enabled)
#         self.screenshot_btn.setEnabled(enabled)
#         if enabled:
#             self.auto_select_model_device.emit()

#     def update_display(self, pixmap):
#         """Update display with new frame"""
#         if pixmap:
#             if self.multi_camera_mode:
#                 # In multi-camera mode, distribute to different camera views
#                 # For now, just update the single view
#                 scaled = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
#                 self.video_label.setPixmap(scaled)
#             else:
#                 scaled = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
#                 self.video_label.setPixmap(scaled)
#             self._set_controls_enabled(True)
#             self.video_label.setStyleSheet("background: transparent; color: #888; font-size: 18px;")
#         else:
#             self.video_label.clear()
#             self.video_label.setText("No video loaded. Please select a video file.")
#             self._set_controls_enabled(False)
#             self.video_label.setStyleSheet("background: transparent; color: #F44336; font-size: 18px;")

#     def _set_controls_enabled(self, enabled):
#         for btn in [self.play_btn, self.pause_btn, self.stop_btn, self.progress, self.detection_toggle, self.screenshot_btn]:
#             btn.setEnabled(enabled)

#     def update_stats(self, stats):
#         """Update statistics display"""
#         if self.smart_intersection_mode:
#             # Update smart intersection overlay
#             scene_data = {
#                 'active_tracks': stats.get('total_objects', 0),
#                 'roi_events': stats.get('roi_events', 0),
#                 'cameras': {
#                     'north': stats.get('north_objects', 0),
#                     'east': stats.get('east_objects', 0),
#                     'south': stats.get('south_objects', 0),
#                     'west': stats.get('west_objects', 0)
#                 },
#                 'analytics': {
#                     'crosswalk_events': stats.get('crosswalk_events', 0),
#                     'lane_events': stats.get('lane_events', 0),
#                     'safety_events': stats.get('safety_events', 0)
#                 }
#             }
#             self.smart_overlay.update_smart_intersection(scene_data)
            
#             # Update right panel analytics
#             self.total_objects_label.setText(f"Total Objects: {stats.get('total_objects', 0)}")
#             self.crosswalk_events_label.setText(f"Crosswalk Events: {stats.get('crosswalk_events', 0)}")
#             self.lane_events_label.setText(f"Lane Violations: {stats.get('lane_events', 0)}")
#             self.safety_alerts_label.setText(f"Safety Alerts: {stats.get('safety_events', 0)}")
#         else:
#             # Update standard overlay
#             cars = stats.get('cars', 0)
#             trucks = stats.get('trucks', 0)
#             peds = stats.get('peds', 0)
#             tlights = stats.get('tlights', 0)
#             motorcycles = stats.get('motorcycles', 0)
#             model = stats.get('model', stats.get('model_name', '-'))
#             device = stats.get('device', stats.get('device_name', '-'))
#             self.standard_overlay.update_overlay(model, device, cars, trucks, peds, tlights, motorcycles)
        
#         # Update performance badges
#         fps = stats.get('fps', None)
#         inference = stats.get('inference', stats.get('detection_time', stats.get('detection_time_ms', None)))
        
#         if fps is not None:
#             self.fps_badge.setText(f"FPS: {fps:.2f}")
#         else:
#             self.fps_badge.setText("FPS: --")
            
#         if inference is not None:
#             self.inference_badge.setText(f"Inference: {inference:.1f} ms")
#         else:
#             self.inference_badge.setText("Inference: -- ms")
        
#         # Update performance panel
#         self.gpu_usage_label.setText(f"GPU Usage: {stats.get('gpu_usage', 0):.1f}%")
#         self.memory_usage_label.setText(f"Memory: {stats.get('memory_usage', 0):.1f} MB")
#         self.processing_time_label.setText(f"Processing: {stats.get('processing_time', 0):.1f} ms")

#     def update_progress(self, value, max_value, timestamp):
#         self.progress.setMaximum(max_value)
#         self.progress.setValue(value)
#         if isinstance(timestamp, float) or isinstance(timestamp, int):
#             timestamp_str = f"{timestamp:.2f}"
#         else:
#             timestamp_str = str(timestamp)
#         self.timestamp.setText(timestamp_str)
    
#     def update_multi_camera_feed(self, camera_position, pixmap, object_count=0):
#         """Update specific camera feed in multi-camera mode"""
#         if self.multi_camera_mode:
#             self.multi_cam_widget.update_camera_feed(camera_position, pixmap, object_count)
    
#     def get_smart_intersection_config(self):
#         """Get current smart intersection configuration"""
#         return {
#             'enabled': self.smart_intersection_mode,
#             'multi_camera': self.multi_camera_mode,
#             'scene_analytics': self.scene_analytics_toggle.isChecked(),
#             'multi_tracking': self.multi_tracking_toggle.isChecked(),
#             'speed_estimation': self.speed_estimation_toggle.isChecked()
#         }
