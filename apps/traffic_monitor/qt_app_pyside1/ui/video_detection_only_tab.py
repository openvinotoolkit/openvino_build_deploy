from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSlider, QCheckBox, 
    QFileDialog, QSizePolicy, QFrame, QTabWidget, QSplitter
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QPixmap, QIcon
import json
import os
from pathlib import Path


class EnhancedPerformanceOverlay(QFrame):
    """Enhanced performance metrics overlay with traffic light status and fixed positioning."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QFrame {
                background: rgba(20, 30, 40, 0.95);
                border: 2px solid #03DAC5;
                border-radius: 12px;
                color: #fff;
                font-family: 'Segoe UI', 'Arial', sans-serif;
            }
        """)
        # Fixed size to prevent overlay from changing size
        self.setFixedSize(400, 140)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(8)
        
        # Title row
        title_layout = QHBoxLayout()
        title = QLabel("📊 Real-time Performance Metrics")
        title.setStyleSheet("""
            color: #03DAC5; 
            font-weight: bold; 
            font-size: 14px;
            margin-bottom: 4px;
        """)
        title_layout.addWidget(title)
        title_layout.addStretch()
        
        # Traffic light status
        self.traffic_light_status = QLabel("🚦 Traffic: Unknown")
        self.traffic_light_status.setStyleSheet("""
            color: #FFD700; 
            font-weight: bold; 
            font-size: 13px;
            background: rgba(0,0,0,0.3);
            padding: 4px 8px;
            border-radius: 6px;
        """)
        title_layout.addWidget(self.traffic_light_status)
        layout.addLayout(title_layout)
        
        # Performance metrics row
        perf_layout = QHBoxLayout()
        perf_layout.setSpacing(16)
        
        # FPS and Inference in badges
        self.fps_label = QLabel("FPS: --")
        self.fps_label.setStyleSheet("""
            background: #27AE60; 
            color: white; 
            font-weight: bold; 
            font-size: 13px;
            padding: 6px 12px;
            border-radius: 8px;
            min-width: 70px;
        """)
        self.fps_label.setAlignment(Qt.AlignCenter)
        
        self.inference_label = QLabel("Inference: -- ms")
        self.inference_label.setStyleSheet("""
            background: #3498DB; 
            color: white; 
            font-weight: bold; 
            font-size: 13px;
            padding: 6px 12px;
            border-radius: 8px;
            min-width: 110px;
        """)
        self.inference_label.setAlignment(Qt.AlignCenter)
        
        perf_layout.addWidget(self.fps_label)
        perf_layout.addWidget(self.inference_label)
        perf_layout.addStretch()
        layout.addLayout(perf_layout)
        
        # System info row
        system_layout = QHBoxLayout()
        self.model_label = QLabel("Model: -")
        self.model_label.setStyleSheet("""
            color: #E74C3C; 
            font-weight: bold; 
            font-size: 12px;
            background: rgba(231, 76, 60, 0.1);
            padding: 4px 8px;
            border-radius: 6px;
        """)
        
        self.device_label = QLabel("Device: -")
        self.device_label.setStyleSheet("""
            color: #9B59B6; 
            font-weight: bold; 
            font-size: 12px;
            background: rgba(155, 89, 182, 0.1);
            padding: 4px 8px;
            border-radius: 6px;
        """)
        
        system_layout.addWidget(self.model_label)
        system_layout.addWidget(self.device_label)
        system_layout.addStretch()
        layout.addLayout(system_layout)
        
        # Vehicle counts row
        self.vehicle_stats_label = QLabel("🚗 Vehicles: 0 | 🚛 Trucks: 0 | 🚶 Pedestrians: 0 | 🏍️ Motorcycles: 0")
        self.vehicle_stats_label.setStyleSheet("""
            color: #F39C12; 
            font-weight: bold; 
            font-size: 12px;
            background: rgba(243, 156, 18, 0.1);
            padding: 6px 10px;
            border-radius: 6px;
        """)
        layout.addWidget(self.vehicle_stats_label)

    def update_overlay(self, model, device, cars, trucks, peds, tlights, motorcycles):
        """Update performance metrics"""
        self.model_label.setText(f"Model: Yolov11x")
        self.device_label.setText(f"Device: {device}")
        self.vehicle_stats_label.setText(f"🚗 Vehicles: {cars} | 🚛 Trucks: {trucks} | 🚶 Pedestrians: {peds} | 🏍️ Motorcycles: {motorcycles}")
    
    def update_performance_metrics(self, fps, inference_time):
        """Update FPS and inference time"""
        if fps is not None:
            self.fps_label.setText(f"FPS: {fps:.1f}")
        else:
            self.fps_label.setText("FPS: --")
            
        if inference_time is not None:
            self.inference_label.setText(f"Inference: {inference_time:.1f} ms")
        else:
            self.inference_label.setText("Inference: -- ms")
    
    def update_traffic_light_status(self, traffic_light_data):
        """Update traffic light status"""
        if traffic_light_data and isinstance(traffic_light_data, dict):
            color = traffic_light_data.get('color', 'unknown')
            confidence = traffic_light_data.get('confidence', 0)
            
            if color.lower() == 'red':
                icon = "🔴"
                text_color = "#E74C3C"
            elif color.lower() == 'yellow':
                icon = "🟡"
                text_color = "#F39C12"
            elif color.lower() == 'green':
                icon = "🟢"
                text_color = "#27AE60"
            else:
                icon = "⚫"
                text_color = "#95A5A6"
            
            self.traffic_light_status.setText(f"{icon} Traffic: {color.title()} ({confidence:.2f})")
            self.traffic_light_status.setStyleSheet(f"""
                color: {text_color}; 
                font-weight: bold; 
                font-size: 13px;
                background: rgba(0,0,0,0.3);
                padding: 4px 8px;
                border-radius: 6px;
            """)
        else:
            self.traffic_light_status.setText("🚦 Traffic: Unknown")
            self.traffic_light_status.setStyleSheet("""
                color: #95A5A6; 
                font-weight: bold; 
                font-size: 13px;
                background: rgba(0,0,0,0.3);
                padding: 4px 8px;
                border-radius: 6px;
            """)


class PerformanceStatsWidget(QFrame):
    """Compact performance statistics widget to replace analytics tables."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QFrame {
                background: #1a1a1a;
                border: 1px solid #424242;
                border-radius: 8px;
                color: #fff;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        
        # Header
        header = QLabel("⚡ Performance Metrics")
        header.setStyleSheet("""
            color: #03DAC5; 
            font-weight: bold; 
            font-size: 14px;
            padding: 8px 0px;
            border-bottom: 2px solid #03DAC5;
            margin-bottom: 8px;
        """)
        layout.addWidget(header)
        
        # Real-time stats
        self.fps_stat = QLabel("🎯 FPS: --")
        self.inference_stat = QLabel("⚡ Inference: -- ms")
        self.device_stat = QLabel("🖥️ Device: --")
        self.model_stat = QLabel("🤖 Model: --")
        
        # Vehicle counts
        self.vehicles_stat = QLabel("🚗 Vehicles: 0")
        self.pedestrians_stat = QLabel("🚶 Pedestrians: 0")
        self.traffic_lights_stat = QLabel("🚦 Traffic Lights: 0")
        
        # Traffic status
        self.traffic_status_stat = QLabel("🚦 Traffic Status: Unknown")
        
        stats = [
            self.fps_stat, self.inference_stat, self.device_stat, self.model_stat,
            self.vehicles_stat, self.pedestrians_stat, self.traffic_lights_stat,
            self.traffic_status_stat
        ]
        
        for stat in stats:
            stat.setStyleSheet("""
                color: #fff; 
                font-size: 10px; 
                padding: 5px 7px;
                background: rgba(255,255,255,0.05);
                border-radius: 6px;
                margin: 2px 0px;
            """)
            layout.addWidget(stat)
        
        layout.addStretch()
    
    def update_stats(self, stats_data):
        """Update all statistics"""
        # Debug: print what data we're receiving
        print(f"[PERF STATS DEBUG] Received stats keys: {list(stats_data.keys()) if stats_data else 'None'}")
        
        # Performance metrics - try multiple field names
        fps = stats_data.get('fps', 0)
        inference = stats_data.get('inference', stats_data.get('detection_time', stats_data.get('detection_time_ms', stats_data.get('inference_time', 0))))
        
        # Try different field names for device and model
        device = (stats_data.get('device') or 
                 stats_data.get('device_name') or 
                 stats_data.get('processing_device') or 
                 'GPU')  # Default to GPU since we see it's working
        
        model = (stats_data.get('model') or 
                stats_data.get('model_name') or 
                stats_data.get('ai_model') or 
                'YOLO11')  # Default to YOLO11 since that's what we're using
        
        # Extract model info from stats 
        model = stats_data.get('model_name', 'Unknown')
        print(f"🔧 DEBUG UI: Received model_name='{model}' from stats")
        
        # If model name not available in stats, try to extract from model_path
        if model == 'Unknown':
            model_path = str(stats_data.get('model_path', ''))
            print(f"🔧 DEBUG UI: Fallback to model_path='{model_path}'")
            if 'yolo11n' in model_path.lower():
                model = 'YOLO11n'
            elif 'yolo11x' in model_path.lower():
                model = 'YOLO11x'
            elif 'yolo11s' in model_path.lower():
                model = 'YOLO11s'
            elif 'yolo11m' in model_path.lower():
                model = 'YOLO11m'
            elif 'yolo11l' in model_path.lower():
                model = 'YOLO11l'
            else:
                model = 'YOLO11n'  # Default fallback
        
        print(f"🔧 DEBUG UI: Final model name for display: '{model}'")
        
        print(f"[PERF STATS DEBUG] Device: {device}, Model: {model}, FPS: {fps}, Inference: {inference}")
        
        self.fps_stat.setText(f"🎯 FPS: {fps:.1f}" if fps else "🎯 FPS: --")
        self.inference_stat.setText(f"⚡ Inference: {inference:.1f} ms" if inference else "⚡ Inference: -- ms")
        self.device_stat.setText(f"🖥️ Device: {device}")
        self.model_stat.setText(f"🤖 Model: {model}")
        
        # Vehicle counts
        cars = stats_data.get('cars', 0)
        trucks = stats_data.get('trucks', 0)
        motorcycles = stats_data.get('motorcycles', 0)
        peds = stats_data.get('peds', 0)
        tlights = stats_data.get('tlights', 0)
        
        total_vehicles = cars + trucks + motorcycles
        self.vehicles_stat.setText(f"🚗 Vehicles: {total_vehicles}")
        self.pedestrians_stat.setText(f"🚶 Pedestrians: {peds}")
        self.traffic_lights_stat.setText(f"🚦 Traffic Lights: {tlights}")
        
        # Traffic status - try different field names
        traffic_light = (stats_data.get('traffic_light') or 
                        stats_data.get('traffic_light_data') or 
                        stats_data.get('latest_traffic_light'))
        
        print(f"[PERF STATS DEBUG] Traffic light data: {traffic_light}")
        
        if traffic_light and isinstance(traffic_light, dict):
            color = traffic_light.get('color', 'Unknown')
            confidence = traffic_light.get('confidence', 0)
            
            # Handle case where color might also be a dict or other type
            if isinstance(color, str):
                self.traffic_status_stat.setText(f"🚦 Traffic Status: {color.title()} ({confidence:.2f})")
            elif isinstance(color, dict):
                # Color is nested dict, try to extract actual color
                actual_color = color.get('color', 'Unknown')
                if isinstance(actual_color, str):
                    self.traffic_status_stat.setText(f"🚦 Traffic Status: {actual_color.title()} ({confidence:.2f})")
                else:
                    self.traffic_status_stat.setText("🚦 Traffic Status: Unknown")
            else:
                self.traffic_status_stat.setText(f"🚦 Traffic Status: {str(color)} ({confidence:.2f})")
        else:
            # Try to get traffic light info from other fields
            if 'traffic_light_color' in stats_data:
                color = stats_data.get('traffic_light_color', 'Unknown')
                if isinstance(color, str):
                    self.traffic_status_stat.setText(f"🚦 Traffic Status: {color.title()}")
                else:
                    self.traffic_status_stat.setText(f"🚦 Traffic Status: {str(color)}")
            else:
                self.traffic_status_stat.setText("🚦 Traffic Status: Unknown")


class VideoDetectionOnlyTab(QWidget):
    """Standard video detection tab without smart intersection features"""
    file_selected = Signal(str)
    play_clicked = Signal()
    pause_clicked = Signal()
    stop_clicked = Signal()
    detection_toggled = Signal(bool)
    screenshot_clicked = Signal()
    seek_changed = Signal(int)
    auto_select_model_device = Signal()

    def __init__(self):
        super().__init__()
        self.video_loaded = False
        
        # Main layout with splitter for video and analytics
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(16)
        
        # File selection bar
        file_bar = self._create_file_bar()
        main_layout.addWidget(file_bar)
        
        # Create splitter for video and analytics
        splitter = QSplitter(Qt.Horizontal)
        
        # Left side - Video display
        video_widget = QWidget()
        video_layout = QVBoxLayout(video_widget)
        video_layout.setContentsMargins(0, 0, 0, 0)
        
        # Video display area
        video_frame = self._create_video_frame()
        video_layout.addWidget(video_frame, 1)
        
        # Control bar
        control_bar = self._create_control_bar()
        video_layout.addWidget(control_bar)
        
        splitter.addWidget(video_widget)
        
        # Right side - Performance Stats (replacing analytics tables)
        self.performance_stats = PerformanceStatsWidget()
        splitter.addWidget(self.performance_stats)
        
        # Set splitter proportions (70% video, 30% analytics)
        splitter.setSizes([700, 300])
        splitter.setCollapsible(0, False)  # Video section can't be collapsed
        splitter.setCollapsible(1, True)   # Analytics can be collapsed
        
        main_layout.addWidget(splitter)
        
    def _create_file_bar(self):
        """Create file selection bar"""
        widget = QWidget()
        bar = QHBoxLayout(widget)
        
        self.file_btn = QPushButton()
        self.file_btn.setIcon(QIcon.fromTheme("folder-video"))
        self.file_btn.setText("Select Video")
        self.file_btn.setStyleSheet("padding: 8px 18px; border-radius: 8px; background: #232323; color: #fff;")
        self.file_label = QLabel("No file selected")
        self.file_label.setStyleSheet("color: #bbb; font-size: 13px;")
        self.file_btn.clicked.connect(self._select_file)
        
        bar.addWidget(self.file_btn)
        bar.addWidget(self.file_label)
        bar.addStretch()
        
        return widget
        
    def _create_video_frame(self):
        """Create video display frame with fixed size"""
        video_frame = QFrame()
        video_frame.setStyleSheet("""
            background: #121212;
            border: 1px solid #424242;
            border-radius: 8px;
        """)
        # Set fixed size to prevent resizing issues
        video_frame.setFixedSize(800, 450)  # 16:9 aspect ratio, fixed size
        video_frame.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        
        video_layout = QVBoxLayout(video_frame)
        video_layout.setContentsMargins(0, 0, 0, 0)
        video_layout.setAlignment(Qt.AlignCenter)
        
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background: transparent; color: #888; font-size: 18px;")
        self.video_label.setText("No video loaded. Please select a file.")
        # Set fixed size for video label to match frame
        self.video_label.setFixedSize(800, 450)
        self.video_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.video_label.setScaledContents(True)  # Scale content to fit
        video_layout.addWidget(self.video_label)
        
        return video_frame
        
    def _create_control_bar(self):
        """Create control bar"""
        widget = QWidget()
        control_bar = QHBoxLayout(widget)
        control_bar.setContentsMargins(0, 16, 0, 0)
        
        # Playback controls
        self.play_btn = QPushButton()
        self.play_btn.setIcon(QIcon.fromTheme("media-playback-start"))
        self.play_btn.setToolTip("Play")
        self.play_btn.setFixedSize(48, 48)
        self.play_btn.setEnabled(False)
        self.play_btn.setStyleSheet(self._button_style())
        
        self.pause_btn = QPushButton()
        self.pause_btn.setIcon(QIcon.fromTheme("media-playback-pause"))
        self.pause_btn.setToolTip("Pause")
        self.pause_btn.setFixedSize(48, 48)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setStyleSheet(self._button_style())
        
        self.stop_btn = QPushButton()
        self.stop_btn.setIcon(QIcon.fromTheme("media-playback-stop"))
        self.stop_btn.setToolTip("Stop")
        self.stop_btn.setFixedSize(48, 48)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet(self._button_style())
        
        for btn, sig in zip([self.play_btn, self.pause_btn, self.stop_btn], 
                           [self.play_clicked.emit, self.pause_clicked.emit, self.stop_clicked.emit]):
            btn.clicked.connect(sig)
        
        control_bar.addWidget(self.play_btn)
        control_bar.addWidget(self.pause_btn)
        control_bar.addWidget(self.stop_btn)
        control_bar.addSpacing(16)
        
        # Progress bar
        self.progress = QSlider(Qt.Horizontal)
        self.progress.setStyleSheet("""
            QSlider::groove:horizontal { 
                height: 6px; 
                background: #232323; 
                border-radius: 3px; 
            } 
            QSlider::handle:horizontal { 
                background: #03DAC5; 
                border-radius: 8px; 
                width: 18px; 
            }
        """)
        self.progress.setMinimumWidth(240)
        self.progress.setEnabled(False)
        self.progress.valueChanged.connect(self.seek_changed.emit)
        control_bar.addWidget(self.progress, 2)
        
        self.timestamp = QLabel("00:00 / 00:00")
        self.timestamp.setStyleSheet("color: #bbb; font-size: 13px;")
        control_bar.addWidget(self.timestamp)
        control_bar.addSpacing(16)
        
        # Detection toggle & screenshot
        self.detection_toggle = QCheckBox("Enable Detection")
        self.detection_toggle.setChecked(True)
        self.detection_toggle.setStyleSheet("color: #fff; font-size: 14px;")
        self.detection_toggle.setEnabled(False)
        self.detection_toggle.toggled.connect(self.detection_toggled.emit)
        control_bar.addWidget(self.detection_toggle)
        
        self.screenshot_btn = QPushButton()
        self.screenshot_btn.setIcon(QIcon.fromTheme("camera-photo"))
        self.screenshot_btn.setText("Screenshot")
        self.screenshot_btn.setToolTip("Save current frame as image")
        self.screenshot_btn.setEnabled(False)
        self.screenshot_btn.setStyleSheet(self._button_style())
        self.screenshot_btn.clicked.connect(self.screenshot_clicked.emit)
        control_bar.addWidget(self.screenshot_btn)
        control_bar.addStretch()
        
        return widget
        
    def _button_style(self):
        return """
            QPushButton {
                background: #232323;
                border-radius: 24px;
                color: #fff;
                font-size: 15px;
                border: none;
            }
            QPushButton:hover {
                background: #03DAC5;
                color: #222;
            }
            QPushButton:pressed {
                background: #018786;
            }
        """

    def _select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Video File", 
            "", 
            "Video Files (*.mp4 *.avi *.mov *.mkv *.webm);;All Files (*)"
        )
        if file_path:
            self.file_label.setText(file_path)
            self.file_selected.emit(file_path)
            self.video_loaded = True
            self._enable_controls(True)
            self.video_label.setText("")
            self.auto_select_model_device.emit()

    def _enable_controls(self, enabled):
        self.play_btn.setEnabled(enabled)
        self.pause_btn.setEnabled(enabled)
        self.stop_btn.setEnabled(enabled)
        self.progress.setEnabled(enabled)
        self.detection_toggle.setEnabled(enabled)
        self.screenshot_btn.setEnabled(enabled)
        if enabled:
            self.auto_select_model_device.emit()

    def update_display(self, pixmap):
        """Update display with new frame"""
        if pixmap:
            scaled = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.video_label.setPixmap(scaled)
            self._set_controls_enabled(True)
            self.video_label.setStyleSheet("background: transparent; color: #888; font-size: 18px;")
        else:
            self.video_label.clear()
            self.video_label.setText("No video loaded. Please select a video file.")
            self._set_controls_enabled(False)
            self.video_label.setStyleSheet("background: transparent; color: #F44336; font-size: 18px;")

    def _set_controls_enabled(self, enabled):
        for btn in [self.play_btn, self.pause_btn, self.stop_btn, self.progress, self.detection_toggle, self.screenshot_btn]:
            btn.setEnabled(enabled)

    def update_stats(self, stats):
        """Update statistics display"""
        # Update right panel performance stats only
        self.performance_stats.update_stats(stats)

    def update_progress(self, value, max_value, timestamp):
        self.progress.setMaximum(max_value)
        self.progress.setValue(value)
        if isinstance(timestamp, float) or isinstance(timestamp, int):
            timestamp_str = f"{timestamp:.2f}"
        else:
            timestamp_str = str(timestamp)
        self.timestamp.setText(timestamp_str)
