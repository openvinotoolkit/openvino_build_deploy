import sys
import os
import time
import json
import csv
import platform
import psutil
import cv2
import numpy as np
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, 
                             QLabel, QFrame, QFileDialog, QGroupBox, QWidget,
                             QScrollArea, QProgressBar, QComboBox, QCheckBox,
                             QDoubleSpinBox, QSpinBox, QSlider, QTextEdit, QTabWidget,
                             QApplication)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QPixmap, QImage, QFont, QIcon
from PyQt6.QtWidgets import QStyle

# Use the correct engine class from your project
from gesture_engine import CompleteGestureEngine
from config_manager import config_manager

from openvino import opset13 as opset
from openvino import Model

class DeviceConfigDialog(QDialog):
    """Dialog for configuring inference devices for each model."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🔧 Device Configuration")
        self.setModal(True)
        self.setMinimumSize(500, 250)
        
        # Get available devices
        self.available_devices = self._get_available_devices()
        self.device_configs = {}
        
        self.setup_ui()
        self.load_current_config()

    def _get_available_devices(self):
        """Get list of available OpenVINO devices."""
        try:
            import openvino as ov
            core = ov.Core()
            devices = list(core.available_devices)
            
            # Add device details
            device_info = {}
            for device in devices:
                try:
                    full_name = core.get_property(device, "FULL_DEVICE_NAME")
                    device_info[device] = f"{device}: {full_name}"
                except:
                    device_info[device] = device
            
            return device_info
        except Exception as e:
            print(f"Error getting devices: {e}")
            return {"CPU": "CPU: Default Processor", "AUTO": "AUTO: Automatic Selection"}

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Header
        header = QLabel("🔧 Model Device Configuration")
        header.setStyleSheet("""
            font-size: 16px; 
            font-weight: bold; 
            color: #00ffff; 
            margin-bottom: 10px;
            padding: 8px;
            background-color: #2d2d30;
            border-radius: 6px;
        """)
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)

        # Model configuration
        models_group = QGroupBox("Model Device Assignment")
        models_layout = QGridLayout(models_group)
        
        # Define models and their descriptions
        self.model_configs = {
            'palm_detection': {
                'name': 'Palm Detection Model',
                'description': 'Detects hand regions in the frame',
                'default': 'CPU'
            },
            'hand_landmarks': {
                'name': 'Hand Landmarks Model', 
                'description': 'Extracts 21 hand keypoints',
                'default': 'CPU'
            },
            'gesture_embedder': {
                'name': 'Gesture Embedder Model',
                'description': 'Creates gesture feature embeddings',
                'default': 'AUTO'
            },
            'gesture_classifier': {
                'name': 'Gesture Classifier Model',
                'description': 'Classifies gestures from embeddings',
                'default': 'AUTO'
            }
        }

        self.device_combos = {}
        
        row = 0
        for model_key, model_info in self.model_configs.items():
            # Model name
            name_label = QLabel(model_info['name'])
            name_label.setStyleSheet("font-weight: bold; color: #ffffff;")
            models_layout.addWidget(name_label, row, 0)
            
            # Device dropdown
            device_combo = QComboBox()
            device_combo.addItems(list(self.available_devices.keys()))
            device_combo.setCurrentText(model_info['default'])
            device_combo.setMinimumWidth(120)
            models_layout.addWidget(device_combo, row, 1)
            
            # Description
            desc_label = QLabel(model_info['description'])
            desc_label.setStyleSheet("color: #aaaaaa; font-style: italic;")
            models_layout.addWidget(desc_label, row, 2)
            
            self.device_combos[model_key] = device_combo
            row += 1

        layout.addWidget(models_group)

        # Action buttons
        button_layout = QHBoxLayout()
        
        cancel_btn = QPushButton("❌ Cancel")
        cancel_btn.clicked.connect(self.reject)
        
        apply_btn = QPushButton("✅ Apply")
        apply_btn.clicked.connect(self.accept)
        
        button_layout.addStretch()
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(apply_btn)
        
        layout.addLayout(button_layout)

    def load_current_config(self):
        """Load current configuration - placeholder method."""
        # This method can be used to load saved device configurations
        # For now, it's a placeholder to prevent the AttributeError
        pass

    def get_device_configuration(self):
        """Get the current device configuration."""
        return {model_key: combo.currentText() 
                for model_key, combo in self.device_combos.items()}


class BenchmarkWorker(QThread):
    """Runs the benchmark in a separate thread to avoid freezing the GUI."""
    progress_updated = pyqtSignal(int, int, str)
    frame_processed = pyqtSignal(np.ndarray, dict)
    benchmark_finished = pyqtSignal(dict)

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.is_running = True

    def run(self):
        try:
            # Create a dedicated engine instance for the benchmark
            engine = CompleteGestureEngine()
            
            # --- NEW: Set device configuration for each model ---
            device_config = self.config.get('device_configuration', {})
            if device_config:
                print(f"🔧 Passing device configuration to ModelManager: {device_config}")
                engine.model_manager.set_device_configuration(device_config)
            
            # Set global device (fallback for any models not specifically configured)
            global_device = self.config.get('inference_device', 'CPU')
            engine.model_manager.set_device(global_device)
            # --- END NEW ---
            
            engine.params = self.config['engine_params']
            
            # Initialize in benchmark mode (no camera)
            # The ModelManager will now use the device_config set above
            if not engine.initialize(benchmark_mode=True):
                self.benchmark_finished.emit({'error': 'Benchmark engine failed to initialize.'})
                return

            # Initialize psutil and get CPU core count
            process = psutil.Process(os.getpid())
            cpu_count = psutil.cpu_count() or 1
            process.cpu_percent(interval=None)

            source_path = self.config['source_path']
            is_video = any(source_path.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mov', '.mkv'])
            
            all_metrics = []
            
            if is_video:
                cap = cv2.VideoCapture(source_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                original_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                
                frame_delay = (1.0 / original_fps) if self.config.get('realtime_playback', True) else 0
                
                frame_num = 0
                start_time = time.time()
                
                while cap.isOpened() and self.is_running:
                    ret, frame = cap.read()
                    if not ret: break
                    
                    frame_num += 1
                    self.progress_updated.emit(frame_num, total_frames, os.path.basename(source_path))
                    
                    processed_frame, frame_metrics = engine.process_single_frame_benchmark(frame)
                    
                    # Normalize CPU usage by core count
                    frame_metrics['cpu_percent'] = process.cpu_percent(interval=None) / cpu_count
                    frame_metrics['memory_mb'] = process.memory_info().rss / (1024 * 1024)

                    all_metrics.append(frame_metrics)
                    self.frame_processed.emit(processed_frame, frame_metrics)
                    
                    if frame_delay > 0:
                        expected_time = start_time + (frame_num * frame_delay)
                        sleep_time = expected_time - time.time()
                        if sleep_time > 0:
                            time.sleep(sleep_time)
                
                cap.release()
            else: # Image folder
                image_files = sorted([os.path.join(source_path, f) for f in os.listdir(source_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                total_frames = len(image_files)
                display_time = self.config.get('image_display_time', 2.0)
                
                for i, img_path in enumerate(image_files):
                    if not self.is_running: break
                    self.progress_updated.emit(i + 1, total_frames, os.path.basename(img_path))
                    frame = cv2.imread(img_path)
                    if frame is None: continue
                    
                    processed_frame, frame_metrics = engine.process_single_frame_benchmark(frame)

                    # Normalize CPU usage by core count
                    frame_metrics['cpu_percent'] = process.cpu_percent(interval=None) / cpu_count
                    frame_metrics['memory_mb'] = process.memory_info().rss / (1024 * 1024)

                    all_metrics.append(frame_metrics)
                    self.frame_processed.emit(processed_frame, frame_metrics)
                    
                    if display_time > 0:
                        time.sleep(display_time)

            final_report = self._aggregate_report(all_metrics)
            
            # --- NEW: Add device configuration to report ---
            final_report['device_configuration'] = device_config
            # --- END NEW ---
            
            self.benchmark_finished.emit(final_report)

        except Exception as e:
            print(f"Benchmark worker error: {e}")
            import traceback
            traceback.print_exc()
            self.benchmark_finished.emit({'error': str(e)})

    def _aggregate_report(self, all_metrics):
        """Creates a final summary report from all frame metrics."""
        if not all_metrics: return {"error": "No frames processed."}
        
        report = {}
        num_frames = len(all_metrics)
        
        # System info
        try:
            import cpuinfo
            cpu_info = cpuinfo.get_cpu_info()
            report['system_cpu'] = cpu_info.get('brand_raw', 'Unknown')
            report['system_cpu_arch'] = cpu_info.get('arch', 'Unknown')
            report['system_cpu_cores'] = cpu_info.get('count', 'Unknown')
        except:
            report['system_cpu'] = 'Unknown'
            report['system_cpu_arch'] = 'Unknown'
            report['system_cpu_cores'] = 'Unknown'
        
        report['system_ram_gb'] = f"{psutil.virtual_memory().total / (1024**3):.1f}"
        report['system_os'] = f"{platform.system()} {platform.release()}"
        report['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
        
        # Performance metrics
        keys_to_agg = [k for k in all_metrics[0].keys() if 'time' in k or 'cpu' in k or 'memory' in k]
        for key in keys_to_agg:
            values = [m.get(key, 0) for m in all_metrics if m.get(key) is not None]
            if values:
                report[f'avg_{key}'] = np.mean(values)
                report[f'max_{key}'] = np.max(values)
                report[f'min_{key}'] = np.min(values)
                report[f'std_{key}'] = np.std(values)

        total_duration_s = sum(m.get('total_engine_time_ms', 0) for m in all_metrics) / 1000
        report['avg_fps'] = num_frames / total_duration_s if total_duration_s > 0 else 0
        report['total_frames'] = num_frames
        report['total_duration_s'] = total_duration_s
        
        # Calculate percentiles for key metrics
        key_metrics = ['total_engine_time_ms', 'palm_detection_inference_ms', 'landmark_inference_ms']
        for metric in key_metrics:
            values = [m.get(metric, 0) for m in all_metrics if m.get(metric) is not None]
            if values:
                report[f'{metric}_p50'] = np.percentile(values, 50)
                report[f'{metric}_p95'] = np.percentile(values, 95)
                report[f'{metric}_p99'] = np.percentile(values, 99)
        
        return report

    def stop(self):
        self.is_running = False

class BenchmarkDialog(QDialog):
    """A dialog for running performance benchmarks on the gesture pipeline."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🔬 Benchmark Studio")
        
        # NEW: Smart window sizing based on screen dimensions
        self._setup_window_size()
        
        self.worker = None
        self.final_report = {}
        self.source_path = ""
        self.device_config = {}  # Store device configuration
        self.setup_ui()
        self.connect_signals()

    def _setup_window_size(self):
        """Set window size based on current screen size for safety"""
        # Get the primary screen geometry
        screen = QApplication.primaryScreen()
        screen_geometry = screen.geometry()
        screen_width = screen_geometry.width()
        screen_height = screen_geometry.height()
        
        print(f"🖥️ Detected screen size: {screen_width}x{screen_height}")
        
        # Calculate safe window dimensions (further reduced height)
        window_width = int(screen_width * 0.70)   # Reduced from 0.75
        window_height = int(screen_height * 0.60)  # Reduced from 0.65
        
        # Ensure minimum size for usability (further reduced minimums)
        min_width = 850   # Reduced from 900
        min_height = 500  # Reduced from 550
        
        window_width = max(window_width, min_width)
        window_height = max(window_height, min_height)
        
        # Ensure it doesn't exceed screen bounds
        window_width = min(window_width, screen_width - 150)
        window_height = min(window_height, screen_height - 150)
        
        print(f"🔧 Setting benchmark window size: {window_width}x{window_height}")
        
        # Set the window size and center it
        self.resize(window_width, window_height)
        
        # Center the window on screen
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.move(x, y)
        
        # Set minimum size to prevent it from being too small
        self.setMinimumSize(min_width, min_height)

    def setup_ui(self):
        main_layout = QHBoxLayout(self)
        
        # --- Left Panel: Configuration (adjust width based on window size) ---
        config_panel = QFrame()
        # Make config panel width proportional to window width
        config_width = min(350, self.width() // 4)
        config_panel.setFixedWidth(config_width)
        config_panel.setFrameShape(QFrame.Shape.StyledPanel)
        config_layout = QVBoxLayout(config_panel)
        
        # Input Source
        input_group = QGroupBox("Input Source")
        input_layout = QVBoxLayout()
        self.source_path_label = QLabel("No source selected.")
        self.source_path_label.setWordWrap(True)
        self.browse_folder_btn = QPushButton("📁 Select Image Folder")
        self.browse_video_btn = QPushButton("🎥 Select Video File")
        input_layout.addWidget(self.source_path_label)
        input_layout.addWidget(self.browse_folder_btn)
        input_layout.addWidget(self.browse_video_btn)
        input_group.setLayout(input_layout)
        
        # --- NEW: Device Configuration Section ---
        device_group = QGroupBox("Device Configuration")
        device_layout = QVBoxLayout()
        
        self.device_status_label = QLabel("Using default device configuration")
        self.device_status_label.setStyleSheet("color: #aaaaaa; font-style: italic;")
        self.device_status_label.setWordWrap(True)
        
        self.configure_devices_btn = QPushButton("🔧 Configure Model Devices")
        self.configure_devices_btn.setToolTip("Set which device each model should use for inference")
        
        device_layout.addWidget(self.device_status_label)
        device_layout.addWidget(self.configure_devices_btn)
        device_group.setLayout(device_layout)
        # --- END NEW ---
        
        # Pipeline Parameters
        params_group = QGroupBox("Pipeline Parameters")
        params_layout = QGridLayout()
        
        self.always_palm_cb = QCheckBox("Always Run Palm Detection")
        self.always_palm_cb.setChecked(config_manager.detection.always_run_palm_detection)
        
        self.input_size_combo = QComboBox()
        self.input_size_combo.addItems(["128", "192", "224", "256"])
        self.input_size_combo.setCurrentText(str(config_manager.detection.input_size))
        
        self.score_thresh_spin = QDoubleSpinBox()
        self.score_thresh_spin.setRange(0.1, 1.0)
        self.score_thresh_spin.setSingleStep(0.05)
        self.score_thresh_spin.setValue(config_manager.detection.score_threshold)
        
        # Global device selection (fallback for non-configured models)
        self.device_combo = QComboBox()
        self.device_combo.addItems(["CPU", "AUTO"])
        try:
            import openvino as ov
            core = ov.Core()
            available_devices = core.available_devices
            if available_devices:
                self.device_combo.clear()
                for device in available_devices:
                    self.device_combo.addItem(device)
        except:
            pass
        
        params_layout.addWidget(self.always_palm_cb, 0, 0, 1, 2)
        params_layout.addWidget(QLabel("Input Size:"), 1, 0)
        params_layout.addWidget(self.input_size_combo, 1, 1)
        params_layout.addWidget(QLabel("Score Threshold:"), 2, 0)
        params_layout.addWidget(self.score_thresh_spin, 2, 1)
        params_layout.addWidget(QLabel("Global Device:"), 3, 0)
        params_layout.addWidget(self.device_combo, 3, 1)
        params_group.setLayout(params_layout)
        
        # Timing Controls
        timing_group = QGroupBox("Timing Controls")
        timing_layout = QGridLayout()
        
        self.realtime_playback_cb = QCheckBox("Real-time Video Playback")
        self.realtime_playback_cb.setChecked(True)
        self.realtime_playback_cb.setToolTip("Enable to maintain original video frame rate")
        
        self.image_display_label = QLabel("Image Display Time: 2.0s")
        self.image_display_slider = QSlider(Qt.Orientation.Horizontal)
        self.image_display_slider.setRange(5, 100)
        self.image_display_slider.setValue(20)
        self.image_display_slider.valueChanged.connect(
            lambda v: self.image_display_label.setText(f"Image Display Time: {v/10:.1f}s")
        )
        
        timing_layout.addWidget(self.realtime_playback_cb, 0, 0, 1, 2)
        timing_layout.addWidget(self.image_display_label, 1, 0, 1, 2)
        timing_layout.addWidget(self.image_display_slider, 2, 0, 1, 2)
        timing_group.setLayout(timing_layout)

        # --- BUTTONS MOVED TO RIGHT PANEL ---
        self.start_btn = QPushButton("▶️ START TEST")
        self.start_btn.setEnabled(False)
        self.start_btn.setMinimumHeight(40)
        
        self.stop_btn = QPushButton("⏹️ STOP TEST")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setMinimumHeight(40)
        
        self.export_btn = QPushButton("💾 EXPORT RESULTS")
        self.export_btn.setEnabled(False)
        self.export_btn.setMinimumHeight(40)

        config_layout.addWidget(input_group)
        config_layout.addWidget(device_group)
        config_layout.addWidget(params_group)
        config_layout.addWidget(timing_group)
        config_layout.addStretch()

        # --- Center Panel: Visualizer (responsive sizing) ---
        center_panel = QFrame()
        center_panel.setFrameShape(QFrame.Shape.StyledPanel)
        center_layout = QVBoxLayout(center_panel)
        
        self.visualizer_label = QLabel("Visualizer will appear here.")
        self.visualizer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Make visualizer size responsive to window size
        viz_min_width = max(400, (self.width() - config_width - 400) // 2)
        viz_min_height = max(300, self.height() // 2)
        self.visualizer_label.setMinimumSize(viz_min_width, viz_min_height)
        self.visualizer_label.setStyleSheet("background-color: #1e1e1e; border: 1px solid #3e3e42;")
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimumHeight(25)
        
        self.progress_label = QLabel("Idle")
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        center_layout.addWidget(self.visualizer_label, 1)
        center_layout.addWidget(self.progress_bar)
        center_layout.addWidget(self.progress_label)

        # --- Right Panel: Results (responsive width) ---
        results_panel = QFrame()
        # Make results panel width proportional to window width (REDUCED SIZE)
        results_width = min(320, self.width() // 4)
        results_panel.setFixedWidth(results_width)
        results_panel.setFrameShape(QFrame.Shape.StyledPanel)
        results_layout = QVBoxLayout(results_panel)
        
        # System Info
        sys_info_group = QGroupBox("System Information")
        sys_info_layout = QVBoxLayout()
        try:
            import cpuinfo
            cpu = cpuinfo.get_cpu_info()['brand_raw']
        except Exception: 
            cpu = "N/A"
        ram = f"{psutil.virtual_memory().total / (1024**3):.1f} GB"
        
        sys_info_layout.addWidget(QLabel(f"<b>CPU:</b> {cpu}"))
        sys_info_layout.addWidget(QLabel(f"<b>RAM:</b> {ram}"))
        sys_info_layout.addWidget(QLabel(f"<b>OS:</b> {platform.system()} {platform.release()}"))
        sys_info_group.setLayout(sys_info_layout)

        # Results Display
        results_group = QGroupBox("Benchmark Results")
        results_group_layout = QVBoxLayout(results_group)
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setPlainText("Run a test to see detailed results here.")
        
        # Make results text height responsive
        results_min_height = max(200, self.height() // 3)
        self.results_text.setMinimumHeight(results_min_height)
        
        # Set monospace font for better alignment (smaller size for better fit)
        font = QFont("Consolas", 8)
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.results_text.setFont(font)
        
        results_group_layout.addWidget(self.results_text)

        results_layout.addWidget(sys_info_group)
        results_layout.addWidget(results_group, 1)
        
        # --- BUTTONS MOVED HERE ---
        results_layout.addStretch()
        results_layout.addWidget(self.start_btn)
        results_layout.addWidget(self.stop_btn)
        results_layout.addWidget(self.export_btn)

        main_layout.addWidget(config_panel)
        main_layout.addWidget(center_panel, 1)
        main_layout.addWidget(results_panel)

    def connect_signals(self):
        self.browse_folder_btn.clicked.connect(lambda: self.browse_source(is_folder=True))
        self.browse_video_btn.clicked.connect(lambda: self.browse_source(is_folder=False))
        self.configure_devices_btn.clicked.connect(self.open_device_config)  # NEW
        self.start_btn.clicked.connect(self.start_benchmark)
        self.stop_btn.clicked.connect(self.stop_benchmark)
        self.export_btn.clicked.connect(self.export_results)

    def open_device_config(self):
        """Open the device configuration dialog."""
        dialog = DeviceConfigDialog(self)
        
        # Load current configuration if available
        if self.device_config:
            for model_key, device in self.device_config.items():
                if model_key in dialog.device_combos:
                    dialog.device_combos[model_key].setCurrentText(device)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.device_config = dialog.get_device_configuration()
            self.update_device_status_display()

    def update_device_status_display(self):
        """Update the device status label with current configuration."""
        if self.device_config:
            config_lines = []
            for model, device in self.device_config.items():
                model_name = model.replace('_', ' ').title()
                config_lines.append(f"{model_name}: {device}")
            
            status_text = "Custom configuration:\n" + "\n".join(config_lines)
            self.device_status_label.setText(status_text)
            self.device_status_label.setStyleSheet("color: #00ff00; font-weight: bold;")
        else:
            self.device_status_label.setText("Using default device configuration")
            self.device_status_label.setStyleSheet("color: #aaaaaa; font-style: italic;")

    def browse_source(self, is_folder=False):
        if is_folder:
            path = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov *.mkv)")
        
        if path:
            self.source_path = path
            self.source_path_label.setText(f"Source: {os.path.basename(path)}")
            self.start_btn.setEnabled(True)

    def start_benchmark(self):
        # Create a temporary params dict for the benchmark run
        params = config_manager.get_legacy_params_dict()
        params.update({
            "always_run_palm_detection": self.always_palm_cb.isChecked(),
            "input_size": int(self.input_size_combo.currentText()),
            "score_threshold": self.score_thresh_spin.value(),
        })
        
        config = {
            "source_path": self.source_path, 
            "engine_params": params,
            "realtime_playback": self.realtime_playback_cb.isChecked(),
            "image_display_time": self.image_display_slider.value() / 10.0,
            "inference_device": self.device_combo.currentText(),
            "device_configuration": self.device_config  # NEW: Include device config
        }
        
        self.worker = BenchmarkWorker(config)
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.frame_processed.connect(self.update_frame)
        self.worker.benchmark_finished.connect(self.display_final_report)
        self.worker.start()
        
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.export_btn.setEnabled(False)
        self.results_text.setPlainText("Benchmark running...\nProcessing frames and collecting metrics...")

    def stop_benchmark(self):
        if self.worker:
            self.worker.stop()
            self.stop_btn.setEnabled(False)
            self.start_btn.setEnabled(True)
            self.progress_label.setText("Benchmark stopped by user.")

    def update_progress(self, current, total, filename):
        if total > 0: 
            self.progress_bar.setValue(int(current * 100 / total))
        self.progress_label.setText(f"Processing: {filename} ({current}/{total})")

    def update_frame(self, frame, metrics):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        qt_image = QImage(rgb_image.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image).scaled(
            self.visualizer_label.size(), 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        )
        self.visualizer_label.setPixmap(pixmap)

    def display_final_report(self, report):
        self.final_report = report
        
        if "error" in report:
            self.results_text.setPlainText(f"❌ Error: {report['error']}")
        else:
            # Create detailed formatted report
            report_text = self._format_detailed_report(report)
            self.results_text.setPlainText(report_text)
        
        self.stop_btn.setEnabled(False)
        self.start_btn.setEnabled(True)
        self.export_btn.setEnabled(True)

    def _format_detailed_report(self, report):
        """Format a comprehensive report for display."""
        lines = []
        lines.append("🔬 BENCHMARK REPORT")
        lines.append("=" * 50)
        lines.append("")
        
        # System Information
        lines.append("💻 SYSTEM INFORMATION")
        lines.append("-" * 30)
        lines.append(f"CPU:           {report.get('system_cpu', 'Unknown')}")
        lines.append(f"Architecture:  {report.get('system_cpu_arch', 'Unknown')}")
        lines.append(f"Cores:         {report.get('system_cpu_cores', 'Unknown')}")
        lines.append(f"RAM:           {report.get('system_ram_gb', 'Unknown')} GB")
        lines.append(f"OS:            {report.get('system_os', 'Unknown')}")
        lines.append(f"Global Device: {self.device_combo.currentText()}")
        lines.append(f"Timestamp:     {report.get('timestamp', 'Unknown')}")
        lines.append("")
        
        # --- NEW: Device Configuration Section ---
        if self.device_config:
            lines.append("🔧 DEVICE CONFIGURATION")
            lines.append("-" * 30)
            for model, device in self.device_config.items():
                model_name = model.replace('_', ' ').title()
                lines.append(f"{model_name:<20} {device}")
            lines.append("")
        # --- END NEW ---
        
        # Overall Performance
        lines.append("⚡ OVERALL PERFORMANCE")
        lines.append("-" * 30)
        lines.append(f"Total Frames:  {report.get('total_frames', 0)}")
        lines.append(f"Duration:      {report.get('total_duration_s', 0):.2f} seconds")
        lines.append(f"Average FPS:   {report.get('avg_fps', 0):.2f}")
        lines.append("")
        
        # Timing Breakdown
        lines.append("⏱️ TIMING BREAKDOWN (milliseconds)")
        lines.append("-" * 40)
        
        timing_metrics = [
            ('Total Engine Time', 'total_engine_time_ms'),
            ('Palm Detection', 'palm_detection_inference_ms'),
            ('Landmark Inference', 'landmark_inference_ms')
        ]
        
        for label, key in timing_metrics:
            avg_key = f'avg_{key}'
            min_key = f'min_{key}'
            max_key = f'max_{key}'
            p95_key = f'{key}_p95'
            
            if avg_key in report:
                lines.append(f"{label}:")
                lines.append(f"  Average:     {report[avg_key]:.2f} ms")
                lines.append(f"  Min:         {report.get(min_key, 0):.2f} ms")
                lines.append(f"  Max:         {report.get(max_key, 0):.2f} ms")
                if p95_key in report:
                    lines.append(f"  95th %ile:   {report[p95_key]:.2f} ms")
                lines.append("")
        
        # Resource Usage
        lines.append("📊 RESOURCE USAGE (% of Total System)")
        lines.append("-" * 40)
        lines.append(f"Peak CPU:      {report.get('max_cpu_percent', 0):.1f}%")
        lines.append(f"Avg CPU:       {report.get('avg_cpu_percent', 0):.1f}%")
        lines.append(f"Peak Memory:   {report.get('max_memory_mb', 0):.1f} MB")
        lines.append(f"Avg Memory:    {report.get('avg_memory_mb', 0):.1f} MB")
        lines.append("")
        
        # Configuration
        lines.append("⚙️ TEST CONFIGURATION")
        lines.append("-" * 30)
        lines.append(f"Input Size:    {self.input_size_combo.currentText()}x{self.input_size_combo.currentText()}")
        lines.append(f"Score Threshold: {self.score_thresh_spin.value():.2f}")
        lines.append(f"Always Palm Detection: {'Yes' if self.always_palm_cb.isChecked() else 'No'}")
        lines.append(f"Source:        {os.path.basename(self.source_path) if self.source_path else 'Unknown'}")
        lines.append("")
        
        return "\n".join(lines)

    def export_results(self):
        if not self.final_report or "error" in self.final_report: 
            return
        
        # Get file path with timestamp
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        default_name = f"benchmark_report_{timestamp}"
        
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Benchmark Report", default_name, 
            "JSON Files (*.json);;CSV Files (*.csv);;Text Files (*.txt)"
        )
        if not path: 
            return
        
        try:
            if path.endswith('.json'):
                # Add configuration to the report
                export_data = dict(self.final_report)
                export_data['configuration'] = {
                    'input_size': int(self.input_size_combo.currentText()),
                    'score_threshold': self.score_thresh_spin.value(),
                    'always_palm_detection': self.always_palm_cb.isChecked(),
                    'global_device': self.device_combo.currentText(),
                    'device_configuration': self.device_config,  # NEW
                    'source_file': os.path.basename(self.source_path) if self.source_path else 'Unknown'
                }
                
                with open(path, 'w') as f:
                    json.dump(export_data, f, indent=4)
                    
            elif path.endswith('.csv'):
                with open(path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Metric', 'Value'])
                    for key, value in self.final_report.items():
                        writer.writerow([key, value])
                        
            elif path.endswith('.txt'):
                with open(path, 'w') as f:
                    f.write(self._format_detailed_report(self.final_report))
            
            self.progress_label.setText(f"✅ Results exported to: {os.path.basename(path)}")
            
        except Exception as e:
            self.progress_label.setText(f"❌ Export failed: {str(e)}")

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
        super().closeEvent(event)