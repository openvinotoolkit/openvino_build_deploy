from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import QWidget, QGridLayout, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QFrame, QComboBox, QCheckBox
import cv2
import numpy as np

class CameraFeedWidget(QFrame):
    settings_clicked = Signal(int)
    detection_toggled = Signal(int, bool)
    def __init__(self, cam_number):
        super().__init__()
        self.cam_number = cam_number
        self.setFrameShape(QFrame.Box)
        self.setLineWidth(3)
        self.setStyleSheet("QFrame { border: 3px solid gray; border-radius: 8px; }")
        layout = QVBoxLayout()
        top_bar = QHBoxLayout()
        self.overlay_label = QLabel(f"<b>CAM {cam_number}</b>")
        self.gear_btn = QPushButton()
        self.gear_btn.setIcon(QIcon.fromTheme("settings"))
        self.gear_btn.setFixedSize(24,24)
        self.gear_btn.clicked.connect(lambda: self.settings_clicked.emit(self.cam_number))
        top_bar.addWidget(self.overlay_label)
        top_bar.addStretch()
        top_bar.addWidget(self.gear_btn)
        layout.addLayout(top_bar)
        self.video_label = QLabel("No Feed")
        self.video_label.setMinimumHeight(160)
        self.fps_label = QLabel("FPS: 0")
        self.count_label = QLabel("Cars: 0 | Trucks: 0 | Ped: 0 | TLights: 0 | Moto: 0")
        self.detection_toggle = QCheckBox("Detection ON")
        self.detection_toggle.setChecked(True)
        self.detection_toggle.toggled.connect(lambda checked: self.detection_toggled.emit(self.cam_number, checked))
        self.start_stop_btn = QPushButton("Start")
        layout.addWidget(self.video_label)
        layout.addWidget(self.fps_label)
        layout.addWidget(self.count_label)
        layout.addWidget(self.detection_toggle)
        layout.addWidget(self.start_stop_btn)
        self.setLayout(layout)
    def set_active(self, active):
        color = "#00FF00" if active else "gray"
        self.setStyleSheet(f"QFrame {{ border: 3px solid {color}; border-radius: 8px; }}")

class LiveMultiCamTab(QWidget):
    source_changed = Signal(int, object)  # cam_number, source
    run_requested = Signal(int, bool)     # cam_number, start/stop
    detection_toggled = Signal(int, bool) # cam_number, enabled
    settings_clicked = Signal(int)
    global_detection_toggled = Signal(bool)
    device_changed = Signal(str)
    video_dropped = Signal(int, object)  # cam_number, dropped source
    snapshot_requested = Signal(int)     # cam_number
    def __init__(self):
        super().__init__()
        # Info bar at the top (only for Live Detection tab)
        info_bar = QHBoxLayout()
        self.model_label = QLabel("Model: -")
        self.device_label = QLabel("Device: -")
        self.yolo_label = QLabel("YOLO Version: -")
        self.resolution_label = QLabel("Resolution: -")
        self.cam1_fps = QLabel("CAM 1 FPS: -")
        self.cam2_fps = QLabel("CAM 2 FPS: -")
        self.cam3_fps = QLabel("CAM 3 FPS: -")
        self.cam4_fps = QLabel("CAM 4 FPS: -")
        info_bar.addWidget(self.model_label)
        info_bar.addWidget(self.device_label)
        info_bar.addWidget(self.yolo_label)
        info_bar.addWidget(self.resolution_label)
        info_bar.addWidget(self.cam1_fps)
        info_bar.addWidget(self.cam2_fps)
        info_bar.addWidget(self.cam3_fps)
        info_bar.addWidget(self.cam4_fps)
        info_bar.addStretch()
        grid = QGridLayout()
        self.cameras = []
        for i in range(4):
            cam_widget = CameraFeedWidget(i+1)
            cam_widget.start_stop_btn.clicked.connect(lambda checked, n=i+1: self._handle_start_stop(n))
            cam_widget.settings_clicked.connect(self.settings_clicked.emit)
            cam_widget.detection_toggled.connect(self.detection_toggled.emit)
            # Add snapshot button for each camera
            snapshot_btn = QPushButton("Snapshot")
            snapshot_btn.clicked.connect(lambda checked=False, n=i+1: self.snapshot_requested.emit(n))
            cam_widget.layout().addWidget(snapshot_btn)
            self.cameras.append(cam_widget)
            grid.addWidget(cam_widget, i//2, i%2)
        controls = QHBoxLayout()
        self.start_all_btn = QPushButton("Start All")
        self.stop_all_btn = QPushButton("Stop All")
        self.global_detection_toggle = QCheckBox("Detection ON (All)")
        self.global_detection_toggle.setChecked(True)
        self.device_selector = QComboBox()
        self.device_selector.addItems(["CPU", "GPU", "NPU"])
        self.start_all_btn.clicked.connect(lambda: self._handle_all(True))
        self.stop_all_btn.clicked.connect(lambda: self._handle_all(False))
        self.global_detection_toggle.toggled.connect(self.global_detection_toggled.emit)
        self.device_selector.currentTextChanged.connect(self.device_changed.emit)
        controls.addWidget(self.start_all_btn)
        controls.addWidget(self.stop_all_btn)
        controls.addWidget(self.global_detection_toggle)
        controls.addWidget(QLabel("Device:"))
        controls.addWidget(self.device_selector)
        main_layout = QVBoxLayout()
        main_layout.addLayout(info_bar)
        main_layout.addLayout(grid)
        main_layout.addLayout(controls)
        self.setLayout(main_layout)
    def _handle_start_stop(self, cam_number):
        btn = self.cameras[cam_number-1].start_stop_btn
        start = btn.text() == "Start"
        self.run_requested.emit(cam_number, start)
        btn.setText("Stop" if start else "Start")
    def _handle_all(self, start):
        for i, cam in enumerate(self.cameras):
            self.run_requested.emit(i+1, start)
            cam.start_stop_btn.setText("Stop" if start else "Start")
    def update_display(self, cam_number, pixmap):
        # If pixmap is None, show a user-friendly message and disable controls
        if pixmap is None:
            self.cameras[cam_number-1].video_label.setText("No feed. Click 'Start' to connect a camera or select a video.")
            self.cameras[cam_number-1].video_label.setStyleSheet("color: #F44336; font-size: 15px; background: transparent;")
            self._set_controls_enabled(cam_number-1, False)
        else:
            self.cameras[cam_number-1].video_label.setPixmap(pixmap)
            self.cameras[cam_number-1].video_label.setStyleSheet("background: transparent;")
            self._set_controls_enabled(cam_number-1, True)
    def _set_controls_enabled(self, cam_idx, enabled):
        for btn in [self.cam_widgets[cam_idx]['start_btn'], self.cam_widgets[cam_idx]['snapshot_btn']]:
            btn.setEnabled(enabled)
    def update_display_np(self, np_frame):
        """Display a NumPy frame in CAM 1 (single source live mode)."""
        import cv2
        import numpy as np
        if np_frame is None or not isinstance(np_frame, np.ndarray) or np_frame.size == 0:
            print(f"[LiveMultiCamTab] ⚠️ Received None or empty frame for CAM 1")
            return
        try:
            rgb_frame = cv2.cvtColor(np_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            from PySide6.QtGui import QImage, QPixmap
            from PySide6.QtCore import Qt
            bytes_per_line = ch * w
            qimg = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            scaled_pixmap = pixmap.scaled(
                self.cameras[0].video_label.width(),
                self.cameras[0].video_label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.cameras[0].video_label.setPixmap(scaled_pixmap)
            self.cameras[0].video_label.update()
            print(f"[LiveMultiCamTab] 🟢 Frame displayed for CAM 1")
        except Exception as e:
            print(f"[LiveMultiCamTab] ❌ Error displaying frame for CAM 1: {e}")
            import traceback
            traceback.print_exc()
    def update_fps(self, cam_number, fps):
        self.cameras[cam_number-1].fps_label.setText(f"FPS: {fps}")
    def update_counts(self, cam_number, cars, trucks, peds, tlights, motorcycles):
        self.cameras[cam_number-1].count_label.setText(
            f"Cars: {cars} | Trucks: {trucks} | Ped: {peds} | TLights: {tlights} | Moto: {motorcycles}")
    def update_stats(self, cam_number, stats):
        # Placeholder: expects stats dict with keys: cars, trucks, peds, tlights, motorcycles, fps
        self.update_counts(cam_number, stats.get('cars', 0), stats.get('trucks', 0), stats.get('peds', 0), stats.get('tlights', 0), stats.get('motorcycles', 0))
        self.update_fps(cam_number, stats.get('fps', 0))
    def set_detection_active(self, cam_number, active):
        self.cameras[cam_number-1].set_active(active)
