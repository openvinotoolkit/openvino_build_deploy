from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QIcon, QImage, QPixmap, QFont, QPainter, QPen, QBrush
from PySide6.QtWidgets import (
    QWidget, QGridLayout, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QFrame, QComboBox, QCheckBox, QTabWidget, QSlider, QSpinBox, QGroupBox,
    QScrollArea, QSizePolicy, QSpacerItem
)
import cv2
import numpy as np

class ModernToggleSwitch(QCheckBox):
    """Custom toggle switch with modern styling"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QCheckBox {
                spacing: 0px;
            }
            QCheckBox::indicator {
                width: 50px;
                height: 24px;
                border-radius: 12px;
                background: #2C2C2C;
                border: 2px solid #2C2C2C;
            }
            QCheckBox::indicator:checked {
                background: #007BFF;
                border: 2px solid #007BFF;
            }
            QCheckBox::indicator:checked:hover {
                background: #3399FF;
                border: 2px solid #3399FF;
            }
        """)

class CameraPanel(QFrame):
    """Modern camera panel with rounded corners and shadow effect"""
    settings_clicked = Signal(int)
    detection_toggled = Signal(int, bool)
    start_stop_clicked = Signal(int)
    snapshot_clicked = Signal(int)
    
    def __init__(self, cam_number):
        super().__init__()
        self.cam_number = cam_number
        self.is_running = False
        self.setupUI()
        
    def setupUI(self):
        self.setObjectName("CameraPanel")
        self.setStyleSheet(self._get_panel_style())
        
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Title bar
        title_bar = QFrame()
        title_bar.setObjectName("PanelTitle")
        title_bar.setStyleSheet("""
            QFrame#PanelTitle {
                background: #2C2C2C;
                border-top-left-radius: 12px;
                border-top-right-radius: 12px;
                min-height: 32px;
                max-height: 32px;
            }
        """)
        
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(12, 6, 12, 6)
        
        self.title_label = QLabel(f"CAM {self.cam_number}")
        self.title_label.setStyleSheet("""
            QLabel {
                color: #FFFFFF;
                font-weight: bold;
                font-size: 14px;
                background: transparent;
            }
        """)
        
        title_layout.addWidget(self.title_label)
        title_layout.addStretch()
        
        layout.addWidget(title_bar)
        
        # Video display area
        video_container = QFrame()
        video_container.setStyleSheet("background: #121212; border: none;")
        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(8, 8, 8, 8)
        
        self.video_label = QLabel("No Feed")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumHeight(180)
        self.video_label.setStyleSheet("""
            QLabel {
                color: #B0B0B0;
                font-size: 16px;
                background: #1E1E1E;
                border: 1px dashed #2C2C2C;
                border-radius: 8px;
            }
        """)
        
        video_layout.addWidget(self.video_label)
        layout.addWidget(video_container)
        
        # Metrics section
        metrics_frame = QFrame()
        metrics_frame.setStyleSheet("background: transparent; border: none;")
        metrics_layout = QVBoxLayout(metrics_frame)
        metrics_layout.setContentsMargins(12, 8, 12, 8)
        metrics_layout.setSpacing(4)
        
        # FPS display
        self.fps_label = QLabel("FPS: 0")
        self.fps_label.setStyleSheet("""
            QLabel {
                color: #FFD700;
                font-size: 12px;
                font-weight: bold;
                background: transparent;
            }
        """)
        
        # Detection counts
        self.count_label = QLabel("Cars: 0 | Trucks: 0 | Ped: 0 | TLights: 0 | Moto: 0")
        self.count_label.setStyleSheet("""
            QLabel {
                color: #B0B0B0;
                font-size: 11px;
                background: transparent;
            }
        """)
        
        metrics_layout.addWidget(self.fps_label)
        metrics_layout.addWidget(self.count_label)
        layout.addWidget(metrics_frame)
        
        # Controls section
        controls_frame = QFrame()
        controls_frame.setStyleSheet("background: transparent; border: none;")
        controls_layout = QVBoxLayout(controls_frame)
        controls_layout.setContentsMargins(12, 4, 12, 12)
        controls_layout.setSpacing(8)
        
        # Detection toggle
        toggle_layout = QHBoxLayout()
        toggle_label = QLabel("Detection")
        toggle_label.setStyleSheet("color: #FFFFFF; font-size: 12px; background: transparent;")
        
        self.detection_toggle = ModernToggleSwitch()
        self.detection_toggle.setChecked(True)
        self.detection_toggle.toggled.connect(
            lambda checked: self.detection_toggled.emit(self.cam_number, checked)
        )
        
        toggle_layout.addWidget(toggle_label)
        toggle_layout.addStretch()
        toggle_layout.addWidget(self.detection_toggle)
        
        # Buttons
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(8)
        
        self.start_btn = QPushButton("Start")
        self.start_btn.setObjectName("primary")
        self.start_btn.clicked.connect(self._handle_start_stop)
        
        self.snapshot_btn = QPushButton("Snapshot")
        self.snapshot_btn.setObjectName("secondary")
        self.snapshot_btn.clicked.connect(
            lambda: self.snapshot_clicked.emit(self.cam_number)
        )
        
        buttons_layout.addWidget(self.start_btn)
        buttons_layout.addWidget(self.snapshot_btn)
        
        controls_layout.addLayout(toggle_layout)
        controls_layout.addLayout(buttons_layout)
        layout.addWidget(controls_frame)
        
        self.setLayout(layout)
        
    def _get_panel_style(self):
        return """
            QFrame#CameraPanel {
                background: #1E1E1E;
                border-radius: 12px;
                border: 1px solid #2C2C2C;
            }
            
            QPushButton[objectName="primary"] {
                background: #007BFF;
                color: #FFFFFF;
                border-radius: 6px;
                font-weight: bold;
                padding: 8px 16px;
                font-size: 12px;
                border: none;
            }
            QPushButton[objectName="primary"]:hover {
                background: #3399FF;
            }
            
            QPushButton[objectName="secondary"] {
                background: #2ECC71;
                color: #FFFFFF;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 12px;
                border: none;
            }
            QPushButton[objectName="secondary"]:hover {
                background: #48D187;
            }
            
            QPushButton[objectName="warning"] {
                background: #E74C3C;
                color: #FFFFFF;
                border-radius: 6px;
                font-weight: bold;
                padding: 8px 16px;
                font-size: 12px;
                border: none;
            }
            QPushButton[objectName="warning"]:hover {
                background: #FF6B5A;
            }
        """
        
    def _handle_start_stop(self):
        self.is_running = not self.is_running
        self.start_btn.setText("Stop" if self.is_running else "Start")
        self.start_btn.setObjectName("warning" if self.is_running else "primary")
        self.start_btn.setStyleSheet(self._get_panel_style())
        self.start_stop_clicked.emit(self.cam_number)
        
    def update_display(self, pixmap):
        if pixmap is None:
            self.video_label.setText("No Feed")
            self.video_label.setStyleSheet("""
                QLabel {
                    color: #B0B0B0;
                    font-size: 16px;
                    background: #1E1E1E;
                    border: 1px dashed #2C2C2C;
                    border-radius: 8px;
                }
            """)
        else:
            self.video_label.setPixmap(pixmap)
            self.video_label.setStyleSheet("background: transparent; border: none;")
            
    def update_fps(self, fps):
        self.fps_label.setText(f"FPS: {fps:.1f}")
        
    def update_counts(self, cars, trucks, peds, tlights, motorcycles):
        # Color-coded counts
        counts_html = f"""
        <span style="color: #00FFFF;">Cars: {cars}</span> | 
        <span style="color: #FFA500;">Ped: {peds}</span> | 
        <span style="color: #FFD700;">TLights: {tlights}</span> | 
        <span style="color: #A020F0;">Moto: {motorcycles}</span>
        """
        self.count_label.setText(counts_html)

class ModernFooterBar(QFrame):
    """Modern footer bar with global controls"""
    start_all_clicked = Signal()
    stop_all_clicked = Signal()
    device_changed = Signal(str)
    global_detection_toggled = Signal(bool)
    
    def __init__(self):
        super().__init__()
        self.setupUI()
        
    def setupUI(self):
        self.setObjectName("FooterBar")
        self.setStyleSheet(self._get_footer_style())
        self.setFixedHeight(60)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 10, 20, 10)
        layout.setSpacing(16)
        
        # Start/Stop All buttons
        self.start_all_btn = QPushButton("Start All")
        self.start_all_btn.setObjectName("primary")
        self.start_all_btn.clicked.connect(self.start_all_clicked.emit)
        
        self.stop_all_btn = QPushButton("Stop All")
        self.stop_all_btn.setObjectName("warning")
        self.stop_all_btn.clicked.connect(self.stop_all_clicked.emit)
        
        # Device selector
        device_label = QLabel("Device:")
        device_label.setStyleSheet("color: #FFFFFF; font-size: 12px; font-weight: bold;")
        
        self.device_combo = QComboBox()
        self.device_combo.addItems(["CPU", "GPU", "NPU"])
        self.device_combo.currentTextChanged.connect(self.device_changed.emit)
        
        # Global detection toggle
        detection_label = QLabel("Global Detection:")
        detection_label.setStyleSheet("color: #FFFFFF; font-size: 12px; font-weight: bold;")
        
        self.global_detection_toggle = ModernToggleSwitch()
        self.global_detection_toggle.setChecked(True)
        self.global_detection_toggle.toggled.connect(self.global_detection_toggled.emit)
        
        # Layout
        layout.addWidget(self.start_all_btn)
        layout.addWidget(self.stop_all_btn)
        layout.addStretch()
        layout.addWidget(device_label)
        layout.addWidget(self.device_combo)
        layout.addWidget(detection_label)
        layout.addWidget(self.global_detection_toggle)
        
    def _get_footer_style(self):
        return """
            QFrame#FooterBar {
                background: #1E1E1E;
                border-top: 1px solid #2C2C2C;
                border-radius: 0px;
            }
            
            QPushButton[objectName="primary"] {
                background: #007BFF;
                color: #FFFFFF;
                border-radius: 8px;
                font-weight: bold;
                padding: 10px 20px;
                font-size: 13px;
                border: none;
            }
            QPushButton[objectName="primary"]:hover {
                background: #3399FF;
            }
            
            QPushButton[objectName="warning"] {
                background: #E74C3C;
                color: #FFFFFF;
                border-radius: 8px;
                font-weight: bold;
                padding: 10px 20px;
                font-size: 13px;
                border: none;
            }
            QPushButton[objectName="warning"]:hover {
                background: #FF6B5A;
            }
            
            QComboBox {
                background: #1E1E1E;
                border: 1px solid #00E6E6;
                border-radius: 6px;
                padding: 6px 12px;
                color: #FFFFFF;
                font-size: 12px;
                min-width: 80px;
            }
        """

class ModernLiveDetectionTab(QWidget):
    """Modern live detection tab with 2x2 grid layout"""
    
    # Signals
    source_changed = Signal(int, object)
    run_requested = Signal(int, bool)
    detection_toggled = Signal(int, bool)
    settings_clicked = Signal(int)
    global_detection_toggled = Signal(bool)
    device_changed = Signal(str)
    snapshot_requested = Signal(int)
    video_dropped = Signal(int, object)  # Added missing signal
    
    def __init__(self):
        super().__init__()
        self.setupUI()
        
    def setupUI(self):
        # Set dark background
        self.setStyleSheet("QWidget { background: #121212; }")
        
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(16)
        
        # 2x2 Camera grid
        camera_grid = QGridLayout()
        camera_grid.setSpacing(16)
        
        self.camera_panels = []
        for i in range(4):
            panel = CameraPanel(i + 1)
            panel.start_stop_clicked.connect(self._handle_camera_action)
            panel.detection_toggled.connect(self.detection_toggled.emit)
            panel.snapshot_clicked.connect(self.snapshot_requested.emit)
            
            self.camera_panels.append(panel)
            camera_grid.addWidget(panel, i // 2, i % 2)
            
        main_layout.addLayout(camera_grid)
        
        # Footer bar
        self.footer_bar = ModernFooterBar()
        self.footer_bar.start_all_clicked.connect(self._start_all)
        self.footer_bar.stop_all_clicked.connect(self._stop_all)
        self.footer_bar.device_changed.connect(self.device_changed.emit)
        self.footer_bar.global_detection_toggled.connect(self.global_detection_toggled.emit)
        
        main_layout.addWidget(self.footer_bar)
        
    def _handle_camera_action(self, cam_number):
        """Handle start/stop button clicks from camera panels"""
        panel = self.camera_panels[cam_number - 1]
        self.run_requested.emit(cam_number, panel.is_running)
        
    def _start_all(self):
        """Start all cameras"""
        for i, panel in enumerate(self.camera_panels):
            if not panel.is_running:
                panel._handle_start_stop()
                
    def _stop_all(self):
        """Stop all cameras"""
        for i, panel in enumerate(self.camera_panels):
            if panel.is_running:
                panel._handle_start_stop()
                
    def update_display(self, cam_number, pixmap):
        """Update camera display"""
        if 1 <= cam_number <= 4:
            self.camera_panels[cam_number - 1].update_display(pixmap)
            
    def update_fps(self, cam_number, fps):
        """Update FPS display"""
        if 1 <= cam_number <= 4:
            self.camera_panels[cam_number - 1].update_fps(fps)
            
    def update_counts(self, cam_number, cars, trucks, peds, tlights, motorcycles):
        """Update detection counts"""
        if 1 <= cam_number <= 4:
            self.camera_panels[cam_number - 1].update_counts(cars, trucks, peds, tlights, motorcycles)
            
    def update_stats(self, cam_number, stats):
        """Update all stats for a camera"""
        if 1 <= cam_number <= 4:
            panel = self.camera_panels[cam_number - 1]
            panel.update_fps(stats.get('fps', 0))
            panel.update_counts(
                stats.get('cars', 0),
                stats.get('trucks', 0), 
                stats.get('peds', 0),
                stats.get('tlights', 0),
                stats.get('motorcycles', 0)
            )
            
    def update_display_np(self, np_frame):
        """Display a NumPy frame in CAM 1 (single source live mode)."""
        import cv2
        import numpy as np
        if np_frame is None or not isinstance(np_frame, np.ndarray) or np_frame.size == 0:
            print(f"[ModernLiveDetectionTab] ⚠️ Received None or empty frame for CAM 1")
            return
        try:
            rgb_frame = cv2.cvtColor(np_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            from PySide6.QtGui import QImage, QPixmap
            from PySide6.QtCore import Qt
            bytes_per_line = ch * w
            qimg = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            
            # Scale pixmap to fit the video label
            video_label = self.camera_panels[0].video_label
            scaled_pixmap = pixmap.scaled(
                video_label.width(),
                video_label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            video_label.setPixmap(scaled_pixmap)
            video_label.update()
            print(f"[ModernLiveDetectionTab] 🟢 Frame displayed for CAM 1")
        except Exception as e:
            print(f"[ModernLiveDetectionTab] ❌ Error displaying frame for CAM 1: {e}")
            import traceback
            traceback.print_exc()
            
    def set_detection_active(self, cam_number, active):
        """Set detection active state for a camera"""
        if 1 <= cam_number <= 4:
            panel = self.camera_panels[cam_number - 1]
            panel.detection_toggle.setChecked(active)
