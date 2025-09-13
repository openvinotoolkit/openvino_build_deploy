from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout
from PySide6.QtCore import QTimer
import psutil

class PerformanceOverlay(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(self.windowFlags() | 0x00080000)  # Qt.ToolTip
        layout = QVBoxLayout(self)
        self.cpu_label = QLabel("CPU: --%")
        self.ram_label = QLabel("RAM: --%")
        self.fps_label = QLabel("FPS: --")
        self.infer_label = QLabel("Inference: -- ms")
        layout.addWidget(self.cpu_label)
        layout.addWidget(self.ram_label)
        layout.addWidget(self.fps_label)
        layout.addWidget(self.infer_label)
        self.fps = None
        self.infer_time = None
        self.update_stats()
        # Add timer for auto-refresh
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_stats)
        self.timer.start(1000)  # Update every second

    def update_stats(self):
        self.cpu_label.setText(f"CPU: {psutil.cpu_percent()}%")
        self.ram_label.setText(f"RAM: {psutil.virtual_memory().percent}%")
        if self.fps is not None:
            self.fps_label.setText(f"FPS: {self.fps:.1f}")
        else:
            self.fps_label.setText("FPS: --")
        if self.infer_time is not None:
            self.infer_label.setText(f"Inference: {self.infer_time:.1f} ms")
        else:
            self.infer_label.setText("Inference: -- ms")

    def set_video_stats(self, fps, inference_time):
        self.fps = fps
        self.infer_time = inference_time
        self.update_stats()
