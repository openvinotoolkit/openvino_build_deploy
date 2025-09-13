from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel

class GlobalStatusPanel(QWidget):
    def __init__(self):
        super().__init__()
        layout = QHBoxLayout()
        self.model_label = QLabel("Model: -")
        self.device_label = QLabel("Device: -")
        self.yolo_label = QLabel("YOLO Version: -")
        self.resolution_label = QLabel("Resolution: -")
        self.fps_labels = [QLabel(f"CAM {i+1} FPS: -") for i in range(4)]
        layout.addWidget(self.model_label)
        layout.addWidget(self.device_label)
        layout.addWidget(self.yolo_label)
        layout.addWidget(self.resolution_label)
        for lbl in self.fps_labels:
            layout.addWidget(lbl)
        self.setLayout(layout)
    def update_status(self, model, device, yolo, resolution, fps_list):
        self.model_label.setText(f"Model: {model}")
        self.device_label.setText(f"Device: {device}")
        self.yolo_label.setText(f"YOLO Version: {yolo}")
        self.resolution_label.setText(f"Resolution: {resolution}")
        for i, fps in enumerate(fps_list):
            self.fps_labels[i].setText(f"CAM {i+1} FPS: {fps}")
