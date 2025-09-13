"""
Detection Overlay Widget - Displays object detection bounding boxes and labels
"""

from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout, QFrame
from PySide6.QtCore import Qt, Signal, QRect, QPoint, QTimer
from PySide6.QtGui import QPainter, QPen, QBrush, QColor, QFont
import json

class DetectionOverlayWidget(QWidget):
    """
    Widget for displaying object detection overlays on video frames
    
    Features:
    - Bounding box visualization
    - Class labels and confidence scores
    - Tracking IDs for multi-object tracking
    - Color-coded detection classes
    - Customizable overlay styles
    """
    
    # Signals
    detection_clicked = Signal(dict)  # Emitted when a detection is clicked
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.detections = []
        self.class_colors = {
            'person': QColor(255, 0, 0),      # Red
            'bicycle': QColor(0, 255, 0),     # Green
            'car': QColor(0, 0, 255),         # Blue
            'motorcycle': QColor(255, 255, 0), # Yellow
            'bus': QColor(255, 0, 255),       # Magenta
            'truck': QColor(0, 255, 255),     # Cyan
            'traffic_light': QColor(255, 165, 0), # Orange
            'stop_sign': QColor(128, 0, 128),  # Purple
        }
        
        self.show_labels = True
        self.show_confidence = True
        self.show_tracking_id = True
        self.overlay_opacity = 0.8
        
        # Make widget transparent for overlay
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.setStyleSheet("background-color: transparent;")
    
    def set_detections(self, detections):
        """
        Set detection results to display
        
        Args:
            detections: List of detection dictionaries with format:
                {
                    'bbox': [x1, y1, x2, y2],
                    'class': 'person',
                    'confidence': 0.95,
                    'track_id': 123 (optional)
                }
        """
        self.detections = detections
        self.update()  # Trigger repaint
    
    def set_overlay_options(self, show_labels=True, show_confidence=True, 
                          show_tracking_id=True, opacity=0.8):
        """Configure overlay display options"""
        self.show_labels = show_labels
        self.show_confidence = show_confidence
        self.show_tracking_id = show_tracking_id
        self.overlay_opacity = opacity
        self.update()
    
    def add_class_color(self, class_name, color):
        """Add or update color for a detection class"""
        self.class_colors[class_name] = color
    
    def paintEvent(self, event):
        """Paint detection overlays"""
        if not self.detections:
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Set up font for labels
        font = QFont("Arial", 10, QFont.Bold)
        painter.setFont(font)
        
        for detection in self.detections:
            self._draw_detection(painter, detection)
    
    def _draw_detection(self, painter, detection):
        """Draw a single detection"""
        # Extract detection info
        bbox = detection.get('bbox', [0, 0, 100, 100])
        class_name = detection.get('class', 'unknown')
        confidence = detection.get('confidence', 0.0)
        track_id = detection.get('track_id', None)
        
        x1, y1, x2, y2 = bbox
        
        # Get color for this class
        color = self.class_colors.get(class_name, QColor(128, 128, 128))
        
        # Set up pen for bounding box
        pen = QPen(color, 3)
        painter.setPen(pen)
        
        # Draw bounding box
        rect = QRect(int(x1), int(y1), int(x2-x1), int(y2-y1))
        painter.drawRect(rect)
        
        # Prepare label text
        label_parts = []
        if self.show_labels:
            label_parts.append(class_name)
        if self.show_confidence:
            label_parts.append(f"{confidence:.2f}")
        if self.show_tracking_id and track_id is not None:
            label_parts.append(f"ID:{track_id}")
        
        if not label_parts:
            return
        
        label_text = " | ".join(label_parts)
        
        # Calculate label background
        metrics = painter.fontMetrics()
        label_width = metrics.horizontalAdvance(label_text) + 10
        label_height = metrics.height() + 6
        
        # Draw label background
        label_rect = QRect(int(x1), int(y1-label_height), label_width, label_height)
        painter.fillRect(label_rect, color)
        
        # Draw label text
        painter.setPen(QPen(QColor(255, 255, 255), 1))
        text_rect = QRect(int(x1+5), int(y1-label_height+3), label_width-10, label_height-6)
        painter.drawText(text_rect, Qt.AlignLeft | Qt.AlignVCenter, label_text)
    
    def mousePressEvent(self, event):
        """Handle mouse clicks on detections"""
        if event.button() == Qt.LeftButton:
            click_pos = event.pos()
            
            # Check if click is within any detection bbox
            for detection in self.detections:
                bbox = detection.get('bbox', [0, 0, 100, 100])
                x1, y1, x2, y2 = bbox
                
                if x1 <= click_pos.x() <= x2 and y1 <= click_pos.y() <= y2:
                    self.detection_clicked.emit(detection)
                    break
    
    def clear_detections(self):
        """Clear all detections"""
        self.detections = []
        self.update()
    
    def get_detection_count(self):
        """Get number of current detections"""
        return len(self.detections)
    
    def get_class_counts(self):
        """Get count of detections by class"""
        counts = {}
        for detection in self.detections:
            class_name = detection.get('class', 'unknown')
            counts[class_name] = counts.get(class_name, 0) + 1
        return counts
