"""
Video Display Widget - Modern video player with detection overlays
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame
from PySide6.QtCore import Qt, Signal, QTimer, QRect
from PySide6.QtGui import QPainter, QPixmap, QColor, QFont, QPen, QBrush

class VideoDisplayWidget(QWidget):
    """
    Modern video display widget with detection overlays and controls
    
    Features:
    - Video frame display with aspect ratio preservation
    - Detection bounding boxes with confidence scores
    - Tracking ID display
    - Recording indicator
    - Camera status overlay
    - Fullscreen support via double-click
    """
    
    # Signals
    double_clicked = Signal()
    recording_toggled = Signal(bool)
    snapshot_requested = Signal()
    
    def __init__(self, camera_id="camera_1", parent=None):
        super().__init__(parent)
        
        self.camera_id = camera_id
        self.current_frame = None
        self.detections = []
        self.confidence_threshold = 0.5
        self.is_recording = False
        self.camera_status = "offline"
        
        # Overlay settings
        self.overlay_settings = {
            'show_boxes': True,
            'show_tracks': True,
            'show_speed': False,
            'show_confidence': True
        }
        
        # Colors for different detection classes
        self.class_colors = {
            'car': QColor('#3498db'),      # Blue
            'truck': QColor('#e74c3c'),    # Red
            'bus': QColor('#f39c12'),      # Orange
            'motorcycle': QColor('#9b59b6'), # Purple
            'bicycle': QColor('#1abc9c'),   # Turquoise
            'person': QColor('#2ecc71'),    # Green
            'default': QColor('#95a5a6')    # Gray
        }
        
        self._setup_ui()
        
        print(f"📺 Video Display Widget initialized for {camera_id}")
    
    def _setup_ui(self):
        """Setup the video display UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        
        # Header with camera info
        header = self._create_header()
        layout.addWidget(header)
        
        # Video display area
        self.video_frame = QFrame()
        self.video_frame.setMinimumSize(320, 240)
        self.video_frame.setStyleSheet("""
            QFrame {
                background-color: #2c3e50;
                border: 1px solid #34495e;
                border-radius: 4px;
            }
        """)
        layout.addWidget(self.video_frame, 1)
        
        # Footer with controls
        footer = self._create_footer()
        layout.addWidget(footer)
    
    def _create_header(self):
        """Create header with camera info"""
        header = QFrame()
        header.setFixedHeight(25)
        header.setStyleSheet("background-color: rgba(0, 0, 0, 0.7); border-radius: 2px;")
        
        layout = QHBoxLayout(header)
        layout.setContentsMargins(5, 2, 5, 2)
        
        # Camera name
        self.camera_label = QLabel(self.camera_id.replace('_', ' ').title())
        self.camera_label.setStyleSheet("color: white; font-weight: bold; font-size: 8pt;")
        layout.addWidget(self.camera_label)
        
        layout.addStretch()
        
        # Status indicator
        self.status_label = QLabel("●")
        self.status_label.setStyleSheet("color: #e74c3c; font-size: 10pt;")
        layout.addWidget(self.status_label)
        
        # Recording indicator
        self.recording_indicator = QLabel()
        self.recording_indicator.setStyleSheet("color: #e74c3c; font-size: 8pt;")
        self.recording_indicator.hide()
        layout.addWidget(self.recording_indicator)
        
        return header
    
    def _create_footer(self):
        """Create footer with quick controls"""
        footer = QFrame()
        footer.setFixedHeight(20)
        footer.setStyleSheet("background-color: rgba(0, 0, 0, 0.5); border-radius: 2px;")
        
        layout = QHBoxLayout(footer)
        layout.setContentsMargins(2, 1, 2, 1)
        
        # Quick action buttons (small)
        snapshot_btn = QPushButton("📸")
        snapshot_btn.setFixedSize(16, 16)
        snapshot_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: none;
                color: white;
                font-size: 6pt;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 0.2);
                border-radius: 2px;
            }
        """)
        snapshot_btn.clicked.connect(self.take_snapshot)
        layout.addWidget(snapshot_btn)
        
        layout.addStretch()
        
        # FPS counter
        self.fps_label = QLabel("-- fps")
        self.fps_label.setStyleSheet("color: white; font-size: 6pt;")
        layout.addWidget(self.fps_label)
        
        return footer
    
    def set_frame(self, frame):
        """Set the current video frame"""
        self.current_frame = frame
        self.update()  # Trigger repaint
    
    def add_detections(self, detections):
        """Add detection results to display"""
        self.detections = detections
        self.update()  # Trigger repaint
    
    def set_confidence_threshold(self, threshold):
        """Set confidence threshold for detection display"""
        self.confidence_threshold = threshold
        self.update()
    
    def update_overlay_settings(self, settings):
        """Update overlay display settings"""
        self.overlay_settings.update(settings)
        self.update()
    
    def set_recording_indicator(self, recording):
        """Set recording status indicator"""
        self.is_recording = recording
        if recording:
            self.recording_indicator.setText("🔴 REC")
            self.recording_indicator.show()
        else:
            self.recording_indicator.hide()
    
    def set_camera_status(self, status):
        """Set camera status (online/offline/error)"""
        self.camera_status = status
        
        status_colors = {
            'online': '#27ae60',    # Green
            'offline': '#e74c3c',   # Red
            'error': '#f39c12',     # Orange
            'connecting': '#3498db' # Blue
        }
        
        color = status_colors.get(status, '#e74c3c')
        self.status_label.setStyleSheet(f"color: {color}; font-size: 10pt;")
    
    def take_snapshot(self):
        """Take a snapshot of current frame"""
        self.snapshot_requested.emit()
        
        # Visual feedback
        self.setStyleSheet("border: 2px solid white;")
        QTimer.singleShot(200, lambda: self.setStyleSheet(""))
    
    def mouseDoubleClickEvent(self, event):
        """Handle double-click for fullscreen"""
        if event.button() == Qt.LeftButton:
            self.double_clicked.emit()
    
    def paintEvent(self, event):
        """Custom paint event for video and overlays"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Get drawing area
        rect = self.video_frame.geometry()
        
        # Draw video frame if available
        if self.current_frame is not None:
            # Scale frame to fit widget while maintaining aspect ratio
            scaled_pixmap = self.current_frame.scaled(
                rect.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            
            # Center the image
            x = rect.x() + (rect.width() - scaled_pixmap.width()) // 2
            y = rect.y() + (rect.height() - scaled_pixmap.height()) // 2
            
            painter.drawPixmap(x, y, scaled_pixmap)
            
            # Draw detection overlays
            if self.detections and any(self.overlay_settings.values()):
                self._draw_detections(painter, rect, scaled_pixmap.size())
        else:
            # Draw placeholder
            painter.fillRect(rect, QColor(44, 62, 80))
            painter.setPen(QColor(255, 255, 255))
            painter.setFont(QFont("Arial", 12))
            painter.drawText(rect, Qt.AlignCenter, "No Signal")
    
    def _draw_detections(self, painter, display_rect, frame_size):
        """Draw detection overlays"""
        # Calculate scaling factors
        scale_x = frame_size.width() / display_rect.width()
        scale_y = frame_size.height() / display_rect.height()
        
        for detection in self.detections:
            # Skip low confidence detections
            confidence = detection.get('confidence', 0)
            if confidence < self.confidence_threshold:
                continue
            
            # Get detection info
            bbox = detection.get('bbox', [0, 0, 0, 0])  # [x, y, w, h]
            class_name = detection.get('class', 'unknown')
            track_id = detection.get('track_id', None)
            speed = detection.get('speed', None)
            
            # Scale bounding box to display coordinates
            x = int(bbox[0] / scale_x) + display_rect.x()
            y = int(bbox[1] / scale_y) + display_rect.y()
            w = int(bbox[2] / scale_x)
            h = int(bbox[3] / scale_y)
            
            # Get color for this class
            color = self.class_colors.get(class_name, self.class_colors['default'])
            
            # Draw bounding box
            if self.overlay_settings['show_boxes']:
                pen = QPen(color, 2)
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
                painter.drawRect(x, y, w, h)
            
            # Draw labels
            labels = []
            
            if self.overlay_settings['show_confidence']:
                labels.append(f"{class_name}: {confidence:.2f}")
            
            if self.overlay_settings['show_tracks'] and track_id is not None:
                labels.append(f"ID: {track_id}")
            
            if self.overlay_settings['show_speed'] and speed is not None:
                labels.append(f"{speed:.1f} km/h")
            
            if labels:
                self._draw_label(painter, x, y, labels, color)
    
    def _draw_label(self, painter, x, y, labels, color):
        """Draw detection label with background"""
        text = " | ".join(labels)
        
        # Set font
        font = QFont("Arial", 8, QFont.Bold)
        painter.setFont(font)
        
        # Calculate text size
        fm = painter.fontMetrics()
        text_rect = fm.boundingRect(text)
        
        # Draw background
        bg_rect = QRect(x, y - text_rect.height() - 4, 
                       text_rect.width() + 8, text_rect.height() + 4)
        
        bg_color = QColor(color)
        bg_color.setAlpha(180)
        painter.fillRect(bg_rect, bg_color)
        
        # Draw text
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(x + 4, y - 4, text)
    
    def update_fps(self, fps):
        """Update FPS display"""
        self.fps_label.setText(f"{fps:.1f} fps")
