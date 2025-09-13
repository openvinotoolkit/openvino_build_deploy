from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QSizePolicy,
    QGraphicsView, QGraphicsScene
)
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QPixmap, QImage, QPainter

import cv2
import numpy as np

class SimpleLiveDisplay(QWidget):
    """Enhanced implementation for video display using QGraphicsView"""
    
    video_dropped = Signal(str)  # For drag and drop compatibility
    
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # Create QGraphicsView and QGraphicsScene
        self.graphics_view = QGraphicsView()
        self.graphics_scene = QGraphicsScene()
        self.graphics_view.setScene(self.graphics_scene)        
        self.graphics_view.setMinimumSize(640, 480)
        self.graphics_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.graphics_view.setStyleSheet("background-color: black;")
        self.graphics_view.setRenderHint(QPainter.Antialiasing)
        self.graphics_view.setRenderHint(QPainter.SmoothPixmapTransform)
        
        # Create backup label (in case QGraphicsView doesn't work)
        self.display_label = QLabel()
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setMinimumSize(640, 480)
        self.display_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.display_label.setStyleSheet("background-color: black;")
          # Set up drag and drop
        self.setAcceptDrops(True)
        
        # Add QGraphicsView to layout (primary display)
        self.layout.addWidget(self.graphics_view)
        
        # Don't add label to layout, we'll only use it as fallback if needed
        
    def update_frame(self, pixmap):
        """Update the display with a new frame"""
        if pixmap and not pixmap.isNull():
            print(f"DEBUG: SimpleLiveDisplay updating with pixmap {pixmap.width()}x{pixmap.height()}")
            
            try:
                # Method 1: Using QGraphicsScene
                self.graphics_scene.clear()
                self.graphics_scene.addPixmap(pixmap)
                self.graphics_view.fitInView(self.graphics_scene.itemsBoundingRect(), Qt.KeepAspectRatio)
                
                # Force an immediate update
                self.graphics_view.update()
                self.repaint()  # Force a complete repaint
                print("DEBUG: SimpleLiveDisplay - pixmap displayed successfully in QGraphicsView")
                
            except Exception as e:
                print(f"ERROR in QGraphicsView display: {e}, falling back to QLabel")
                try:
                    # Fallback method: Using QLabel
                    scaled_pixmap = pixmap.scaled(
                        self.display_label.size(),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation
                    )
                    self.display_label.setPixmap(scaled_pixmap)
                    self.display_label.update()
                except Exception as e2:
                    print(f"ERROR in QLabel fallback: {e2}")
                    import traceback
                    traceback.print_exc()
        else:
            print("DEBUG: SimpleLiveDisplay received null or invalid pixmap")
    
    def resizeEvent(self, event):
        """Handle resize events"""
        super().resizeEvent(event)
        # If we have a pixmap, rescale it to fit the new size
        if not self.display_label.pixmap() or self.display_label.pixmap().isNull():
            return
            
        scaled_pixmap = self.display_label.pixmap().scaled(
            self.display_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.display_label.setPixmap(scaled_pixmap)
        
    def reset_display(self):
        """Reset display to black"""
        blank = QPixmap(self.display_label.size())
        blank.fill(Qt.black)
        self.display_label.setPixmap(blank)
        
    def dragEnterEvent(self, event):
        """Handle drag enter events"""
        if event.mimeData().hasUrls():
            url = event.mimeData().urls()[0].toLocalFile()
            if url.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                event.acceptProposedAction()
                
    def dropEvent(self, event):
        """Handle drop events"""
        if event.mimeData().hasUrls():
            url = event.mimeData().urls()[0].toLocalFile()
            if url.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                self.video_dropped.emit(url)
                  
                  
    def display_frame(self, frame: np.ndarray):
        """Display a NumPy OpenCV frame directly (converts to QPixmap and calls update_frame)"""
        if frame is None:
            print("⚠️ Empty frame received")
            return
            
        # Force a debug print with the frame shape
        print(f"🟢 display_frame CALLED with frame: type={type(frame)}, shape={getattr(frame, 'shape', None)}")
        
        try:
            # Make a copy of the frame to ensure we're not using memory that might be released
            frame_copy = frame.copy()
            
            # Convert BGR to RGB (OpenCV uses BGR, Qt uses RGB)
            rgb_frame = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
            
            # Print shape info
            h, w, ch = rgb_frame.shape
            print(f"📊 Frame dimensions: {w}x{h}, channels: {ch}")
            
            # Force continuous array for QImage
            if not rgb_frame.flags['C_CONTIGUOUS']:
                rgb_frame = np.ascontiguousarray(rgb_frame)
                
            # Create QImage - critical to use .copy() to ensure Qt owns the data
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
            
            # Check if QImage is valid
            if qt_image.isNull():
                print("⚠️ Failed to create QImage")
                return
                
            # Create QPixmap from QImage
            pixmap = QPixmap.fromImage(qt_image)
            
            # METHOD 1: Display using QGraphicsScene/View
            try:
                self.graphics_scene.clear()
                self.graphics_scene.addPixmap(pixmap)
                self.graphics_view.setScene(self.graphics_scene)
                
                # Set the view to fit the content
                self.graphics_view.fitInView(self.graphics_scene.itemsBoundingRect(), Qt.KeepAspectRatio)
                
                # Force immediate updates
                self.graphics_view.viewport().update()
                self.graphics_view.update()
                print("✅ Frame displayed in QGraphicsView")
            except Exception as e:
                print(f"⚠️ Error in QGraphicsView display: {e}")
                
                # METHOD 2: Fall back to QLabel if QGraphicsView fails
                try:
                    # Add to layout if not already there
                    if self.display_label not in self.layout.children():
                        self.layout.addWidget(self.display_label)
                        self.graphics_view.hide()
                        self.display_label.show()
                    
                    # Scale pixmap for display
                    scaled_pixmap = pixmap.scaled(
                        max(self.display_label.width(), 640),
                        max(self.display_label.height(), 480),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation
                    )
                    
                    self.display_label.setPixmap(scaled_pixmap)
                    self.display_label.setScaledContents(True)
                    self.display_label.update()
                    print("✅ Frame displayed in QLabel (fallback)")
                except Exception as e2:
                    print(f"❌ ERROR in QLabel fallback: {e2}")
                    import traceback
                    traceback.print_exc()
                    
        except Exception as main_error:
            print(f"❌ CRITICAL ERROR in display_frame: {main_error}")
            import traceback
            traceback.print_exc()
