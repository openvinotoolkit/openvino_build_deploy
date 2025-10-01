from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QSizePolicy,
    QGraphicsView, QGraphicsScene
)
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QPixmap, QImage, QPainter

import cv2
import numpy as np
import time

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
        
        # Track frame update times
        self.last_update = time.time()
        self.frame_count = 0
        self.fps = 0.0
          
        # Set up drag and drop
        self.setAcceptDrops(True)
        
        # Add QGraphicsView to layout (primary display)
        self.layout.addWidget(self.graphics_view)
        
        # Don't add label to layout, we'll only use it as fallback if needed
        
    def update_frame(self, pixmap, overlay_states=None):
        """Update the display with a new frame, using overlay_states to control overlays"""
        if overlay_states is None:
            overlay_states = {
                'show_vehicles': True,
                'show_ids': True,
                'show_red_light': True,
                'show_violation': True,
            }
        if pixmap and not pixmap.isNull():
            print(f"DEBUG: SimpleLiveDisplay updating with pixmap {pixmap.width()}x{pixmap.height()}")
            # Here you would use overlay_states to control what is drawn
            # For example, in your actual drawing logic:
            # if overlay_states['show_vehicles']:
            #     draw detection boxes
            # if overlay_states['show_ids']:
            #     draw IDs
            # if overlay_states['show_red_light']:
            #     draw traffic light color
            # if overlay_states['show_violation']:
            #     draw violation line
            try:
                self.graphics_scene.clear()
                self.graphics_scene.addPixmap(pixmap)
                self.graphics_view.fitInView(self.graphics_scene.itemsBoundingRect(), Qt.KeepAspectRatio)
                self.graphics_view.update()
                self.graphics_view.viewport().update()
                print("DEBUG: SimpleLiveDisplay - pixmap displayed successfully in QGraphicsView")
            except Exception as e:
                print(f"ERROR in QGraphicsView display: {e}, falling back to QLabel")
                try:
                    scaled_pixmap = pixmap.scaled(
                        self.display_label.width() or pixmap.width(),
                        self.display_label.height() or pixmap.height(),
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
        # If we have content in the scene, resize it to fit
        if not self.graphics_scene.items():
            return
            
        self.graphics_view.fitInView(self.graphics_scene.itemsBoundingRect(), Qt.KeepAspectRatio)
        
    def reset_display(self):
        """Reset display to black"""
        blank = QPixmap(self.width(), self.height())
        blank.fill(Qt.black)
        self.update_frame(blank)
        
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
        """Display a NumPy OpenCV frame directly (converts to QPixmap and displays)"""
        # Check for frame validity
        if frame is None:
            print("⚠️ Empty frame received")
            return
            
        # Calculate FPS
        now = time.time()
        time_diff = now - self.last_update
        self.frame_count += 1
        if time_diff >= 1.0:
            self.fps = self.frame_count / time_diff
            print(f"🎬 Display FPS: {self.fps:.2f}")
            self.frame_count = 0
            self.last_update = now
            
        # Print debug info about the frame
        print(f"🟢 display_frame: frame shape={getattr(frame, 'shape', None)}, dtype={getattr(frame, 'dtype', None)}")
        print(f"💾 Frame memory address: {hex(id(frame))}")
        
        try:
            print("💻 Processing frame for display...")
            # Make a copy of the frame to ensure we're not using memory that might be released
            frame_copy = frame.copy()
            
            # Convert BGR to RGB (OpenCV uses BGR, Qt uses RGB)
            rgb_frame = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
            
            # Force continuous array for QImage
            is_contiguous = rgb_frame.flags.c_contiguous
            print(f"🔄 RGB frame is contiguous: {is_contiguous}")
            if not is_contiguous:
                print("⚙️ Making frame contiguous...")
                rgb_frame = np.ascontiguousarray(rgb_frame)
                
            # Get dimensions
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            print(f"📏 Frame dimensions: {w}x{h}, channels: {ch}, bytes_per_line: {bytes_per_line}")
            
            # Create QImage - use .copy() to ensure Qt owns the data
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
            
            if qt_image.isNull():
                print("⚠️ Failed to create QImage")
                return
                
            # Create QPixmap and update display
            pixmap = QPixmap.fromImage(qt_image)
            print(f"📊 Created pixmap: {pixmap.width()}x{pixmap.height()}, isNull: {pixmap.isNull()}")            # Method 1: Use graphics scene (preferred)
            try:
                self.graphics_scene.clear()
                self.graphics_scene.addPixmap(pixmap)
                self.graphics_view.fitInView(self.graphics_scene.itemsBoundingRect(), Qt.KeepAspectRatio)
                self.graphics_view.update()
                self.graphics_view.viewport().update()
                
                # Draw simple FPS counter on the view
                fps_text = f"Display: {self.fps:.1f} FPS"
                self.graphics_scene.addText(fps_text)
                print("✅ Frame displayed in graphics view")
                
            except Exception as e:
                print(f"⚠️ QGraphicsView error: {e}, using QLabel fallback")
                
                # Method 2: Fall back to QLabel
                if self.display_label.parent() is None:
                    self.layout.removeWidget(self.graphics_view)
                    self.graphics_view.hide()
                    self.layout.addWidget(self.display_label)
                    self.display_label.show()
                
                # Set pixmap on the label
                self.display_label.setPixmap(pixmap)
                self.display_label.setScaledContents(True)
                print("✅ Frame displayed in label (fallback)")
                
        except Exception as e:
            print(f"❌ Critical error in display_frame: {e}")
            import traceback
            traceback.print_exc()
