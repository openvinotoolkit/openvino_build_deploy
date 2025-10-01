import cv2
import numpy as np
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt
from typing import Dict, List, Any

# Color mapping for traffic-related classes
COLORS = {
    'person': (255, 165, 0),         # Orange
    'bicycle': (255, 0, 255),        # Magenta
    'car': (0, 255, 0),              # Green
    'motorcycle': (255, 255, 0),     # Cyan
    'bus': (0, 0, 255),              # Red
    'truck': (0, 128, 255),          # Orange-Blue
    'traffic light': (0, 165, 255),  # Orange
    'stop sign': (0, 0, 139),        # Dark Red
    'parking meter': (128, 0, 128),  # Purple
    'default': (0, 255, 255)         # Yellow as default
}

VIOLATION_COLORS = {
    'red_light_violation': (0, 0, 255),    # Red
    'stop_sign_violation': (0, 100, 255),  # Orange-Red
    'speed_violation': (0, 255, 255),      # Yellow
    'lane_violation': (255, 0, 255),       # Magenta
    'default': (255, 0, 0)                 # Red as default
}

def draw_detections(frame: np.ndarray, detections: List[Dict], 
                  draw_labels: bool = True, draw_confidence: bool = True) -> np.ndarray:
    """
    Draw detection bounding boxes on the frame.
    
    Args:
        frame: Input video frame
        detections: List of detection dictionaries
        draw_labels: Whether to draw class labels
        draw_confidence: Whether to draw confidence scores
        
    Returns:
        Annotated frame
    """
    if frame is None or not isinstance(frame, np.ndarray):
        return np.zeros((300, 300, 3), dtype=np.uint8)
    
    annotated_frame = frame.copy()
    
    for det in detections:
        if 'bbox' not in det:
            continue
        
        try:
            bbox = det['bbox']
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                continue
                
            x1, y1, x2, y2 = map(int, bbox)
            if x1 >= x2 or y1 >= y2:
                continue
                
            class_name = det.get('class_name', 'unknown')
            confidence = det.get('confidence', 0.0)
            track_id = det.get('track_id', None)
            
            # Get color based on class
            color = COLORS.get(class_name.lower(), COLORS['default'])
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label text
            label_text = ""
            if draw_labels:
                label_text += class_name
                
                if track_id is not None:
                    label_text += f" #{track_id}"
                    
                if draw_confidence and confidence > 0:
                    label_text += f" {confidence:.2f}"
            
            # Draw label background
            if label_text:
                text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(
                    annotated_frame, 
                    (x1, y1 - text_size[1] - 8), 
                    (x1 + text_size[0] + 8, y1), 
                    color, 
                    -1
                )
                cv2.putText(
                    annotated_frame,
                    label_text,
                    (x1 + 4, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2
                )
        except Exception as e:
            print(f"Error drawing detection: {e}")
    
    return annotated_frame

def draw_violations(frame: np.ndarray, violations: List[Dict]) -> np.ndarray:
    """
    Draw violation indicators on the frame.
    (Currently disabled - just returns the original frame)
    
    Args:
        frame: Input video frame
        violations: List of violation dictionaries
        
    Returns:
        Annotated frame
    """
    # Violation detection is disabled - simply return the original frame
    if frame is None or not isinstance(frame, np.ndarray):
        return np.zeros((300, 300, 3), dtype=np.uint8)
    
    # Just return a copy of the frame without drawing violations
    return frame.copy()

def draw_performance_metrics(frame: np.ndarray, metrics: Dict) -> np.ndarray:
    """
    Draw performance metrics overlay on the frame.
    
    Args:
        frame: Input video frame
        metrics: Dictionary of performance metrics
        
    Returns:
        Annotated frame
    """
    if frame is None or not isinstance(frame, np.ndarray):
        return np.zeros((300, 300, 3), dtype=np.uint8)
    
    annotated_frame = frame.copy()
    height = annotated_frame.shape[0]
    
    # Create semi-transparent overlay
    overlay = annotated_frame.copy()
    cv2.rectangle(overlay, (10, height - 140), (250, height - 20), (0, 0, 0), -1)
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0, annotated_frame)
    
    # Draw metrics
    text_y = height - 120
    for metric, value in metrics.items():
        text = f"{metric}: {value}"
        cv2.putText(
            annotated_frame,
            text,
            (20, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )
        text_y += 25
        
    return annotated_frame

def convert_cv_to_qimage(cv_img):
    """
    Convert OpenCV image to QImage for display in Qt widgets.
    
    Args:
        cv_img: OpenCV image (numpy array)
        
    Returns:
        QImage object
    """
    if cv_img is None or not isinstance(cv_img, np.ndarray):
        return QImage(1, 1, QImage.Format_RGB888)
    
    try:
        # Make a copy to ensure the data stays in scope
        img_copy = cv_img.copy()
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        
        # Create QImage - this approach ensures continuous memory layout
        # which is important for QImage to work correctly
        qimage = QImage(rgb_image.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Return a copy to ensure it remains valid
        return qimage.copy()
    except Exception as e:
        print(f"Error converting image: {e}")
        return QImage(1, 1, QImage.Format_RGB888)

def convert_cv_to_pixmap(cv_img, target_width=None):
    """
    Convert OpenCV image to QPixmap for display in Qt widgets.
    
    Args:
        cv_img: OpenCV image (numpy array)
        target_width: Optional width to resize to (maintains aspect ratio)
        
    Returns:
        QPixmap object
    """
    try:
        if cv_img is None:
            print("WARNING: convert_cv_to_pixmap received None image")
            empty_pixmap = QPixmap(640, 480)
            empty_pixmap.fill(Qt.black)
            return empty_pixmap
            
        # Make a copy to ensure the data stays in scope
        img_copy = cv_img.copy()
            
        # Convert BGR to RGB directly
        rgb_image = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        
        # Create QImage using tobytes() to ensure a continuous copy is made
        # This avoids memory layout issues with numpy arrays
        qimg = QImage(rgb_image.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888)
        
        if qimg.isNull():
            print("WARNING: Failed to create QImage")
            empty_pixmap = QPixmap(640, 480)
            empty_pixmap.fill(Qt.black)
            return empty_pixmap
            
        # Resize if needed
        if target_width and qimg.width() > target_width:
            qimg = qimg.scaledToWidth(target_width, Qt.SmoothTransformation)
        
        # Convert to pixmap
        pixmap = QPixmap.fromImage(qimg)
        if pixmap.isNull():
            print("WARNING: Failed to create QPixmap from QImage")
            empty_pixmap = QPixmap(640, 480)
            empty_pixmap.fill(Qt.black)
            return empty_pixmap
            
        return pixmap
        
    except Exception as e:
        print(f"ERROR in convert_cv_to_pixmap: {e}")
        
        # Return a black pixmap as fallback
        empty_pixmap = QPixmap(640, 480)
        empty_pixmap.fill(Qt.black)
        return empty_pixmap

def resize_frame_for_display(frame: np.ndarray, max_width: int = 1280, max_height: int = 720) -> np.ndarray:
    """
    Resize frame for display while maintaining aspect ratio.
    
    Args:
        frame: Input video frame
        max_width: Maximum display width
        max_height: Maximum display height
        
    Returns:
        Resized frame
    """
    if frame is None:
        return np.zeros((300, 300, 3), dtype=np.uint8)
        
    height, width = frame.shape[:2]
    
    # No resize needed if image is already smaller than max dimensions
    if width <= max_width and height <= max_height:
        return frame
    
    # Calculate scale factor to fit within max dimensions
    scale_width = max_width / width if width > max_width else 1.0
    scale_height = max_height / height if height > max_height else 1.0
    
    # Use the smaller scale to ensure image fits within bounds
    scale = min(scale_width, scale_height)
    
    # Resize using calculated scale
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

def pipeline_with_violation_line(frame: np.ndarray, draw_violation_line_func, violation_line_y: int = None) -> QPixmap:
    """
    Example pipeline to ensure violation line is drawn and color order is correct.
    Args:
        frame: Input BGR frame (np.ndarray)
        draw_violation_line_func: Function to draw violation line (should accept BGR frame)
        violation_line_y: Y position for the violation line (int)
    Returns:
        QPixmap ready for display
    """
    annotated_frame = frame.copy()
    if violation_line_y is not None:
        annotated_frame = draw_violation_line_func(annotated_frame, violation_line_y, color=(0, 0, 255), label='VIOLATION LINE')
    display_frame = resize_frame_for_display(annotated_frame, max_width=1280, max_height=720)
    pixmap = convert_cv_to_pixmap(display_frame)
    return pixmap
