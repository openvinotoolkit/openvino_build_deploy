# Utility for drawing detections, tracks, and violations on frames
import numpy as np
from PySide6.QtGui import QPixmap

def enhanced_annotate_frame(app, frame, detections, violations):
    """Enhanced annotation function for drawing detections and violations on frames"""
    import cv2
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return np.zeros((300, 300, 3), dtype=np.uint8)
    
    annotated_frame = frame.copy()
    if detections is None:
        detections = []
    if violations is None:
        violations = []
        
    # Draw detections
    for det in detections:
        if 'bbox' not in det:
            continue
        try:
            bbox = det['bbox']
            if len(bbox) < 4:
                continue
            x1, y1, x2, y2 = bbox
            conf = det.get('confidence', 0.0)
            class_name = det.get('class_name', 'unknown')
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(annotated_frame, label, (int(x1), int(y1) - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        except Exception:
            continue
    
    # Draw violations
    for violation in violations:
        try:
            if 'bbox' in violation:
                bbox = violation['bbox']
                x1, y1, x2, y2 = bbox
                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
                cv2.putText(annotated_frame, "VIOLATION", (int(x1), int(y1) - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        except Exception:
            continue
    
    return annotated_frame

def resize_frame_for_display(frame, max_width=1280, max_height=720):
    """Resize frame while maintaining aspect ratio"""
    if frame is None:
        return np.zeros((300, 300, 3), dtype=np.uint8)
    
    h, w = frame.shape[:2]
    if w <= max_width and h <= max_height:
        return frame
    
    # Calculate scaling factor
    scale_w = max_width / w
    scale_h = max_height / h
    scale = min(scale_w, scale_h)
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    import cv2
    return cv2.resize(frame, (new_w, new_h))

def convert_cv_to_pixmap(cv_frame):
    """Convert OpenCV frame (BGR) to QPixmap"""
    if cv_frame is None or cv_frame.size == 0:
        # Return a blank pixmap
        pixmap = QPixmap(300, 300)
        pixmap.fill()
        return pixmap
    
    try:
        import cv2
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QImage
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(cv_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        
        # Create QImage
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Convert to QPixmap
        pixmap = QPixmap.fromImage(qt_image)
        return pixmap
    except Exception:
        # Return a blank pixmap on error
        pixmap = QPixmap(300, 300)
        pixmap.fill()
        return pixmap
