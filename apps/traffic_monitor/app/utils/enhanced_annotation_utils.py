import cv2
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt

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

# Enhanced class colors for consistent visualization
def get_enhanced_class_color(class_name: str, class_id: int = -1) -> Tuple[int, int, int]:
    """
    Get color for class with enhanced mapping (traffic classes only)
    
    Args:
        class_name: Name of the detected class
        class_id: COCO class ID
        
    Returns:
        BGR color tuple
    """
    # Only traffic class IDs/colors
    enhanced_colors = {
        0: (255, 165, 0),      # person - Orange
        1: (255, 0, 255),      # bicycle - Magenta
        2: (0, 255, 0),        # car - Green
        3: (255, 255, 0),      # motorcycle - Cyan
        4: (0, 0, 255),        # bus - Red
        5: (0, 128, 255),      # truck - Orange-Blue
        6: (0, 165, 255),      # traffic light - Orange
        7: (0, 0, 139),        # stop sign - Dark Red
        8: (128, 0, 128),      # parking meter - Purple
    }
    
    # Get color from class name if available
    if class_name and class_name.lower() in COLORS:
        return COLORS[class_name.lower()]
    
    # Get color from class ID if available
    if isinstance(class_id, int) and class_id in enhanced_colors:
        return enhanced_colors[class_id]
    
    # Default color
    return COLORS['default']

def enhanced_draw_detections(frame: np.ndarray, detections: List[Dict], 
                           draw_labels: bool = True, draw_confidence: bool = True) -> np.ndarray:
    """
    Enhanced version of draw_detections with better visualization
    
    Args:
        frame: Input video frame
        detections: List of detection dictionaries
        draw_labels: Whether to draw class labels
        draw_confidence: Whether to draw confidence scores
        
    Returns:
        Annotated frame
    """
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        print("Warning: Invalid frame provided to enhanced_draw_detections")
        return np.zeros((300, 300, 3), dtype=np.uint8)  # Return blank frame as fallback
        
    annotated_frame = frame.copy()
    
    # Handle case when detections is None or empty
    if detections is None or len(detections) == 0:
        return annotated_frame
    
    # Get frame dimensions for validation
    h, w = frame.shape[:2]
    
    for detection in detections:
        if not isinstance(detection, dict):
            continue
            
        try:
            # Skip detection if it doesn't have bbox or has invalid confidence
            if 'bbox' not in detection:
                continue
                
            # Skip if confidence is below threshold (don't rely on external filtering)
            confidence = detection.get('confidence', 0.0)
            if confidence < 0.1:  # Apply a minimal threshold to ensure we're not drawing noise
                continue
            
            bbox = detection['bbox']
            class_name = detection.get('class_name', 'unknown')
            class_id = detection.get('class_id', -1)
            
            # Get color for class
            color = get_enhanced_class_color(class_name, class_id)
            
            # Ensure bbox has enough coordinates and they are numeric values
            if len(bbox) < 4 or not all(isinstance(coord, (int, float)) for coord in bbox[:4]):
                continue
                
            # Convert coordinates to integers
            try:
                x1, y1, x2, y2 = map(int, bbox[:4])
            except (ValueError, TypeError):
                print(f"Warning: Invalid bbox format: {bbox}")
                continue
                
            # Validate coordinates are within frame bounds
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            
            # Ensure x2 > x1 and y2 > y1 (at least 1 pixel width/height)
            if x2 <= x1 or y2 <= y1:
                # Instead of skipping, fix the coordinates to ensure at least 1 pixel width/height
                x2 = max(x1 + 1, x2)
                y2 = max(y1 + 1, y2)
                                
            # Draw bounding box with thicker line for better visibility
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label text
            label_parts = []
            if draw_labels:
                # Display proper class name
                display_name = class_name.replace('_', ' ').title()
                label_parts.append(display_name)
                
                # Add tracking ID if available
                track_id = detection.get('track_id')
                if track_id is not None:
                    label_parts[-1] += f" #{track_id}"
            
            if draw_confidence and confidence > 0:
                label_parts.append(f"{confidence:.2f}")
            
            # Draw traffic light color indicator if available
            if class_name == 'traffic light' and 'traffic_light_color' in detection:
                light_color = detection['traffic_light_color']
                
                # Add traffic light color to label
                if light_color != 'unknown':
                    # Set color indicator based on traffic light state
                    if light_color == 'red':
                        color_indicator = (0, 0, 255)  # Red
                        label_parts.append("🔴 RED")
                    elif light_color == 'yellow':
                        color_indicator = (0, 255, 255)  # Yellow
                        label_parts.append("🟡 YELLOW")
                    elif light_color == 'green':
                        color_indicator = (0, 255, 0)  # Green
                        label_parts.append("🟢 GREEN")
                
                # Draw traffic light visual indicator (circle with detected color)
                circle_y = y1 - 15
                circle_x = x1 + 10
                circle_radius = 10
                
                if light_color == 'red':
                    cv2.circle(annotated_frame, (circle_x, circle_y), circle_radius, (0, 0, 255), -1)
                elif light_color == 'yellow':
                    cv2.circle(annotated_frame, (circle_x, circle_y), circle_radius, (0, 255, 255), -1)
                elif light_color == 'green':
                    cv2.circle(annotated_frame, (circle_x, circle_y), circle_radius, (0, 255, 0), -1)
            
            # Draw label if we have any text
            if label_parts:
                label = " ".join(label_parts)
                
                try:
                    # Get text size for background
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    
                    # Ensure label position is within frame
                    text_y = max(text_height + 10, y1)
                    
                    # Draw label background (use colored background)
                    bg_color = tuple(int(c * 0.7) for c in color)  # Darker version of box color
                    cv2.rectangle(
                        annotated_frame,
                        (x1, text_y - text_height - 10),
                        (x1 + text_width + 10, text_y),
                        bg_color,
                        -1
                    )
                    # Draw label text (white text on colored background)
                    cv2.putText(
                        annotated_frame,
                        label,
                        (x1 + 5, text_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),  # White text
                        2
                    )
                except Exception as e:
                    print(f"Error drawing label: {e}")
                    
        except Exception as e:
            print(f"Error drawing detection: {e}")
            continue
    
    return annotated_frame

def draw_performance_overlay(frame: np.ndarray, metrics: Dict) -> np.ndarray:
    """
    Draw enhanced performance metrics overlay on the frame.
    
    Args:
        frame: Input video frame
        metrics: Dictionary of performance metrics
        
    Returns:
        Annotated frame
    """
    if frame is None or not isinstance(frame, np.ndarray):
        return np.zeros((300, 300, 3), dtype=np.uint8)
    
    annotated_frame = frame.copy()
    height, width = annotated_frame.shape[:2]
    
    # Create semi-transparent overlay for metrics panel
    overlay = annotated_frame.copy()
    
    # Calculate panel size based on metrics count
    text_height = 25
    padding = 10
    metrics_count = len(metrics)
    panel_height = metrics_count * text_height + 2 * padding
    panel_width = 220  # Fixed width
    
    # Position panel at bottom left
    panel_x = 10
    panel_y = height - panel_height - 10
    
    # Draw background panel with transparency
    cv2.rectangle(
        overlay,
        (panel_x, panel_y),
        (panel_x + panel_width, panel_y + panel_height),
        (0, 0, 0),
        -1
    )
    
    # Apply transparency
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0, annotated_frame)
    
    # Draw metrics with custom formatting
    text_y = panel_y + padding + text_height
    for metric, value in metrics.items():
        # Format metric name and value
        metric_text = f"{metric}: {value}"
        
        # Choose color based on metric type
        if "FPS" in metric:
            color = (0, 255, 0)  # Green for FPS
        elif "ms" in str(value):
            color = (0, 255, 255)  # Yellow for timing metrics
        else:
            color = (255, 255, 255)  # White for other metrics
        
        # Draw text with drop shadow for better readability
        cv2.putText(
            annotated_frame,
            metric_text,
            (panel_x + padding + 1, text_y + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),  # Black shadow
            2
        )
        cv2.putText(
            annotated_frame,
            metric_text,
            (panel_x + padding, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )
        text_y += text_height
    
    return annotated_frame

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

def enhanced_cv_to_qimage(cv_img: np.ndarray) -> QImage:
    """
    Enhanced converter from OpenCV image to QImage with robust error handling.
    
    Args:
        cv_img: OpenCV image (numpy array)
        
    Returns:
        QImage object
    """
    if cv_img is None or not isinstance(cv_img, np.ndarray):
        print("Warning: Invalid image in enhanced_cv_to_qimage")
        # Return a small black image as fallback
        return QImage(10, 10, QImage.Format_RGB888)
    
    try:
        # Get image dimensions and verify its validity
        h, w, ch = cv_img.shape
        if h <= 0 or w <= 0 or ch != 3:
            raise ValueError(f"Invalid image dimensions: {h}x{w}x{ch}")
        
        # OpenCV uses BGR, Qt uses RGB format, so convert
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        
        # Calculate bytes per line
        bytes_per_line = ch * w
        
        # Use numpy array data directly
        # This avoids a copy, but ensures the data is properly aligned
        # by creating a contiguous array
        contiguous_data = np.ascontiguousarray(rgb_image)
        
        # Create QImage from numpy array
        q_image = QImage(contiguous_data.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Create a copy to ensure the data stays valid when returning
        return q_image.copy()
    except Exception as e:
        print(f"Error in enhanced_cv_to_qimage: {e}")
        # Return a small black image as fallback
        return QImage(10, 10, QImage.Format_RGB888)

def enhanced_cv_to_pixmap(cv_img: np.ndarray, target_width: int = None) -> QPixmap:
    """
    Enhanced converter from OpenCV image to QPixmap with robust error handling.
    
    Args:
        cv_img: OpenCV image (numpy array)
        target_width: Optional width to resize to (maintains aspect ratio)
        
    Returns:
        QPixmap object
    """
    if cv_img is None or not isinstance(cv_img, np.ndarray):
        print("Warning: Invalid image in enhanced_cv_to_pixmap")
        # Create an empty pixmap with visual indication of error
        empty_pixmap = QPixmap(640, 480)
        empty_pixmap.fill(Qt.black)
        return empty_pixmap
    
    try:
        # First convert to QImage
        q_image = enhanced_cv_to_qimage(cv_img)
        
        if q_image.isNull():
            raise ValueError("Generated null QImage")
            
        # Resize if needed
        if target_width and q_image.width() > target_width:
            q_image = q_image.scaledToWidth(target_width, Qt.SmoothTransformation)
        
        # Convert to QPixmap
        pixmap = QPixmap.fromImage(q_image)
        
        if pixmap.isNull():
            raise ValueError("Generated null QPixmap")
            
        return pixmap
    except Exception as e:
        print(f"Error in enhanced_cv_to_pixmap: {e}")
        # Create an empty pixmap with visual indication of error
        empty_pixmap = QPixmap(640, 480)
        empty_pixmap.fill(Qt.black)
        return empty_pixmap
