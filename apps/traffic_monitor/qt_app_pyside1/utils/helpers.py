import json
import os
import sys
import time
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta

def bbox_iou(box1, box2):
    """
    Calculate IoU (Intersection over Union) between two bounding boxes
    
    Args:
        box1: First bounding box in format [x1, y1, x2, y2]
        box2: Second bounding box in format [x1, y1, x2, y2]
        
    Returns:
        IoU score between 0 and 1
    """
    # Ensure boxes are in [x1, y1, x2, y2] format and have valid dimensions
    if len(box1) < 4 or len(box2) < 4:
        return 0.0
        
    # Convert to float and ensure x2 > x1 and y2 > y1
    x1_1, y1_1, x2_1, y2_1 = map(float, box1[:4])
    x1_2, y1_2, x2_2, y2_2 = map(float, box2[:4])
    
    if x2_1 <= x1_1 or y2_1 <= y1_1 or x2_2 <= x1_2 or y2_2 <= y1_2:
        return 0.0
    
    # Calculate area of each box
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    if area1 <= 0 or area2 <= 0:
        return 0.0
    
    # Calculate intersection area
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0  # No intersection
    
    intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate IoU
    union_area = area1 + area2 - intersection_area
    
    if union_area <= 0:
        return 0.0
    
    iou = intersection_area / union_area
    return iou

def load_configuration(config_file: str) -> Dict:
    """
    Load configuration from JSON file.
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    default_config = {
        "detection": {
            "confidence_threshold": 0.5,
            "enable_ocr": True,
            "enable_tracking": True,
            "model_path": None
        },
        "violations": {
            "red_light_grace_period": 2.0,
            "stop_sign_duration": 2.0,
            "speed_tolerance": 5
        },
        "display": {
            "max_display_width": 800,
            "show_confidence": True,
            "show_labels": True,
            "show_license_plates": True
        },
        "performance": {
            "max_history_frames": 1000,
            "cleanup_interval": 3600
        }
    }
    
    if not os.path.exists(config_file):
        return default_config
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
            
        # Merge with defaults
        for section in default_config:
            if section in config:
                default_config[section].update(config[section])
            
        return default_config
    except Exception as e:
        print(f"Error loading config: {e}")
        return default_config

def save_configuration(config: Dict, config_file: str) -> bool:
    """
    Save configuration to JSON file.
    
    Args:
        config: Configuration dictionary
        config_file: Path to save configuration file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving config: {e}")
        return False

def format_timestamp(timestamp: float) -> str:
    """
    Format timestamp as readable string.
    
    Args:
        timestamp: Unix timestamp
        
    Returns:
        Formatted timestamp string
    """
    dt = datetime.fromtimestamp(timestamp)
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def format_duration(seconds: float) -> str:
    """
    Format duration in seconds as readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def create_export_csv(detections: List[Dict], filename: str) -> bool:
    """
    Export detections to CSV file.
    
    Args:
        detections: List of detection dictionaries
        filename: Output CSV filename
        
    Returns:
        True if successful, False otherwise
    """
    try:
        import pandas as pd
        
        # Create DataFrame from detections
        rows = []
        for det in detections:
            row = {
                'timestamp': det.get('timestamp', 0),
                'class': det.get('class_name', 'unknown'),
                'confidence': det.get('confidence', 0),
                'x1': det.get('bbox', [0, 0, 0, 0])[0],
                'y1': det.get('bbox', [0, 0, 0, 0])[1],
                'x2': det.get('bbox', [0, 0, 0, 0])[2],
                'y2': det.get('bbox', [0, 0, 0, 0])[3]
            }
            rows.append(row)
            
        df = pd.DataFrame(rows)
        
        # Save to CSV
        df.to_csv(filename, index=False)
        return True
    except Exception as e:
        print(f"Error exporting to CSV: {e}")
        return False

def create_export_json(data: Dict, filename: str) -> bool:
    """
    Export data to JSON file.
    
    Args:
        data: Data to export
        filename: Output JSON filename
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        print(f"Error exporting to JSON: {e}")
        return False

def create_unique_filename(prefix: str, ext: str) -> str:
    """
    Create unique filename with timestamp.
    
    Args:
        prefix: Filename prefix
        ext: File extension
        
    Returns:
        Unique filename
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{ext}"

def save_snapshot(frame: np.ndarray, filename: str = None) -> str:
    """
    Save video frame as image file.
    
    Args:
        frame: Video frame
        filename: Output filename (optional)
        
    Returns:
        Path to saved image
    """
    if filename is None:
        filename = create_unique_filename("snapshot", "jpg")
        
    try:
        cv2.imwrite(filename, frame)
        return filename
    except Exception as e:
        print(f"Error saving snapshot: {e}")
        return None

def get_video_properties(source):
    """
    Get video file properties.
    
    Args:
        source: Video source (file path or device number)
        
    Returns:
        Dictionary of video properties
    """
    try:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            return {}
            
        props = {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        }
        
        cap.release()
        return props
    except Exception as e:
        print(f"Error getting video properties: {e}")
        return {}
