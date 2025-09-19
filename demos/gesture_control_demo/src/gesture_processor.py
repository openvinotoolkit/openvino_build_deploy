import cv2
import numpy as np
import time
import math
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from config_manager import config_manager
from event_system import event_bus, GestureEvent
from openvino_models import model_manager
from volume_controller import get_volume_controller


from hand_landmark import *



def calculate_angle(p1, p2, p3):
    """Calculate angle at point p2 formed by p1-p2-p3"""
    v1 = [p1[0] - p2[0], p1[1] - p2[1]]
    v2 = [p3[0] - p2[0], p3[1] - p2[1]]
    
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
    
    if mag1 == 0 or mag2 == 0:
        return 180
    
    cos_angle = dot_product / (mag1 * mag2)
    cos_angle = max(-1, min(1, cos_angle))
    angle = math.degrees(math.acos(cos_angle))
    
    return angle

def detect_index_finger_bend(landmarks):
    """Detect if index finger is bent using angle calculation"""
    try:
        mcp = landmarks[5]  
        pip = landmarks[6]  
        dip = landmarks[8]  
        
        angle = calculate_angle(mcp, pip, dip)
        return angle
        
    except Exception as e:
        print(f"Error in finger bend detection: {e}")
        return 180

def detect_middle_finger_bend(landmarks):
    """Detect if middle finger is bent using angle calculation"""
    try:
        mcp = landmarks[9]   
        pip = landmarks[10]  
        dip = landmarks[12]  
        
        angle = calculate_angle(mcp, pip, dip)
        return angle
        
    except Exception as e:
        print(f"Error in middle finger bend detection: {e}")
        return 180

def detect_finger_perpendicularity(landmarks):
    """Detect angle between index and middle finger directions"""
    try:
        
        index_mcp = landmarks[5]   
        index_pip = landmarks[6]   
        
        
        middle_mcp = landmarks[9]  
        middle_pip = landmarks[10] 
        
        
        index_vector = [index_pip[0] - index_mcp[0], index_pip[1] - index_mcp[1]]
        middle_vector = [middle_pip[0] - middle_mcp[0], middle_pip[1] - middle_mcp[1]]
        
        
        dot_product = index_vector[0] * middle_vector[0] + index_vector[1] * middle_vector[1]
        
        
        index_mag = math.sqrt(index_vector[0]**2 + index_vector[1]**2)
        middle_mag = math.sqrt(middle_vector[0]**2 + middle_vector[1]**2)
        
        if index_mag == 0 or middle_mag == 0:
            return 90  
        
        
        cos_angle = dot_product / (index_mag * middle_mag)
        cos_angle = max(-1, min(1, cos_angle))  
        angle_between = math.degrees(math.acos(abs(cos_angle)))
        
        return angle_between
        
    except Exception as e:
        print(f"Error in perpendicularity detection: {e}")
        return 90  

def are_fingers_parallel(landmarks, parallel_threshold=30):
    """Check if fingers are parallel (can do both-finger gestures)"""
    angle_between = detect_finger_perpendicularity(landmarks)
    return angle_between < parallel_threshold

def are_fingers_perpendicular(landmarks, perpendicular_threshold=20):
    """Check if fingers are perpendicular (only index gesture)"""
    angle_between = detect_finger_perpendicularity(landmarks)
    return abs(angle_between - 90) < perpendicular_threshold

def detect_gesture_type(landmarks, bend_threshold):
    """Detect gesture type based on finger angles and relationships"""
    try:
        
        index_angle = detect_index_finger_bend(landmarks)
        middle_angle = detect_middle_finger_bend(landmarks)
        
        
        fingers_parallel = are_fingers_parallel(landmarks)
        fingers_perpendicular = are_fingers_perpendicular(landmarks)
        finger_angle_between = detect_finger_perpendicularity(landmarks)
        
        
        def get_finger_state(angle):
            if angle < 60:
                return "RELAXED"      
            elif angle < 130:
                return "BENT"         
            else:
                return "EXTENDED"     
        
        index_state = get_finger_state(index_angle)
        middle_state = get_finger_state(middle_angle)
        
        
        gesture_type = "none"
        
        if index_state == "BENT":
            if fingers_parallel and middle_state == "BENT":
                gesture_type = "index_middle_both"
            elif fingers_perpendicular:
                gesture_type = "index_only"
            elif not fingers_parallel and middle_state in ["EXTENDED", "RELAXED"]:
                gesture_type = "index_only"
        
        return {
            'index_angle': index_angle,
            'middle_angle': middle_angle,
            'index_state': index_state,
            'middle_state': middle_state,
            'gesture_type': gesture_type,
            'finger_angle_between': finger_angle_between,
            'fingers_parallel': fingers_parallel,
            'fingers_perpendicular': fingers_perpendicular
        }
        
    except Exception as e:
        print(f"Error in gesture detection: {e}")
        return {
            'index_angle': 180,
            'middle_angle': 180,
            'index_state': "EXTENDED",
            'middle_state': "EXTENDED", 
            'gesture_type': "none",
            'finger_angle_between': 90,
            'fingers_parallel': False,
            'fingers_perpendicular': True
        }

def process_finger_detection(region, params):
    """Process finger detection with gesture mapping system"""
    if not (hasattr(region, 'landmarks') and params['enable_finger_detection']):
        return
    
    try:
        
        gesture_data = detect_gesture_type(region.landmarks, params['bend_angle_threshold'])
        hand_type = "right" if region.handedness > 0.5 else "left"
        
        
        region.index_angle = gesture_data['index_angle']
        region.middle_angle = gesture_data['middle_angle']
        region.index_state = gesture_data['index_state']
        region.middle_state = gesture_data['middle_state']
        region.gesture_type = gesture_data['gesture_type']
        region.hand_type = hand_type
        
        
        region.finger_angle_between = gesture_data['finger_angle_between']
        region.fingers_parallel = gesture_data['fingers_parallel']
        region.fingers_perpendicular = gesture_data['fingers_perpendicular']

        
        
        
    except Exception as e:
        print(f"Error processing finger detection: {e}")



def process_application_modes(region, params, app_modes):
    """Simplified application modes processing for Phase 2"""
    if not hasattr(region, 'landmarks'):
        return
    
    current_mode = app_modes['current_mode']
    if current_mode == 'disabled' or current_mode not in app_modes:
        return
    
    
    
    region.processed_for_app_mode = True

