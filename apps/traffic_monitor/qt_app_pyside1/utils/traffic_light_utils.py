"""
Traffic light color detection utilities
"""

import cv2
import numpy as np
import os
import time
from typing import Dict, List, Tuple, Optional
import logging
from collections import Counter, deque

# HSV thresholds as config constants
HSV_THRESHOLDS = {
    "red": [
        (np.array([0, 40, 40]), np.array([15, 255, 255])),   # Lower red range (more permissive)
        (np.array([160, 40, 40]), np.array([180, 255, 255])) # Upper red range (more permissive)
    ],
    "yellow": [
        (np.array([15, 50, 50]), np.array([40, 255, 255]))   # Wider yellow range
    ],
    "green": [
        (np.array([35, 25, 25]), np.array([95, 255, 255]))   # More permissive green range
    ]
}

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# History buffer for smoothing (can be used in controller)
COLOR_HISTORY = []
HISTORY_SIZE = 5

# Global color history for temporal smoothing
COLOR_HISTORY_DICT = {}
HISTORY_LEN = 7  # Number of frames to smooth over

def get_light_id(bbox):
    # Use bbox center as a simple unique key (rounded to nearest 10 pixels)
    x1, y1, x2, y2 = bbox
    cx = int((x1 + x2) / 2 // 10 * 10)
    cy = int((y1 + y2) / 2 // 10 * 10)
    return (cx, cy)

def detect_dominant_color(hsv_img):
    """
    Detect the dominant color in a traffic light based on simple HSV thresholding.
    Useful as a fallback for small traffic lights where circle detection may fail.
    """
    h, w = hsv_img.shape[:2]
    
    # Create masks for each color
    color_masks = {}
    color_areas = {}
    
    # Create a visualization image for debugging
    debug_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    
    for color, thresholds in HSV_THRESHOLDS.items():
        mask = np.zeros((h, w), dtype=np.uint8)
        
        for lower, upper in thresholds:
            color_mask = cv2.inRange(hsv_img, lower, upper)
            mask = cv2.bitwise_or(mask, color_mask)
        
        # Calculate the percentage of pixels matching each color
        color_areas[color] = np.count_nonzero(mask) / (h * w) if h * w > 0 else 0
        
        # Create a colored mask for visualization
        color_viz = np.zeros((h, w, 3), dtype=np.uint8)
        if color == "red":
            color_viz[:, :] = [0, 0, 255]  # BGR red
        elif color == "yellow":
            color_viz[:, :] = [0, 255, 255]  # BGR yellow
        elif color == "green":
            color_viz[:, :] = [0, 255, 0]  # BGR green
            
        # Apply the mask to the color
        color_viz = cv2.bitwise_and(color_viz, color_viz, mask=mask)
        
        # Blend with debug image for visualization
        alpha = 0.5
        mask_expanded = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
        debug_img = debug_img * (1 - alpha * mask_expanded) + color_viz * (alpha * mask_expanded)
    
    # Show debug visualization
    cv2.imshow(f"Color Masks", debug_img.astype(np.uint8))
    cv2.waitKey(1)
    
    # Debug output
    print(f"Color areas: Red={color_areas.get('red', 0):.3f}, Yellow={color_areas.get('yellow', 0):.3f}, Green={color_areas.get('green', 0):.3f}")
    
    # If any color exceeds the threshold, consider it detected
    best_color = max(color_areas.items(), key=lambda x: x[1]) if color_areas else ("unknown", 0)
    
    # Only return a color if it has a minimum area percentage
    if best_color[1] > 0.02:  # at least 2% of pixels match the color (reduced from 3%)
        return best_color[0], best_color[1]
    
    return "unknown", 0

def detect_traffic_light_color(frame: np.ndarray, bbox: list) -> dict:
    from collections import Counter
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = frame.shape[:2]
    x1 = max(0, min(x1, w-1))
    y1 = max(0, min(y1, h-1))
    x2 = max(0, min(x2, w-1))
    y2 = max(0, min(y2, h-1))
    if x2 <= x1 or y2 <= y1:
        return {"color": "unknown", "confidence": 0.0}
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return {"color": "unknown", "confidence": 0.0}
    roi = cv2.resize(roi, (32, 64))
    roi = cv2.GaussianBlur(roi, (5, 5), 0)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    hsv[..., 2] = clahe.apply(hsv[..., 2])
    red_lower1 = np.array([0, 120, 120])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([160, 120, 120])
    red_upper2 = np.array([180, 255, 255])
    yellow_lower = np.array([18, 110, 110])
    yellow_upper = np.array([38, 255, 255])
    green_lower = np.array([42, 90, 90])
    green_upper = np.array([90, 255, 255])
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    red_count = cv2.countNonZero(red_mask)
    yellow_count = cv2.countNonZero(yellow_mask)
    green_count = cv2.countNonZero(green_mask)
    total_pixels = hsv.shape[0] * hsv.shape[1]
    red_ratio = red_count / total_pixels
    yellow_ratio = yellow_count / total_pixels
    green_ratio = green_count / total_pixels
    color_counts = {'red': red_count, 'yellow': yellow_count, 'green': green_count}
    color_ratios = {'red': red_ratio, 'yellow': yellow_ratio, 'green': green_ratio}
    print(f"[DEBUG] ratios: red={red_ratio:.3f}, yellow={yellow_ratio:.3f}, green={green_ratio:.3f}")

    # --- Improved Decision Logic ---
    min_area = 0.025  # 2.5% of ROI must be the color
    dominance_margin = 1.5  # Must be 50% more pixels than next best
    detected_color = "unknown"
    confidence = 0.0
    if green_ratio > min_area:
        if red_ratio < 2 * green_ratio:
            detected_color = "green"
            confidence = float(green_ratio)
    if detected_color == "unknown" and yellow_ratio > min_area:
        if red_ratio < 1.5 * yellow_ratio:
            detected_color = "yellow"
            confidence = float(yellow_ratio)
    if detected_color == "unknown" and red_ratio > min_area and red_ratio > green_ratio and red_ratio > yellow_ratio:
        detected_color = "red"
        confidence = float(red_ratio)
    # Fallbacks (vertical thirds, hough, etc.)
    if detected_color == "unknown":
        # Fallback: vertical thirds (classic traffic light layout)
        h_roi, w_roi = roi.shape[:2]
        top_roi = roi[0:h_roi//3, :]
        middle_roi = roi[h_roi//3:2*h_roi//3, :]
        bottom_roi = roi[2*h_roi//3:, :]
        try:
            top_hsv = cv2.cvtColor(top_roi, cv2.COLOR_BGR2HSV)
            middle_hsv = cv2.cvtColor(middle_roi, cv2.COLOR_BGR2HSV)
            bottom_hsv = cv2.cvtColor(bottom_roi, cv2.COLOR_BGR2HSV)
            top_avg = np.mean(top_hsv, axis=(0,1))
            middle_avg = np.mean(middle_hsv, axis=(0,1))
            bottom_avg = np.mean(bottom_hsv, axis=(0,1))
            if (top_avg[0] <= 15 or top_avg[0] >= 160) and top_avg[1] > 40:
                detected_color = "red"
                confidence = 0.7
            elif 18 <= middle_avg[0] <= 38 and middle_avg[1] > 40:
                detected_color = "yellow"
                confidence = 0.7
            elif 42 <= bottom_avg[0] <= 90 and bottom_avg[1] > 35:
                detected_color = "green"
                confidence = 0.7
        except Exception as e:
            print(f"[DEBUG] thirds fallback error: {e}")
    # If still unknown, try Hough Circle fallback
    if detected_color == "unknown":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=5,
            param1=50, param2=10, minRadius=3, maxRadius=15)
        detected_colors = []
        if circles is not None:
            for circle in circles[0, :]:
                cx, cy, r = map(int, circle)
                if 0 <= cy < hsv.shape[0] and 0 <= cx < hsv.shape[1]:
                    h, s, v = hsv[cy, cx]
                    if (h <= 10 or h >= 160):
                        detected_colors.append("red")
                    elif 18 <= h <= 38:
                        detected_colors.append("yellow")
                    elif 42 <= h <= 90:
                        detected_colors.append("green")
        if detected_colors:
            counter = Counter(detected_colors)
            detected_color, count = counter.most_common(1)[0]
            confidence = count / len(detected_colors)

    # --- Temporal Consistency Filtering ---
    light_id = get_light_id(bbox)
    if light_id not in COLOR_HISTORY_DICT:
        COLOR_HISTORY_DICT[light_id] = deque(maxlen=HISTORY_LEN)
    if detected_color != "unknown":
        COLOR_HISTORY_DICT[light_id].append(detected_color)
    # Soft voting
    if len(COLOR_HISTORY_DICT[light_id]) > 0:
        most_common = Counter(COLOR_HISTORY_DICT[light_id]).most_common(1)[0][0]
        # Optionally, only output if the most common color is at least 2/3 of the buffer
        count = Counter(COLOR_HISTORY_DICT[light_id])[most_common]
        if count >= (len(COLOR_HISTORY_DICT[light_id]) // 2 + 1):
            return {"color": most_common, "confidence": confidence}
    # If not enough history, return current detected color
    return {"color": detected_color, "confidence": confidence}

def detect_traffic_light_color_old(frame: np.ndarray, bbox: list) -> dict:
    print("[DEBUG] detect_traffic_light_color called")
    """
    Hybrid robust traffic light color detection:
    1. Preprocess ROI (resize, blur, CLAHE, HSV)
    2. Pixel-ratio HSV masking and thresholding (fast, robust)
    3. If ambiguous, fallback to Hough Circle detection
    Returns: {"color": str, "confidence": float}
    """
    import cv2
    import numpy as np
    from collections import Counter

    # --- Preprocessing ---
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = frame.shape[:2]
    x1 = max(0, min(x1, w-1))
    y1 = max(0, min(y1, h-1))
    x2 = max(0, min(x2, w-1))
    y2 = max(0, min(y2, h-1))
    if x2 <= x1 or y2 <= y1:
        return {"color": "unknown", "confidence": 0.0}
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return {"color": "unknown", "confidence": 0.0}
    roi = cv2.resize(roi, (32, 64))
    roi = cv2.GaussianBlur(roi, (5, 5), 0)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # CLAHE on V channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    hsv[..., 2] = clahe.apply(hsv[..., 2])

    # --- HSV Masking ---
    # Refined thresholds
    red_lower1 = np.array([0, 110, 110])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([160, 110, 110])
    red_upper2 = np.array([180, 255, 255])
    yellow_lower = np.array([18, 110, 110])
    yellow_upper = np.array([38, 255, 255])
    green_lower = np.array([42, 80, 80])
    green_upper = np.array([90, 255, 255])
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    green_mask = cv2.inRange(hsv, green_lower, green_upper)

    # --- Pixel Counting ---
    red_count = cv2.countNonZero(red_mask)
    yellow_count = cv2.countNonZero(yellow_mask)
    green_count = cv2.countNonZero(green_mask)
    total_pixels = hsv.shape[0] * hsv.shape[1]
    red_ratio = red_count / total_pixels
    yellow_ratio = yellow_count / total_pixels
    green_ratio = green_count / total_pixels
    # Stricter threshold for red, slightly relaxed for green/yellow
    thresholds = {'red': 0.04, 'yellow': 0.02, 'green': 0.02}  # 4% for red, 2% for others

    color = "unknown"
    confidence = 0.0
    # Prefer green/yellow if their ratio is close to red (within 80%)
    if green_ratio > thresholds['green'] and green_ratio >= 0.8 * red_ratio:
        color = "green"
        confidence = green_ratio
    elif yellow_ratio > thresholds['yellow'] and yellow_ratio >= 0.8 * red_ratio:
        color = "yellow"
        confidence = yellow_ratio
    elif red_ratio > thresholds['red']:
        color = "red"
        confidence = red_ratio

    # --- If strong color found, return ---
    if color != "unknown" and confidence > 0.01:
        print(f"[DEBUG] detect_traffic_light_color result: {color}, confidence: {confidence:.2f}")
        return {"color": color, "confidence": float(confidence)}

    # --- Fallback: Hough Circle Detection ---
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=5,
        param1=50, param2=10, minRadius=3, maxRadius=15)
    detected_colors = []
    if circles is not None:
        for circle in circles[0, :]:
            cx, cy, r = map(int, circle)
            if 0 <= cy < hsv.shape[0] and 0 <= cx < hsv.shape[1]:
                h, s, v = hsv[cy, cx]
                if (h <= 10 or h >= 160):
                    detected_colors.append("red")
                elif 18 <= h <= 38:
                    detected_colors.append("yellow")
                elif 42 <= h <= 90:
                    detected_colors.append("green")
    if detected_colors:
        counter = Counter(detected_colors)
        final_color, count = counter.most_common(1)[0]
        confidence = count / len(detected_colors)
        print(f"[DEBUG] detect_traffic_light_color (hough): {final_color}, confidence: {confidence:.2f}")
        return {"color": final_color, "confidence": float(confidence)}

    # --- If still unknown, return unknown ---
    print("[DEBUG] detect_traffic_light_color result: unknown")
    return {"color": "unknown", "confidence": 0.0}

def draw_traffic_light_status(frame: np.ndarray, bbox: List[int], color_info) -> np.ndarray:
    """
    Draw traffic light status on the frame with confidence score.
    
    Args:
        frame: Image to draw on
        bbox: Bounding box coordinates [x1, y1, x2, y2]
        color_info: Either a string ("red", "yellow", "green", "unknown") or 
                   a dict {"color": str, "confidence": float}
        
    Returns:
        Frame with color status drawn
    """
    try:
        # Handle both string and dictionary formats
        if isinstance(color_info, dict):
            color = color_info.get("color", "unknown")
            confidence = color_info.get("confidence", 0.0)
            confidence_text = f"{confidence:.2f}"
        else:
            color = color_info
            confidence_text = ""
        
        # Debug message
        print(f"📝 Drawing traffic light status: {color} at bbox {bbox}")
        
        # Parse and validate bbox
        x1, y1, x2, y2 = [int(c) for c in bbox]
        
        # Define color for drawing
        status_colors = {
            "red": (0, 0, 255),      # BGR: Red
            "yellow": (0, 255, 255), # BGR: Yellow
            "green": (0, 255, 0),    # BGR: Green
            "unknown": (255, 255, 255)  # BGR: White
        }
        
        draw_color = status_colors.get(color, (255, 255, 255))
        
        # Draw rectangle with color-specific border (thicker for visibility)
        cv2.rectangle(frame, (x1, y1), (x2, y2), draw_color, 3)
        
        # Add text label with the color and confidence if available
        if confidence_text:
            label = f"Traffic Light: {color.upper()} ({confidence_text})"
        else:
            label = f"Traffic Light: {color.upper()}"
            
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        
        # Draw background rectangle for text
        cv2.rectangle(
            frame, 
            (x1, y1 - text_size[1] - 10), 
            (x1 + text_size[0], y1), 
            draw_color, 
            -1
        )
        
        # Draw text
        cv2.putText(
            frame, 
            label, 
            (x1, y1 - 5), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 0, 0),  # Black text
            2
        )
        
        # Also draw a large indicator at the top of the frame for high visibility
        indicator_size = 30
        margin = 10
        
        # Draw colored circle indicator at top-right
        cv2.circle(
            frame, 
            (frame.shape[1] - margin - indicator_size, margin + indicator_size), 
            indicator_size, 
            draw_color, 
            -1
        )
        
        # Remove the extra white rectangle/text from the UI overlay
        # In draw_traffic_light_status, the white rectangle and text are likely drawn by this block:
        # cv2.circle(
        #     frame, 
        #     (frame.shape[1] - margin - indicator_size, margin + indicator_size), 
        #     indicator_size, 
        #     draw_color, 
        #     -1
        # )
        # cv2.putText(
        #     frame, 
        #     color.upper(), 
        #     (frame.shape[1] - margin - indicator_size*2 - 80, margin + indicator_size + 10), 
        #     cv2.FONT_HERSHEY_SIMPLEX, 
        #     1.0, 
        #     draw_color, 
        #     3
        # )
        # To remove the white overlay, comment out or remove the cv2.putText line for the color text at the top.
        # Only keep the circle indicator if you want, or remove both if you want no indicator at the top.
        # Let's remove the cv2.putText for color at the top.
        
        return frame
        
    except Exception as e:
        print(f"❌ Error drawing traffic light status: {e}")
        import traceback
        traceback.print_exc()
        return frame

def ensure_traffic_light_color(frame, bbox):
    print("[DEBUG] ensure_traffic_light_color called")
    """
    Emergency function to always return a traffic light color even with poor quality crops.
    This function is less strict and will fall back to enforced color detection.
    """
    try:
        # First try the regular detection
        result = detect_traffic_light_color(frame, bbox)
        if isinstance(result, dict) and result.get('color', 'unknown') != 'unknown':
            print(f"[DEBUG] ensure_traffic_light_color result (from detect): {result}")
            return result
        # If we got unknown, extract traffic light region again
        x1, y1, x2, y2 = [int(c) for c in bbox]
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(0, min(x2, w-1))
        y2 = max(0, min(y2, h-1))
        if x2 <= x1 or y2 <= y1:
            print("❌ Invalid bbox for traffic light")
            return {"color": "unknown", "confidence": 0.0}
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            print("❌ Empty ROI for traffic light")
            return {"color": "unknown", "confidence": 0.0}
        # Try analyzing by vertical thirds (typical traffic light pattern)
        h_roi, w_roi = roi.shape[:2]
        top_roi = roi[0:h_roi//3, :]
        middle_roi = roi[h_roi//3:2*h_roi//3, :]
        bottom_roi = roi[2*h_roi//3:, :]
        try:
            top_hsv = cv2.cvtColor(top_roi, cv2.COLOR_BGR2HSV)
            middle_hsv = cv2.cvtColor(middle_roi, cv2.COLOR_BGR2HSV)
            bottom_hsv = cv2.cvtColor(bottom_roi, cv2.COLOR_BGR2HSV)
            top_avg = np.mean(top_hsv, axis=(0,1))
            middle_avg = np.mean(middle_hsv, axis=(0,1))
            bottom_avg = np.mean(bottom_hsv, axis=(0,1))
            print(f"Traffic light regions - Top HSV: {top_avg}, Middle HSV: {middle_avg}, Bottom HSV: {bottom_avg}")
            # Check for red in top
            if (top_avg[0] <= 15 or top_avg[0] >= 160) and top_avg[1] > 40:
                return {"color": "red", "confidence": 0.7}
            # Check for yellow in middle
            if 18 <= middle_avg[0] <= 38 and middle_avg[1] > 40:
                return {"color": "yellow", "confidence": 0.7}
            # Check for green in bottom
            if 42 <= bottom_avg[0] <= 90 and bottom_avg[1] > 35:
                return {"color": "green", "confidence": 0.7}
        except:
            pass
        # If we still haven't found a color, look at overall color distribution
        try:
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            very_permissive_red1 = cv2.inRange(hsv_roi, np.array([0, 30, 30]), np.array([20, 255, 255]))
            very_permissive_red2 = cv2.inRange(hsv_roi, np.array([155, 30, 30]), np.array([180, 255, 255]))
            very_permissive_red = cv2.bitwise_or(very_permissive_red1, very_permissive_red2)
            very_permissive_yellow = cv2.inRange(hsv_roi, np.array([10, 30, 30]), np.array([45, 255, 255]))
            very_permissive_green = cv2.inRange(hsv_roi, np.array([30, 20, 20]), np.array([100, 255, 255]))
            red_count = cv2.countNonZero(very_permissive_red)
            yellow_count = cv2.countNonZero(very_permissive_yellow)
            green_count = cv2.countNonZero(very_permissive_green)
            total_pixels = hsv_roi.shape[0] * hsv_roi.shape[1]
            print(f"Very permissive detection: Red={red_count/total_pixels:.3f}, Yellow={yellow_count/total_pixels:.3f}, Green={green_count/total_pixels:.3f}")
            max_count = max(red_count, yellow_count, green_count)
            if max_count > 0:
                # Prefer green/yellow if close to red
                if green_count == max_count and green_count >= 0.9 * red_count:
                    return {"color": "green", "confidence": 0.5 * green_count/total_pixels}
                elif yellow_count == max_count and yellow_count >= 0.9 * red_count:
                    return {"color": "yellow", "confidence": 0.5 * yellow_count/total_pixels}
                elif red_count == max_count:
                    return {"color": "red", "confidence": 0.5 * red_count/total_pixels}
        except Exception as e:
            print(f"❌ Error in permissive analysis: {e}")
        # Last resort - analyze mean color
        mean_color = np.mean(roi, axis=(0,1))
        b, g, r = mean_color
        if r > g and r > b and r > 60:
            return {"color": "red", "confidence": 0.4}
        elif g > r and g > b and g > 60:
            return {"color": "green", "confidence": 0.4}
        elif r > 70 and g > 70 and r/g > 0.7 and r/g < 1.3:
            return {"color": "yellow", "confidence": 0.4}
        print("[DEBUG] ensure_traffic_light_color fallback to unknown")
        return {"color": "unknown", "confidence": 0.0}
    except Exception as e:
        print(f"❌ Error in ensure_traffic_light_color: {e}")
        import traceback
        traceback.print_exc()
        return {"color": "unknown", "confidence": 0.0}