def group_score(group, w, h):
    if len(group) < 2:  # Reduced minimum requirement
        return 0
    heights = [r[3] for r in group]
    x_centers = [r[0] + r[2]//2 for r in group]
    y_centers = [r[1] + r[3]//2 for r in group]
    # Stripe count (normalized) - more permissive
    count_score = min(len(group) / 4, 1.0)  # Reduced from 6 to 4
    # Height consistency
    if len(heights) > 1:
        height_score = 1.0 - min(np.std(heights) / (np.mean(heights) + 1e-6), 1.0)
    else:
        height_score = 0.5
    # Horizontal alignment (zebra stripes should be roughly aligned)
    if len(y_centers) > 1:
        y_score = 1.0 - min(np.std(y_centers) / (h * 0.1), 1.0)
    else:
        y_score = 0.5
    # Regular spacing between stripes
    if len(group) >= 3:
        x_sorted = sorted([r[0] for r in group])
        gaps = [x_sorted[i+1] - x_sorted[i] for i in range(len(x_sorted)-1)]
        gap_consistency = 1.0 - min(np.std(gaps) / (np.mean(gaps) + 1e-6), 1.0)
    else:
        gap_consistency = 0.3
    # Area coverage (zebra crossing should cover reasonable area)
    total_area = sum(r[2] * r[3] for r in group)
    area_score = min(total_area / (w * h * 0.05), 1.0)  # At least 5% of frame
    # Final score (weighted sum)
    score = (0.3*count_score + 0.2*height_score + 0.2*y_score +
            0.15*gap_consistency + 0.15*area_score)
    return score
print("� [CROSSWALK_UTILS2] This is d:/Downloads/finale6/Khatam final/khatam/qt_app_pyside/utils/crosswalk_utils2.py LOADED")
import cv2
import numpy as np
from typing import Tuple, Optional

def detect_crosswalk_and_violation_line(frame: np.ndarray, traffic_light_position: Optional[Tuple[int, int]] = None, perspective_M: Optional[np.ndarray] = None):
    """
    Detects crosswalk (zebra crossing) or fallback stop line in a traffic scene using classical CV.
    Args:
        frame: BGR image frame from video feed
        traffic_light_position: Optional (x, y) of traffic light in frame
        perspective_M: Optional 3x3 homography matrix for bird's eye view normalization
    Returns:
        result_frame: frame with overlays (for visualization)
        crosswalk_bbox: (x, y, w, h) or None if fallback used
        violation_line_y: int (y position for violation check)
        debug_info: dict (for visualization/debugging)
    """
    # --- PROCESS CROSSWALK DETECTION REGARDLESS OF TRAFFIC LIGHT ---
    print(f"[CROSSWALK DEBUG] Starting crosswalk detection. Traffic light: {traffic_light_position}")
    if traffic_light_position is None:
        print("[CROSSWALK DEBUG] No traffic light detected, but proceeding with crosswalk detection")
    debug_info = {}
    orig_frame = frame.copy()
    h, w = frame.shape[:2]

    # 1. Perspective Normalization (Bird's Eye View)
    if perspective_M is not None:
        frame = cv2.warpPerspective(frame, perspective_M, (w, h))
        debug_info['perspective_warped'] = True
    else:
        debug_info['perspective_warped'] = False

    # 1. Enhanced White Color Filtering (more permissive for zebra stripes)
    mask_white = cv2.inRange(frame, (140, 140, 140), (255, 255, 255))
    debug_info['mask_white_ratio'] = np.sum(mask_white > 0) / (h * w)

    # 2. Grayscale for adaptive threshold
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Enhance contrast for night/low-light
    if np.mean(gray) < 80:
        gray = cv2.equalizeHist(gray)
        debug_info['hist_eq'] = True
    else:
        debug_info['hist_eq'] = False
    
    # 3. Adaptive threshold (more permissive)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 3)
    # Combine with color mask
    combined = cv2.bitwise_and(thresh, mask_white)
    
    # 4. Better morphology for zebra stripe detection
    # Horizontal kernel to connect zebra stripes
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    morph = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_h, iterations=1)
    
    # Vertical kernel to separate stripes
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel_v, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    zebra_rects = []
    
    # Focus on lower half of frame where crosswalks typically are
    roi_y_start = int(h * 0.4)  # Start from 40% down
    
    for cnt in contours:
        x, y, w, h_rect = cv2.boundingRect(cnt)
        
        # Skip if in upper part of frame
        if y < roi_y_start:
            continue
            
        aspect_ratio = w / max(h_rect, 1)
        area = w * h_rect
        
        # More permissive criteria for zebra stripe detection
        min_area = 300   # Smaller minimum area
        max_area = 0.3 * frame.shape[0] * frame.shape[1]  # Larger max area
        min_aspect = 2.0  # Lower aspect ratio requirement
        max_height = 40   # Allow taller stripes
        
        if (aspect_ratio > min_aspect and 
            min_area < area < max_area and 
            h_rect < max_height and
            w > 50):  # Minimum width for zebra stripe
            
            angle = 0  # For simplicity, assume horizontal stripes
            zebra_rects.append((x, y, w, h_rect, angle))
            
    print(f"[CROSSWALK DEBUG] Found {len(zebra_rects)} zebra stripe candidates")
    # --- Enhanced Grouping and Scoring for Crosswalk Detection ---
    
    # 4. More flexible grouping
    crosswalk_bbox = None
    violation_line_y = None
    
    if len(zebra_rects) >= 2:  # Reduced minimum requirement from 3 to 2
        # Sort by y-coordinate for grouping
        zebra_rects = sorted(zebra_rects, key=lambda r: r[1])
        
        # Group stripes that are horizontally aligned
        y_tolerance = int(h * 0.08)  # Increased tolerance to 8%
        groups = []
        
        if zebra_rects:
            group = [zebra_rects[0]]
            for rect in zebra_rects[1:]:
                # Check if this stripe is roughly at the same y-level as the group
                group_y_avg = sum(r[1] for r in group) / len(group)
                if abs(rect[1] - group_y_avg) < y_tolerance:
                    group.append(rect)
                else:
                    if len(group) >= 2:  # Reduced from 3 to 2
                        groups.append(group)
                    group = [rect]
            
            # Don't forget the last group
            if len(group) >= 2:
                groups.append(group)
        
        # Score all groups
        scored_groups = [(group_score(g, w, h), g) for g in groups]
        # More permissive threshold
        scored_groups = [(s, g) for s, g in scored_groups if s > 0.05]  # Reduced from 0.1
        
        print(f"[CROSSWALK DEBUG] Found {len(groups)} potential crosswalk groups")
        print(f"[CROSSWALK DEBUG] scored_groups: {[round(s, 3) for s, _ in scored_groups]}")
        if scored_groups:
            scored_groups.sort(reverse=True, key=lambda x: x[0])
            best_score, best_group = scored_groups[0]
            print(f"[CROSSWALK DEBUG] Best crosswalk group score: {best_score:.3f}")
            print(f"[CROSSWALK DEBUG] Best group has {len(best_group)} stripes")
            
            # Calculate crosswalk bounding box
            xs = [r[0] for r in best_group] + [r[0] + r[2] for r in best_group]
            ys = [r[1] for r in best_group] + [r[1] + r[3] for r in best_group]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            crosswalk_bbox = (x1, y1, x2 - x1, y2 - y1)
            
            # Place violation line just before the crosswalk
            violation_line_y = y1 - 15  # 15 pixels before crosswalk starts
            
            debug_info['crosswalk_group'] = best_group
            debug_info['crosswalk_score'] = best_score
            debug_info['crosswalk_bbox'] = crosswalk_bbox
            print(f"[CROSSWALK DEBUG] CROSSWALK DETECTED at bbox: {crosswalk_bbox}")
            print(f"[CROSSWALK DEBUG] Violation line at y={violation_line_y}")
            
        else:
            print("[CROSSWALK DEBUG] No valid crosswalk groups found")
    # --- Fallback: Improved Stop line detection ---
    if crosswalk_bbox is None:
        # Enhanced edge detection for stop lines
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Focus on lower half of frame where stop lines typically are
        roi_height = int(h * 0.6)  # Lower 60% of frame
        roi_y = h - roi_height
        roi_edges = edges[roi_y:h, :]
        
        # Detect horizontal lines (stop lines)
        lines = cv2.HoughLinesP(roi_edges, 1, np.pi / 180, 
                               threshold=50, minLineLength=100, maxLineGap=30)
        stop_lines = []
        
        if lines is not None:
            for l in lines:
                x1, y1, x2, y2 = l[0]
                # Convert back to full frame coordinates
                y1 += roi_y
                y2 += roi_y
                
                # Check if line is horizontal (stop line characteristic)
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                line_length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                
                if (abs(angle) < 15 or abs(angle) > 165) and line_length > 80:
                    stop_lines.append((x1, y1, x2, y2))
        
        debug_info['stop_lines'] = stop_lines
        print(f"[CROSSWALK DEBUG] stop_lines: {len(stop_lines)} found")
        
        if stop_lines:
            # Choose the best stop line based on traffic light position or bottom-most line
            if traffic_light_position:
                tx, ty = traffic_light_position
                # Find line closest to traffic light but below it
                valid_lines = [l for l in stop_lines if ((l[1]+l[3])//2) > ty + 50]
                if valid_lines:
                    best_line = min(valid_lines, key=lambda l: abs(((l[1]+l[3])//2) - (ty + 100)))
                else:
                    best_line = min(stop_lines, key=lambda l: abs(((l[1]+l[3])//2) - ty))
            else:
                # Use the bottom-most horizontal line as stop line
                best_line = max(stop_lines, key=lambda l: max(l[1], l[3]))
            
            x1, y1, x2, y2 = best_line
            crosswalk_bbox = None
            # Place violation line slightly above the detected stop line
            violation_line_y = min(y1, y2) - 10
            debug_info['stop_line'] = best_line
            print(f"[CROSSWALK DEBUG] using stop_line: {best_line}")
            print(f"[CROSSWALK DEBUG] violation line placed at y={violation_line_y}")
    # Draw violation line on the frame for visualization
    result_frame = orig_frame.copy()
    if violation_line_y is not None:
        print(f"[CROSSWALK DEBUG] Drawing VIOLATION LINE at y={violation_line_y}")
        result_frame = draw_violation_line(result_frame, violation_line_y, 
                                         color=(0, 0, 255), thickness=8, 
                                         style='solid', label='VIOLATION LINE')

    return result_frame, crosswalk_bbox, violation_line_y, debug_info

def draw_violation_line(frame: np.ndarray, y: int, color=(0, 0, 255), thickness=8, style='solid', label='Violation Line'):
    """
    Draws a thick, optionally dashed, labeled violation line at the given y-coordinate.
    Args:
        frame: BGR image
        y: y-coordinate for the line
        color: BGR color tuple
        thickness: line thickness
        style: 'solid' or 'dashed'
        label: Optional label to draw above the line
    Returns:
        frame with line overlay
    """
    import cv2
    h, w = frame.shape[:2]
    x1, x2 = 0, w
    overlay = frame.copy()
    if style == 'dashed':
        dash_len = 30
        gap = 20
        for x in range(x1, x2, dash_len + gap):
            x_end = min(x + dash_len, x2)
            cv2.line(overlay, (x, y), (x_end, y), color, thickness, lineType=cv2.LINE_AA)
    else:
        cv2.line(overlay, (x1, y), (x2, y), color, thickness, lineType=cv2.LINE_AA)
    # Blend for semi-transparency
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    # Draw label
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size, _ = cv2.getTextSize(label, font, 0.8, 2)
        text_x = max(10, (w - text_size[0]) // 2)
        text_y = max(0, y - 12)
        cv2.rectangle(frame, (text_x - 5, text_y - text_size[1] - 5), (text_x + text_size[0] + 5, text_y + 5), (0,0,0), -1)
        cv2.putText(frame, label, (text_x, text_y), font, 0.8, color, 2, cv2.LINE_AA)
    return frame

def get_violation_line_y(frame, traffic_light_bbox=None, crosswalk_bbox=None):
    """
    Returns the y-coordinate of the violation line using the following priority:
    1. Crosswalk bbox (most accurate)
    2. Stop line detection via image processing (CV)
    3. Traffic light bbox heuristic
    4. Fallback (default)
    """
    height, width = frame.shape[:2]
    # 1. Crosswalk bbox
    if crosswalk_bbox is not None and len(crosswalk_bbox) == 4:
        return int(crosswalk_bbox[1]) - 15
    # 2. Stop line detection (CV)
    roi_height = int(height * 0.4)
    roi_y = height - roi_height
    roi = frame[roi_y:height, 0:width]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, -2
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stop_line_candidates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / max(h, 1)
        normalized_width = w / width
        if (aspect_ratio > 5 and normalized_width > 0.3 and h < 15 and y > roi_height * 0.5):
            abs_y = y + roi_y
            stop_line_candidates.append((abs_y, w))
    if stop_line_candidates:
        stop_line_candidates.sort(key=lambda x: x[1], reverse=True)
        return stop_line_candidates[0][0]
    # 3. Traffic light bbox heuristic
    if traffic_light_bbox is not None and len(traffic_light_bbox) == 4:
        traffic_light_bottom = traffic_light_bbox[3]
        traffic_light_height = traffic_light_bbox[3] - traffic_light_bbox[1]
        estimated_distance = min(5 * traffic_light_height, height * 0.3)
        return min(int(traffic_light_bottom + estimated_distance), height - 20)
  

# Example usage:
# bbox, vline, dbg = detect_crosswalk_and_violation_line(frame, (tl_x, tl_y), perspective_M)