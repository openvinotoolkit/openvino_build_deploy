# print("🟡 [CROSSWALK_UTILS] This is d:/Downloads/finale6/Khatam final/khatam/qt_app_pyside/utils/crosswalk_utils.py LOADED")
# import cv2
# import numpy as np

# def detect_crosswalk_and_violation_line(frame, traffic_light_detected=False, perspective_M=None, debug=False):
#     """
#     Detects crosswalk (zebra crossing) or fallback stop line in a traffic scene using classical CV.
#     Only runs crosswalk detection if a traffic light is present in the frame.
#     If no traffic light is present, no violation line is drawn or returned.
#     Returns:
#         result_frame: frame with overlays (for visualization)
#         crosswalk_bbox: (x, y, w, h) or None if fallback used
#         violation_line_y: int (y position for violation check) or None if not applicable
#         debug_info: dict (for visualization/debugging)
#     """
#     debug_info = {}
#     orig_frame = frame.copy()
#     h, w = frame.shape[:2]

#     if not traffic_light_detected:
#         # No traffic light: do not draw or return any violation line
#         debug_info['crosswalk_bbox'] = None
#         debug_info['violation_line_y'] = None
#         debug_info['note'] = 'No traffic light detected, no violation line.'
#         return orig_frame, None, None, debug_info

#     # 1. Perspective Normalization (Bird's Eye View)
#     if perspective_M is not None:
#         frame = cv2.warpPerspective(frame, perspective_M, (w, h))
#         debug_info['perspective_warped'] = True
#     else:
#         debug_info['perspective_warped'] = False

#     # 2. White Color Filtering (relaxed)
#     mask_white = cv2.inRange(frame, (160, 160, 160), (255, 255, 255))
#     debug_info['mask_white_ratio'] = np.sum(mask_white > 0) / (h * w)

#     # 3. Grayscale for adaptive threshold
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     if np.mean(gray) < 80:
#         gray = cv2.equalizeHist(gray)
#         debug_info['hist_eq'] = True
#     else:
#         debug_info['hist_eq'] = False

#     # 4. Adaptive threshold (tuned)
#     thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                    cv2.THRESH_BINARY, 15, 5)
#     combined = cv2.bitwise_and(thresh, mask_white)

#     # 5. Morphology (tuned)
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
#     morph = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=1)

#     # 6. Find contours for crosswalk bars
#     contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     zebra_rects = []
#     for cnt in contours:
#         x, y, rw, rh = cv2.boundingRect(cnt)
#         aspect_ratio = rw / max(rh, 1)
#         area = rw * rh
#         if aspect_ratio > 3 and 1000 < area < 0.5 * h * w and rh < 60:
#             zebra_rects.append((x, y, rw, rh))

#     # 7. Group crosswalk bars by y (vertical alignment)
#     y_tolerance = int(h * 0.05)
#     crosswalk_bbox = None
#     violation_line_y = None
#     if len(zebra_rects) >= 3:
#         zebra_rects = sorted(zebra_rects, key=lambda r: r[1])
#         groups = []
#         group = [zebra_rects[0]]
#         for rect in zebra_rects[1:]:
#             if abs(rect[1] - group[-1][1]) < y_tolerance:
#                 group.append(rect)
#             else:
#                 if len(group) >= 3:
#                     groups.append(group)
#                 group = [rect]
#         if len(group) >= 3:
#             groups.append(group)
#         # Use the largest group
#         if groups:
#             best_group = max(groups, key=len)
#             xs = [r[0] for r in best_group] + [r[0] + r[2] for r in best_group]
#             ys = [r[1] for r in best_group] + [r[1] + r[3] for r in best_group]
#             x1, x2 = min(xs), max(xs)
#             y1, y2 = min(ys), max(ys)
#             crosswalk_bbox = (x1, y1, x2 - x1, y2 - y1)
#             violation_line_y = min(y2 + 5, h - 1) # Place just before crosswalk
#             # Draw crosswalk region
#             overlay = orig_frame.copy()
#             cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
#             orig_frame = cv2.addWeighted(overlay, 0.2, orig_frame, 0.8, 0)
#             cv2.rectangle(orig_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(orig_frame, "Crosswalk", (10, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#     # --- Fallback: Stop line detection ---
#     if crosswalk_bbox is None:
#         edges = cv2.Canny(gray, 80, 200)
#         lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=60, maxLineGap=20)
#         stop_lines = []
#         if lines is not None:
#             for l in lines:
#                 x1, y1, x2, y2 = l[0]
#                 angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
#                 if abs(angle) < 20 or abs(angle) > 160:  # horizontal
#                     if y1 > h // 2 or y2 > h // 2:  # lower half
#                         stop_lines.append((x1, y1, x2, y2))
#         if stop_lines:
#             best_line = max(stop_lines, key=lambda l: max(l[1], l[3]))
#             x1, y1, x2, y2 = best_line
#             violation_line_y = min(y1, y2) - 5
#             cv2.line(orig_frame, (0, violation_line_y), (w, violation_line_y), (0, 255, 255), 8)
#             cv2.putText(orig_frame, "Fallback Stop Line", (10, violation_line_y - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
#         else:
#             # Final fallback: bottom third
#             violation_line_y = int(h * 0.75)
#             cv2.line(orig_frame, (0, violation_line_y), (w, violation_line_y), (0, 0, 255), 3)
#             cv2.putText(orig_frame, "Default Violation Line", (10, violation_line_y - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#     # Always draw the violation line if found
#     if violation_line_y is not None and crosswalk_bbox is not None:
#         cv2.line(orig_frame, (0, violation_line_y), (w, violation_line_y), (0, 0, 255), 3)
#         cv2.putText(orig_frame, "Violation Line", (10, violation_line_y - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#     debug_info['crosswalk_bbox'] = crosswalk_bbox
#     debug_info['violation_line_y'] = violation_line_y

#     return orig_frame, crosswalk_bbox, violation_line_y, debug_info

# def draw_violation_line(frame: np.ndarray, y: int, color=(0, 0, 255), thickness=4, style='solid', label='Violation Line'):
#     h, w = frame.shape[:2]
#     x1, x2 = 0, w
#     overlay = frame.copy()
#     if style == 'dashed':
#         dash_len = 30
#         gap = 20
#         for x in range(x1, x2, dash_len + gap):
#             x_end = min(x + dash_len, x2)
#             cv2.line(overlay, (x, y), (x_end, y), color, thickness, lineType=cv2.LINE_AA)
#     else:
#         cv2.line(overlay, (x1, y), (x2, y), color, thickness, lineType=cv2.LINE_AA)
#     cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
#     if label:
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         text_size, _ = cv2.getTextSize(label, font, 0.8, 2)
#         text_x = max(10, (w - text_size[0]) // 2)
#         text_y = max(0, y - 12)
#         cv2.rectangle(frame, (text_x - 5, text_y - text_size[1] - 5), (text_x + text_size[0] + 5, text_y + 5), (0,0,0), -1)
#         cv2.putText(frame, label, (text_x, text_y), font, 0.8, color, 2, cv2.LINE_AA)
#     return frame

# def get_violation_line_y(frame, traffic_light_bbox=None, crosswalk_bbox=None):
#     """
#     Returns the y-coordinate of the violation line using the following priority:
#     1. Crosswalk bbox (most accurate)
#     2. Stop line detection via image processing (CV)
#     3. Traffic light bbox heuristic
#     4. Fallback (default)
#     """
#     height, width = frame.shape[:2]
#     # 1. Crosswalk bbox
#     if crosswalk_bbox is not None and len(crosswalk_bbox) == 4:
#         return int(crosswalk_bbox[1]) - 15
#     # 2. Stop line detection (CV)
#     roi_height = int(height * 0.4)
#     roi_y = height - roi_height
#     roi = frame[roi_y:height, 0:width]
#     gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#     binary = cv2.adaptiveThreshold(
#         gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         cv2.THRESH_BINARY, 15, -2
#     )
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
#     processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
#     contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     stop_line_candidates = []
#     for cnt in contours:
#         x, y, w, h = cv2.boundingRect(cnt)
#         aspect_ratio = w / max(h, 1)
#         normalized_width = w / width
#         if (aspect_ratio > 5 and normalized_width > 0.3 and h < 15 and y > roi_height * 0.5):
#             abs_y = y + roi_y
#             stop_line_candidates.append((abs_y, w))
#     if stop_line_candidates:
#         stop_line_candidates.sort(key=lambda x: x[1], reverse=True)
#         return stop_line_candidates[0][0]
#     # 3. Traffic light bbox heuristic
#     if traffic_light_bbox is not None and len(traffic_light_bbox) == 4:
#         traffic_light_bottom = traffic_light_bbox[3]
#         traffic_light_height = traffic_light_bbox[3] - traffic_light_bbox[1]
#         estimated_distance = min(5 * traffic_light_height, height * 0.3)
#         return min(int(traffic_light_bottom + estimated_distance), height - 20)
#     # 4. Fallback
#     return int(height * 0.75)

# # Example usage:
# # bbox, vline, dbg = detect_crosswalk_and_violation_line(frame, (tl_x, tl_y), perspective_M)
print("🟡 [CROSSWALK_UTILS]222 This is d:/Downloads/finale6/Khatam final/khatam/qt_app_pyside/utils/crosswalk_utils.py LOADED")
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
    debug_info = {}
    orig_frame = frame.copy()
    h, w = frame.shape[:2]

    # 1. Perspective Normalization (Bird's Eye View)
    if perspective_M is not None:
        frame = cv2.warpPerspective(frame, perspective_M, (w, h))
        debug_info['perspective_warped'] = True
    else:
        debug_info['perspective_warped'] = False

    # 1. White Color Filtering (relaxed)
    mask_white = cv2.inRange(frame, (160, 160, 160), (255, 255, 255))
    debug_info['mask_white_ratio'] = np.sum(mask_white > 0) / (h * w)

    # 2. Grayscale for adaptive threshold
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Enhance contrast for night/low-light
    if np.mean(gray) < 80:
        gray = cv2.equalizeHist(gray)
        debug_info['hist_eq'] = True
    else:
        debug_info['hist_eq'] = False
    # 5. Adaptive threshold (tuned)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 15, 5)
    # Combine with color mask
    combined = cv2.bitwise_and(thresh, mask_white)
    # 2. Morphology (tuned)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    morph = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=1)
    # Find contours
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    zebra_rects = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / max(h, 1)
        area = w * h
        angle = 0  # For simplicity, assume horizontal stripes
        # Heuristic: wide, short, and not too small
        if aspect_ratio > 3 and 1000 < area < 0.5 * frame.shape[0] * frame.shape[1] and h < 60:
            zebra_rects.append((x, y, w, h, angle))
            cv2.rectangle(orig_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # --- Overlay drawing for debugging: draw all zebra candidates ---
    for r in zebra_rects:
        x, y, rw, rh, _ = r
        cv2.rectangle(orig_frame, (x, y), (x+rw, y+rh), (0, 255, 0), 2)
    # Draw all zebra candidate rectangles for debugging (no saving)
    for r in zebra_rects:
        x, y, rw, rh, _ = r
        cv2.rectangle(orig_frame, (x, y), (x+rw, y+rh), (0, 255, 0), 2)
    # --- Probabilistic Scoring for Groups ---
    def group_score(group):
        if len(group) < 3:
            return 0
        heights = [r[3] for r in group]
        x_centers = [r[0] + r[2]//2 for r in group]
        angles = [r[4] for r in group]
        # Stripe count (normalized)
        count_score = min(len(group) / 6, 1.0)
        # Height consistency
        height_score = 1.0 - min(np.std(heights) / (np.mean(heights) + 1e-6), 1.0)
        # X-center alignment
        x_score = 1.0 - min(np.std(x_centers) / (w * 0.2), 1.0)
        # Angle consistency (prefer near 0 or 90)
        mean_angle = np.mean([abs(a) for a in angles])
        angle_score = 1.0 - min(np.std(angles) / 10.0, 1.0)
        # Whiteness (mean mask_white in group area)
        whiteness = 0
        for r in group:
            x, y, rw, rh, _ = r
            whiteness += np.mean(mask_white[y:y+rh, x:x+rw]) / 255
        whiteness_score = whiteness / len(group)
        # Final score (weighted sum)
        score = 0.25*count_score + 0.2*height_score + 0.2*x_score + 0.15*angle_score + 0.2*whiteness_score
        return score
    # 4. Dynamic grouping tolerance
    y_tolerance = int(h * 0.05)
    crosswalk_bbox = None
    violation_line_y = None
    best_score = 0
    best_group = None
    if len(zebra_rects) >= 3:
        zebra_rects = sorted(zebra_rects, key=lambda r: r[1])
        groups = []
        group = [zebra_rects[0]]
        for rect in zebra_rects[1:]:
            if abs(rect[1] - group[-1][1]) < y_tolerance:
                group.append(rect)
            else:
                if len(group) >= 3:
                    groups.append(group)
                group = [rect]
        if len(group) >= 3:
            groups.append(group)
        # Score all groups
        scored_groups = [(group_score(g), g) for g in groups if group_score(g) > 0.1]
        print(f"[CROSSWALK DEBUG] scored_groups: {[s for s, _ in scored_groups]}")
        if scored_groups:
            scored_groups.sort(reverse=True, key=lambda x: x[0])
            best_score, best_group = scored_groups[0]
            print("Best group score:", best_score)
            # Visualization for debugging
            debug_vis = orig_frame.copy()
            for r in zebra_rects:
                x, y, rw, rh, _ = r
                cv2.rectangle(debug_vis, (x, y), (x+rw, y+rh), (255, 0, 255), 2)
            for r in best_group:
                x, y, rw, rh, _ = r
                cv2.rectangle(debug_vis, (x, y), (x+rw, y+rh), (0, 255, 255), 3)
            cv2.imwrite(f"debug_crosswalk_group.png", debug_vis)
            # Optionally, filter by vanishing point as before
            # ...existing vanishing point code...
            xs = [r[0] for r in best_group] + [r[0] + r[2] for r in best_group]
            ys = [r[1] for r in best_group] + [r[1] + r[3] for r in best_group]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            crosswalk_bbox = (x1, y1, x2 - x1, y2 - y1)
            violation_line_y = y2 - 5
            debug_info['crosswalk_group'] = best_group
            debug_info['crosswalk_score'] = best_score
            debug_info['crosswalk_angles'] = [r[4] for r in best_group]
    # --- Fallback: Stop line detection ---
    if crosswalk_bbox is None:
        edges = cv2.Canny(gray, 80, 200)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=60, maxLineGap=20)
        stop_lines = []
        if lines is not None:
            for l in lines:
                x1, y1, x2, y2 = l[0]
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                if abs(angle) < 20 or abs(angle) > 160:  # horizontal
                    if y1 > h // 2 or y2 > h // 2:  # lower half
                        stop_lines.append((x1, y1, x2, y2))
        debug_info['stop_lines'] = stop_lines
        print(f"[CROSSWALK DEBUG] stop_lines: {len(stop_lines)} found")
        if stop_lines:
            if traffic_light_position:
                tx, ty = traffic_light_position
                best_line = min(stop_lines, key=lambda l: abs(((l[1]+l[3])//2) - ty))
            else:
                best_line = max(stop_lines, key=lambda l: max(l[1], l[3]))
            x1, y1, x2, y2 = best_line
            crosswalk_bbox = None
            violation_line_y = min(y1, y2) - 5
            debug_info['stop_line'] = best_line
            print(f"[CROSSWALK DEBUG] using stop_line: {best_line}")
    # Draw fallback violation line overlay for debugging (no saving)
    if crosswalk_bbox is None and violation_line_y is not None:
        print(f"[DEBUG] Drawing violation line at y={violation_line_y} (frame height={orig_frame.shape[0]})")
        if 0 <= violation_line_y < orig_frame.shape[0]:
            orig_frame = draw_violation_line(orig_frame, violation_line_y, color=(0, 255, 255), thickness=8, style='solid', label='Fallback Stop Line')
        else:
            print(f"[WARNING] Invalid violation line position: {violation_line_y}")
    # --- Manual overlay for visualization pipeline test ---
    # Removed fake overlays that could overwrite the real violation line
    print(f"[CROSSWALK DEBUG] crosswalk_bbox: {crosswalk_bbox}, violation_line_y: {violation_line_y}")
    return orig_frame, crosswalk_bbox, violation_line_y, debug_info

def draw_violation_line(frame: np.ndarray, y: int, color=(0, 255, 255), thickness=8, style='solid', label='Violation Line'):
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
    # 4. Fallback
    return int(height * 0.75)

# Example usage:
# bbox, vline, dbg = detect_crosswalk_and_violation_line(frame, (tl_x, tl_y), perspective_M)