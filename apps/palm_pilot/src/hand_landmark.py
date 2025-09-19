import cv2
import numpy as np
from collections import namedtuple
from math import ceil, sqrt, exp, pi, floor, sin, cos, atan2

class HandRegion:
    def __init__(self, pd_score, pd_box, pd_kps=None):
        self.pd_score = pd_score
        self.pd_box = pd_box
        self.pd_kps = pd_kps if pd_kps is not None else []

    def print(self):
        attrs = vars(self)
        print('\n'.join("%s: %s" % item for item in attrs.items()))


SSDAnchorOptions = namedtuple('SSDAnchorOptions', [
    'num_layers',
    'min_scale',
    'max_scale',
    'input_size_height',
    'input_size_width',
    'anchor_offset_x',
    'anchor_offset_y',
    'strides',
    'aspect_ratios',
    'reduce_boxes_in_lowest_layer',
    'interpolated_scale_aspect_ratio',
    'fixed_anchor_size'
])

def calculate_scale(min_scale, max_scale, stride_index, num_strides):
    if num_strides == 1:
        return (min_scale + max_scale) / 2
    else:
        return min_scale + (max_scale - min_scale) * stride_index / (num_strides - 1)

def generate_anchors(options):
    anchors = []
    layer_id = 0
    n_strides = len(options.strides)
    while layer_id < n_strides:
        anchor_height = []
        anchor_width = []
        aspect_ratios = []
        scales = []
        last_same_stride_layer = layer_id
        while last_same_stride_layer < n_strides and \
                options.strides[last_same_stride_layer] == options.strides[layer_id]:
            scale = calculate_scale(options.min_scale, options.max_scale, last_same_stride_layer, n_strides)
            if last_same_stride_layer == 0 and options.reduce_boxes_in_lowest_layer:
                aspect_ratios += [1.0, 2.0, 0.5]
                scales += [0.1, scale, scale]
            else:
                aspect_ratios += options.aspect_ratios
                scales += [scale] * len(options.aspect_ratios)
                if options.interpolated_scale_aspect_ratio > 0:
                    if last_same_stride_layer == n_strides -1:
                        scale_next = 1.0
                    else:
                        scale_next = calculate_scale(options.min_scale, options.max_scale, last_same_stride_layer+1, n_strides)
                    scales.append(sqrt(scale * scale_next))
                    aspect_ratios.append(options.interpolated_scale_aspect_ratio)
            last_same_stride_layer += 1

        for i, r in enumerate(aspect_ratios):
            ratio_sqrts = sqrt(r)
            anchor_height.append(scales[i] / ratio_sqrts)
            anchor_width.append(scales[i] * ratio_sqrts)

        stride = options.strides[layer_id]
        feature_map_height = ceil(options.input_size_height / stride)
        feature_map_width = ceil(options.input_size_width / stride)

        for y in range(feature_map_height):
            for x in range(feature_map_width):
                for anchor_id in range(len(anchor_height)):
                    x_center = (x + options.anchor_offset_x) / feature_map_width
                    y_center = (y + options.anchor_offset_y) / feature_map_height
                    if options.fixed_anchor_size:
                        new_anchor = [x_center, y_center, 1.0, 1.0]
                    else:
                        new_anchor = [x_center, y_center, anchor_width[anchor_id], anchor_height[anchor_id]]
                    anchors.append(new_anchor)
        layer_id = last_same_stride_layer
    return anchors


def decode_bboxes(score_thresh, scores, bboxes, anchors):
    """
    wi, hi : NN input shape
    mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.cc
    
    
    

    https://github.com/google/mediapipe/blob/master/mediapipe/modules/palm_detection/palm_detection_cpu.pbtxt :
    node {
        calculator: "TensorsToDetectionsCalculator"
        input_stream: "TENSORS:detection_tensors"
        input_side_packet: "ANCHORS:anchors"
        output_stream: "DETECTIONS:unfiltered_detections"
        options: {
            [mediapipe.TensorsToDetectionsCalculatorOptions.ext] {
            num_classes: 1
            num_boxes: 896
            num_coords: 18
            box_coord_offset: 0
            keypoint_coord_offset: 4
            num_keypoints: 7
            num_values_per_keypoint: 2
            sigmoid_score: true
            score_clipping_thresh: 100.0
            reverse_output_order: true

            x_scale: 128.0
            y_scale: 128.0
            h_scale: 128.0
            w_scale: 128.0
            min_score_thresh: 0.5
            }
        }
    }

    scores: shape = [number of anchors 896]
    bboxes: shape = [ number of anchors x 18], 18 = 4 (bounding box : (cx,cy,w,h) + 14 (7 palm keypoints)
    """
    regions = []
    scores = 1 / (1 + np.exp(-scores))
    detection_mask = scores > score_thresh
    det_scores = scores[detection_mask]
    if det_scores.size == 0: return regions
    det_bboxes = bboxes[detection_mask]
    det_anchors = anchors[detection_mask]
    scale = 128   
    det_bboxes = det_bboxes* np.tile(det_anchors[:,2:4], 9) / scale + np.tile(det_anchors[:,0:2],9)
    det_bboxes[:,2:4] = det_bboxes[:,2:4] - det_anchors[:,0:2]
    det_bboxes[:,0:2] = det_bboxes[:,0:2] - det_bboxes[:,3:4] * 0.5
    for i in range(det_bboxes.shape[0]):
        score = det_scores[i]
        box = det_bboxes[i,0:4]
        kps = []
        for kp in range(7):
            kps.append(det_bboxes[i,4+kp*2:6+kp*2])
        regions.append(HandRegion(float(score), box, kps))
    return regions

def non_max_suppression(regions, nms_thresh):
    boxes = [ [int(x*1000) for x in r.pd_box] for r in regions]
    scores = [r.pd_score for r in regions]
    indices = cv2.dnn.NMSBoxes(boxes, scores, 0, nms_thresh)
    return [regions[i] for i in indices]

def normalize_radians(angle):
    return angle - 2 * pi * floor((angle + pi) / (2 * pi))

def rot_vec(vec, rotation):
    vx, vy = vec
    return [vx * cos(rotation) - vy * sin(rotation), vx * sin(rotation) + vy * cos(rotation)]

def detections_to_rect(regions):
    target_angle = pi * 0.5
    for region in regions:
        region.rect_w = region.pd_box[2]
        region.rect_h = region.pd_box[3]
        region.rect_x_center = region.pd_box[0] + region.rect_w / 2
        region.rect_y_center = region.pd_box[1] + region.rect_h / 2
        x0, y0 = region.pd_kps[0]
        x1, y1 = region.pd_kps[2]
        rotation = target_angle - atan2(-(y1 - y0), x1 - x0)
        region.rotation = normalize_radians(rotation)


def rotated_rect_to_points(cx, cy, w, h, rotation, wi, hi):
    b = cos(rotation) * 0.5
    a = sin(rotation) * 0.5
    points = []
    p0x = cx - a*h - b*w
    p0y = cy + b*h - a*w
    p1x = cx + a*h - b*w
    p1y = cy - b*h - a*w
    p2x = int(2*cx - p0x)
    p2y = int(2*cy - p0y)
    p3x = int(2*cx - p1x)
    p3y = int(2*cy - p1y)
    p0x, p0y, p1x, p1y = int(p0x), int(p0y), int(p1x), int(p1y)
    return [(p0x,p0y), (p1x,p1y), (p2x,p2y), (p3x,p3y)]

def rect_transformation(regions, w, h):
    scale_x = 1.4  
    scale_y = 2.4 
    shift_x = 0
    shift_y = -0.4
    for region in regions:
        width = region.rect_w
        height = region.rect_h
        rotation = region.rotation 
        
        
        region.rect_x_center_a = (region.rect_x_center + width * shift_x) * w
        region.rect_y_center_a = (region.rect_y_center + height * shift_y) * h
        
        
        
            
            
            
            
            
        long_side = max(width * w, height * h)
        region.rect_w_a = long_side * scale_x
        region.rect_h_a = long_side * scale_y
        region.rect_points = rotated_rect_to_points(region.rect_x_center_a, region.rect_y_center_a, region.rect_w_a, region.rect_h_a, region.rotation, w, h)

def warp_rect_img(rect_points, img, w, h):
    src = np.array(rect_points[1:], dtype=np.float32)
    dst = np.array([(0, 0), (h, 0), (h, w)], dtype=np.float32)
    mat = cv2.getAffineTransform(src, dst)
    return cv2.warpAffine(img, mat, (w, h))

def distance(a, b):
    return np.linalg.norm(a-b)

def angle(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


options = SSDAnchorOptions(
    num_layers=4,
    min_scale=0.1484375,
    max_scale=0.75,
    input_size_height=192,
    input_size_width=192,
    anchor_offset_x=0.5,
    anchor_offset_y=0.5,
    strides=[8, 16, 16, 16],
    aspect_ratios=[1.0],
    reduce_boxes_in_lowest_layer=False,
    interpolated_scale_aspect_ratio=1.0,
    fixed_anchor_size=True
)



def calculate_iou(box1, box2):
    """
    Calculates Intersection over Union (IoU) between two bounding boxes.
    Box format: [x_min, y_min, width, height] (normalized coordinates)
    """
    x1_min, y1_min, w1, h1 = box1
    x1_max, y1_max = x1_min + w1, y1_min + h1

    x2_min, y2_min, w2, h2 = box2
    x2_max, y2_max = x2_min + w2, y2_min + h2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_w = max(0, inter_x_max - inter_x_min)
    inter_h = max(0, inter_y_max - inter_y_min)
    intersection_area = inter_w * inter_h

    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - intersection_area

    if union_area == 0:
        return 0.0
    return intersection_area / union_area

def normalize_landmarks_correctly(landmarks_px, image_width=224, image_height=224, normalize_z_factor=1.0):
    """Follow MediaPipe's exact normalization from C++ source.
    landmarks_px is a list/array of [x_px, y_px, z_px] coordinates.
    """
    normalized_landmarks = []
    for landmark in landmarks_px:
        x, y, z = landmark[:3]
        
        norm_x = x / image_width
        norm_y = y / image_height  
        norm_z = (z / image_width) / normalize_z_factor 
        
        normalized_landmarks.append([norm_x, norm_y, norm_z])
    return normalized_landmarks

def apply_sigmoid_activation(raw_scores):
    """Apply sigmoid activation to raw scores."""
    return [1.0 / (1.0 + np.exp(-score)) for score in raw_scores]

def should_run_palm_detection(tracked_regions, lm_score_threshold=0.7):
    """Determine if palm detection is needed."""
    if not tracked_regions:
        
        return True  
    
    for region in tracked_regions:
        if not hasattr(region, 'lm_score') or region.lm_score < lm_score_threshold:
            
            return True  
            
    
    return False 


def lm_postprocess(current_region, inference, previous_frame_regions, 
                   alpha=0.6, iou_threshold=0.4, crop_width=224, crop_height=224):
    """Process landmark model output, applying EMA smoothing."""
    
    
    current_region.lm_score = float(np.squeeze(inference['Identity_1']))
    
    
    current_region.handedness = float(np.squeeze(inference['Identity_2']))
    
    
    pixel_landmarks_raw = np.squeeze(inference['Identity']) 
    
    
    pixel_landmarks_reshaped = pixel_landmarks_raw.reshape(21, 3)
    
    
    current_raw_normalized_landmarks = normalize_landmarks_correctly(
        pixel_landmarks_reshaped, 
        image_width=crop_width, 
        image_height=crop_height
    )
    
    
    best_match_prev_region = None
    max_iou = 0.0

    if previous_frame_regions and hasattr(current_region, 'pd_box'):
        for prev_region in previous_frame_regions:
            if hasattr(prev_region, 'pd_box'):
                
                
                
                iou = calculate_iou(current_region.pd_box, prev_region.pd_box)
                if iou > max_iou:
                    max_iou = iou
                    best_match_prev_region = prev_region
    
    final_landmarks_to_set = list(current_raw_normalized_landmarks) 

    if best_match_prev_region and max_iou >= iou_threshold and \
       hasattr(best_match_prev_region, 'landmarks') and \
       best_match_prev_region.landmarks is not None and \
       len(best_match_prev_region.landmarks) == len(current_raw_normalized_landmarks):
        
        prev_smoothed_landmarks = best_match_prev_region.landmarks
        
        temp_smoothed_ema = []
        for i in range(len(current_raw_normalized_landmarks)):
            smooth_pt = [
                alpha * current_raw_normalized_landmarks[i][k] + (1 - alpha) * prev_smoothed_landmarks[i][k]
                for k in range(3) 
            ]
            temp_smoothed_ema.append(smooth_pt)
        final_landmarks_to_set = temp_smoothed_ema
        
    
        
        
    current_region.landmarks = final_landmarks_to_set 
    
    
    return current_region




def lm_render(frame, region, lm_score_threshold=0.5):
    """Render landmarks on the frame with correct coordinate transformation"""
    if not hasattr(region, 'lm_score') or region.lm_score < lm_score_threshold:
        return
    
    if not hasattr(region, 'landmarks') or not hasattr(region, 'rect_points'):
        return
    
    if len(region.landmarks) != 21:
        print(f"Warning: Expected 21 landmarks, got {len(region.landmarks)}")
        return
    
    try:
        rect_points = region.rect_points
        
        
        lm_xy_crop_pixels = np.array([(l[0] * 224.0, l[1] * 224.0) for l in region.landmarks], dtype=np.float32)
        
        
        src_crop_coords = np.array([(0, 0), (224, 0), (224, 224)], dtype=np.float32)
        dst_rect_coords = np.array([rect_points[1], rect_points[2], rect_points[3]], dtype=np.float32)
        
        mat = cv2.getAffineTransform(src_crop_coords, dst_rect_coords)
        lm_xy_transformed = cv2.transform(np.expand_dims(lm_xy_crop_pixels, axis=0), mat)
        lm_xy_final = np.squeeze(lm_xy_transformed).astype(np.int32)
        
        
        connections = [
            [0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [5, 9, 10, 11, 12],
            [9, 13, 14, 15, 16], [13, 17], [0, 17, 18, 19, 20]
        ]
        
        
        for connection in connections:
            for i in range(len(connection) - 1):
                if (connection[i] < len(lm_xy_final) and connection[i + 1] < len(lm_xy_final)):
                    pt1 = tuple(lm_xy_final[connection[i]])
                    pt2 = tuple(lm_xy_final[connection[i + 1]])
                    cv2.line(frame, pt1, pt2, (255, 0, 0), 2)
        
        
        colors = [
            (255,0,255), (0,255,0), (255,255,0), (0,255,255), (255,0,0), (128,0,128)
        ]
        
        for i, (x, y) in enumerate(lm_xy_final):
            color_idx = 0
            if 1 <= i <= 4: color_idx = 1      
            elif 5 <= i <= 8: color_idx = 2    
            elif 9 <= i <= 12: color_idx = 3   
            elif 13 <= i <= 16: color_idx = 4  
            elif 17 <= i <= 20: color_idx = 5  
            cv2.circle(frame, (x, y), 4, colors[color_idx], -1)
            # cv2.putText(frame, str(i), (x+5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
        
        
        hand_type = "RIGHT" if region.handedness > 0.5 else "LEFT"
        text_pos_x = int(rect_points[0][0])
        text_pos_y = int(rect_points[0][1] - 30)
        if text_pos_y < 10: 
            text_pos_y = int(rect_points[0][1] + 30)

        cv2.putText(frame, f"{hand_type} ({region.handedness:.2f})", 
                   (text_pos_x, text_pos_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
    except Exception as e:
        print(f"Error in landmark rendering: {e}")
        import traceback
        traceback.print_exc()


gesture_labels = [
    "None", "Closed_Fist", "Open_Palm", "Pointing_Up", 
    "Thumb_Down", "Thumb_Up", "Victory", "ILoveYou"
]

def classify_gesture(gesture_embedding, gesture_classifier_model, gesture_labels, confidence_threshold=0.5):
    """Classify gesture from embedding"""
    try:
        
        classification_results = gesture_classifier_model([gesture_embedding])
        
        
        raw_probabilities = list(classification_results.values())[0][0]  
        
        
        final_probabilities = apply_sigmoid_activation(raw_probabilities)
        
        
        predicted_class = np.argmax(final_probabilities)
        confidence = final_probabilities[predicted_class]
        
        
        if confidence > confidence_threshold:
            if predicted_class < len(gesture_labels):
                gesture_name = gesture_labels[predicted_class]
            else:
                gesture_name = f"Unknown_{predicted_class}"
        else:
            gesture_name = "None" 
            
            
            
            
            
            
        return gesture_name, confidence, final_probabilities 
    except Exception as e:
        print(f"Error in gesture classification: {e}")
        return "Error", 0.0, [0.0] * len(gesture_labels) 


def mediapipe_style_gesture_processing(region, lm_results, gesture_model, classifier_model, gesture_labels_list):
    """Extract gesture embedding and classify using MediaPipe-style logic."""
    if not hasattr(region, 'landmarks') or len(region.landmarks) != 21:
        region.gesture_name = "None"
        region.gesture_confidence = 0.0
        return None
    
    
    hand_landmarks_normalized = np.array(region.landmarks, dtype=np.float32).reshape(1, 21, 3)
    handedness = np.array([[region.handedness]], dtype=np.float32)
    
    
    if 'Identity_3' in lm_results:
        world_landmarks_raw = np.squeeze(lm_results['Identity_3'])
        
        
        world_landmarks = world_landmarks_raw.reshape(1, 21, 3).astype(np.float32)
    else:
        
        
        print("Warning: 'Identity_3' (world landmarks) not found in landmark model results. Using approximation.")
        world_landmarks = hand_landmarks_normalized * 0.1 
    
    try:
        gesture_embedding_results = gesture_model([hand_landmarks_normalized, handedness, world_landmarks])
        embedding = list(gesture_embedding_results.values())[0]
        region.gesture_embedding = embedding
        
        
        if classifier_model and gesture_labels_list:
            gesture_name, confidence, probabilities = classify_gesture(embedding, classifier_model, gesture_labels_list)
            region.gesture_name = gesture_name
            region.gesture_confidence = confidence
            region.gesture_probabilities = probabilities
        else:
            region.gesture_name = "None" 
            region.gesture_confidence = 0.0
        
        return embedding
    except Exception as e:
        print(f"Error in gesture embedding/classification: {e}")
        region.gesture_name = "Error"
        region.gesture_confidence = 0.0
        return None



def lm_postprocess_with_gesture_classification(current_region, inference, previous_frame_regions, 
                                              gesture_model, classifier_model, 
                                              alpha=0.6, iou_threshold=0.4, crop_width=224, crop_height=224):
    """Process landmark model output with gesture embedding and classification"""
    
    
    lm_postprocess(current_region, inference, previous_frame_regions, alpha, iou_threshold, crop_width, crop_height)
    
    
    if hasattr(current_region, 'landmarks') and current_region.lm_score > 0.5 : 
        embedding = mediapipe_style_gesture_processing(current_region, inference, gesture_model, classifier_model, gesture_labels) 
            
    return current_region


