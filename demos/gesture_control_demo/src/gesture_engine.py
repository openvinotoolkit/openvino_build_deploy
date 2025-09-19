import cv2
import numpy as np
import time
import os
import math
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from config_manager import config_manager
from event_system import event_bus, GestureEvent
from openvino_models import model_manager

from gesture_processor import process_finger_detection
from hand_landmark import *
from application_modes import ApplicationModeManager
import pyautogui
from game_controller import get_game_controller
import time

import os
import sys

SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "utils")
sys.path.append(os.path.dirname(SCRIPT_DIR))

try:
    from utils import demo_utils as utils
    print("demo_utils loaded successfully.")
except ImportError:
    utils = None
    print("⚠️ demo_utils not available. OpenVINO watermark will not be displayed.")



class CompleteGestureEngine:
    """Complete gesture detection engine with full visual rendering like your notebook"""
    
    def __init__(self):
        self.config_manager = config_manager
        self.event_bus = event_bus
        self.model_manager = model_manager
        
        
        self.running = False
        self.paused = False
        self.frame_count = 0
        self.start_time = None
        self.fps = 0
        
        
        self.cap = None
        self.current_frame = None
        
        
        self.params = None
        self.app_modes = None
        
        
        self.anchors2_np = None
        self.app_mode_manager = None

        self.qr_code = None
        self.initialize_qr_code()


    def initialize_qr_code(self):
        """Initialize QR code for the demo"""
        if utils is not None:
            try:
                self.qr_code = utils.get_qr_code(
                    'https://github.com/openvinotoolkit/openvino_build_deploy/tree/master/demos/gesture_control_demo',
                    with_embedded_image=True
                )
                print("✅ QR code generated successfully.")
            except Exception as e:
                print(f"⚠️  QR code generation failed: {e}")
                self.qr_code = None
        else:
            print("⚠️  QR code not available (utils not loaded)")

     
    def initialize(self, benchmark_mode: bool = False) -> bool:
        """Initialize the complete gesture engine"""
        print("🔧 Initializing Complete Gesture Engine...")
        
        
        validation = self.config_manager.validate_config()
        if validation['errors']:
            print(f"❌ Configuration errors: {validation['errors']}")
            return False
        
        
        self.params = self.config_manager.get_legacy_params_dict()
        self.app_modes = self.params['app_modes']
        self.app_mode_manager = ApplicationModeManager(self.app_modes)
        self.app_mode_manager.set_engine_params(self.params)
        
        
        model_paths = {
            'palm_detection': 'mediapipeModels/hand_detector.xml',
            'hand_landmarks': 'mediapipeModels/hand_landmarks_detector.xml',
            'gesture_embedder': 'mediapipeModels/gesture_embedder.xml',
            'gesture_classifier': 'mediapipeModels/canned_gesture_classifier.xml'
        }
        
        if not self.model_manager.initialize_models(model_paths):
            print("❌ Model initialization failed!")
            return False
        
        
        if not benchmark_mode:
            # Check for camera ID from command line launcher
            preferred_camera = os.environ.get('PALMPLIOT_CAMERA_ID', '0')
            
            # Convert to int if numeric, keep as string for video files
            if preferred_camera.isnumeric():
                preferred_camera = int(preferred_camera)
            
            self.cap = None
            
            # Try preferred camera first
            print(f"🎥 Trying preferred camera/source: {preferred_camera}")
            try:
                test_cap = cv2.VideoCapture(preferred_camera, cv2.CAP_DSHOW if isinstance(preferred_camera, int) else 0)
                if test_cap.isOpened():
                    ret, test_frame = test_cap.read()
                    if ret and test_frame is not None:
                        print(f"✅ Preferred camera {preferred_camera} working!")
                        test_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        test_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        test_cap.set(cv2.CAP_PROP_FPS, 30)
                        self.cap = test_cap
                    else:
                        test_cap.release()
                        print(f"⚠️ Preferred camera {preferred_camera} opened but no frames")
                else:
                    test_cap.release()
                    print(f"⚠️ Preferred camera {preferred_camera} failed to open")
            except Exception as e:
                print(f"⚠️ Preferred camera {preferred_camera} error: {e}")
            
            # Fall back to auto-detection if preferred camera failed
            if self.cap is None:
                print("🔍 Falling back to auto-detection...")
                for camera_id in [0, 1, -1]:  
                    try:
                        test_cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
                        if test_cap.isOpened():
                            ret, test_frame = test_cap.read()
                            if ret and test_frame is not None:
                                print(f"✅ Fallback camera {camera_id} working!")
                                test_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                                test_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                                test_cap.set(cv2.CAP_PROP_FPS, 30)
                                self.cap = test_cap
                                break
                            else:
                                test_cap.release()
                        else:
                            test_cap.release()
                    except Exception as e:
                        print(f"Fallback camera {camera_id} error: {e}")
                        continue
            
            if not self.cap or not self.cap.isOpened():
                print("❌ Camera initialization failed!")
                return False
        else:
            print("🔧 Benchmark mode: Skipping camera initialization")
            self.cap = None
        
        
        
        anchors2 = generate_anchors(options)
        self.anchors2_np = np.array(anchors2)
        
        
        self.event_bus.start_processing()
        
        print("✅ Complete Gesture Engine initialized!")
        return True
    
    def start(self):
        """Start the engine"""
        
        if self.cap is None or not self.cap.isOpened():
            print("Re-initializing camera for start...")
            
            
            self.initialize() 

        self.running = True
        self.paused = False
        if self.start_time is None:
            self.start_time = time.time()
        print("▶️ Complete Engine started!")
    
    def pause(self):
        """Pause the engine"""
        self.paused = True
        print("⏸️ Complete Engine paused!")
    
    def resume(self):
        """Resume the engine"""
        self.paused = False
        print("▶️ Complete Engine resumed!")
    
    def stop(self):
        """Stop the engine"""
        self.running = False
        self.paused = False
        if self.cap:
            self.cap.release()
            
            self.cap = None
        self.event_bus.stop_processing()
        print("⏹️ Complete Engine stopped!")

    def process_single_frame_benchmark(self, frame: np.ndarray):
        """
        Processes a single frame for benchmarking, returning the annotated frame and performance timings.
        This method does NOT use the camera and does NOT trigger application mode actions.
        """
        timings = {}
        overall_start_time = time.perf_counter()
        
        try:
            original_frame = frame.copy()
            frame_h, frame_w = original_frame.shape[:2]
            resized_frame_for_input = cv2.resize(frame, (self.params['input_size'], self.params['input_size']))

            
            pd_start_time = time.perf_counter()
            
            need_palm_detection = self.params.get('always_run_palm_detection', True)
            
            current_regions_for_processing = []
            if need_palm_detection:
                regions_nms = self._run_palm_detection(resized_frame_for_input)
                self._smooth_detection_boxes(regions_nms)
                current_regions_for_processing = regions_nms
            else:
                
                
                regions_nms = self._run_palm_detection(resized_frame_for_input)
                current_regions_for_processing = regions_nms

            timings['palm_detection_inference_ms'] = (time.perf_counter() - pd_start_time) * 1000

            if current_regions_for_processing:
                detections_to_rect(current_regions_for_processing)
                rect_transformation(current_regions_for_processing, self.params['input_size'], self.params['input_size'])
            
            
            lm_start_time = time.perf_counter()
            processed_regions = self._process_landmarks_and_gestures(current_regions_for_processing, resized_frame_for_input)
            timings['landmark_inference_ms'] = (time.perf_counter() - lm_start_time) * 1000
            
            
            self._render_results_complete(original_frame, processed_regions, frame_w, frame_h)
            
            
            self.params['previous_frame_processed_regions'] = list(processed_regions)
            
            timings['total_engine_time_ms'] = (time.perf_counter() - overall_start_time) * 1000
            
            return original_frame, timings

        except Exception as e:
            print(f"Error in benchmark frame processing: {e}")
            return frame, {'error': str(e)}

    
    def get_frame_with_complete_processing(self):
        """Get frame with COMPLETE processing and rendering exactly like your notebook"""
        if not self.cap or not self.cap.isOpened():
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        original_frame = frame.copy()
        
        if not self.running or self.paused:
         
            cv2.putText(original_frame, "ENGINE PAUSED", 
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            
           
            if utils is not None:
                try:
                    # Draw OpenVINO watermark
                    utils.draw_ov_watermark(frame, alpha=0.35, size=0.2)
                    
                    # Draw QR code in bottom-right corner
                    if self.qr_code is not None:
                        utils.draw_qr_code(frame, self.qr_code)
                except Exception as e:
                    # Silently fail if watermark/QR code can't be drawn
                    pass
                    
            return original_frame
        
        try:
            frame_h, frame_w = original_frame.shape[:2]
            resized_frame_for_input = cv2.resize(frame, (self.params['input_size'], self.params['input_size']))

            
            current_hand_count = len(self.params['previous_frame_processed_regions'])

            
            in_game_mode = (self.app_mode_manager and 
                           self.app_mode_manager.app_modes.current_mode == 'game_mode')
            
            if in_game_mode:
                
                if current_hand_count < 2:
                    need_palm_detection = True  
                else:
                    
                    need_palm_detection = should_run_palm_detection(
                        self.params['previous_frame_processed_regions'], 
                        self.params['landmark_score_for_palm_redetection_threshold'] * 0.8  
                    )
            else:
                
                need_palm_detection = self._smart_palm_detection_state_machine(current_hand_count)
                if not need_palm_detection:
                    need_palm_detection = (
                        self.params['always_run_palm_detection'] or
                        should_run_palm_detection(
                            self.params['previous_frame_processed_regions'], 
                            self.params['landmark_score_for_palm_redetection_threshold']
                        )
                    )
                        
            current_regions_for_processing = []
            
            if need_palm_detection:
                
                regions_nms = self._run_palm_detection(resized_frame_for_input)
                
                
                self._smooth_detection_boxes(regions_nms)
                
                current_regions_for_processing = regions_nms
                if current_regions_for_processing:
                    detections_to_rect(current_regions_for_processing)
                    rect_transformation(current_regions_for_processing, self.params['input_size'], self.params['input_size'])
            else:
                
                current_regions_for_processing = self.params['previous_frame_processed_regions']
                if current_regions_for_processing:
                    detections_to_rect(current_regions_for_processing)
                    rect_transformation(current_regions_for_processing, self.params['input_size'], self.params['input_size'])
            
            
            if (self.app_mode_manager and 
                self.app_mode_manager.app_modes.current_mode == 'game_mode'):
                
                
                
                for region in current_regions_for_processing:
                    if not hasattr(region, 'rect_x_center_a') or not hasattr(region, 'rect_y_center_a'):
                        
                        detections_to_rect([region])
                        rect_transformation([region], self.params['input_size'], self.params['input_size'])
                
                
                processed_regions = current_regions_for_processing
                
                
                if len(processed_regions) > 0:
                    self.app_mode_manager.process_game_mode_regions(processed_regions)
                
            else:
                
                processed_regions = self._process_landmarks_and_gestures(current_regions_for_processing, resized_frame_for_input)
                
                
                for region in processed_regions:
                    self._process_application_modes(region)
            
            
            self._reset_hand_tracking(processed_regions)
            
            
            self._render_results_complete(original_frame, processed_regions, frame_w, frame_h)
            
            self._render_game_controller_overlay(original_frame)
            
            self.params['previous_frame_processed_regions'] = list(processed_regions)
            
            
            self.frame_count += 1
            if self.start_time:
                elapsed = time.time() - self.start_time
                if elapsed > 0:
                    self.fps = self.frame_count / elapsed

            if utils is not None:
                try:
                    # Draw OpenVINO watermark
                    utils.draw_ov_watermark(frame, alpha=0.35, size=0.2)
                    
                    # Draw QR code in bottom-right corner
                    if self.qr_code is not None:
                        utils.draw_qr_code(frame, self.qr_code)
                except Exception as e:
                    # Silently fail if watermark/QR code can't be drawn
                    pass
            
            return original_frame
            
        except Exception as e:
            print(f"Error in complete frame processing: {e}")
            import traceback
            traceback.print_exc()
            cv2.putText(original_frame, f"ERROR: {str(e)[:50]}", 
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Add watermark even on error frames (ADD THIS)
            if utils is not None:
                try:
                    # Draw OpenVINO watermark
                    utils.draw_ov_watermark(frame, alpha=0.35, size=0.2)
                    
                    # Draw QR code in bottom-right corner
                    if self.qr_code is not None:
                        utils.draw_qr_code(frame, self.qr_code)
                except Exception as e:
                    # Silently fail if watermark/QR code can't be drawn
                    pass
                    
            return original_frame
    
    def _smart_palm_detection_state_machine(self, current_hand_count):
        """Smart state machine exactly like your notebook"""
        current_time = time.time()
        current_state = self.params['palm_detection_state']
        debug = self.params['state_transition_debug']
        
        
        if current_hand_count == 0:
            if current_state != 'NO_HANDS':
                if debug: print(f"🔄 STATE: {current_state} → NO_HANDS")
                self.params['palm_detection_state'] = 'NO_HANDS'
            return True  
            
        elif current_hand_count == 1:
            if current_state == 'NO_HANDS':
                self.params['palm_detection_state'] = 'ONE_HAND_SEARCHING'
                self.params['grace_period_start'] = current_time
                if debug: print(f"🔍 STATE: NO_HANDS → ONE_HAND_SEARCHING")
                return True
                
            elif current_state == 'ONE_HAND_SEARCHING':
                elapsed_time = current_time - self.params['grace_period_start']
                if elapsed_time >= self.params['grace_period_duration']:
                    self.params['palm_detection_state'] = 'ONE_HAND_STABLE'
                    if debug: print(f"⏰ STATE: ONE_HAND_SEARCHING → ONE_HAND_STABLE")
                    return False
                return True
                
            elif current_state == 'ONE_HAND_STABLE':
                self.params['periodic_check_counter'] += 1
                if self.params['periodic_check_counter'] >= self.params['periodic_check_interval']:
                    self.params['periodic_check_counter'] = 0
                    if debug: print(f"👀 Periodic check for 2nd hand")
                    return True
                return False
                
        elif current_hand_count >= 2:
            if current_state != 'TWO_HANDS':
                if debug: print(f"🎉 STATE: {current_state} → TWO_HANDS")
                self.params['palm_detection_state'] = 'TWO_HANDS'
            return False
        
        return False
    
    def _run_palm_detection(self, resized_frame):
    
        try:
            compiled_model = self.model_manager.get_compiled_model('palm_detection')
            if not compiled_model:
                return []
            
            input_tensor = np.expand_dims(resized_frame, axis=0)
            
            output_node_regressors = compiled_model.output("Identity")
            output_node_scores = compiled_model.output("Identity_1")
            results = compiled_model([input_tensor])

            regressors_tensor = results[output_node_regressors]
            scores_tensor = results[output_node_scores]

            raw_scores = scores_tensor[0, :, 0]
            raw_bboxes_and_keypoints = regressors_tensor[0]

            regions_from_palm_detection = decode_bboxes(
                self.params['score_threshold'], raw_scores, raw_bboxes_and_keypoints, self.anchors2_np
            )
            
            regions_nms = []
            if regions_from_palm_detection:
                regions_nms = non_max_suppression(regions_from_palm_detection, self.params['nms_threshold'])

            return regions_nms
            
        except Exception as e:
            print(f"Error in palm detection: {e}")
            return []
    
    def _smooth_detection_boxes(self, regions_nms):
        """Apply smoothing exactly like your notebook"""
        if not (regions_nms and self.params['previous_frame_processed_regions']):
            return
        
        for current_new_region in regions_nms:
            best_match_prev = None
            max_iou_for_smoothing = 0.0
            
            for prev_reg in self.params['previous_frame_processed_regions']:
                if hasattr(prev_reg, 'pd_box'):
                    iou_val = calculate_iou(current_new_region.pd_box, prev_reg.pd_box)
                    if iou_val > max_iou_for_smoothing:
                        max_iou_for_smoothing = iou_val
                        best_match_prev = prev_reg
            
            if best_match_prev and max_iou_for_smoothing > 0.15:
                for i_coord in range(4):
                    current_new_region.pd_box[i_coord] = (
                        self.params['detection_smoothing_alpha'] * best_match_prev.pd_box[i_coord] + 
                        (1 - self.params['detection_smoothing_alpha']) * current_new_region.pd_box[i_coord]
                    )
    
    def _process_landmarks_and_gestures(self, regions, resized_frame):
        """Process landmarks and gestures exactly like your notebook"""
        processed_regions = []
        
        if not (regions and self.params['show_landmarks']):
            return processed_regions
        
        
        compiled_model_landmark = self.model_manager.get_compiled_model('hand_landmarks')
        compiled_model_gesture = self.model_manager.get_compiled_model('gesture_embedder')
        compiled_model_classifier = self.model_manager.get_compiled_model('gesture_classifier')
        
        if not compiled_model_landmark:
            return processed_regions
        
        for region_idx, region_to_process in enumerate(regions):
            if not hasattr(region_to_process, 'rect_points'):
                continue
                
            try:
                
                hand_crop_bgr = warp_rect_img(region_to_process.rect_points, resized_frame, 224, 224)
                hand_crop_rgb = cv2.cvtColor(hand_crop_bgr, cv2.COLOR_BGR2RGB)
                hand_input = np.expand_dims(hand_crop_rgb, axis=0).astype(np.float32) / 255.0
                
                lm_results = compiled_model_landmark([hand_input])
                
                if self.params['show_static_gestures']:
                    lm_postprocess_with_gesture_classification(
                        region_to_process, lm_results, 
                        self.params['previous_frame_processed_regions'],
                        compiled_model_gesture,
                        compiled_model_classifier,
                        alpha=self.params['smoothing_alpha'], 
                        iou_threshold=self.params['iou_match_threshold']
                    )
                    
                    self._apply_gesture_smoothing(region_to_process, region_idx)
                else:
                    lm_postprocess(region_to_process, lm_results, 
                                 self.params['previous_frame_processed_regions'],
                                 alpha=self.params['smoothing_alpha'], 
                                 iou_threshold=self.params['iou_match_threshold'])
                
                
                
                process_finger_detection(region_to_process, self.params)
                
                processed_regions.append(region_to_process)
                
            except Exception as e:
                print(f"Error processing region {region_idx}: {e}")
        
        return processed_regions
    
    def _apply_gesture_smoothing(self, region, region_idx):
        """Apply gesture smoothing exactly like your notebook"""
        if not hasattr(region, 'gesture_name'):
            return
        
        region_id = f"region_{region_idx}"
        
        if region_id not in self.params['gesture_history']:
            self.params['gesture_history'][region_id] = []
        
        self.params['gesture_history'][region_id].append(region.gesture_name)
        if len(self.params['gesture_history'][region_id]) > self.params['gesture_smoothing_frames']:
            self.params['gesture_history'][region_id].pop(0)
        
        if len(self.params['gesture_history'][region_id]) >= 3:
            gesture_counts = {}
            for gest in self.params['gesture_history'][region_id]:
                gesture_counts[gest] = gesture_counts.get(gest, 0) + 1
            
            most_common_gesture = max(gesture_counts, key=lambda k: gesture_counts[k])
            if gesture_counts[most_common_gesture] >= len(self.params['gesture_history'][region_id]) // 2 + 1:
                region.gesture_name = most_common_gesture
    
    def _process_application_modes(self, region):
        """Delegate to application mode manager"""
        if self.app_mode_manager:
            self.app_mode_manager.process_application_modes(region)
    
    def _reset_hand_tracking(self, processed_regions):
        """Reset hand tracking exactly like your notebook"""
        any_gesture = any(hasattr(region, 'gesture_type') and region.gesture_type != "none" 
                          for region in processed_regions)
        
        if not any_gesture:
            self.params['last_pressed_hand'] = None
    
    def _render_results_complete(self, frame, processed_regions, frame_w, frame_h):
        """Render bounding boxes, landmarks, show gesture name, user-friendly gesture, and the actual action performed in the current mode as an overlay, plus FPS overlay."""
        if not processed_regions:
            return

        input_size = self.params['input_size']
        current_mode = None
        mode_action_map = None

        if hasattr(self, 'app_modes') and hasattr(self.app_modes, 'current_mode'):
            current_mode = getattr(self.app_modes, 'current_mode', None)
            mode_config = getattr(self.app_modes, current_mode, None)
            if mode_config and hasattr(mode_config, 'gestures'):
                mode_action_map = mode_config.gestures

        for region in processed_regions:
            if not hasattr(region, 'rect_points'):
                continue

            # Scale rectangle points
            scaled_points = []
            for ptx, pty in region.rect_points:
                scaled_ptx = int(ptx * frame_w / input_size)
                scaled_pty = int(pty * frame_h / input_size)
                scaled_points.append((scaled_ptx, scaled_pty))

            points_array = np.array(scaled_points, np.int32)
            cv2.polylines(frame, [points_array], True, (0, 255, 0), 2)

            # Render landmarks if enabled
            if self.params['show_landmarks'] and hasattr(region, 'landmarks'):
                original_rp_backup = region.rect_points
                region.rect_points = scaled_points
                lm_render(frame, region)
                region.rect_points = original_rp_backup

            # Calculate text position
            y_text = max(pt[1] for pt in scaled_points) + 25
            x_text = min(pt[0] for pt in scaled_points)

            # Get gesture information
            static_gesture_name = getattr(region, 'gesture_name', None)
            dynamic_gesture_type = getattr(region, 'gesture_type', None)
            hand_type = getattr(region, 'hand_type', 'unknown')

            # Determine primary gesture and display name
            primary_gesture = None
            display_name = None
            gesture_source = None

            # Check for dynamic gestures first (they're more immediate)
            if dynamic_gesture_type and dynamic_gesture_type != "none":
                if dynamic_gesture_type == 'index_only':
                    primary_gesture = f"{hand_type}_index_bent"
                    display_name = f"{hand_type.title()} Index Finger Bent"
                    gesture_source = "Dynamic"
                elif dynamic_gesture_type == 'index_middle_both':
                    primary_gesture = f"{hand_type}_index_middle_bent"
                    display_name = f"{hand_type.title()} Index + Middle Bent"
                    gesture_source = "Dynamic"
            
            # If no dynamic gesture, check for static gestures
            elif static_gesture_name and static_gesture_name != "None":
                if static_gesture_name == "Closed_Fist":
                    primary_gesture = 'fist_gesture'
                    display_name = 'Closed Fist'
                    gesture_source = "Static"
                elif static_gesture_name == "Open_Palm":
                    primary_gesture = 'open_palm_gesture'
                    display_name = 'Open Palm'
                    gesture_source = "Static"
                elif static_gesture_name == "ILoveYou":
                    primary_gesture = 'iloveyou_gesture'
                    display_name = 'I Love You Sign'
                    gesture_source = "Static"
                else:
                    primary_gesture = static_gesture_name
                    display_name = static_gesture_name
                    gesture_source = "Static"

            # Display the gesture information
            if primary_gesture and display_name:
                # Main gesture line
                gesture_text = f"Gesture: {display_name}"
                cv2.putText(frame, gesture_text, (x_text, y_text), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Gesture source and type info
                source_text = f"Type: {gesture_source}"
                cv2.putText(frame, source_text, (x_text, y_text + 28), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
                
                # Show mode action if available
                mode_action = None
                if mode_action_map and primary_gesture in mode_action_map:
                    action_config = mode_action_map[primary_gesture]
                    if hasattr(action_config, 'action'):
                        if action_config.action == 'key_press' and hasattr(action_config, 'key'):
                            mode_action = f"Key: {action_config.key.upper()}"
                        elif action_config.action == 'mouse_click' and hasattr(action_config, 'button'):
                            mode_action = f"Click: {action_config.button.upper()}"
                
                if mode_action and current_mode != 'disabled':
                    cv2.putText(frame, f"Action: {mode_action}", (x_text, y_text + 56), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 2)
            else:
                # No gesture detected
                cv2.putText(frame, "No Gesture", (x_text, y_text), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)

        # Display FPS
        fps_val = self.fps if hasattr(self, 'fps') else 0
        utils.draw_text(frame, f"FPS: {fps_val:.1f}", (5, 10), 
                    font_scale=4.0, font_color=(0, 255, 0), with_background=False)
    
        
        # Add OpenVINO watermark (ADD THIS)
        if utils is not None:
            try:
                # Draw OpenVINO watermark
                utils.draw_ov_watermark(frame, alpha=0.35, size=0.2)
                
                # Draw QR code in bottom-right corner
                if self.qr_code is not None:
                    utils.draw_qr_code(frame, self.qr_code)
            except Exception as e:
                # Silently fail if watermark/QR code can't be drawn
                pass

    def _render_game_controller_overlay(self, frame):
        """Renders the two-handed game control visuals."""
        game_controller = get_game_controller()
        if not game_controller or not game_controller.active:
            return

        overlay_data = game_controller.get_overlay_data()
        if not overlay_data:
            return

        
        global_offset_x = self.params.get('hand_label_offset_x', 0.0)
        global_offset_y = self.params.get('hand_label_offset_y', 0.0)
        left_offset_x = self.params.get('left_hand_offset_x', 0.0)
        left_offset_y = self.params.get('left_hand_offset_y', 0.0)
        right_offset_x = self.params.get('right_hand_offset_x', 0.0)
        right_offset_y = self.params.get('right_hand_offset_y', 0.0)

        
        control_lines = overlay_data.get('control_lines', {})
        frame_h, frame_w = frame.shape[:2]

        
        accel_y = int(control_lines.get('accelerate_y', frame_h * 0.25))
        cv2.line(frame, (0, accel_y), (frame_w, accel_y), (0, 255, 0), 3)  
        cv2.putText(frame, "ACCELERATE", (10, accel_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        
        brake_y = int(control_lines.get('brake_y', frame_h * 0.75))
        cv2.line(frame, (0, brake_y), (frame_w, brake_y), (0, 0, 255), 3)  
        cv2.putText(frame, "BRAKE", (10, brake_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        
        left_hand_pos = overlay_data.get('left_hand_pos')
        right_hand_pos = overlay_data.get('right_hand_pos')

        
        if left_hand_pos and right_hand_pos:
            
            left_x = int(left_hand_pos[0] + global_offset_x + left_offset_x)
            left_y = int(left_hand_pos[1] + global_offset_y + left_offset_y)
            right_x = int(right_hand_pos[0] + global_offset_x + right_offset_x)
            right_y = int(right_hand_pos[1] + global_offset_y + right_offset_y)

            
            steering_zone = overlay_data.get('steering_zone', 'neutral')
            if steering_zone == 'neutral':
                line_color = (0, 255, 0)      
            elif steering_zone in ['soft_left', 'soft_right']:
                line_color = (0, 255, 255)    
            elif steering_zone in ['hard_left', 'hard_right']:
                line_color = (0, 0, 255)      
            else:
                line_color = (255, 255, 255)  

            
            cv2.line(frame, (left_x, left_y), (right_x, right_y), line_color, 4)

            
            cv2.circle(frame, (left_x, left_y), 18, (255, 0, 255), 3)
            cv2.circle(frame, (right_x, right_y), 18, (255, 255, 0), 3)

            
            steering_angle = overlay_data.get('steering_angle', 0.0)
            angle_text = f"Angle: {steering_angle:.1f}"
            
            mid_x = int((left_x + right_x) / 2)
            mid_y = int((left_y + right_y) / 2)
            cv2.putText(frame, angle_text, (mid_x - 60, mid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, line_color, 3)

        
        if left_hand_pos:
            label_x = int(left_hand_pos[0] + global_offset_x + left_offset_x)
            label_y = int(left_hand_pos[1] + global_offset_y + left_offset_y)
            cv2.putText(frame, "L", (label_x - 30, label_y - 30), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 0, 255), 3)

        if right_hand_pos:
            label_x = int(right_hand_pos[0] + global_offset_x + right_offset_x)
            label_y = int(right_hand_pos[1] + global_offset_y + right_offset_y)
            cv2.putText(frame, "R", (label_x - 30, label_y - 30), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 0), 3)

        if utils is not None:
            try:
                # Draw OpenVINO watermark
                utils.draw_ov_watermark(frame, alpha=0.35, size=0.2)
                
                # Draw QR code in bottom-right corner
                if self.qr_code is not None:
                    utils.draw_qr_code(frame, self.qr_code)
            except Exception as e:
                # Silently fail if watermark/QR code can't be drawn
                pass
        
        
    
    def get_status(self):
        """Get engine status"""
        return {
            'running': self.running,
            'paused': self.paused,
            'frame_count': self.frame_count,
            'fps': self.fps,
            'models_loaded': self.model_manager.is_initialized(),
            'camera_active': self.cap is not None and self.cap.isOpened()
        }
    def switch_mode(self, mode_name: str):
        """Switch to a new application mode - delegate to manager"""
        print(f"🔄 Engine switching to mode: {mode_name}")
        if self.app_mode_manager:
            result = self.app_mode_manager.switch_mode(mode_name)
            print(f"   Mode switch result: {result}")
            
            
            if mode_name == 'game_mode':
                game_controller = get_game_controller()
                if game_controller:
                    print(f"   Game controller active after switch: {game_controller.active}")
                else:
                    print("   ❌ No game controller found after mode switch")
            
            return result
        return False


complete_engine = CompleteGestureEngine()