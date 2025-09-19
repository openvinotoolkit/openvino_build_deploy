import time
import pyautogui
import numpy as np
import math
from typing import Dict, Any
from config_manager import ApplicationModeGesture, config_manager
from game_controller import get_game_controller
try:
    import pydirectinput
    pydirectinput.PAUSE = 0 
    PYDIRECTINPUT_AVAILABLE = True
except ImportError:
    PYDIRECTINPUT_AVAILABLE = False
    print("⚠️ pydirectinput not available. Fighting arcade mode will use pyautogui fallback.")


class ApplicationModeManager:
    """Manages application-specific gesture modes and actions using dataclass objects."""
    
    def __init__(self, app_modes_config: Any):
        
      
        if isinstance(app_modes_config, dict):
            self.app_modes = config_manager.app_modes
        else:
            self.app_modes = app_modes_config
        
        self.game_controller = None  
        self.params = {}  
        
    def set_engine_params(self, params: Dict[str, Any]):
        """Set reference to engine parameters"""
        self.params = params
        
        self.params.setdefault('volume_control_active_left', False)
        self.params.setdefault('volume_control_active_left_until', 0)

     
        print(f"🔧 Setting engine params:")
        print(f" enable_game_control: {params.get('enable_game_control', False)}")
        print(f" game_control_type: {params.get('game_control_type', 'unknown')}")
        print(f" steering_sensitivity: {params.get('steering_sensitivity', 'unknown')}")

       
        if params.get('enable_game_control', False):
            self.game_controller = get_game_controller(params)
            print(f"🎮 Game controller created: {self.game_controller is not None}")
            if self.game_controller:
                print(f"🎮 Game controller active: {self.game_controller.active}")
             
                if self.app_modes.current_mode == 'game_mode':
                    self.game_controller.activate()
                    print("🎮 Auto-activated game controller for current game mode")
        else:
            print("❌ Game control disabled - no controller will be created")


    def process_application_modes(self, region):
        """Main processor for application modes, using dataclass attribute access."""
        if not hasattr(region, 'landmarks'):
            return

        self._handle_browser_mode_iloveyou(region)

        current_mode_key = self.app_modes.current_mode
        if current_mode_key == 'disabled':
            return

        mode_config = getattr(self.app_modes, current_mode_key, None)
        if not mode_config or not mode_config.enabled:
            return

        hand_type = "right" if region.handedness > 0.5 else "left"

       
        media_volume_active = False
        if current_mode_key == 'media_mode':
       
            if (hasattr(region, 'gesture_name') and 
                region.gesture_name == "ILoveYou" and 
                hand_type == "left"):
                self.params['volume_control_active_left'] = True
                self.params['volume_control_active_left_until'] = time.time() + 0.8
                print("🎵 Left-hand volume control activated in media mode")
            
            
            media_volume_active = (
                self.params.get('volume_control_active_left', False) and 
                time.time() < self.params.get('volume_control_active_left_until', 0)
            )
            
            if media_volume_active and hand_type == "left":
                self.handle_pinch_volume_control(region)
          
                return

        if current_mode_key == 'volume_mode':
            self.handle_pinch_volume_control(region)
            return

     
        if current_mode_key == 'game_mode':
          
            return 

       
        detected_gestures = []


        if current_mode_key == 'fighting_arcade_mode':
            if hasattr(region, 'gesture_type'):
                if region.gesture_type == "index_only":
                    detected_gestures.append(f"{hand_type}_index_bent")
                elif region.gesture_type == "index_middle_both":
                    detected_gestures.append(f"{hand_type}_index_middle_bent")

           
            if hasattr(region, 'gesture_name'):
                if region.gesture_name == "Closed_Fist" and hand_type == "left":
                    detected_gestures.append('left_fist_gesture')
                elif region.gesture_name == "Closed_Fist" and hand_type == "right":
                    detected_gestures.append('fist_gesture') 
                elif region.gesture_name == "ILoveYou" and hand_type == "right":
                    detected_gestures.append('iloveyou_gesture') 
                elif region.gesture_name == "ILoveYou":
                
                    pass

        else:
        
            if hasattr(region, 'gesture_type'):
             
                skip_left_gestures = (current_mode_key == 'media_mode' and 
                                    media_volume_active and 
                                    hand_type == "left")
                
                if region.gesture_type == "index_only" and not skip_left_gestures:
                    detected_gestures.append(f"{hand_type}_index_bent")
                elif region.gesture_type == "index_middle_both" and not skip_left_gestures:
                    detected_gestures.append(f"{hand_type}_index_middle_bent")

            if hasattr(region, 'gesture_name'):
                if region.gesture_name == "Closed_Fist":
                    detected_gestures.append('fist_gesture')
                elif region.gesture_name == "Open_Palm":
                    detected_gestures.append('open_palm_gesture')
                elif region.gesture_name == "ILoveYou":
                 
                    if current_mode_key == 'browser_mode':
                        if hand_type == 'left':
                         
                            detected_gestures.append('iloveyou_gesture')
                        else:
                           
                            self._handle_browser_mode_iloveyou(region)
                    else:
                       
                     
                        if current_mode_key != 'media_mode' or hand_type != "left":
                            detected_gestures.append('iloveyou_gesture')

 
        if (current_mode_key == 'media_mode' and 
            time.time() >= self.params.get('volume_control_active_left_until', 0)):
            self.params['volume_control_active_left'] = False

        for gesture_id in set(detected_gestures):
            if gesture_id in mode_config.gestures:
                gesture_data = mode_config.gestures[gesture_id]
                success = self._execute_application_gesture(gesture_id, gesture_data)
                if success:
                    print(f"🎯 Executed {gesture_id} in mode {current_mode_key}")

        
        if current_mode_key == 'browser_mode' and hand_type == "right":
            right_mode = self.app_modes.browser_right_hand_mode
            if right_mode == 'cursor':
                self.handle_cursor_control(region, force_enable=True)
            elif right_mode == 'scroll':
                self.handle_scroll_control(region, force_enable=True)


    def process_game_mode_regions(self, regions_list):
        """Process all regions for game mode using two-handed control"""
        current_mode_key = self.app_modes.current_mode
        if current_mode_key == 'game_mode':
            self._handle_game_mode(regions_list)

    def _handle_game_mode(self, regions_list):
        """Handle game mode with two-handed steering wheel logic"""
        if not self.game_controller:
            print(f"❌ Game controller not available! Current controller: {self.game_controller}")
            return

        self.game_controller.update_controls_from_hands(regions_list)

    def switch_mode(self, mode_name: str) -> bool:
        """Switch to a new application mode using dataclass attribute access."""
        current_time = time.time()
        if current_time - self.app_modes.last_mode_switch < self.app_modes.mode_switch_cooldown:
            return False
        

        current_mode_key = self.app_modes.current_mode
        if current_mode_key != 'disabled':
            current_mode_obj = getattr(self.app_modes, current_mode_key, None)
            if current_mode_obj:
                current_mode_obj.enabled = False
            
         
            if current_mode_key == 'game_mode' and self.game_controller:
                self.game_controller.deactivate()
        
        
        self.app_modes.current_mode = mode_name
        new_mode_obj = None
        if mode_name != 'disabled':
            new_mode_obj = getattr(self.app_modes, mode_name, None)
            if new_mode_obj:
                new_mode_obj.enabled = True
            
            
            if mode_name == 'game_mode' and self.game_controller:
                self.game_controller.activate()
        
        self.app_modes.last_mode_switch = current_time
        
        mode_display = new_mode_obj.name if new_mode_obj else 'Disabled'
        print(f"🔄 MODE SWITCH: {mode_display}")
        return True
    
        
    def handle_pinch_volume_control(self, region):
        params = self.params
        if not params.get('enable_volume_control', False):
            return

        if not hasattr(region, 'landmarks') or len(region.landmarks) < 9:
            return

        hand_pref = params.get('volume_control_hand', 'any')
        if hand_pref != 'any' and region.hand_type != hand_pref:
            return

        thumb_tip = np.array(region.landmarks[4][:2])
        index_tip = np.array(region.landmarks[8][:2])
        pinch_dist = np.linalg.norm(thumb_tip - index_tip)

        
        smoothing_alpha = 0.5  
        if not hasattr(region, 'smoothed_pinch_dist'):
            region.smoothed_pinch_dist = pinch_dist
        else:
            region.smoothed_pinch_dist = (
                smoothing_alpha * pinch_dist + (1 - smoothing_alpha) * region.smoothed_pinch_dist
            )
        smooth_dist = region.smoothed_pinch_dist

        
        if not hasattr(region, 'pinch_state'):
            region.pinch_state = {'active': False, 'last_dist': smooth_dist, 'last_change_time': 0}

        state = region.pinch_state
        current_time = time.time()

        start_thresh = params.get('pinch_threshold_start', 0.28)  
        stop_thresh = params.get('pinch_threshold_stop', 0.32)    

        if state['active']:
            if smooth_dist > stop_thresh:
                state['active'] = False
                state['last_dist'] = smooth_dist
                return

            if current_time - state['last_change_time'] < params.get('volume_change_cooldown', 0.05):
                return

            dist_change = smooth_dist - state['last_dist']
            sensitivity = params.get('volume_sensitivity', 1.5)
            volume_change = dist_change * 100 * sensitivity

            if abs(volume_change) > 0.1:
                from volume_controller import get_volume_controller
                vc = get_volume_controller()
                if vc.is_valid:
                    vc.change_volume(volume_change)
                    state['last_change_time'] = current_time

        else:
            if smooth_dist < start_thresh:
                state['active'] = True
                print(f"🤏 Pinch gesture activated on {region.hand_type} hand.")

        state['last_dist'] = smooth_dist

    def _handle_browser_mode_iloveyou(self, region):
        """ILoveYou gesture logic updated for dataclass attribute access."""
        if not self.app_modes.browser_mode.enabled:
            return
        if not hasattr(region, 'gesture_name') or region.gesture_name != "ILoveYou":
            return
        
        hand_type = "right" if region.handedness > 0.5 else "left"
        
        
        if hand_type != "right":
            return
        
        current_time = time.time()

        if current_time - self.app_modes.browser_last_iloveyou_switch < self.app_modes.browser_iloveyou_switch_cooldown:
            return

        current_rh_mode = self.app_modes.browser_right_hand_mode
        new_rh_mode = 'scroll' if current_rh_mode == 'cursor' else 'cursor'
        self.app_modes.browser_right_hand_mode = new_rh_mode
        self.app_modes.browser_last_iloveyou_switch = current_time
        print(f"🤟 BROWSER: Right-hand ILoveYou toggled right-hand mode → {new_rh_mode.upper()}")

    def _execute_application_gesture(self, gesture_id: str, gesture_config: ApplicationModeGesture):
        """Gesture execution logic with proper validation and pydirectinput support for fighting mode."""
        current_time = time.time()
        
        if current_time - self.app_modes.gesture_timings.get(gesture_id, 0) < gesture_config.cooldown:
            return False

        try:
            action = gesture_config.action
            
            if action == 'key_press' and gesture_config.key:
                
                is_fighting_mode = self.app_modes.current_mode == 'fighting_arcade_mode'
                
                if '+' in gesture_config.key:
                    keys_to_press = [k.strip() for k in gesture_config.key.split('+')]
                    
                    if is_fighting_mode and PYDIRECTINPUT_AVAILABLE:
                        
                        for key in keys_to_press:
                            pydirectinput.keyDown(key)
                        for key in reversed(keys_to_press):
                            pydirectinput.keyUp(key)
                        print(f"🥊 {gesture_config.description}: PyDirectInput Combo {gesture_config.key.upper()}")
                    else:
                        
                        pyautogui.hotkey(*keys_to_press)
                        print(f"🎯 {gesture_config.description}: Hotkey {gesture_config.key.upper()}")
                else:
                    if is_fighting_mode and PYDIRECTINPUT_AVAILABLE:
                        
                        pydirectinput.press(gesture_config.key)
                        print(f"🥊 {gesture_config.description}: PyDirectInput Key {gesture_config.key.upper()}")
                    else:
                        
                        pyautogui.press(gesture_config.key)
                        print(f"🎯 {gesture_config.description}: Key Press {gesture_config.key.upper()}")
                        
            elif action == 'mouse_click' and gesture_config.button:
                
                valid_buttons = ('left', 'middle', 'right', 'primary', 'secondary')
                if gesture_config.button not in valid_buttons:
                    print(f"❌ Invalid mouse button '{gesture_config.button}' for gesture '{gesture_id}'. Skipping.")
                    return False
                    
                pyautogui.click(button=gesture_config.button)
                print(f"🖱️ {gesture_config.description}: {gesture_config.button.upper()} CLICK")
            else:
                print(f"❌ Invalid gesture configuration for {gesture_id}: action='{action}', key='{gesture_config.key}', button='{gesture_config.button}'")
                return False

            self.app_modes.gesture_timings[gesture_id] = current_time
            return True
            
        except Exception as e:
            print(f"❌ Error executing gesture {gesture_id}: {e}")
            return False


    def handle_cursor_control(self, region, force_enable=False):
        """OPTIMIZED: Your cursor control logic, now non-blocking and efficient."""
        if not force_enable and not self.params.get('enable_cursor_control', False):
            return
        if not hasattr(region, 'landmarks') or len(region.landmarks) < 9:
            return
        
        try:
            index_tip = region.landmarks[8]
            
            target_x = self.params['screen_width'] * (1 - index_tip[0])
            target_y = self.params['screen_height'] * index_tip[1]
            
            if self.params.get('previous_cursor_pos') is not None:
                prev_x, prev_y = self.params['previous_cursor_pos']
                smooth_x = self.params['cursor_smoothing'] * prev_x + (1 - self.params['cursor_smoothing']) * target_x
                smooth_y = self.params['cursor_smoothing'] * prev_y + (1 - self.params['cursor_smoothing']) * target_y
            else:
                smooth_x, smooth_y = target_x, target_y
            
            pyautogui.moveTo(smooth_x, smooth_y, duration=0)
            
            self.params['previous_cursor_pos'] = (smooth_x, smooth_y)
            
        except Exception as e:
            print(f"Error in cursor control: {e}")

    def handle_scroll_control(self, region, force_enable=False):
        """FIXED: Scroll control logic with proper gesture detection."""
        if not force_enable and not self.params.get('enable_scroll_control', False):
            return
        if not hasattr(region, 'landmarks') or len(region.landmarks) < 13:
            return
        
        try:
            index_tip = region.landmarks[8]
            middle_tip = region.landmarks[12]
            
            
            finger_distance = math.sqrt((index_tip[0] - middle_tip[0])**2 + (index_tip[1] - middle_tip[1])**2)
            current_hand = "right" if region.handedness > 0.5 else "left"
            
            
            scroll_hand_pref = self.params.get('scroll_hand_preference', 'any')
            if scroll_hand_pref != 'any' and current_hand != scroll_hand_pref:
                return
            
            
            if 'scroll_state' not in self.params:
                self.params['scroll_state'] = {
                    'is_scrolling': False, 
                    'start_pos': None, 
                    'locked_direction': None, 
                    'active_hand': None,
                    'last_scroll_time': 0
                }
            
            scroll_state = self.params['scroll_state']
            center_y = (index_tip[1] + middle_tip[1]) / 2
            current_time = time.time()
            
            
            scroll_threshold = self.params.get('scroll_threshold', 0.04)
            
            print(f"[DEBUG] Hand: {current_hand}, Distance: {finger_distance:.4f}, Threshold: {scroll_threshold:.4f}")
            
            
            if finger_distance < scroll_threshold:
                if not scroll_state['is_scrolling']:
                    
                    scroll_state.update({
                        'is_scrolling': True, 
                        'start_pos': center_y, 
                        'active_hand': current_hand,
                        'locked_direction': None
                    })
                    print(f"🔒 {current_hand.upper()} SCROLL START (threshold: {scroll_threshold:.3f})")
                    return
                
                
                if scroll_state['active_hand'] != current_hand:
                    return
                
                
                delta_from_start = center_y - scroll_state['start_pos']
                
                
                if scroll_state['locked_direction'] is None and abs(delta_from_start) > 0.03:
                    scroll_state['locked_direction'] = "down" if delta_from_start > 0 else "up"
                    print(f"🎯 {current_hand.upper()} DIRECTION LOCKED: {scroll_state['locked_direction'].upper()}")
                
                
                if scroll_state['locked_direction'] is not None:
                    current_direction = "down" if delta_from_start > 0 else "up"
                    
                    
                    if current_direction == scroll_state['locked_direction']:
                        
                        if current_time - scroll_state['last_scroll_time'] > 0.1:  
                            scroll_sensitivity = self.params.get('scroll_sensitivity', 6)
                            scroll_amount = int(-delta_from_start * scroll_sensitivity * 80)
                            
                            if abs(scroll_amount) > 2:  
                                pyautogui.scroll(scroll_amount)
                                scroll_state['last_scroll_time'] = current_time
                                direction_arrow = "↑" if scroll_amount > 0 else "↓"
                                print(f"📜 {current_hand.upper()} SCROLL {direction_arrow} (amount: {scroll_amount})")
            else:
                
                if scroll_state['is_scrolling']:
                    print(f"🔓 {scroll_state['active_hand'].upper()} SCROLL END")
                    scroll_state.update({
                        'is_scrolling': False, 
                        'start_pos': None, 
                        'locked_direction': None, 
                        'active_hand': None
                    })
                    
        except Exception as e:
            print(f"❌ Error in scroll control: {e}")
            import traceback
            traceback.print_exc()