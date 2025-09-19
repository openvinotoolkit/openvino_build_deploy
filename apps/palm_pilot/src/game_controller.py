"""
Two-handed game controller for racing games, based on intuitive steering wheel-like motions.
Enhanced with performance optimizations, robustness features, and advanced control algorithms.
"""

import time
import math
from typing import Dict, Any, Optional, Tuple, List

try:
    import pydirectinput
    pydirectinput.PAUSE = 0 
    DIRECTINPUT_AVAILABLE = True
except ImportError:
    DIRECTINPUT_AVAILABLE = False
    print("⚠️ pydirectinput not available. Using keyboard fallback.")

class GameController:
    """
    A two-handed game control system that simulates a steering wheel.
    - Steering: Tilting the hands controls the car's direction.
    - Acceleration/Braking: Moving both hands up or down controls speed.
    """
    
    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.active = False

        
        self.left_hand_pos = None  
        self.right_hand_pos = None  
        self.steering_angle = 0.0
        self.is_accelerating = False
        self.is_braking = False

        
        self._left_hand_pos_f = None  
        self._right_hand_pos_f = None  
        self._steering_angle_f = 0.0
        self._last_update_ts = 0.0

        
        self._hands_lost_ts = None  
        self._last_valid_steering = 0.0
        self._steering_history = []  
        self._hand_confidences = [0.0, 0.0]  
        
        
        self._steering_zone = 'neutral'  
        self._zone_values = {
            'hard_left': -1.0,
            'soft_left': -0.4,
            'neutral': 0.0,
            'soft_right': 0.4,
            'hard_right': 1.0
        }
        
        
        self._zone_thresholds = {
            'hard_left_enter': self.params.get('hard_left_angle', 35),      
            'soft_left_enter': self.params.get('soft_left_angle', 13),       
            'neutral_dead_zone': self.params.get('neutral_dead_zone', 5),   
            'soft_right_enter': self.params.get('soft_right_angle', 13),     
            'hard_right_enter': self.params.get('hard_right_angle', 35),    
        }
        
        
        self._zone_hysteresis = self.params.get('zone_hysteresis', 1)  

        
        self.control_lines = {}
        self._update_control_lines()

        
        self.pressed_keys = set()
        self._pending_key_changes = {'press': set(), 'release': set()}

    def activate(self):
        """Activate the game controller."""
        self.active = True
        self._update_control_lines()
        print("🎮 Two-Handed Game Controller ACTIVATED")

    def deactivate(self):
        """Deactivate the controller and release all keys."""
        self.active = False
        self._neutralize_controls()
        self._apply_key_changes()  
        self._release_all_keys()  
        print("🎮 Two-Handed Game Controller DEACTIVATED")

    def _update_control_lines(self):
        """Update control line positions from parameters."""
        
        screen_h = self.params.get('camera_frame_height', self.params.get('screen_height', 1080))
        
        accelerate_line_y = self.params.get('accelerate_line_y', 0.25)  
        brake_line_y = self.params.get('brake_line_y', 0.75)  
        self.control_lines = {
            'accelerate_y': screen_h * accelerate_line_y,
            'brake_y': screen_h * brake_line_y,
        }

    def update_controls_from_hands(self, hand_regions: List[Any]):
        """
        The main control logic loop. Takes a list of detected hand regions
        and updates the game controls accordingly.
        """
        if not self.active:
            return

        now = time.time()
        
        
        rate_hz = self.params.get('control_update_rate_hz', 300)  
        min_dt = 1.0 / max(1, rate_hz)
        if now - self._last_update_ts < min_dt:
            return
        self._last_update_ts = now

        
        self._pending_key_changes = {'press': set(), 'release': set()}

        if len(hand_regions) >= 2:
            
            self._hands_lost_ts = None
            
            
            regions_sorted_by_score = sorted(
                [r for r in hand_regions if hasattr(r, 'pd_score')],
                key=lambda r: r.pd_score,
                reverse=True
            )
            regions_pair = regions_sorted_by_score[:2] if len(regions_sorted_by_score) >= 2 else hand_regions[:2]

            
            self._hand_confidences = [
                getattr(regions_pair[0], 'pd_score', 0.8),
                getattr(regions_pair[1], 'pd_score', 0.8)
            ]
            
            
            min_confidence = self.params.get('min_hand_confidence', 0.2)  
            if min(self._hand_confidences) < min_confidence:
                print(f"⚠️ Low confidence: {self._hand_confidences}, min required: {min_confidence}")
                self._handle_hand_loss(now)
                self._apply_key_changes()
                return

            
            hand1_center = self._get_box_center(regions_pair[0])
            hand2_center = self._get_box_center(regions_pair[1])

            
            if self._validate_hand_positions(hand1_center, hand2_center):
                if hand1_center[0] < hand2_center[0]:
                    self.left_hand_pos, self.right_hand_pos = hand1_center, hand2_center
                else:
                    self.left_hand_pos, self.right_hand_pos = hand2_center, hand1_center

                print(f"🎮 Hands: L({self.left_hand_pos[0]:.0f},{self.left_hand_pos[1]:.0f}) R({self.right_hand_pos[0]:.0f},{self.right_hand_pos[1]:.0f}) Conf:{self._hand_confidences}")
                self._apply_smoothing()
                self._update_steering()
                self._update_throttle_brake()
            else:
                
                print(f"❌ Invalid positions: {hand1_center}, {hand2_center}")
                self._handle_hand_loss(now)
        else:
            
            self._handle_hand_loss(now)
        
        
        self._apply_key_changes()

    def _get_box_center(self, region: Any) -> Tuple[float, float]:
        """Calculate hand center in camera-frame pixels for correct overlay alignment."""
        
        frame_w = self.params.get('camera_frame_width', 640)  
        frame_h = self.params.get('camera_frame_height', 480)  
        
        if hasattr(region, 'rect_x_center_a') and hasattr(region, 'rect_y_center_a'):
            
            x_center = float(region.rect_x_center_a)
            y_center = float(region.rect_y_center_a)
            return (x_center, y_center)
        
        if hasattr(region, 'pd_box') and isinstance(region.pd_box, (list, tuple)) and len(region.pd_box) >= 4:
            x_min, y_min, w, h = region.pd_box[:4]
            
            x_center_norm = x_min + w * 0.5
            y_center_norm = y_min + h * 0.5
            
            
            x_center = x_center_norm * frame_w
            y_center = y_center_norm * frame_h
            
            return (x_center, y_center)
        
        
        return (frame_w * 0.5, frame_h * 0.5)


    def _validate_hand_positions(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> bool:
        """Validate that hand positions are reasonable and not teleporting."""
        if self.left_hand_pos is None or self.right_hand_pos is None:
            return True  
        
        
        max_movement = self.params.get('max_hand_movement_per_frame', 300)  
        
        prev_positions = [self.left_hand_pos, self.right_hand_pos]
        new_positions = [pos1, pos2]
        
        for prev, new in zip(prev_positions, new_positions):
            distance = math.sqrt((new[0] - prev[0])**2 + (new[1] - prev[1])**2)
            if distance > max_movement:
                print(f"❌ Teleportation detected: {distance:.1f} > {max_movement}")
                return False
                
        
        frame_w = self.params.get('camera_frame_width', self.params.get('screen_width', 1920))
        frame_h = self.params.get('camera_frame_height', self.params.get('screen_height', 1080))
        edge_margin = 20  
        
        for pos in [pos1, pos2]:
            if (pos[0] < edge_margin or pos[0] > frame_w - edge_margin or
                pos[1] < edge_margin or pos[1] > frame_h - edge_margin):
                print(f"❌ Edge position detected: {pos}")
                return False
                
        return True

    def _handle_hand_loss(self, current_time: float):
        """Handle the case when hands are lost with grace period."""
        grace_period = self.params.get('hand_loss_grace_period', 0.3)  
        
        if self._hands_lost_ts is None:
            self._hands_lost_ts = current_time
            return  
            
        time_lost = current_time - self._hands_lost_ts
        
        if time_lost < grace_period:
            
            if time_lost > grace_period * 0.5:  
                self._steering_zone = 'neutral'
                self.steering_angle = 0.0
                self._queue_key_release('left')
                self._queue_key_release('right')
                
            
        else:
            
            self._neutralize_controls()

    def _apply_smoothing(self):
        """Apply position smoothing with adaptive parameters."""
        
        base_alpha = self.params.get('position_smoothing_alpha', 0.4)
        confidence_factor = min(self._hand_confidences) / 0.8  
        alpha_pos = base_alpha * confidence_factor
        
        if self._left_hand_pos_f is None:
            self._left_hand_pos_f = self.left_hand_pos
        else:
            lx = alpha_pos * self.left_hand_pos[0] + (1 - alpha_pos) * self._left_hand_pos_f[0]
            ly = alpha_pos * self.left_hand_pos[1] + (1 - alpha_pos) * self._left_hand_pos_f[1]
            self._left_hand_pos_f = (lx, ly)

        if self._right_hand_pos_f is None:
            self._right_hand_pos_f = self.right_hand_pos
        else:
            rx = alpha_pos * self.right_hand_pos[0] + (1 - alpha_pos) * self._right_hand_pos_f[0]
            ry = alpha_pos * self.right_hand_pos[1] + (1 - alpha_pos) * self._right_hand_pos_f[1]
            self._right_hand_pos_f = (rx, ry)

    def _update_steering(self):
        """Calculate and apply steering using rotation angle and virtual joystick zones."""
        if self.left_hand_pos is None or self.right_hand_pos is None:
            return

        
        lpos = self._left_hand_pos_f or self.left_hand_pos
        rpos = self._right_hand_pos_f or self.right_hand_pos
        
        
        dx = rpos[0] - lpos[0]  
        dy = rpos[1] - lpos[1]  
        
        
        rotation_angle = math.degrees(math.atan2(dy, dx))
        
        
        
        
        if rotation_angle > 90:
            rotation_angle = 180 - rotation_angle
        elif rotation_angle < -90:
            rotation_angle = -180 - rotation_angle

        
        rotation_angle = -rotation_angle

        print(f"🎮 Rotation angle (swapped): {rotation_angle:.1f}°, Current zone: {self._steering_zone}")
        
        
        target_zone = self._calculate_target_zone(rotation_angle)
        
        
        if target_zone != self._steering_zone:
            if self._should_change_zone(rotation_angle, target_zone):
                self._steering_zone = target_zone
                print(f"🎮 Zone changed to: {self._steering_zone}")
        
        
        self.steering_angle = self._zone_values[self._steering_zone]
        
        
        if self._steering_zone in ['hard_left', 'soft_left']:
            self._queue_key_press('left')
            self._queue_key_release('right')
            print(f"🎮 STEERING LEFT: zone={self._steering_zone}, angle={self.steering_angle}")
        elif self._steering_zone in ['hard_right', 'soft_right']:
            self._queue_key_press('right')
            self._queue_key_release('left')
            print(f"🎮 STEERING RIGHT: zone={self._steering_zone}, angle={self.steering_angle}")
        else:  
            self._queue_key_release('left')
            self._queue_key_release('right')
            print(f"🎮 NEUTRAL: zone={self._steering_zone}")

    def _calculate_target_zone(self, angle: float) -> str:
        """Calculate which zone the rotation angle should be in."""
        abs_angle = abs(angle)
        
        
        if abs_angle >= self._zone_thresholds['hard_left_enter']:
            return 'hard_left' if angle < 0 else 'hard_right'
        
        
        elif abs_angle >= self._zone_thresholds['soft_left_enter']:
            return 'soft_left' if angle < 0 else 'soft_right'
        
        
        elif abs_angle <= self._zone_thresholds['neutral_dead_zone']:
            return 'neutral'
        
        
        return self._steering_zone

    def _should_change_zone(self, angle: float, target_zone: str) -> bool:
        """Determine if we should actually change zones based on hysteresis."""
        current_zone = self._steering_zone
        abs_angle = abs(angle)
        
        
        if current_zone == 'neutral':
            
            if target_zone in ['soft_left', 'soft_right']:
                return abs_angle >= self._zone_thresholds['soft_left_enter']
            elif target_zone in ['hard_left', 'hard_right']:
                return abs_angle >= self._zone_thresholds['hard_left_enter']
        
        
        elif target_zone == 'neutral':
            
            return abs_angle <= (self._zone_thresholds['neutral_dead_zone'] - self._zone_hysteresis)
        
        
        elif current_zone in ['soft_left', 'soft_right'] and target_zone in ['hard_left', 'hard_right']:
            
            return abs_angle >= (self._zone_thresholds['hard_left_enter'] + self._zone_hysteresis)
        
        elif current_zone in ['hard_left', 'hard_right'] and target_zone in ['soft_left', 'soft_right']:
            
            return abs_angle <= (self._zone_thresholds['hard_left_enter'] - self._zone_hysteresis)
        
        
        return True

    def _update_throttle_brake(self):
        """Calculate and apply acceleration or braking with confidence weighting."""
        if self.left_hand_pos is None or self.right_hand_pos is None:
            return

        
        lpos = self._left_hand_pos_f or self.left_hand_pos
        rpos = self._right_hand_pos_f or self.right_hand_pos
        avg_y = (lpos[1] + rpos[1]) / 2.0

        
        confidence_weight = min(self._hand_confidences) / 0.8
        min_confidence_for_throttle = self.params.get('min_confidence_for_throttle', 0.4)
        
        if confidence_weight < min_confidence_for_throttle:
            
            self.is_accelerating = False
            self.is_braking = False
            self._queue_key_release('w')
            self._queue_key_release('s')
            return

        
        if avg_y < self.control_lines['accelerate_y']:
            self.is_accelerating = True
            self.is_braking = False
            self._queue_key_press('w')
            self._queue_key_release('s')
        elif avg_y > self.control_lines['brake_y']:
            self.is_accelerating = False
            self.is_braking = True
            self._queue_key_press('s')
            self._queue_key_release('w')
        else:
            self.is_accelerating = False
            self.is_braking = False
            self._queue_key_release('w')
            self._queue_key_release('s')

    def _queue_key_press(self, key: str):
        """Queue a key press for batch processing."""
        self._pending_key_changes['press'].add(key)
        self._pending_key_changes['release'].discard(key)  

    def _queue_key_release(self, key: str):
        """Queue a key release for batch processing."""
        self._pending_key_changes['release'].add(key)
        self._pending_key_changes['press'].discard(key)  

    def _apply_key_changes(self):
        """Apply all queued key changes in a batch."""
        
        for key in self._pending_key_changes['release']:
            self._release_key(key)
        
        for key in self._pending_key_changes['press']:
            self._press_key(key)

    def _neutralize_controls(self):
        """Release all game control keys and reset state."""
        self._queue_key_release('left')
        self._queue_key_release('right')
        self._queue_key_release('w')
        self._queue_key_release('s')
        
        self.left_hand_pos = None
        self.right_hand_pos = None
        self.steering_angle = 0.0
        self.is_accelerating = False
        self.is_braking = False
        self._steering_zone = 'neutral'
        self._steering_history.clear()
        self._hands_lost_ts = None

    def _press_key(self, key: str):
        """Press and hold a key if not already pressed."""
        if key not in self.pressed_keys:
            try:
                if DIRECTINPUT_AVAILABLE:
                    pydirectinput.keyDown(key)
                else:
                    import pyautogui
                    pyautogui.keyDown(key)
                self.pressed_keys.add(key)
            except Exception as e:
                print(f"❌ Error pressing key {key}: {e}")

    def _release_key(self, key: str):
        """Release a held key if it is pressed."""
        if key in self.pressed_keys:
            try:
                if DIRECTINPUT_AVAILABLE:
                    pydirectinput.keyUp(key)
                else:
                    import pyautogui
                    pyautogui.keyUp(key)
                self.pressed_keys.discard(key)
            except Exception as e:
                print(f"❌ Error releasing key {key}: {e}")
    
    def _release_all_keys(self):
        """Release all currently held keys."""
        
        keys_to_release = list(self.pressed_keys)
        for key in keys_to_release:
            self._release_key(key)
        self.pressed_keys.clear()
    
    def get_overlay_data(self) -> Dict[str, Any]:
        """Get data for rendering the game overlay."""
        if not self.active:
            return {}

        return {
            'control_lines': self.control_lines,
            'left_hand_pos': self.left_hand_pos,  
            'right_hand_pos': self.right_hand_pos,  
            'left_hand_pos_smoothed': self._left_hand_pos_f,  
            'right_hand_pos_smoothed': self._right_hand_pos_f,  
            'steering_angle': self.steering_angle,
            'steering_zone': self._steering_zone,
            'is_accelerating': self.is_accelerating,
            'is_braking': self.is_braking,
        }

_game_controller = None
def get_game_controller(params: Optional[Dict[str, Any]] = None) -> Optional[GameController]:
    """Singleton accessor for the GameController instance."""
    global _game_controller
    if _game_controller is None and params:
        _game_controller = GameController(params)
    return _game_controller
