import json
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict, field
from pathlib import Path
from game_controller import get_game_controller

@dataclass
class DetectionConfig:
    """Core detection parameters"""
    input_size: int = 192
    score_threshold: float = 0.5
    nms_threshold: float = 0.4
    num_hands: int = 2
    enable_dynamic_box_expansion: bool = True
    dynamic_box_expansion_margin: float = 0.2
    dynamic_box_expansion_factor: float = 1.15
    dynamic_box_shrink_factor: float = 0.99
    dynamic_box_default_scale: float = 2.6
    dynamic_box_max_scale: float = 4.0
    smoothing_alpha: float = 0.6
    iou_match_threshold: float = 0.3
    detection_smoothing_alpha: float = 0.5
    gesture_smoothing_frames: int = 8
    landmark_score_for_palm_redetection_threshold: float = 0.5
    always_run_palm_detection: bool = False
    show_landmarks: bool = True
    show_static_gestures: bool = True
    enable_finger_detection: bool = True
    bend_angle_threshold: int = 160
    # Device assignment fields
    enable_device_switching: bool = True
    model_devices: Dict[str, str] = field(default_factory=lambda: {
        'palm_detection': 'CPU',
        'hand_landmarks': 'CPU',
        'gesture_embedder': 'CPU',
        'gesture_classifier': 'CPU'
    })
    fallback_device: str = 'CPU'

@dataclass
class SmartPalmDetectionConfig:
    """Smart palm detection state machine settings"""
    grace_period_duration: float = 0.3
    periodic_check_interval: int = 50
    state_transition_debug: bool = False

@dataclass
class ControlSystemConfig:
    """Control system settings"""
    enable_cursor_control: bool = False
    cursor_smoothing: float = 0.7
    cursor_sensitivity: float = 2.5
    screen_width: int = 1920
    screen_height: int = 1080

    enable_volume_control: bool = True
    volume_control_hand: str = 'any'
    volume_sensitivity: float = 1.5
    pinch_threshold_start: float = 0.1
    pinch_threshold_stop: float = 0.15
    volume_change_cooldown: float = 0.05

    enable_scroll_control: bool = False
    scroll_sensitivity: int = 6
    scroll_threshold: float = 0.1
    scroll_smoothing: float = 0.6
    scroll_hand_preference: str = 'any'

    enable_key_control: bool = False
    key_press_cooldown: float = 0.6

    enable_game_control: bool = True
    game_control_type: str = 'keyboard'

    steering_max_tilt: int = 200
    steering_sensitivity: float = 0.7
    steering_deadzone: float = 0.07
    accelerate_line_y: float = 0.28
    brake_line_y: float = 0.75

    steering_box_width: float = 0.1
    steering_box_height: float = 0.1
    steering_box_x: float = 0.01
    steering_box_y: float = 0.25
    steering_smoothing: float = 0.2
    steering_exponent: float = 1.0
    steering_displacement_amplification: float = 4.0

    open_palm_threshold: float = 0.5
    game_gesture_cooldown: float = 0.2

    hand_label_offset_x: float = 55.0
    hand_label_offset_y: float = 150.0
    left_hand_offset_x: float = 15.0
    left_hand_offset_y: float = 15.0
    right_hand_offset_x: float = 200.0
    right_hand_offset_y: float = 20.0

@dataclass
class GestureDefinition:
    """Individual gesture configuration"""
    name: str
    description: str
    detection_type: str
    hand: str
    enabled: bool = True
    mapped_key: str = 'space'
    cooldown: float = 0.6
    last_triggered: float = 0.0

@dataclass
class GestureMappingConfig:
    """Gesture mapping system configuration"""
    enable_gesture_mapping: bool = True
    global_cooldown: float = 0.1
    last_any_gesture_time: float = 0.0
    
    
    relaxed_threshold: int = 60
    bent_threshold: int = 160
    
    
    require_simultaneous_detection: bool = True
    gesture_stability_frames: int = 2
      
    available_keys: list = field(default_factory=lambda: [
        
        'space', 'enter', 'tab', 'escape', 'backspace', 'delete',
        'left', 'right', 'up', 'down', 'pageup', 'pagedown', 'home', 'end', 'insert',
        
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
        'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
        
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        
        'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12',
        
        'ctrl', 'alt', 'shift', 'win', 'cmd', 'command', 'option'
    ])
    
    
    gesture_definitions: Dict[str, GestureDefinition] = field(default_factory=dict)

@dataclass
class ApplicationModeGesture:
    """Application mode gesture configuration"""
    action: str  
    key: Optional[str] = None
    button: Optional[str] = None
    description: str = ''
    cooldown: float = 0.8

@dataclass
class ApplicationModeConfig:
    """Single application mode configuration"""
    name: str
    enabled: bool = False
    gestures: Dict[str, ApplicationModeGesture] = field(default_factory=dict)

@dataclass
class ApplicationModesConfig:
    """Application modes system configuration"""
    current_mode: str = 'volume_mode'
    mode_switch_cooldown: float = 2.0
    last_mode_switch: float = 0.0
    debug_mode: bool = True
    gesture_timings: Dict[str, float] = field(default_factory=dict)
    
    
    ppt_mode: ApplicationModeConfig = field(default_factory=lambda: ApplicationModeConfig(name='PowerPoint Mode'))
    media_mode: ApplicationModeConfig = field(default_factory=lambda: ApplicationModeConfig(name='Media Player Mode'))
    browser_mode: ApplicationModeConfig = field(default_factory=lambda: ApplicationModeConfig(name='Browser Mode'))
    
    game_mode: Optional[ApplicationModeConfig] = None
    volume_mode: ApplicationModeConfig = field(default_factory=lambda: ApplicationModeConfig(name='Volume Control Mode')) 
    
    browser_right_hand_mode: str = 'cursor'  
    
    browser_iloveyou_switch_cooldown: float = 1.0
    browser_last_iloveyou_switch: float = 0.0

class ConfigurationManager:
    """Central configuration management with save/load capabilities"""
    
    def __init__(self, config_file: str = "gesture_config.json"):
        self.config_file = Path(config_file)
        self.observers = []  
        
        
        self.detection = DetectionConfig()
        self.smart_palm = SmartPalmDetectionConfig()
        self.control_system = ControlSystemConfig()
        self.gesture_mapping = GestureMappingConfig()
        self.app_modes = ApplicationModesConfig()
        
        
        self._initialize_default_gestures()
        self._initialize_default_app_modes()
        
        
        config_existed = self.config_file.exists()
        if config_existed:
            self.load_config()

                
        if self.app_modes.game_mode is None:
            print("🔧 'game_mode' is missing from config. Adding default and saving.")
            self._initialize_default_game_mode() 
            self.save_config() 
        elif not config_existed:
            
            print("🔧 No config file found. Creating one with all defaults.")
            self.save_config()
    

    def _initialize_default_game_mode(self):
        """Initializes ONLY the default game mode configuration."""
        self.app_modes.game_mode = ApplicationModeConfig(name='Game Mode (Racing)')
        self.app_modes.game_mode.gestures = {
            'left_index_bent': ApplicationModeGesture(
                action='key_press', key='x',
                description='Speedbreaker/Handbrake', cooldown=0.2
            ),
            'left_index_middle_bent': ApplicationModeGesture(
                action='key_press', key='z', 
                description='Brake', cooldown=0.1
            ),
            'fist_gesture': ApplicationModeGesture(
                action='key_press', key='shift',
                description='Nitrous', cooldown=0.3
            )
        }

    def _initialize_default_gestures(self):
        """Initialize default gesture definitions to match required JSON"""
        default_gestures = {
            'left_index_bent': GestureDefinition(
                name='Left Index Finger Bent',
                description='Left hand index finger bent',
                detection_type='finger_angle',
                hand='left',
                enabled=True,
                mapped_key='left',
                cooldown=0.6,
                last_triggered=0.0
            ),
            'right_index_bent': GestureDefinition(
                name='Right Index Finger Bent',
                description='Right hand index finger bent',
                detection_type='finger_angle',
                hand='right',
                enabled=True,
                mapped_key='right',
                cooldown=0.6,
                last_triggered=0.0
            ),
            'left_index_middle_bent': GestureDefinition(
                name='Left Index + Middle Bent',
                description='Left hand index and middle fingers bent',
                detection_type='finger_angle',
                hand='left',
                enabled=True,
                mapped_key='up',
                cooldown=0.6,
                last_triggered=0.0
            ),
            'right_index_middle_bent': GestureDefinition(
                name='Right Index + Middle Bent',
                description='Right hand index and middle fingers bent',
                detection_type='finger_angle',
                hand='right',
                enabled=True,
                mapped_key='down',
                cooldown=0.6,
                last_triggered=0.0
            ),
            'fist_gesture': GestureDefinition(
                name='Closed Fist',
                description='closed fist',
                detection_type='mediapipe_static',
                hand='any',
                enabled=True,
                mapped_key='space',
                cooldown=1.0,
                last_triggered=0.0
            ),
            'open_palm_gesture': GestureDefinition(
                name='Open Palm',
                description='Open palm (stop/cancel gesture)',
                detection_type='mediapipe_static',
                hand='any',
                enabled=True,
                mapped_key='escape',
                cooldown=1.0,
                last_triggered=0.0
            ),
            'iloveyou_gesture': GestureDefinition(
                name='I Love You Sign',
                description='ASL I Love You sign (thumb, index, pinky extended)',
                detection_type='mediapipe_static',
                hand='any',
                enabled=True,
                mapped_key='ctrl+s',
                cooldown=1.5,
                last_triggered=0.0
            )
        }
        self.gesture_mapping.gesture_definitions = default_gestures
    
    def _initialize_default_app_modes(self):
        """Initialize default application mode configurations to match required JSON"""
        self.app_modes.ppt_mode.gestures = {
            'right_index_bent': ApplicationModeGesture(
                action='key_press', key='right',
                description='Next slide', cooldown=0.8
            ),
            'left_index_bent': ApplicationModeGesture(
                action='key_press', key='left',
                description='Previous slide', cooldown=0.8
            ),
            'fist_gesture': ApplicationModeGesture(
                action='key_press', key='f5',
                description='Start slideshow', cooldown=2.0
            )
        }

        self.app_modes.media_mode.gestures = {
            'right_index_bent': ApplicationModeGesture(
                action='key_press', key='space',
                description='Play/Pause', cooldown=0.6
            ),
            'right_index_middle_bent': ApplicationModeGesture(
                action='key_press', key='right',
                description='Skip 10 secs forward', cooldown=0.3
            ),
            'left_index_middle_bent': ApplicationModeGesture(
                action='key_press', key='left',
                description='Skip 10 secs back', cooldown=0.3
            ),
            'left_index_bent': ApplicationModeGesture(
                action='key_press', key='m',
                description='Mute', cooldown=0.8
            ),
            'fist_gesture': ApplicationModeGesture(
                action='key_press', key='f',
                description='Fullscreen', cooldown=1.5
            )
        }

        self.app_modes.browser_mode.gestures = {
            'left_index_bent': ApplicationModeGesture(
                action='mouse_click', button='left',
                description='Left click', cooldown=0.4
            ),
            'left_index_middle_bent': ApplicationModeGesture(
                action='mouse_click', button='right',
                description='Right click', cooldown=0.6
            ),
            'fist_gesture': ApplicationModeGesture(
                action='key_press', key='win+h',
                description='Speech to text (Win + H)', cooldown=1.0
            ),
            'iloveyou_gesture': ApplicationModeGesture(
                action='key_press', key='ctrl+shift+tab',
                description='Switch to next browser tab (left-hand ILoveYou)', cooldown=0.5
            )
        }

        self.app_modes.game_mode = ApplicationModeConfig(name='Game Mode (Racing)')
        self.app_modes.game_mode.gestures = {
            'left_index_bent': ApplicationModeGesture(
                action='key_press', key='x',
                description='Speedbreaker/Handbrake', cooldown=0.1
            ),
            'left_index_middle_bent': ApplicationModeGesture(
                action='key_press', key='w',
                description='Accelerate', cooldown=0.1
            ),
            'fist_gesture': ApplicationModeGesture(
                action='key_press', key='shift',
                description='Nitrous', cooldown=0.3
            )
        }

        self.app_modes.volume_mode = ApplicationModeConfig(name='Volume Control Mode', enabled=True, gestures={})

        self.app_modes.browser_right_hand_mode = 'cursor'
        self.app_modes.browser_iloveyou_switch_cooldown = 1.0
        self.app_modes.browser_last_iloveyou_switch = 1756369598.924626

        self.app_modes.current_mode = 'volume_mode'
        self.app_modes.mode_switch_cooldown = 2.0
        self.app_modes.last_mode_switch = 1756369711.155566
        self.app_modes.debug_mode = True
        self.app_modes.gesture_timings = {
            'left_index_bent': 1756369694.0063186,
            'left_index_middle_bent': 1756369705.6043553,
            'right_index_bent': 1756369683.6069806,
            'right_index_middle_bent': 1756369698.7675455,
            'fist_gesture': 1756369707.4752407,
            'open_palm_gesture': 1756042844.102283,
            'iloveyou_gesture': 1756369596.5354419
        }
    
    def save_config(self) -> bool:
        """Save current configuration to file"""
        try:
            config_dict = {
                'detection': asdict(self.detection),
                'smart_palm': asdict(self.smart_palm),
                'control_system': asdict(self.control_system),
                'gesture_mapping': asdict(self.gesture_mapping),
                'app_modes': asdict(self.app_modes)
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(config_dict, f, indent=4)
            
            print(f"✅ Configuration saved to {self.config_file}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to save configuration: {e}")
            return False
    
    def load_config(self) -> bool:
        """Load configuration from file"""
        try:
            with open(self.config_file, 'r') as f:
                config_dict = json.load(f)
            
            
            if 'detection' in config_dict:
                self.detection = DetectionConfig(**config_dict['detection'])
            
            if 'smart_palm' in config_dict:
                self.smart_palm = SmartPalmDetectionConfig(**config_dict['smart_palm'])
            
            if 'control_system' in config_dict:
                self.control_system = ControlSystemConfig(**config_dict['control_system'])
            
            if 'gesture_mapping' in config_dict:
                gm_dict = config_dict['gesture_mapping']
                
                if 'gesture_definitions' in gm_dict:
                    gesture_defs = {}
                    for key, value in gm_dict['gesture_definitions'].items():
                        gesture_defs[key] = GestureDefinition(**value)
                    gm_dict['gesture_definitions'] = gesture_defs
                
                self.gesture_mapping = GestureMappingConfig(**gm_dict)
            
            if 'app_modes' in config_dict:
                am_dict = config_dict['app_modes']
                
                
                self.app_modes = ApplicationModesConfig()
                
                
                for mode_key, mode_data in am_dict.items():
                    if mode_key.endswith('_mode') and isinstance(mode_data, dict):
                        
                        if 'gestures' in mode_data:
                            gestures = {}
                            for key, value in mode_data['gestures'].items():
                                gestures[key] = ApplicationModeGesture(**value)
                            mode_data['gestures'] = gestures
                        
                        
                        mode_config = ApplicationModeConfig(**mode_data)
                        setattr(self.app_modes, mode_key, mode_config)
                        print(f"✅ Loaded mode: {mode_key} - {mode_config.name}")
                
                
                for key, value in am_dict.items():
                    if not key.endswith('_mode') and hasattr(self.app_modes, key):
                        setattr(self.app_modes, key, value)
            
            print(f"✅ Configuration loaded from {self.config_file}")
            self._notify_observers('config_loaded')
            return True
            
        except Exception as e:
            print(f"❌ Failed to load configuration: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def reset_to_defaults(self):
        """Reset all configuration to default values"""
        self.detection = DetectionConfig()
        self.smart_palm = SmartPalmDetectionConfig()
        self.control_system = ControlSystemConfig()
        self.gesture_mapping = GestureMappingConfig()
        self.app_modes = ApplicationModesConfig()
        
        self._initialize_default_gestures()
        self._initialize_default_app_modes()
        
        self._notify_observers('config_reset')
        print("🔄 Configuration reset to defaults")
    
    def add_observer(self, callback):
        """Add observer for configuration changes"""
        self.observers.append(callback)
    
    def remove_observer(self, callback):
        """Remove observer"""
        if callback in self.observers:
            self.observers.remove(callback)
    
    def _notify_observers(self, event_type: str, data: Any = None):
        """Notify all observers of configuration changes"""
        for callback in self.observers:
            try:
                callback(event_type, data)
            except Exception as e:
                print(f"Error notifying observer: {e}")
    
    def update_gesture_key_mapping(self, gesture_id: str, new_key: str):
        """Update key mapping for a specific gesture"""
        if gesture_id in self.gesture_mapping.gesture_definitions:
            self.gesture_mapping.gesture_definitions[gesture_id].mapped_key = new_key
            self._notify_observers('gesture_mapping_changed', {'gesture_id': gesture_id, 'new_key': new_key})
    
    def toggle_gesture_enabled(self, gesture_id: str):
        """Toggle enabled state for a gesture"""
        if gesture_id in self.gesture_mapping.gesture_definitions:
            gesture = self.gesture_mapping.gesture_definitions[gesture_id]
            gesture.enabled = not gesture.enabled
            self._notify_observers('gesture_enabled_changed', {'gesture_id': gesture_id, 'enabled': gesture.enabled})
    
    def get_legacy_params_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to legacy params dict format for backward compatibility.
        FIXED: This now correctly includes custom modes and handles dynamic attributes.
        """
        params = {}
        
        
        params.update(asdict(self.detection))
        
        
        params.update({
            'palm_detection_state': 'NO_HANDS',
            'grace_period_start': 0,
            'grace_period_duration': self.smart_palm.grace_period_duration,
            'last_hand_count': 0,
            'periodic_check_counter': 0,
            'periodic_check_interval': self.smart_palm.periodic_check_interval,
            'state_transition_debug': self.smart_palm.state_transition_debug,
        })
                
        params['scroll_state'] = {
            'is_scrolling': False,
            'start_pos': None,
            'locked_direction': None,
            'active_hand': None,
            'last_scroll_time': 0
        }
        
        
        
        params.update(asdict(self.control_system))
        params['previous_cursor_pos'] = None
        params['previous_scroll_pos'] = None
        params['last_key_press_time'] = 0
        params['last_pressed_hand'] = None
        
        
        params['previous_frame_processed_regions'] = []
        params['gesture_history'] = {}
        
        
        gesture_mapping_dict = {
            'enable_gesture_mapping': self.gesture_mapping.enable_gesture_mapping,
            'global_cooldown': self.gesture_mapping.global_cooldown,
            'last_any_gesture_time': self.gesture_mapping.last_any_gesture_time,
            'relaxed_threshold': self.gesture_mapping.relaxed_threshold,
            'bent_threshold': self.gesture_mapping.bent_threshold,
            'require_simultaneous_detection': self.gesture_mapping.require_simultaneous_detection,
            'gesture_stability_frames': self.gesture_mapping.gesture_stability_frames,
            'available_keys': list(self.gesture_mapping.available_keys),
            'gesture_definitions': {}
        }
        
        
        for key, gesture_def in self.gesture_mapping.gesture_definitions.items():
            gesture_mapping_dict['gesture_definitions'][key] = {
                'name': gesture_def.name,
                'description': gesture_def.description,
                'detection_type': gesture_def.detection_type,
                'hand': gesture_def.hand,
                'enabled': gesture_def.enabled,
                'mapped_key': gesture_def.mapped_key,
                'cooldown': gesture_def.cooldown,
                'last_triggered': gesture_def.last_triggered
            }
        
        params['gesture_mapping'] = gesture_mapping_dict
        
        
        app_modes_dict = {
            'current_mode': self.app_modes.current_mode,
            'mode_switch_cooldown': self.app_modes.mode_switch_cooldown,
            'last_mode_switch': self.app_modes.last_mode_switch,
            'debug_mode': self.app_modes.debug_mode,
            'gesture_timings': dict(self.app_modes.gesture_timings),
            'browser_right_hand_mode': self.app_modes.browser_right_hand_mode,
            'browser_iloveyou_switch_cooldown': self.app_modes.browser_iloveyou_switch_cooldown,
            'browser_last_iloveyou_switch': self.app_modes.browser_last_iloveyou_switch,
        }
        
        
        for mode_key in dir(self.app_modes):
            if mode_key.endswith('_mode') and not mode_key.startswith('_'):
                mode_config = getattr(self.app_modes, mode_key)
                
                if isinstance(mode_config, ApplicationModeConfig):
                    
                    mode_dict = {
                        'name': mode_config.name,
                        'enabled': mode_config.enabled,
                        'gestures': {}
                    }
                    
                    
                    for gesture_key, gesture_obj in mode_config.gestures.items():
                        mode_dict['gestures'][gesture_key] = {
                            'action': gesture_obj.action,
                            'key': gesture_obj.key,
                            'button': gesture_obj.button,
                            'description': gesture_obj.description,
                            'cooldown': gesture_obj.cooldown
                        }
                    
                    app_modes_dict[mode_key] = mode_dict
        
        params['app_modes'] = app_modes_dict
        
        return params
    
    def validate_config(self) -> Dict[str, list]:
        """Validate configuration and return any errors/warnings"""
        errors = []
        warnings = []
        
        
        if not 0.0 <= self.detection.score_threshold <= 1.0:
            errors.append("Detection score threshold must be between 0.0 and 1.0")
        
        if not 0.0 <= self.detection.nms_threshold <= 1.0:
            errors.append("NMS threshold must be between 0.0 and 1.0")
        
        if self.control_system.screen_width <= 0 or self.control_system.screen_height <= 0:
            errors.append("Screen dimensions must be positive")
        
        
        for gesture_id, gesture_def in self.gesture_mapping.gesture_definitions.items():
            keys_to_check = [k.strip() for k in gesture_def.mapped_key.split('+')]
            for key in keys_to_check:
                if key not in self.gesture_mapping.available_keys:
                    warnings.append(f"Gesture '{gesture_id}' uses potentially unknown key '{key}' in mapping '{gesture_def.mapped_key}'")
                    break 
        
        
        for mode_key in dir(self.app_modes):
            if mode_key.endswith('_mode') and not mode_key.startswith('_'):
                mode_config = getattr(self.app_modes, mode_key)
                if isinstance(mode_config, ApplicationModeConfig):
                    for gesture_key, gesture_data in mode_config.gestures.items():
                        if gesture_data.action == 'key_press' and gesture_data.key:
                            keys_to_check = [k.strip() for k in gesture_data.key.split('+')]
                            for key in keys_to_check:
                                if key not in self.gesture_mapping.available_keys:
                                    warnings.append(f"Mode '{mode_config.name}', gesture '{gesture_key}' uses potentially unknown key '{key}' in mapping '{gesture_data.key}'")
                                    break 
        
        return {'errors': errors, 'warnings': warnings}


config_manager = ConfigurationManager()