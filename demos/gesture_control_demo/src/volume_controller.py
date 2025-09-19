from ctypes import cast, POINTER
import numpy as np


try:
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    PYCAW_AVAILABLE = True
except (ImportError, OSError):
    PYCAW_AVAILABLE = False
    print("⚠️ pycaw library not found or failed to initialize. Volume control will be disabled.")
    print("   To enable, please run: pip install pycaw")

class VolumeController:
    """A cross-platform system volume controller using pycaw."""
    def __init__(self):
        self.is_valid = False
        self.volume = None
        self.min_vol = -65.25  
        self.max_vol = 0.0     

        if PYCAW_AVAILABLE:
            try:
                devices = AudioUtilities.GetSpeakers()
                interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                self.volume = cast(interface, POINTER(IAudioEndpointVolume))
                self.min_vol, self.max_vol, _ = self.volume.GetVolumeRange()
                self.is_valid = True
                print("✅ Volume controller initialized successfully.")
            except Exception as e:
                print(f"❌ Could not initialize volume controller (pycaw): {e}")

    def set_volume(self, level_percent: float):
        """Sets master volume to a percentage (0-100)."""
        if not self.is_valid: return
        
        
        scalar_volume = self.min_vol + (self.max_vol - self.min_vol) * (level_percent / 100.0)
        self.volume.SetMasterVolumeLevel(scalar_volume, None)

    def get_volume(self) -> float:
        """Gets current master volume as a percentage (0-100)."""
        if not self.is_valid: return 0.0
        
        current_scalar = self.volume.GetMasterVolumeLevel()
        
        level_percent = ((current_scalar - self.min_vol) / (self.max_vol - self.min_vol)) * 100.0
        return level_percent

    def change_volume(self, change_amount_percent: float):
        """Changes volume by a relative percentage amount (+ or -)."""
        if not self.is_valid: return
        
        current_level = self.get_volume()
        new_level = np.clip(current_level + change_amount_percent, 0, 100)
        self.set_volume(new_level)


_volume_controller_instance = None

def get_volume_controller():
    """Provides access to the singleton VolumeController instance."""
    global _volume_controller_instance
    if _volume_controller_instance is None:
        _volume_controller_instance = VolumeController()
    return _volume_controller_instance