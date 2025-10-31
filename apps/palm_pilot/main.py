#!/usr/bin/env python3
"""
PalmPilot - Gesture Control with OpenVINO
Demo launcher with automatic model downloading using demo_utils
"""

import argparse
import sys
import os
from pathlib import Path

# Add src directory to Python path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

# OpenVINO demo utils integration
SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils")
sys.path.append(os.path.dirname(SCRIPT_DIR))

try:
    from utils import demo_utils as utils
except ImportError:
    utils = None
    print("⚠️ demo_utils not available. Model downloading will be limited.")

def download_models() -> bool:
    """Download all required gesture recognition models using demo_utils"""
    
    # Using your actual GitHub repository
    BASE_URL = "https://raw.githubusercontent.com/itachi202/palm-pilot/main"
    
    model_files = [
        "hand_detector.xml",
        "hand_detector.bin", 
        "hand_landmarks_detector.xml",
        "hand_landmarks_detector.bin",
        "gesture_embedder.xml", 
        "gesture_embedder.bin",
        "canned_gesture_classifier.xml",
        "canned_gesture_classifier.bin"
    ]
    
    model_dir = Path("mediapipeModels")
    model_dir.mkdir(exist_ok=True)
    
    print("Checking gesture recognition models...")
    
    # Check if all models exist
    missing_models = []
    for filename in model_files:
        if not (model_dir / filename).exists():
            missing_models.append(filename)
    
    if not missing_models:
        print("✅ All gesture recognition models are present!")
        return True
    
    print(f"📥 Downloading {len(missing_models)} missing models...")
    
    # Download missing models using demo_utils
    if not utils:
        print(" demo_utils not available. Cannot download models automatically.")
        print("   Please manually place model files in 'mediapipeModels' folder.")
        return False
    
    for filename in missing_models:
        try:
            url = f"{BASE_URL}/{filename}"
            print(f"   Downloading {filename}...")
            
            # Use demo_utils.download_file with proper parameters
            downloaded_path = utils.download_file(
                url=url,
                filename=filename,
                directory=model_dir,
                show_progress=True,
                timeout=60
            )
            
            if downloaded_path:
                print(f"    Downloaded {filename}")
            else:
                print(f"   Failed to download {filename}")
                return False
                
        except Exception as e:
            print(f"   Error downloading {filename}: {e}")
            return False
    
    print("✅ All gesture recognition models downloaded successfully!")
    return True

def parse_args():
    parser = argparse.ArgumentParser(
        description="PalmPilot - Gesture Control with OpenVINO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py --stream 0
    python main.py --stream 1
    python main.py --stream webcam.mp4
        """
    )
    
    parser.add_argument(
        '--stream', 
        default="0", 
        type=str, 
        help="Camera stream ID or video file path (default: 0)"
    )
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("🖐️ PalmPilot - Gesture Control with OpenVINO")
    print(f"   Camera: {args.stream}")
    
    # Download models automatically using demo_utils (AUTOMATIC!)
    if not download_models():
        print(" Failed to download required models. Please check:")
        print("   1. Internet connection")
        print("   2. GitHub repository is accessible")
        print("   3. demo_utils is available")
        print("\n   You can manually place model files in 'mediapipeModels' folder:")
        print("   - hand_detector.xml/.bin")
        print("   - hand_landmarks_detector.xml/.bin")
        print("   - gesture_embedder.xml/.bin") 
        print("   - canned_gesture_classifier.xml/.bin")
        return 1
    
    # Set environment variables for the engine
    os.environ['PALMPLIOT_CAMERA_ID'] = args.stream
    
    # Import and launch GUI from src directory
    try:
        from gui_main import GestureDashboard
        from PyQt6.QtWidgets import QApplication
        
        app = QApplication(sys.argv)
        app.setApplicationName("PalmPilot - Gesture Control")
        
        window = GestureDashboard()
        window.show()
        
        return app.exec()
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("   Make sure all required modules are in the src/ directory")
        return 1

if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code or 0)
    except KeyboardInterrupt:
        print("\n Interrupted")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
