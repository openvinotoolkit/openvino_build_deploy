from PySide6.QtWidgets import QApplication
import json
import os
import sys
import time
import traceback
from pathlib import Path

# Add current directory to Python path for module discovery
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# === PyInstaller Hidden Import Fix ===
# These imports help PyInstaller discover all modules during static analysis
# to eliminate "ERROR: Hidden import '...' not found" warnings
try:
    # Utils modules
    import utils.annotation_utils
    import utils.helpers
    import utils.enhanced_annotation_utils
    import utils.data_publisher
    import utils.mqtt_publisher
    import utils.traffic_light_utils
    import utils.scene_analytics
    import utils.crosswalk_utils
    import utils.crosswalk_utils2
    import utils.enhanced_tracker
    import utils.embedder_openvino
    
    # Controllers modules
    import openvino_build_deploy.apps.traffic_monitor.app.controllers.video_controller
    import controllers.model_manager
    import controllers.analytics_controller
    import controllers.performance_overlay
    import controllers.vlm_controller
    import controllers.smart_intersection_controller
    
    # UI modules
    import ui.main_window
    import ui.analytics_tab
    import ui.violations_tab
    import ui.export_tab
    import ui.modern_config_panel
    import ui.modern_live_detection_tab
    
    print("✅ All hidden imports loaded successfully for PyInstaller")
except ImportError as e:
    print(f"⚠️ Some modules not available during import: {e}")
# === End PyInstaller Fix ===

print("=== DEBUG INFO ===")
print(f"Python executable: {sys.executable}")
print(f"Current working dir: {os.getcwd()}")
print(f"Script location: {os.path.dirname(os.path.abspath(__file__))}")
print(f"sys.path: {sys.path[:3]}...")  # First 3 paths
print("=== STARTING APP ===")

def main():
    # Create application instance first
    app = QApplication.instance() or QApplication(sys.argv)
    
    # Show splash screen if available
    splash = None
    try:
        from splash import show_splash
        result = show_splash(app)
        if result:
            splash, app = result
            if splash is None:
                print("No splash image found, continuing without splash")
    except Exception as e:
        print(f"Could not show splash screen: {e}")

    # Add a short delay to show the splash screen
    if splash:
        print("[DEBUG] Splash screen shown, sleeping for 0.2s (reduced)")
        time.sleep(0.2)

    try:
        # Load standard MainWindow
        from ui.main_window import MainWindow
        print("✅ Using standard MainWindow")
    except Exception as e:
        print(f"❌ Could not load MainWindow: {e}")
        sys.exit(1)

    try:
        print("[DEBUG] Instantiating MainWindow...")
        # Initialize main window
        window = MainWindow()
        print("[DEBUG] MainWindow instantiated.")
        # Close splash if it exists
        if splash:
            print("[DEBUG] Closing splash screen.")
            splash.finish(window)
        # Show main window
        print("[DEBUG] Showing main window.")
        window.show()
        # Start application event loop
        print("[DEBUG] Entering app.exec() loop.")
        sys.exit(app.exec())
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
