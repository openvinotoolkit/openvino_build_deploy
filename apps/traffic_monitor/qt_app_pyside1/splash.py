from PySide6.QtWidgets import QApplication, QSplashScreen
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap
import sys
import os

def show_splash(existing_app=None):
    # Use existing app if provided, otherwise create a new one
    app = existing_app or QApplication(sys.argv)
    
    # Get the directory of the executable or script
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        app_dir = os.path.dirname(sys.executable)
    else:
        # Running as script
        app_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Look for splash image
    splash_image = os.path.join(app_dir, 'resources', 'splash.png')
    if not os.path.exists(splash_image):
        splash_image = os.path.join(app_dir, 'splash.png')
        if not os.path.exists(splash_image):
            print("No splash image found, skipping splash screen")
            return None, app
    
    # Create splash screen
    pixmap = QPixmap(splash_image)
    splash = QSplashScreen(pixmap, Qt.WindowStaysOnTopHint)
    splash.show()
    app.processEvents()
    
    return splash, app

if __name__ == "__main__":
    # This is for testing the splash screen independently
    splash, app = show_splash()
    
    # Close the splash after 3 seconds
    QTimer.singleShot(3000, splash.close)
    
    sys.exit(app.exec())
