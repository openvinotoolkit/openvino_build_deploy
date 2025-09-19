from PyQt6.QtCore import QObject, pyqtSignal, QThread
from gesture_engine import complete_engine
import cv2
import time

class GestureEngineWorker(QObject):
    """Runs the gesture engine in a separate thread."""
    # Signals to send data back to the GUI
    new_frame = pyqtSignal(object)  # Emits the processed frame (numpy array)
    status_update = pyqtSignal(str) # Emits status messages
    
    def __init__(self):
        super().__init__()
        self.engine = complete_engine
        self._running = False

    def run(self):
        """The main work loop."""
        self._running = True
        
        # Try to initialize the engine if not already done
        try:
            if not self.engine.initialize():
                self.status_update.emit("Engine Failed to Initialize!")
                self._running = False
                return
        except Exception as e:
            self.status_update.emit(f"Engine initialization error: {str(e)}")
            self._running = False
            return
        
        self.engine.start()
        self.status_update.emit("Engine Started")

        while self._running:
            try:
                frame = self.engine.get_frame_with_complete_processing()
                if frame is not None:
                    # Convert BGR (OpenCV) to RGB (Qt)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.new_frame.emit(rgb_frame)
                else:
                    # Handle case where camera disconnects
                    self.status_update.emit("Camera feed lost!")
                    break
            except Exception as e:
                self.status_update.emit(f"Processing error: {str(e)}")
                break
                
            time.sleep(0.01) # Small sleep to yield thread

        self.engine.stop()
        self.status_update.emit("Engine Stopped")

    def stop(self):
        """Stops the loop."""
        self._running = False