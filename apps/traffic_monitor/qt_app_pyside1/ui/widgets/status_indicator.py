"""
Status Indicator Widget - Modern status display with animations
"""

from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QFrame
from PySide6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, Property
from PySide6.QtGui import QPainter, QColor, QFont

class StatusIndicator(QWidget):
    """
    Modern status indicator with animated status changes
    
    States:
    - running (green pulse)
    - stopped (red solid)
    - warning (yellow blink)
    - error (red blink)
    - loading (blue pulse)
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setFixedSize(24, 24)
        self._status = "stopped"
        self._opacity = 1.0
        
        # Animation for pulsing/blinking effects
        self.animation = QPropertyAnimation(self, b"opacity")
        self.animation.setDuration(1000)
        self.animation.setEasingCurve(QEasingCurve.InOutQuad)
        
        # Status colors
        self.colors = {
            'running': QColor('#27ae60'),    # Green
            'stopped': QColor('#e74c3c'),    # Red
            'warning': QColor('#f39c12'),    # Orange
            'error': QColor('#c0392b'),      # Dark red
            'loading': QColor('#3498db'),    # Blue
        }
        
        print("✅ Status Indicator initialized")
    
    def get_opacity(self):
        return self._opacity
    
    def set_opacity(self, opacity):
        self._opacity = opacity
        self.update()
    
    opacity = Property(float, get_opacity, set_opacity)
    
    def set_status(self, status):
        """Set the status and start appropriate animation"""
        if status not in self.colors:
            status = 'stopped'
        
        self._status = status
        
        # Configure animation based on status
        if status == 'running':
            # Slow pulse for running
            self.animation.setStartValue(0.3)
            self.animation.setEndValue(1.0)
            self.animation.setLoopCount(-1)  # Infinite
            self.animation.start()
        elif status in ['warning', 'error']:
            # Fast blink for alerts
            self.animation.setStartValue(0.2)
            self.animation.setEndValue(1.0)
            self.animation.setDuration(500)
            self.animation.setLoopCount(-1)
            self.animation.start()
        elif status == 'loading':
            # Medium pulse for loading
            self.animation.setStartValue(0.4)
            self.animation.setEndValue(1.0)
            self.animation.setDuration(800)
            self.animation.setLoopCount(-1)
            self.animation.start()
        else:
            # Solid color for stopped
            self.animation.stop()
            self._opacity = 1.0
            self.update()
    
    def paintEvent(self, event):
        """Custom paint event for the status indicator"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Get current color
        color = self.colors.get(self._status, self.colors['stopped'])
        color.setAlphaF(self._opacity)
        
        # Draw the status circle
        painter.setBrush(color)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(2, 2, 20, 20)
        
        # Add inner highlight for 3D effect
        highlight = QColor(255, 255, 255, int(80 * self._opacity))
        painter.setBrush(highlight)
        painter.drawEllipse(4, 4, 8, 8)
