from PySide6.QtGui import QIcon, QPixmap, QPainter, QColor, QFont, QBrush, QPen
from PySide6.QtCore import Qt, QSize, QRect
import os

def generate_app_icon(size=512):
    """Generate a simple app icon if none is available"""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing, True)
    
    # Background
    painter.setBrush(QBrush(QColor(40, 120, 200)))
    painter.setPen(Qt.NoPen)
    painter.drawEllipse(10, 10, size-20, size-20)
    
    # Traffic light circle
    painter.setBrush(QBrush(QColor(50, 50, 50)))
    painter.setPen(QPen(QColor(30, 30, 30), 10))
    painter.drawEllipse(size//4, size//4, size//2, size//2)
    
    # Red light
    painter.setBrush(QBrush(QColor(240, 30, 30)))
    painter.setPen(Qt.NoPen)
    painter.drawEllipse(size//2.5, size//3.5, size//5, size//5)
    
    # Yellow light
    painter.setBrush(QBrush(QColor(240, 240, 30)))
    painter.setPen(Qt.NoPen)
    painter.drawEllipse(size//2.5, size//2.3, size//5, size//5)
    
    # Green light
    painter.setBrush(QBrush(QColor(30, 200, 30)))
    painter.setPen(Qt.NoPen)
    painter.drawEllipse(size//2.5, size//1.7, size//5, size//5)
    
    painter.end()
    
    return pixmap

def create_app_icons(output_dir):
    """Create application icons in various formats"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create icons in different sizes
    sizes = [16, 32, 48, 64, 128, 256, 512]
    for size in sizes:
        icon = generate_app_icon(size)
        icon.save(os.path.join(output_dir, f"icon_{size}.png"))
    
    # Save main icon
    icon = generate_app_icon(512)
    icon.save(os.path.join(output_dir, "icon.png"))
    
    print(f"App icons created in {output_dir}")
    return os.path.join(output_dir, "icon.png")

def create_splash_image(output_dir, width=600, height=350):
    """Create a splash screen image"""
    os.makedirs(output_dir, exist_ok=True)
    
    pixmap = QPixmap(width, height)
    pixmap.fill(QColor(40, 40, 45))
    
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing, True)
    
    # Draw app icon at the top
    app_icon = generate_app_icon(120)
    painter.drawPixmap(width//2 - 60, 30, app_icon)
    
    # Draw text
    painter.setPen(QColor(240, 240, 240))
    
    title_font = QFont("Arial", 24)
    title_font.setBold(True)
    painter.setFont(title_font)
    painter.drawText(QRect(0, 160, width, 40), Qt.AlignCenter, "Traffic Monitoring System")
    
    subtitle_font = QFont("Arial", 12)
    painter.setFont(subtitle_font)
    painter.drawText(QRect(0, 210, width, 30), Qt.AlignCenter, "Advanced traffic analysis with OpenVINO acceleration")
    
    version_font = QFont("Arial", 10)
    painter.setFont(version_font)
    painter.drawText(QRect(0, height-30, width, 20), Qt.AlignCenter, "Version 1.0")
    
    painter.end()
    
    # Save splash image
    output_path = os.path.join(output_dir, "splash.png")
    pixmap.save(output_path)
    
    print(f"Splash image created at {output_path}")
    return output_path

if __name__ == "__main__":
    # For testing icon generation
    import sys
    from PySide6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    resources_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "resources")
    
    # Create icons
    create_app_icons(os.path.join(resources_dir, "icons"))
    
    # Create splash image
    create_splash_image(resources_dir)
    
    print("Resource generation complete!")
