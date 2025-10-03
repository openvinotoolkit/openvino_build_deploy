"""
Statistics Panel Widget - Real-time traffic statistics display
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                               QFrame, QProgressBar, QGridLayout)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QColor, QPalette

class StatisticsPanel(QWidget):
    """
    Real-time statistics panel for traffic monitoring
    
    Features:
    - Vehicle counts by type
    - Traffic flow rates
    - Violation statistics
    - Performance metrics
    - Historical trends
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.stats_data = {
            'vehicle_count': 0,
            'car_count': 0,
            'truck_count': 0,
            'motorcycle_count': 0,
            'person_count': 0,
            'violation_count': 0,
            'avg_speed': 0.0,
            'traffic_density': 0.0,
            'fps': 0.0,
            'processing_time': 0.0
        }
        
        self._setup_ui()
        
        print("📊 Statistics Panel initialized")
    
    def _setup_ui(self):
        """Setup the statistics panel UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)
        
        # Vehicle counts section
        vehicle_section = self._create_vehicle_section()
        layout.addWidget(vehicle_section)
        
        # Traffic metrics section
        metrics_section = self._create_metrics_section()
        layout.addWidget(metrics_section)
        
        # Performance section
        performance_section = self._create_performance_section()
        layout.addWidget(performance_section)
        
        layout.addStretch()
    
    def _create_vehicle_section(self):
        """Create vehicle statistics section"""
        section = QFrame()
        section.setFrameStyle(QFrame.Box)
        section.setStyleSheet("""
            QFrame {
                border: 1px solid #bdc3c7;
                border-radius: 6px;
                background-color: rgba(236, 240, 241, 0.3);
            }
        """)
        
        layout = QVBoxLayout(section)
        layout.setContentsMargins(8, 8, 8, 8)
        
        # Section title
        title = QLabel("🚗 Vehicle Detection")
        title.setFont(QFont("Segoe UI", 9, QFont.Bold))
        layout.addWidget(title)
        
        # Vehicle type grid
        grid = QGridLayout()
        grid.setSpacing(5)
        
        # Total vehicles
        self.total_label = self._create_stat_label("Total", "0")
        grid.addWidget(QLabel("Total:"), 0, 0)
        grid.addWidget(self.total_label, 0, 1)
        
        # Cars
        self.cars_label = self._create_stat_label("Cars", "0")
        grid.addWidget(QLabel("Cars:"), 1, 0)
        grid.addWidget(self.cars_label, 1, 1)
        
        # Trucks
        self.trucks_label = self._create_stat_label("Trucks", "0")
        grid.addWidget(QLabel("Trucks:"), 2, 0)
        grid.addWidget(self.trucks_label, 2, 1)
        
        # Motorcycles
        self.motorcycles_label = self._create_stat_label("Motorcycles", "0")
        grid.addWidget(QLabel("Motorcycles:"), 3, 0)
        grid.addWidget(self.motorcycles_label, 3, 1)
        
        # Pedestrians
        self.pedestrians_label = self._create_stat_label("Pedestrians", "0")
        grid.addWidget(QLabel("Pedestrians:"), 4, 0)
        grid.addWidget(self.pedestrians_label, 4, 1)
        
        layout.addLayout(grid)
        
        return section
    
    def _create_metrics_section(self):
        """Create traffic metrics section"""
        section = QFrame()
        section.setFrameStyle(QFrame.Box)
        section.setStyleSheet("""
            QFrame {
                border: 1px solid #bdc3c7;
                border-radius: 6px;
                background-color: rgba(52, 152, 219, 0.1);
            }
        """)
        
        layout = QVBoxLayout(section)
        layout.setContentsMargins(8, 8, 8, 8)
        
        # Section title
        title = QLabel("📈 Traffic Metrics")
        title.setFont(QFont("Segoe UI", 9, QFont.Bold))
        layout.addWidget(title)
        
        # Metrics grid
        grid = QGridLayout()
        grid.setSpacing(5)
        
        # Average speed
        self.speed_label = self._create_stat_label("Speed", "0.0 km/h")
        grid.addWidget(QLabel("Avg Speed:"), 0, 0)
        grid.addWidget(self.speed_label, 0, 1)
        
        # Traffic density
        self.density_label = self._create_stat_label("Density", "Low")
        grid.addWidget(QLabel("Density:"), 1, 0)
        grid.addWidget(self.density_label, 1, 1)
        
        # Violations
        self.violations_label = self._create_stat_label("Violations", "0")
        grid.addWidget(QLabel("Violations:"), 2, 0)
        grid.addWidget(self.violations_label, 2, 1)
        
        layout.addLayout(grid)
        
        # Traffic density bar
        density_layout = QHBoxLayout()
        density_layout.addWidget(QLabel("Traffic Flow:"))
        
        self.density_bar = QProgressBar()
        self.density_bar.setRange(0, 100)
        self.density_bar.setValue(0)
        self.density_bar.setMaximumHeight(15)
        density_layout.addWidget(self.density_bar)
        
        layout.addLayout(density_layout)
        
        return section
    
    def _create_performance_section(self):
        """Create performance metrics section"""
        section = QFrame()
        section.setFrameStyle(QFrame.Box)
        section.setStyleSheet("""
            QFrame {
                border: 1px solid #bdc3c7;
                border-radius: 6px;
                background-color: rgba(39, 174, 96, 0.1);
            }
        """)
        
        layout = QVBoxLayout(section)
        layout.setContentsMargins(8, 8, 8, 8)
        
        # Section title
        title = QLabel("⚡ Performance")
        title.setFont(QFont("Segoe UI", 9, QFont.Bold))
        layout.addWidget(title)
        
        # Performance grid
        grid = QGridLayout()
        grid.setSpacing(5)
        
        # FPS
        self.fps_label = self._create_stat_label("FPS", "0.0")
        grid.addWidget(QLabel("FPS:"), 0, 0)
        grid.addWidget(self.fps_label, 0, 1)
        
        # Processing time
        self.proc_time_label = self._create_stat_label("Processing", "0.0 ms")
        grid.addWidget(QLabel("Proc Time:"), 1, 0)
        grid.addWidget(self.proc_time_label, 1, 1)
        
        layout.addLayout(grid)
        
        # FPS progress bar
        fps_layout = QHBoxLayout()
        fps_layout.addWidget(QLabel("FPS Health:"))
        
        self.fps_bar = QProgressBar()
        self.fps_bar.setRange(0, 30)  # Target 30 FPS
        self.fps_bar.setValue(0)
        self.fps_bar.setMaximumHeight(15)
        fps_layout.addWidget(self.fps_bar)
        
        layout.addLayout(fps_layout)
        
        return section
    
    def _create_stat_label(self, name, value):
        """Create a styled statistics label"""
        label = QLabel(value)
        label.setFont(QFont("Segoe UI", 9, QFont.Bold))
        label.setAlignment(Qt.AlignRight)
        label.setStyleSheet("color: #2c3e50; padding: 2px;")
        return label
    
    def update_data(self, new_data):
        """Update statistics with new data"""
        # Update internal data
        self.stats_data.update(new_data)
        
        # Update vehicle counts
        self.total_label.setText(str(self.stats_data.get('vehicle_count', 0)))
        self.cars_label.setText(str(self.stats_data.get('car_count', 0)))
        self.trucks_label.setText(str(self.stats_data.get('truck_count', 0)))
        self.motorcycles_label.setText(str(self.stats_data.get('motorcycle_count', 0)))
        self.pedestrians_label.setText(str(self.stats_data.get('person_count', 0)))
        
        # Update traffic metrics
        avg_speed = self.stats_data.get('avg_speed', 0.0)
        self.speed_label.setText(f"{avg_speed:.1f} km/h")
        
        # Update traffic density
        density = self.stats_data.get('traffic_density', 0.0)
        density_text = self._get_density_text(density)
        self.density_label.setText(density_text)
        self.density_bar.setValue(int(density * 100))
        
        # Update violations
        violations = self.stats_data.get('violation_count', 0)
        self.violations_label.setText(str(violations))
        
        # Update performance metrics
        fps = self.stats_data.get('fps', 0.0)
        self.fps_label.setText(f"{fps:.1f}")
        self.fps_bar.setValue(int(fps))
        
        # Style FPS bar based on performance
        if fps >= 25:
            self.fps_bar.setStyleSheet("QProgressBar::chunk { background-color: #27ae60; }")
        elif fps >= 15:
            self.fps_bar.setStyleSheet("QProgressBar::chunk { background-color: #f39c12; }")
        else:
            self.fps_bar.setStyleSheet("QProgressBar::chunk { background-color: #e74c3c; }")
        
        proc_time = self.stats_data.get('processing_time', 0.0)
        self.proc_time_label.setText(f"{proc_time:.1f} ms")
    
    def _get_density_text(self, density):
        """Convert density value to text description"""
        if density < 0.3:
            return "Low"
        elif density < 0.6:
            return "Medium"
        elif density < 0.8:
            return "High"
        else:
            return "Very High"
    
    def get_current_stats(self):
        """Get current statistics data"""
        return self.stats_data.copy()
    
    def reset_stats(self):
        """Reset all statistics to zero"""
        self.stats_data = {key: 0 if isinstance(value, (int, float)) else value 
                          for key, value in self.stats_data.items()}
        self.update_data({})
        print("📊 Statistics reset")
