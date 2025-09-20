"""
Real-time performance graphs for inference latency analysis
Shows when latency spikes occur with different resolutions and devices
Completely rewritten with proper Qt widget lifecycle management
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QGroupBox, QTabWidget, QFrame, QSplitter, QScrollArea
)
from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtGui import QPainter, QPen, QBrush, QColor, QFont, QLinearGradient
import numpy as np
from collections import deque
from typing import Dict, List, Any, Optional
import time

# Try to import psutil for system monitoring, use fallback if not available
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️ psutil not available - system monitoring will use fallback values")

class RealTimeGraph(QWidget):
    """Custom widget for drawing real-time graphs with enhanced styling"""
    
    def __init__(self, title: str = "Graph", y_label: str = "Value", max_points: int = 300, parent: QWidget = None):
        super().__init__(parent)
        self.title = title
        self.y_label = y_label
        self.max_points = max_points
        
        # Data storage
        self.x_data = deque(maxlen=max_points)
        self.y_data = deque(maxlen=max_points)
        self.spike_markers = deque(maxlen=max_points)  # Mark spikes
        self.device_markers = deque(maxlen=max_points)  # Mark device changes
        self.resolution_markers = deque(maxlen=max_points)  # Mark resolution changes
        
        # Enhanced styling colors
        self.bg_color = QColor(18, 18, 18)  # Very dark background
        self.grid_color = QColor(40, 40, 45)  # Subtle grid
        self.line_color = QColor(0, 230, 255)  # Bright cyan
        self.spike_color = QColor(255, 77, 77)   # Bright red for spikes
        self.cpu_color = QColor(120, 180, 255)  # Light blue for CPU
        self.gpu_color = QColor(255, 165, 0)    # Orange for GPU
        self.text_color = QColor(220, 220, 220)  # Light gray text
        self.accent_color = QColor(255, 215, 0)  # Gold accent
        
        # Graph settings
        self.y_min = 0
        self.y_max = 100
        self.auto_scale = True
        self.grid_lines = True
        self.show_markers = True
        
        # Statistics tracking
        self.spike_count = 0
        self.device_switches = 0
        self.resolution_changes = 0
        
        # Set minimum size
        self.setMinimumSize(300, 150)
        
        # Enable mouse tracking for tooltips
        self.setMouseTracking(True)
        
    def clear_data(self):
        """Clear the graph data"""
        self.x_data.clear()
        self.y_data.clear()
        self.spike_markers.clear()
        self.device_markers.clear()
        self.resolution_markers.clear()
        self.spike_count = 0
        self.device_switches = 0
        self.resolution_changes = 0
        self.update()
        
    def add_data_point(self, x: float, y: float, is_spike: bool = False, device: str = "CPU", is_res_change: bool = False):
        """Add a new data point to the graph"""
        try:
            self.x_data.append(x)
            self.y_data.append(y)
            self.spike_markers.append(is_spike)
            self.device_markers.append(device)
            self.resolution_markers.append(is_res_change)
            
            # Auto-scale Y axis
            if self.auto_scale and self.y_data:
                data_max = max(self.y_data)
                data_min = min(self.y_data)
                padding = (data_max - data_min) * 0.1 if data_max > data_min else 10
                self.y_max = data_max + padding if data_max > 0 else 100
                self.y_min = max(0, data_min - padding)
            
            self.update()
        except Exception as e:
            print(f"[GRAPH ERROR] Failed to add data point: {e}")
        
    def paintEvent(self, event):
        """Override paint event to draw the graph with enhanced styling"""
        try:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            width = self.width()
            height = self.height()
            margin = 50
            graph_width = width - 2 * margin
            graph_height = height - 2 * margin
            
            # Enhanced background with subtle gradient
            gradient = QLinearGradient(0, 0, 0, height)
            gradient.setColorAt(0, QColor(25, 25, 30))
            gradient.setColorAt(1, QColor(15, 15, 20))
            painter.fillRect(0, 0, width, height, QBrush(gradient))
            
            # Draw title
            painter.setPen(QPen(self.accent_color, 2))
            font = QFont("Arial", 11, QFont.Bold)
            painter.setFont(font)
            painter.drawText(10, 20, self.title)
            
            # Draw graph area background
            graph_rect = painter.boundingRect(margin, margin, graph_width, graph_height, Qt.AlignLeft, "")
            painter.fillRect(margin, margin, graph_width, graph_height, QBrush(self.bg_color))
            painter.setPen(QPen(self.grid_color, 1))
            painter.drawRect(margin, margin, graph_width, graph_height)
            
            # Draw grid lines if enabled
            if self.grid_lines and graph_width > 0 and graph_height > 0:
                painter.setPen(QPen(self.grid_color, 1, Qt.DotLine))
                # Horizontal grid lines
                for i in range(1, 5):
                    y = margin + (graph_height * i / 5)
                    painter.drawLine(margin, int(y), margin + graph_width, int(y))
                # Vertical grid lines
                for i in range(1, 6):
                    x = margin + (graph_width * i / 6)
                    painter.drawLine(int(x), margin, int(x), margin + graph_height)
            
            # Draw Y-axis labels
            if graph_height > 0:
                painter.setPen(QPen(self.text_color, 1))
                font = QFont("Arial", 8)
                painter.setFont(font)
                for i in range(6):
                    y_val = self.y_min + (self.y_max - self.y_min) * i / 5
                    y_pos = margin + graph_height - (graph_height * i / 5)
                    painter.drawText(5, int(y_pos + 5), f"{y_val:.1f}")
            
            # Draw data if available
            if len(self.x_data) > 1 and len(self.y_data) > 1 and graph_width > 0 and graph_height > 0:
                points = []
                
                # Calculate data range
                x_min = min(self.x_data) if self.x_data else 0
                x_max = max(self.x_data) if self.x_data else 1
                x_range = x_max - x_min if x_max > x_min else 1
                y_range = self.y_max - self.y_min if self.y_max > self.y_min else 1
                
                # Create points for the line
                for i, (x, y) in enumerate(zip(self.x_data, self.y_data)):
                    try:
                        x_pixel = margin + ((x - x_min) / x_range) * graph_width
                        y_pixel = margin + graph_height - ((y - self.y_min) / y_range) * graph_height
                        
                        # Ensure coordinates are within bounds
                        x_pixel = max(margin, min(margin + graph_width, x_pixel))
                        y_pixel = max(margin, min(margin + graph_height, y_pixel))
                        
                        points.append((x_pixel, y_pixel))
                        
                        # Draw spike markers
                        if self.show_markers and i < len(self.spike_markers) and self.spike_markers[i]:
                            painter.setPen(QPen(self.spike_color, 3))
                            painter.drawEllipse(int(x_pixel - 3), int(y_pixel - 3), 6, 6)
                        
                        # Draw device change markers
                        if (self.show_markers and i < len(self.device_markers) and 
                            i > 0 and i < len(self.device_markers) and 
                            self.device_markers[i] != self.device_markers[i-1]):
                            color = self.gpu_color if self.device_markers[i] == "GPU" else self.cpu_color
                            painter.setPen(QPen(color, 2))
                            painter.drawLine(int(x_pixel), margin, int(x_pixel), margin + graph_height)
                            
                    except (ValueError, ZeroDivisionError) as e:
                        print(f"[GRAPH PAINT ERROR] Point calculation failed: {e}")
                        continue
                
                # Draw the main line
                if len(points) > 1:
                    painter.setPen(QPen(self.line_color, 2))
                    for i in range(len(points) - 1):
                        try:
                            x1, y1 = points[i]
                            x2, y2 = points[i + 1]
                            painter.drawLine(int(x1), int(y1), int(x2), int(y2))
                        except Exception as e:
                            print(f"[GRAPH LINE ERROR] Line drawing failed: {e}")
                            continue
            
            # Draw statistics in corner
            painter.setPen(QPen(self.text_color, 1))
            font = QFont("Arial", 8)
            painter.setFont(font)
            stats_text = f"Points: {len(self.y_data)} | Spikes: {self.spike_count}"
            painter.drawText(width - 200, height - 10, stats_text)
            
        except Exception as e:
            print(f"[GRAPH PAINT ERROR] Paint event failed: {e}")
            # Draw error message
            painter.setPen(QPen(QColor(255, 0, 0), 2))
            painter.drawText(10, height // 2, f"Graph Error: {str(e)[:50]}")

class PerformanceGraphsWidget(QWidget):
    """Enhanced performance monitoring widget with real-time graphs"""
    
    # Enhanced signals for better integration
    performance_data_updated = Signal(dict)
    spike_detected = Signal(dict)
    device_switched = Signal(str)
    
    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        print("[PERF GRAPHS] Initializing new performance graphs widget")
        
        # Initialize data structures first
        self.latest_data = {}
        self.start_time = time.time()
        self.frame_counter = 0
        self.spike_threshold = 100.0  # Default spike threshold in ms
        self.previous_device = "CPU"  # Track device changes
        
        # Enhanced data tracking
        self.cpu_usage_history = deque(maxlen=300)
        self.ram_usage_history = deque(maxlen=300)
        
        # Enhanced latency statistics
        self.latency_stats = {
            'max': 0,
            'min': float('inf'),
            'avg': 0,
            'spike_count': 0
        }
        
        # Graph references - initialize as None
        self.latency_graph: Optional[RealTimeGraph] = None
        self.fps_graph: Optional[RealTimeGraph] = None
        self.device_graph: Optional[RealTimeGraph] = None
        self.latency_stats_label: Optional[QLabel] = None
        self.fps_stats: Optional[QLabel] = None
        self.cpu_ram_stats: Optional[QLabel] = None
        
        # Setup UI first
        self.setup_ui()
        
        # Enhanced timer setup - only start after UI is ready
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_graphs)
        self.system_timer = QTimer(self)
        self.system_timer.timeout.connect(self.update_system_metrics)
        
        try:
            self.update_timer.start(500)  # Update graphs every 500ms
            self.system_timer.start(1000)  # Update system metrics every second
            print("[PERF GRAPHS] Timers started successfully")
        except Exception as e:
            print(f"❌ Error starting performance graph timers: {e}")
        
    def setup_ui(self):
        """Setup the enhanced UI with proper widget hierarchy"""
        print("[PERF GRAPHS] Setting up UI")
        
        # Create main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(0)
        
        # Create scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #121212;
            }
            QScrollBar:vertical {
                background-color: #2C2C2C;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #555555;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #777777;
            }
        """)
        
        # Create content widget for scroll area
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(10, 10, 10, 10)
        content_layout.setSpacing(15)
        
        # Create splitter for three graphs
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #333333;
                width: 2px;
            }
            QSplitter::handle:hover {
                background-color: #555555;
            }
        """)
        
        # 1. Latency Graph Section
        latency_frame = QFrame()
        latency_frame.setMinimumHeight(250)
        latency_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(40, 25, 25, 255),
                    stop:1 rgba(30, 15, 15, 255));
                border: 2px solid #FF4444;
                border-radius: 10px;
            }
        """)
        latency_layout = QVBoxLayout(latency_frame)
        
        # Create latency graph with proper parent
        self.latency_graph = RealTimeGraph(
            "Inference Latency Analysis", 
            "Latency (ms)", 
            max_points=300,
            parent=latency_frame
        )
        self.latency_graph.setMinimumHeight(180)
        latency_layout.addWidget(self.latency_graph)
        
        # Latency statistics
        self.latency_stats_label = QLabel(
            "Max: 0ms | Min: 0ms | Avg: 0ms | Spikes: 0"
        )
        self.latency_stats_label.setStyleSheet("""
            color: #FF4444; 
            font-size: 12px; 
            font-weight: bold; 
            margin: 4px 8px;
            padding: 4px 8px;
            background-color: rgba(255, 68, 68, 0.15);
            border-radius: 4px;
        """)
        latency_layout.addWidget(self.latency_stats_label)
        splitter.addWidget(latency_frame)
        
        # 2. FPS Graph Section  
        fps_frame = QFrame()
        fps_frame.setMinimumHeight(250)
        fps_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(25, 40, 25, 255),
                    stop:1 rgba(15, 30, 15, 255));
                border: 2px solid #00FF00;
                border-radius: 10px;
            }
        """)
        fps_layout = QVBoxLayout(fps_frame)
        
        # Create FPS graph with proper parent
        self.fps_graph = RealTimeGraph(
            "FPS & Resolution Impact", 
            "FPS", 
            max_points=300,
            parent=fps_frame
        )
        self.fps_graph.setMinimumHeight(180)
        fps_layout.addWidget(self.fps_graph)
        
        # FPS statistics
        self.fps_stats = QLabel("Current FPS: 0 | Resolution: - | Device: - | Model: -")
        self.fps_stats.setStyleSheet("""
            color: #00FF00; 
            font-size: 12px; 
            font-weight: bold; 
            margin: 4px 8px;
            padding: 4px 8px;
            background-color: rgba(0, 255, 0, 0.15);
            border-radius: 4px;
        """)
        fps_layout.addWidget(self.fps_stats)
        splitter.addWidget(fps_frame)
        
        # 3. Device Switching Graph Section
        device_frame = QFrame()
        device_frame.setMinimumHeight(250)
        device_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(35, 30, 30, 255),
                    stop:1 rgba(25, 20, 20, 255));
                border: 2px solid #FFB300;
                border-radius: 10px;
            }
        """)
        device_layout = QVBoxLayout(device_frame)
        
        # Create device graph with proper parent
        self.device_graph = RealTimeGraph(
            "Device Switching & System Usage", 
            "Usage %", 
            max_points=300,
            parent=device_frame
        )
        self.device_graph.setMinimumHeight(180)
        device_layout.addWidget(self.device_graph)
        
        # Device statistics
        self.cpu_ram_stats = QLabel("CPU: 0% | RAM: 0% | GPU: Available")
        self.cpu_ram_stats.setStyleSheet("""
            color: #FFB300; 
            font-size: 12px; 
            font-weight: bold; 
            margin: 4px 8px;
            padding: 4px 8px;
            background-color: rgba(255, 179, 0, 0.15);
            border-radius: 4px;
        """)
        device_layout.addWidget(self.cpu_ram_stats)
        splitter.addWidget(device_frame)
        
        # Set splitter proportions
        splitter.setSizes([350, 350, 300])
        content_layout.addWidget(splitter)
        
        # Set scroll area content
        scroll_area.setWidget(content_widget)
        main_layout.addWidget(scroll_area)
        
        print("[PERF GRAPHS] UI setup completed")
        
    @Slot(dict)
    def update_performance_data(self, data: dict):
        """Update performance data from external sources"""
        try:
            print(f"[PERF GRAPHS] Received performance data: {list(data.keys())}")
            
            # Store the latest data
            self.latest_data = data.copy()
            
            # Extract real-time data
            real_time_data = data.get('real_time_data', {})
            current_metrics = data.get('current_metrics', {})
            
            # Emit performance data updated signal
            self.performance_data_updated.emit(data)
            
        except Exception as e:
            print(f"[PERF GRAPHS ERROR] Failed to update performance data: {e}")
    
    @Slot()
    def update_graphs(self):
        """Update graphs with latest data - with improved error handling"""
        try:
            if not self.latest_data:
                return
                
            print(f"[PERF GRAPHS] Updating graphs...")
            
            # Check which graphs are still valid
            latency_valid = self._is_graph_valid(self.latency_graph, "latency")
            fps_valid = self._is_graph_valid(self.fps_graph, "fps") 
            device_valid = self._is_graph_valid(self.device_graph, "device")
            
            # If no graphs are valid, stop timers
            if not (latency_valid or fps_valid or device_valid):
                print("[PERF GRAPHS] No valid graphs, stopping timers")
                self.update_timer.stop()
                self.system_timer.stop()
                return
            
            # Get data
            real_time_data = self.latest_data.get('real_time_data', {})
            current_metrics = self.latest_data.get('current_metrics', {})
            
            if not real_time_data.get('timestamps'):
                return
                
            timestamps = real_time_data.get('timestamps', [])
            if not timestamps:
                return
                
            current_time = timestamps[-1] - self.start_time if self.start_time else timestamps[-1]
            
            # Update latency graph (only if valid)
            if latency_valid and 'inference_latency' in real_time_data:
                self._update_latency_graph(real_time_data, current_metrics, current_time)
            
            # Update FPS graph (only if valid)  
            if fps_valid and 'fps' in real_time_data:
                self._update_fps_graph(real_time_data, current_metrics, current_time)
            
            # Update device graph (only if valid)
            if device_valid and 'device_usage' in real_time_data:
                self._update_device_graph(real_time_data, current_metrics, current_time)
                
            print(f"[PERF GRAPHS] Graph update completed - L:{latency_valid} F:{fps_valid} D:{device_valid}")
            
        except Exception as e:
            print(f"[PERF GRAPHS ERROR] Failed to update graphs: {e}")
    
    def _is_graph_valid(self, graph: Optional[RealTimeGraph], name: str) -> bool:
        """Check if a graph widget is still valid"""
        try:
            if graph is None:
                return False
            # Try to access a property to test if the C++ object still exists
            _ = graph.objectName()
            return True
        except RuntimeError:
            print(f"[PERF GRAPHS] {name}_graph has been deleted")
            return False
        except Exception as e:
            print(f"[PERF GRAPHS] {name}_graph validation error: {e}")
            return False
    
    def _update_latency_graph(self, real_time_data: dict, current_metrics: dict, current_time: float):
        """Update the latency graph"""
        try:
            latency_values = real_time_data['inference_latency']
            if latency_values:
                latest_latency = latency_values[-1]
                is_spike = latest_latency > self.spike_threshold
                device = current_metrics.get('device', 'CPU')
                
                self.latency_graph.add_data_point(
                    current_time, 
                    latest_latency, 
                    is_spike=is_spike,
                    device=device
                )
                
                # Update statistics
                self.latency_stats['max'] = max(self.latency_stats['max'], latest_latency)
                self.latency_stats['min'] = min(self.latency_stats['min'], latest_latency)
                
                if is_spike:
                    self.latency_stats['spike_count'] += 1
                    self.spike_detected.emit({
                        'latency': latest_latency,
                        'timestamp': current_time,
                        'device': device
                    })
                
                # Calculate running average
                if hasattr(self.latency_graph, 'y_data') and self.latency_graph.y_data:
                    self.latency_stats['avg'] = sum(self.latency_graph.y_data) / len(self.latency_graph.y_data)
                
                # Update stats label
                if self.latency_stats_label:
                    self.latency_stats_label.setText(
                        f"Max: {self.latency_stats['max']:.1f}ms | "
                        f"Min: {self.latency_stats['min']:.1f}ms | "
                        f"Avg: {self.latency_stats['avg']:.1f}ms | "
                        f"Spikes: {self.latency_stats['spike_count']}"
                    )
                
                print(f"[PERF GRAPHS] Updated latency: {latest_latency:.2f}ms")
                
        except Exception as e:
            print(f"[PERF GRAPHS ERROR] Latency graph update failed: {e}")
    
    def _update_fps_graph(self, real_time_data: dict, current_metrics: dict, current_time: float):
        """Update the FPS graph"""
        try:
            fps_values = real_time_data['fps']
            if fps_values:
                latest_fps = fps_values[-1]
                device = current_metrics.get('device', 'CPU')
                resolution = current_metrics.get('resolution', 'Unknown')
                
                # Check for device switch
                device_switched = device != self.previous_device
                if device_switched:
                    self.device_switched.emit(device)
                    self.previous_device = device
                
                self.fps_graph.add_data_point(
                    current_time,
                    latest_fps,
                    device=device,
                    is_res_change=False
                )
                
                # Update FPS stats display
                if self.fps_stats:
                    model_name = current_metrics.get('model', 'Unknown')
                    self.fps_stats.setText(
                        f"Current FPS: {latest_fps:.1f} | Resolution: {resolution} | "
                        f"Device: {device} | Model: {model_name}"
                    )
                
                print(f"[PERF GRAPHS] Updated FPS: {latest_fps:.2f}")
                
        except Exception as e:
            print(f"[PERF GRAPHS ERROR] FPS graph update failed: {e}")
    
    def _update_device_graph(self, real_time_data: dict, current_metrics: dict, current_time: float):
        """Update the device usage graph"""
        try:
            device_usage = real_time_data.get('device_usage', [])
            if device_usage:
                latest_usage = device_usage[-1]
                device = current_metrics.get('device', 'CPU')
                
                self.device_graph.add_data_point(
                    current_time,
                    latest_usage * 100,  # Convert to percentage
                    device=device
                )
                
                print(f"[PERF GRAPHS] Updated device usage: {latest_usage * 100:.2f}%")
                
        except Exception as e:
            print(f"[PERF GRAPHS ERROR] Device graph update failed: {e}")
    
    @Slot()
    def update_system_metrics(self):
        """Update system CPU and RAM usage"""
        try:
            # Check if the widget is still valid
            if not self or not hasattr(self, 'isVisible'):
                return
                
            if PSUTIL_AVAILABLE:
                cpu_percent = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()
                ram_percent = memory.percent
            else:
                # Fallback values
                cpu_percent = 0
                ram_percent = 0
            
            # Store in history
            self.cpu_usage_history.append(cpu_percent)
            self.ram_usage_history.append(ram_percent)
            
            # Update display
            if self.cpu_ram_stats:
                self.cpu_ram_stats.setText(f"CPU: {cpu_percent:.1f}% | RAM: {ram_percent:.1f}% | GPU: Available")
                
        except Exception as e:
            print(f"[PERF GRAPHS ERROR] System metrics update failed: {e}")
            # Fallback in case of any error
            if hasattr(self, 'cpu_ram_stats') and self.cpu_ram_stats:
                self.cpu_ram_stats.setText("CPU: -- | RAM: -- (error)")
    
    def clear_all_graphs(self):
        """Clear all graph data"""
        try:
            if self._is_graph_valid(self.latency_graph, "latency"):
                self.latency_graph.clear_data()
            if self._is_graph_valid(self.fps_graph, "fps"):
                self.fps_graph.clear_data()
            if self._is_graph_valid(self.device_graph, "device"):
                self.device_graph.clear_data()
            
            # Reset statistics
            self.latency_stats = {
                'max': 0,
                'min': float('inf'),
                'avg': 0,
                'spike_count': 0
            }
            
            print("[PERF GRAPHS] All graphs cleared")
            
        except Exception as e:
            print(f"[PERF GRAPHS ERROR] Failed to clear graphs: {e}")
    
    def closeEvent(self, event):
        """Clean up when widget is closed"""
        try:
            # Stop timers
            if hasattr(self, 'update_timer') and self.update_timer:
                self.update_timer.stop()
            if hasattr(self, 'system_timer') and self.system_timer:
                self.system_timer.stop()
            print("[PERF GRAPHS] Cleanup completed")
        except Exception as e:
            print(f"[PERF GRAPHS ERROR] Cleanup failed: {e}")
        
        super().closeEvent(event)
