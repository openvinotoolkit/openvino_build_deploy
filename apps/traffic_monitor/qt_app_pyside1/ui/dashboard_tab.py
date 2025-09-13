"""
Local Dashboard Tab - Direct InfluxDB visualization without Grafana
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGridLayout, 
    QGroupBox, QComboBox, QPushButton, QFrame, QScrollArea
)
from PySide6.QtCore import QTimer, QThread, Signal, QObject
from PySide6.QtGui import QFont, QPalette, QColor
from datetime import datetime, timedelta
import time

# Try to import optional dependencies
try:
    from influxdb_client import InfluxDBClient, Point, WritePrecision
    INFLUXDB_AVAILABLE = True
except ImportError:
    print("⚠️ InfluxDB client not available. Dashboard will show placeholder data.")
    INFLUXDB_AVAILABLE = False

try:
    import pyqtgraph as pg
    import numpy as np
    PYQTGRAPH_AVAILABLE = True
except ImportError:
    print("⚠️ PyQtGraph not available. Dashboard will use basic widgets.")
    PYQTGRAPH_AVAILABLE = False

# InfluxDB Configuration
INFLUX_URL = "http://localhost:8086"
INFLUX_TOKEN = "kNFfXEpPQoWrk5Tteowda21Dzv6xD3jY7QHSHHQHb5oYW6VH6mkAgX9ZMjQJkaHHa8FwzmyVFqDG7qqzxN09uQ=="
INFLUX_ORG = "smart-intersection-org"
INFLUX_BUCKET = "traffic_monitoring"

class InfluxDBQueryThread(QThread):
    """Background thread for querying InfluxDB"""
    
    data_ready = Signal(str, list, list)  # query_type, timestamps, values
    error_occurred = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.client = None
        self.query_api = None
        self.running = True
        self.queries = {}
        
    def setup_connection(self):
        """Setup InfluxDB connection"""
        if not INFLUXDB_AVAILABLE:
            self.error_occurred.emit("InfluxDB client not available")
            return False
        try:
            self.client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
            self.query_api = self.client.query_api()
            print("✅ InfluxDB Query Thread connected")
            return True
        except Exception as e:
            print(f"❌ InfluxDB Query Thread connection failed: {e}")
            self.error_occurred.emit(f"Connection failed: {e}")
            return False
    
    def add_query(self, query_type, flux_query):
        """Add a query to be executed"""
        self.queries[query_type] = flux_query
    
    def run(self):
        """Main thread loop"""
        if not self.setup_connection():
            return
            
        while self.running:
            for query_type, flux_query in self.queries.items():
                try:
                    result = self.query_api.query(flux_query, org=INFLUX_ORG)
                    
                    timestamps = []
                    values = []
                    
                    for table in result:
                        for record in table.records:
                            timestamps.append(record.get_time())
                            values.append(record.get_value())
                    
                    self.data_ready.emit(query_type, timestamps, values)
                    
                except Exception as e:
                    print(f"❌ Query error for {query_type}: {e}")
                    self.error_occurred.emit(f"Query error: {e}")
            
            self.msleep(2000)  # Update every 2 seconds
    
    def stop(self):
        """Stop the thread"""
        self.running = False
        if self.client:
            self.client.close()

class DashboardTab(QWidget):
    """Local Dashboard Tab with real-time charts"""
    
    def __init__(self):
        super().__init__()
        self.setObjectName("DashboardTab")
        self.setup_ui()
        self.setup_charts()
        self.setup_influxdb_queries()
        
        # Data storage for charts
        self.chart_data = {
            'performance_fps': {'x': [], 'y': []},
            'performance_latency': {'x': [], 'y': []},
            'vehicle_count': {'x': [], 'y': []},
            'traffic_light': {'x': [], 'y': []},
            'violations': {'x': [], 'y': []}
        }
        
        # Start data fetching
        self.start_data_fetching()
        
    def setup_ui(self):
        """Setup the user interface"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Header
        header_layout = QHBoxLayout()
        
        title_label = QLabel("📊 Real-Time Dashboard")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        
        # Time range selector
        time_range_label = QLabel("Time Range:")
        self.time_range_combo = QComboBox()
        self.time_range_combo.addItems(["Last 5 minutes", "Last 15 minutes", "Last 30 minutes", "Last 1 hour"])
        self.time_range_combo.setCurrentText("Last 15 minutes")
        self.time_range_combo.currentTextChanged.connect(self.on_time_range_changed)
        
        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.clicked.connect(self.refresh_all_charts)
        
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        header_layout.addWidget(time_range_label)
        header_layout.addWidget(self.time_range_combo)
        header_layout.addWidget(refresh_btn)
        
        main_layout.addLayout(header_layout)
        
        # Create scroll area for charts
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # Chart grid
        chart_grid = QGridLayout()
        
        # Performance Charts Group
        performance_group = QGroupBox("🚀 Performance Metrics")
        performance_layout = QHBoxLayout(performance_group)
        
        self.fps_chart_widget = QWidget()
        self.fps_chart_layout = QVBoxLayout(self.fps_chart_widget)
        fps_label = QLabel("FPS")
        fps_label.setStyleSheet("font-weight: bold; color: #00ff00;")
        self.fps_chart_layout.addWidget(fps_label)
        
        self.latency_chart_widget = QWidget()
        self.latency_chart_layout = QVBoxLayout(self.latency_chart_widget)
        latency_label = QLabel("Processing Time (ms)")
        latency_label.setStyleSheet("font-weight: bold; color: #ff8c00;")
        self.latency_chart_layout.addWidget(latency_label)
        
        performance_layout.addWidget(self.fps_chart_widget)
        performance_layout.addWidget(self.latency_chart_widget)
        
        # Detection Charts Group
        detection_group = QGroupBox("🚗 Detection Metrics")
        detection_layout = QHBoxLayout(detection_group)
        
        self.vehicle_chart_widget = QWidget()
        self.vehicle_chart_layout = QVBoxLayout(self.vehicle_chart_widget)
        vehicle_label = QLabel("Vehicle Count")
        vehicle_label.setStyleSheet("font-weight: bold; color: #4da6ff;")
        self.vehicle_chart_layout.addWidget(vehicle_label)
        
        detection_layout.addWidget(self.vehicle_chart_widget)
        
        # Traffic Light Group
        traffic_group = QGroupBox("🚦 Traffic Light Status")
        traffic_layout = QVBoxLayout(traffic_group)
        
        self.traffic_chart_widget = QWidget()
        self.traffic_chart_layout = QVBoxLayout(self.traffic_chart_widget)
        traffic_label = QLabel("Traffic Light Color")
        traffic_label.setStyleSheet("font-weight: bold; color: #ff6b6b;")
        self.traffic_chart_layout.addWidget(traffic_label)
        
        traffic_layout.addWidget(self.traffic_chart_widget)
        
        # Violations Group
        violations_group = QGroupBox("🚨 Violations")
        violations_layout = QVBoxLayout(violations_group)
        
        self.violations_chart_widget = QWidget()
        self.violations_chart_layout = QVBoxLayout(self.violations_chart_widget)
        violations_label = QLabel("Red Light Violations")
        violations_label.setStyleSheet("font-weight: bold; color: #ff4757;")
        self.violations_chart_layout.addWidget(violations_label)
        
        violations_layout.addWidget(self.violations_chart_widget)
        
        # Add groups to scroll layout
        scroll_layout.addWidget(performance_group)
        scroll_layout.addWidget(detection_group)
        scroll_layout.addWidget(traffic_group)
        scroll_layout.addWidget(violations_group)
        scroll_layout.addStretch()
        
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        main_layout.addWidget(scroll_area)
        
        # Status bar
        self.status_label = QLabel("🔄 Connecting to InfluxDB...")
        self.status_label.setStyleSheet("color: #888; font-size: 12px;")
        main_layout.addWidget(self.status_label)
        
    def setup_charts(self):
        """Setup PyQtGraph charts or placeholder widgets"""
        if not PYQTGRAPH_AVAILABLE:
            self.setup_placeholder_charts()
            return
            
        # Configure PyQtGraph
        pg.setConfigOptions(antialias=True, useOpenGL=True)
        pg.setConfigOption('background', '#2b2b2b')
        pg.setConfigOption('foreground', 'w')
        
        # FPS Chart
        self.fps_chart = pg.PlotWidget()
        self.fps_chart.setLabel('left', 'FPS')
        self.fps_chart.setLabel('bottom', 'Time')
        self.fps_chart.setYRange(0, 30)
        self.fps_chart.showGrid(x=True, y=True, alpha=0.3)
        self.fps_plot = self.fps_chart.plot(pen=pg.mkPen('#00ff00', width=2))
        self.fps_chart_layout.addWidget(self.fps_chart)
        
        # Latency Chart
        self.latency_chart = pg.PlotWidget()
        self.latency_chart.setLabel('left', 'Processing Time (ms)')
        self.latency_chart.setLabel('bottom', 'Time')
        self.latency_chart.setYRange(0, 200)
        self.latency_chart.showGrid(x=True, y=True, alpha=0.3)
        self.latency_plot = self.latency_chart.plot(pen=pg.mkPen('#ff8c00', width=2))
        self.latency_chart_layout.addWidget(self.latency_chart)
        
        # Vehicle Count Chart
        self.vehicle_chart = pg.PlotWidget()
        self.vehicle_chart.setLabel('left', 'Vehicle Count')
        self.vehicle_chart.setLabel('bottom', 'Time')
        self.vehicle_chart.setYRange(0, 20)
        self.vehicle_chart.showGrid(x=True, y=True, alpha=0.3)
        self.vehicle_plot = self.vehicle_chart.plot(pen=pg.mkPen('#4da6ff', width=2), symbol='o')
        self.vehicle_chart_layout.addWidget(self.vehicle_chart)
        
        # Traffic Light Chart
        self.traffic_chart = pg.PlotWidget()
        self.traffic_chart.setLabel('left', 'Status')
        self.traffic_chart.setLabel('bottom', 'Time')
        self.traffic_chart.setYRange(0, 4)
        self.traffic_chart.showGrid(x=True, y=True, alpha=0.3)
        # Custom Y-axis labels for traffic light
        self.traffic_chart.getAxis('left').setTicks([[(0, 'Unknown'), (1, 'Red'), (2, 'Yellow'), (3, 'Green')]])
        self.traffic_plot = self.traffic_chart.plot(pen=pg.mkPen('#ff6b6b', width=3), symbol='s', symbolSize=8)
        self.traffic_chart_layout.addWidget(self.traffic_chart)
        
        # Violations Chart (Bar chart style)
        self.violations_chart = pg.PlotWidget()
        self.violations_chart.setLabel('left', 'Violations Count')
        self.violations_chart.setLabel('bottom', 'Time')
        self.violations_chart.setYRange(0, 10)
        self.violations_chart.showGrid(x=True, y=True, alpha=0.3)
        self.violations_plot = self.violations_chart.plot(pen=pg.mkPen('#ff4757', width=2), symbol='t', symbolSize=10)
        self.violations_chart_layout.addWidget(self.violations_chart)
        
    def setup_placeholder_charts(self):
        """Setup placeholder widgets when PyQtGraph is not available"""
        # FPS placeholder
        fps_placeholder = QLabel("📊 FPS Chart\n(Install PyQtGraph for real-time charts)")
        fps_placeholder.setStyleSheet("border: 2px dashed #00ff00; padding: 20px; text-align: center; color: #00ff00; background: #1a1a1a;")
        self.fps_chart_layout.addWidget(fps_placeholder)
        
        # Latency placeholder
        latency_placeholder = QLabel("📊 Latency Chart\n(Install PyQtGraph for real-time charts)")
        latency_placeholder.setStyleSheet("border: 2px dashed #ff8c00; padding: 20px; text-align: center; color: #ff8c00; background: #1a1a1a;")
        self.latency_chart_layout.addWidget(latency_placeholder)
        
        # Vehicle placeholder
        vehicle_placeholder = QLabel("📊 Vehicle Count Chart\n(Install PyQtGraph for real-time charts)")
        vehicle_placeholder.setStyleSheet("border: 2px dashed #4da6ff; padding: 20px; text-align: center; color: #4da6ff; background: #1a1a1a;")
        self.vehicle_chart_layout.addWidget(vehicle_placeholder)
        
        # Traffic Light placeholder
        traffic_placeholder = QLabel("📊 Traffic Light Chart\n(Install PyQtGraph for real-time charts)")
        traffic_placeholder.setStyleSheet("border: 2px dashed #ff6b6b; padding: 20px; text-align: center; color: #ff6b6b; background: #1a1a1a;")
        self.traffic_chart_layout.addWidget(traffic_placeholder)
        
        # Violations placeholder
        violations_placeholder = QLabel("📊 Violations Chart\n(Install PyQtGraph for real-time charts)")
        violations_placeholder.setStyleSheet("border: 2px dashed #ff4757; padding: 20px; text-align: center; color: #ff4757; background: #1a1a1a;")
        self.violations_chart_layout.addWidget(violations_placeholder)
        
    def setup_influxdb_queries(self):
        """Setup InfluxDB queries"""
        if not INFLUXDB_AVAILABLE:
            return
            
        self.query_thread = InfluxDBQueryThread()
        self.query_thread.data_ready.connect(self.on_data_received)
        self.query_thread.error_occurred.connect(self.on_error_occurred)
        
        # Define Flux queries
        time_range = self.get_time_range()
        
        # Performance Queries
        fps_query = f'''
        from(bucket: "{INFLUX_BUCKET}")
        |> range(start: {time_range})
        |> filter(fn: (r) => r._measurement == "performance")
        |> filter(fn: (r) => r._field == "fps")
        |> aggregateWindow(every: 10s, fn: mean, createEmpty: false)
        |> yield(name: "fps")
        '''
        
        latency_query = f'''
        from(bucket: "{INFLUX_BUCKET}")
        |> range(start: {time_range})
        |> filter(fn: (r) => r._measurement == "performance")
        |> filter(fn: (r) => r._field == "processing_time_ms")
        |> aggregateWindow(every: 10s, fn: mean, createEmpty: false)
        |> yield(name: "latency")
        '''
        
        # Vehicle Count Query
        vehicle_query = f'''
        from(bucket: "{INFLUX_BUCKET}")
        |> range(start: {time_range})
        |> filter(fn: (r) => r._measurement == "detection_events")
        |> filter(fn: (r) => r._field == "vehicle_count")
        |> aggregateWindow(every: 30s, fn: mean, createEmpty: false)
        |> yield(name: "vehicles")
        '''
        
        # Traffic Light Query
        traffic_query = f'''
        from(bucket: "{INFLUX_BUCKET}")
        |> range(start: {time_range})
        |> filter(fn: (r) => r._measurement == "traffic_light_status")
        |> filter(fn: (r) => r._field == "color_numeric")
        |> last()
        |> yield(name: "traffic_light")
        '''
        
        # Violations Query
        violations_query = f'''
        from(bucket: "{INFLUX_BUCKET}")
        |> range(start: {time_range})
        |> filter(fn: (r) => r._measurement == "violation_events")
        |> filter(fn: (r) => r.violation_type == "red_light_violation")
        |> aggregateWindow(every: 1m, fn: count, createEmpty: false)
        |> yield(name: "violations")
        '''
        
        # Add queries to thread
        self.query_thread.add_query("fps", fps_query)
        self.query_thread.add_query("latency", latency_query)
        self.query_thread.add_query("vehicles", vehicle_query)
        self.query_thread.add_query("traffic_light", traffic_query)
        self.query_thread.add_query("violations", violations_query)
        
    def start_data_fetching(self):
        """Start fetching data from InfluxDB"""
        if not INFLUXDB_AVAILABLE:
            self.status_label.setText("⚠️ InfluxDB client not available - Install influxdb-client package")
            return
            
        self.query_thread.start()
        self.status_label.setText("✅ Connected to InfluxDB - Real-time data streaming")
        
    def get_time_range(self):
        """Get time range based on combo box selection"""
        time_ranges = {
            "Last 5 minutes": "-5m",
            "Last 15 minutes": "-15m", 
            "Last 30 minutes": "-30m",
            "Last 1 hour": "-1h"
        }
        return time_ranges.get(self.time_range_combo.currentText(), "-15m")
        
    def on_time_range_changed(self):
        """Handle time range change"""
        self.setup_influxdb_queries()
        self.refresh_all_charts()
        
    def on_data_received(self, query_type, timestamps, values):
        """Handle data received from InfluxDB"""
        if not timestamps or not values or not PYQTGRAPH_AVAILABLE:
            return
            
        # Convert timestamps to seconds since epoch for plotting
        x_data = [int(ts.timestamp()) for ts in timestamps]
        y_data = values
        
        # Update chart data
        if query_type == "fps":
            self.chart_data['performance_fps'] = {'x': x_data, 'y': y_data}
            self.fps_plot.setData(x_data, y_data)
            
        elif query_type == "latency":
            self.chart_data['performance_latency'] = {'x': x_data, 'y': y_data}
            self.latency_plot.setData(x_data, y_data)
            
        elif query_type == "vehicles":
            self.chart_data['vehicle_count'] = {'x': x_data, 'y': y_data}
            self.vehicle_plot.setData(x_data, y_data)
            
        elif query_type == "traffic_light":
            self.chart_data['traffic_light'] = {'x': x_data, 'y': y_data}
            self.traffic_plot.setData(x_data, y_data)
            
        elif query_type == "violations":
            self.chart_data['violations'] = {'x': x_data, 'y': y_data}
            self.violations_plot.setData(x_data, y_data)
            
        # Update status
        self.status_label.setText(f"📈 Last updated: {datetime.now().strftime('%H:%M:%S')} - Query: {query_type}")
        
    def on_error_occurred(self, error_msg):
        """Handle errors from InfluxDB"""
        self.status_label.setText(f"❌ Error: {error_msg}")
        
    def refresh_all_charts(self):
        """Refresh all charts"""
        if INFLUXDB_AVAILABLE:
            self.setup_influxdb_queries()
            self.status_label.setText("🔄 Refreshing charts...")
        else:
            self.status_label.setText("⚠️ InfluxDB client not available")
        
    def closeEvent(self, event):
        """Clean up when tab is closed"""
        if hasattr(self, 'query_thread') and INFLUXDB_AVAILABLE:
            self.query_thread.stop()
            self.query_thread.wait(3000)  # Wait up to 3 seconds
        event.accept()
