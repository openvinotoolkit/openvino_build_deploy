"""
System Performance Tab - Grafana-like performance monitoring dashboard
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                               QGroupBox, QLabel, QPushButton, QComboBox,
                               QSlider, QProgressBar, QFrame, QSplitter,
                               QScrollArea, QCheckBox, QSpinBox, QTabWidget)
from PySide6.QtCore import Qt, Signal, QTimer, QDateTime
from PySide6.QtGui import QFont, QColor, QPainter, QPen, QBrush
import random
import math

class MetricCard(QFrame):
    """Individual metric display card"""
    
    def __init__(self, title, value, unit="", trend=None, color="#3498db", parent=None):
        super().__init__(parent)
        
        self.title = title
        self.value = value
        self.unit = unit
        self.trend = trend
        self.color = color
        
        self.setFixedSize(200, 120)
        self._setup_ui()
        self._apply_style()
    
    def _setup_ui(self):
        """Setup metric card UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 10, 15, 10)
        
        # Title
        title_label = QLabel(self.title)
        title_label.setFont(QFont("Segoe UI", 9))
        title_label.setStyleSheet("color: #7f8c8d; font-weight: 500;")
        layout.addWidget(title_label)
        
        # Value with unit
        value_layout = QHBoxLayout()
        
        value_label = QLabel(str(self.value))
        value_label.setFont(QFont("Segoe UI", 20, QFont.Bold))
        value_label.setStyleSheet(f"color: {self.color};")
        value_layout.addWidget(value_label)
        
        if self.unit:
            unit_label = QLabel(self.unit)
            unit_label.setFont(QFont("Segoe UI", 10))
            unit_label.setStyleSheet("color: #95a5a6;")
            unit_label.setAlignment(Qt.AlignBottom)
            value_layout.addWidget(unit_label)
        
        value_layout.addStretch()
        layout.addLayout(value_layout)
        
        # Trend indicator
        if self.trend is not None:
            trend_label = QLabel(f"{'↗' if self.trend > 0 else '↘' if self.trend < 0 else '→'} {abs(self.trend):.1f}%")
            trend_color = "#27ae60" if self.trend > 0 else "#e74c3c" if self.trend < 0 else "#95a5a6"
            trend_label.setStyleSheet(f"color: {trend_color}; font-size: 8pt;")
            layout.addWidget(trend_label)
        
        layout.addStretch()
    
    def _apply_style(self):
        """Apply card styling"""
        self.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 1px solid #e1e8ed;
                border-radius: 8px;
                margin: 5px;
            }
            QFrame:hover {
                border-color: #3498db;
            }
        """)
    
    def update_value(self, value, trend=None):
        """Update metric value"""
        self.value = value
        self.trend = trend
        # Re-setup UI to reflect changes
        # Clear and rebuild
        for i in reversed(range(self.layout().count())):
            child = self.layout().itemAt(i).widget()
            if child:
                child.setParent(None)
        self._setup_ui()

class SimpleChart(QWidget):
    """Simple line chart widget"""
    
    def __init__(self, title="Chart", width=300, height=200, parent=None):
        super().__init__(parent)
        
        self.title = title
        self.data_points = []
        self.max_points = 50
        self.color = QColor(52, 152, 219)
        
        self.setFixedSize(width, height)
        self.setStyleSheet("""
            QWidget {
                background-color: white;
                border: 1px solid #e1e8ed;
                border-radius: 8px;
            }
        """)
    
    def add_data_point(self, value):
        """Add a new data point"""
        self.data_points.append(value)
        if len(self.data_points) > self.max_points:
            self.data_points.pop(0)
        self.update()
    
    def paintEvent(self, event):
        """Custom paint event for chart"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw background
        painter.fillRect(self.rect(), QColor(255, 255, 255))
        
        # Draw title
        painter.setPen(QColor(44, 62, 80))
        painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
        painter.drawText(10, 20, self.title)
        
        if len(self.data_points) < 2:
            return
        
        # Calculate chart area
        chart_rect = self.rect().adjusted(20, 30, -20, -20)
        
        # Find min and max values
        min_val = min(self.data_points)
        max_val = max(self.data_points)
        if min_val == max_val:
            max_val = min_val + 1
        
        # Draw grid lines
        painter.setPen(QPen(QColor(236, 240, 241), 1))
        for i in range(5):
            y = chart_rect.top() + (chart_rect.height() * i / 4)
            painter.drawLine(chart_rect.left(), y, chart_rect.right(), y)
        
        # Draw data line
        painter.setPen(QPen(self.color, 2))
        for i in range(1, len(self.data_points)):
            x1 = chart_rect.left() + (chart_rect.width() * (i-1) / (len(self.data_points)-1))
            y1 = chart_rect.bottom() - (chart_rect.height() * (self.data_points[i-1] - min_val) / (max_val - min_val))
            
            x2 = chart_rect.left() + (chart_rect.width() * i / (len(self.data_points)-1))
            y2 = chart_rect.bottom() - (chart_rect.height() * (self.data_points[i] - min_val) / (max_val - min_val))
            
            painter.drawLine(x1, y1, x2, y2)

class SystemPerformanceTab(QWidget):
    """
    System Performance Tab with Grafana-like monitoring dashboard
    
    Features:
    - Real-time performance metrics
    - Interactive charts and graphs
    - System resource monitoring
    - Performance alerts and thresholds
    - Historical data analysis
    - Export capabilities
    """
    
    # Signals
    performance_alert = Signal(str, str)  # alert_type, message
    threshold_exceeded = Signal(str, float)  # metric_name, value
    export_requested = Signal(str)  # export_type
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.performance_data = {}
        self.alerts_enabled = True
        
        self._setup_ui()
        
        # Timer for real-time updates
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_metrics)
        self.update_timer.start(1000)  # Update every second
        
        print("🔥 System Performance Tab initialized")
    
    def _setup_ui(self):
        """Setup the performance monitoring UI"""
        layout = QVBoxLayout(self)
        
        # Header with controls
        header = self._create_header()
        layout.addWidget(header)
        
        # Main content tabs
        content_tabs = self._create_content_tabs()
        layout.addWidget(content_tabs)
    
    def _create_header(self):
        """Create header with system overview and controls"""
        header = QFrame()
        header.setFixedHeight(60)
        header.setStyleSheet("""
            QFrame {
                background-color: #2c3e50;
                border-radius: 8px;
                margin-bottom: 10px;
            }
        """)
        
        layout = QHBoxLayout(header)
        layout.setContentsMargins(20, 10, 20, 10)
        
        # System status
        status_layout = QVBoxLayout()
        
        system_label = QLabel("🔥 System Performance Monitor")
        system_label.setFont(QFont("Segoe UI", 14, QFont.Bold))
        system_label.setStyleSheet("color: white;")
        status_layout.addWidget(system_label)
        
        self.system_health_label = QLabel("System Health: Optimal")
        self.system_health_label.setFont(QFont("Segoe UI", 9))
        self.system_health_label.setStyleSheet("color: #27ae60;")
        status_layout.addWidget(self.system_health_label)
        
        layout.addLayout(status_layout)
        
        layout.addStretch()
        
        # Quick metrics
        quick_metrics = QHBoxLayout()
        
        # CPU usage
        cpu_layout = QVBoxLayout()
        cpu_layout.addWidget(QLabel("CPU"))
        self.cpu_progress = QProgressBar()
        self.cpu_progress.setRange(0, 100)
        self.cpu_progress.setFixedWidth(80)
        cpu_layout.addWidget(self.cpu_progress)
        quick_metrics.addLayout(cpu_layout)
        
        # Memory usage
        memory_layout = QVBoxLayout()
        memory_layout.addWidget(QLabel("Memory"))
        self.memory_progress = QProgressBar()
        self.memory_progress.setRange(0, 100)
        self.memory_progress.setFixedWidth(80)
        memory_layout.addWidget(self.memory_progress)
        quick_metrics.addLayout(memory_layout)
        
        # GPU usage
        gpu_layout = QVBoxLayout()
        gpu_layout.addWidget(QLabel("GPU"))
        self.gpu_progress = QProgressBar()
        self.gpu_progress.setRange(0, 100)
        self.gpu_progress.setFixedWidth(80)
        gpu_layout.addWidget(self.gpu_progress)
        quick_metrics.addLayout(gpu_layout)
        
        # Style progress bars
        progress_style = """
            QProgressBar {
                border: 1px solid #34495e;
                border-radius: 3px;
                text-align: center;
                color: white;
                font-size: 8pt;
            }
            QProgressBar::chunk {
                background-color: #3498db;
                border-radius: 2px;
            }
            QLabel {
                color: white;
                font-size: 8pt;
                text-align: center;
            }
        """
        for widget in [self.cpu_progress, self.memory_progress, self.gpu_progress]:
            widget.setStyleSheet(progress_style)
        
        layout.addLayout(quick_metrics)
        
        # Control buttons
        controls_layout = QVBoxLayout()
        
        alerts_btn = QPushButton("🔔 Alerts")
        alerts_btn.setFixedSize(80, 25)
        alerts_btn.setCheckable(True)
        alerts_btn.setChecked(True)
        alerts_btn.toggled.connect(self._toggle_alerts)
        controls_layout.addWidget(alerts_btn)
        
        export_btn = QPushButton("📊 Export")
        export_btn.setFixedSize(80, 25)
        export_btn.clicked.connect(self._export_performance_data)
        controls_layout.addWidget(export_btn)
        
        # Style buttons
        button_style = """
            QPushButton {
                background-color: #34495e;
                color: white;
                border: 1px solid #34495e;
                border-radius: 4px;
                font-size: 8pt;
            }
            QPushButton:hover {
                background-color: #4a6741;
            }
            QPushButton:checked {
                background-color: #27ae60;
            }
        """
        alerts_btn.setStyleSheet(button_style)
        export_btn.setStyleSheet(button_style)
        
        layout.addLayout(controls_layout)
        
        return header
    
    def _create_content_tabs(self):
        """Create main content tabs"""
        tabs = QTabWidget()
        
        # Overview tab
        overview_tab = self._create_overview_tab()
        tabs.addTab(overview_tab, "📊 Overview")
        
        # Detailed metrics tab
        metrics_tab = self._create_metrics_tab()
        tabs.addTab(metrics_tab, "📈 Detailed Metrics")
        
        # Alerts tab
        alerts_tab = self._create_alerts_tab()
        tabs.addTab(alerts_tab, "🚨 Alerts")
        
        # Settings tab
        settings_tab = self._create_settings_tab()
        tabs.addTab(settings_tab, "⚙️ Settings")
        
        return tabs
    
    def _create_overview_tab(self):
        """Create overview dashboard tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Metric cards grid
        cards_scroll = QScrollArea()
        cards_widget = QWidget()
        cards_layout = QGridLayout(cards_widget)
        
        # Create metric cards
        self.metric_cards = {}
        
        cards_data = [
            ("FPS", 0, "fps", None, "#3498db"),
            ("Processing Time", 0, "ms", None, "#e74c3c"),
            ("CPU Usage", 0, "%", None, "#f39c12"),
            ("Memory Usage", 0, "%", None, "#9b59b6"),
            ("GPU Usage", 0, "%", None, "#1abc9c"),
            ("Network I/O", 0, "MB/s", None, "#34495e"),
            ("Disk I/O", 0, "MB/s", None, "#95a5a6"),
            ("Temperature", 0, "°C", None, "#e67e22"),
        ]
        
        for i, (title, value, unit, trend, color) in enumerate(cards_data):
            card = MetricCard(title, value, unit, trend, color)
            self.metric_cards[title.lower().replace(" ", "_")] = card
            cards_layout.addWidget(card, i // 4, i % 4)
        
        cards_scroll.setWidget(cards_widget)
        cards_scroll.setMaximumHeight(280)
        layout.addWidget(cards_scroll)
        
        # Charts section
        charts_layout = QHBoxLayout()
        
        # FPS chart
        self.fps_chart = SimpleChart("FPS Over Time", 400, 200)
        self.fps_chart.color = QColor(52, 152, 219)
        charts_layout.addWidget(self.fps_chart)
        
        # CPU/Memory chart
        self.resource_chart = SimpleChart("Resource Usage", 400, 200)
        self.resource_chart.color = QColor(243, 156, 18)
        charts_layout.addWidget(self.resource_chart)
        
        layout.addLayout(charts_layout)
        
        return tab
    
    def _create_metrics_tab(self):
        """Create detailed metrics tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Metrics filter
        filter_layout = QHBoxLayout()
        
        filter_layout.addWidget(QLabel("Category:"))
        category_combo = QComboBox()
        category_combo.addItems(["All", "Performance", "Resources", "Network", "Storage"])
        filter_layout.addWidget(category_combo)
        
        filter_layout.addWidget(QLabel("Time Range:"))
        time_combo = QComboBox()
        time_combo.addItems(["Last 5 minutes", "Last hour", "Last 24 hours", "Last week"])
        filter_layout.addWidget(time_combo)
        
        filter_layout.addStretch()
        
        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.clicked.connect(self._refresh_metrics)
        filter_layout.addWidget(refresh_btn)
        
        layout.addLayout(filter_layout)
        
        # Detailed charts area
        charts_scroll = QScrollArea()
        charts_widget = QWidget()
        charts_layout = QVBoxLayout(charts_widget)
        
        # Create multiple detailed charts
        chart_titles = [
            "Frame Processing Pipeline",
            "Detection Performance",
            "Memory Allocation",
            "GPU Utilization",
            "Network Throughput",
            "Disk I/O Performance"
        ]
        
        self.detailed_charts = {}
        for title in chart_titles:
            chart = SimpleChart(title, 700, 150)
            self.detailed_charts[title.lower().replace(" ", "_")] = chart
            charts_layout.addWidget(chart)
        
        charts_scroll.setWidget(charts_widget)
        layout.addWidget(charts_scroll)
        
        return tab
    
    def _create_alerts_tab(self):
        """Create alerts configuration tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Alert thresholds
        thresholds_group = QGroupBox("Alert Thresholds")
        thresholds_layout = QGridLayout(thresholds_group)
        
        thresholds = [
            ("CPU Usage", 80, "%"),
            ("Memory Usage", 85, "%"),
            ("GPU Usage", 90, "%"),
            ("FPS Drop", 15, "fps"),
            ("Processing Time", 100, "ms"),
            ("Temperature", 75, "°C")
        ]
        
        self.threshold_controls = {}
        for i, (metric, default_value, unit) in enumerate(thresholds):
            # Label
            thresholds_layout.addWidget(QLabel(f"{metric}:"), i, 0)
            
            # Slider
            slider = QSlider(Qt.Horizontal)
            slider.setRange(1, 100)
            slider.setValue(default_value)
            thresholds_layout.addWidget(slider, i, 1)
            
            # Value label
            value_label = QLabel(f"{default_value} {unit}")
            thresholds_layout.addWidget(value_label, i, 2)
            
            # Connect slider to label
            slider.valueChanged.connect(
                lambda v, label=value_label, u=unit: label.setText(f"{v} {u}")
            )
            
            self.threshold_controls[metric.lower().replace(" ", "_")] = slider
        
        layout.addWidget(thresholds_group)
        
        # Alert history
        history_group = QGroupBox("Recent Alerts")
        history_layout = QVBoxLayout(history_group)
        
        self.alerts_text = QLabel("No recent alerts")
        self.alerts_text.setStyleSheet("""
            QLabel {
                background-color: #ecf0f1;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                padding: 10px;
                min-height: 200px;
            }
        """)
        self.alerts_text.setAlignment(Qt.AlignTop)
        self.alerts_text.setWordWrap(True)
        history_layout.addWidget(self.alerts_text)
        
        layout.addWidget(history_group)
        
        return tab
    
    def _create_settings_tab(self):
        """Create performance settings tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Update intervals
        intervals_group = QGroupBox("Update Intervals")
        intervals_layout = QGridLayout(intervals_group)
        
        intervals_layout.addWidget(QLabel("Metrics Update:"), 0, 0)
        metrics_interval = QSpinBox()
        metrics_interval.setRange(500, 10000)
        metrics_interval.setValue(1000)
        metrics_interval.setSuffix(" ms")
        intervals_layout.addWidget(metrics_interval, 0, 1)
        
        intervals_layout.addWidget(QLabel("Charts Update:"), 1, 0)
        charts_interval = QSpinBox()
        charts_interval.setRange(1000, 30000)
        charts_interval.setValue(5000)
        charts_interval.setSuffix(" ms")
        intervals_layout.addWidget(charts_interval, 1, 1)
        
        layout.addWidget(intervals_group)
        
        # Display options
        display_group = QGroupBox("Display Options")
        display_layout = QVBoxLayout(display_group)
        
        self.show_grid_cb = QCheckBox("Show Chart Grid Lines")
        self.show_grid_cb.setChecked(True)
        display_layout.addWidget(self.show_grid_cb)
        
        self.smooth_charts_cb = QCheckBox("Smooth Chart Animations")
        self.smooth_charts_cb.setChecked(True)
        display_layout.addWidget(self.smooth_charts_cb)
        
        self.auto_scale_cb = QCheckBox("Auto-scale Chart Axes")
        self.auto_scale_cb.setChecked(True)
        display_layout.addWidget(self.auto_scale_cb)
        
        layout.addWidget(display_group)
        
        # Performance optimization
        optimization_group = QGroupBox("Performance Optimization")
        optimization_layout = QVBoxLayout(optimization_group)
        
        self.reduce_quality_cb = QCheckBox("Reduce Chart Quality for Better Performance")
        optimization_layout.addWidget(self.reduce_quality_cb)
        
        self.limit_history_cb = QCheckBox("Limit Historical Data (50 points)")
        self.limit_history_cb.setChecked(True)
        optimization_layout.addWidget(self.limit_history_cb)
        
        layout.addWidget(optimization_group)
        
        layout.addStretch()
        
        return tab
    
    def _update_metrics(self):
        """Update all performance metrics (called by timer)"""
        # Simulate real metrics (replace with actual system monitoring)
        import random
        import time
        
        # Generate simulated metrics
        current_time = time.time()
        metrics = {
            'fps': random.uniform(25, 30),
            'processing_time': random.uniform(20, 50),
            'cpu_usage': random.uniform(30, 70),
            'memory_usage': random.uniform(40, 80),
            'gpu_usage': random.uniform(50, 90),
            'network_io': random.uniform(5, 25),
            'disk_io': random.uniform(1, 10),
            'temperature': random.uniform(45, 65)
        }
        
        # Update metric cards
        for metric_name, value in metrics.items():
            if metric_name in self.metric_cards:
                # Calculate trend (simplified)
                old_value = getattr(self, f'_last_{metric_name}', value)
                trend = ((value - old_value) / old_value * 100) if old_value != 0 else 0
                
                self.metric_cards[metric_name].update_value(f"{value:.1f}", trend)
                setattr(self, f'_last_{metric_name}', value)
        
        # Update progress bars in header
        self.cpu_progress.setValue(int(metrics['cpu_usage']))
        self.memory_progress.setValue(int(metrics['memory_usage']))
        self.gpu_progress.setValue(int(metrics['gpu_usage']))
        
        # Update system health
        avg_usage = (metrics['cpu_usage'] + metrics['memory_usage'] + metrics['gpu_usage']) / 3
        if avg_usage < 50:
            health_status = "Optimal"
            health_color = "#27ae60"
        elif avg_usage < 75:
            health_status = "Good"
            health_color = "#f39c12"
        else:
            health_status = "High Load"
            health_color = "#e74c3c"
        
        self.system_health_label.setText(f"System Health: {health_status}")
        self.system_health_label.setStyleSheet(f"color: {health_color};")
        
        # Update charts
        self.fps_chart.add_data_point(metrics['fps'])
        self.resource_chart.add_data_point(metrics['cpu_usage'])
        
        # Update detailed charts
        for chart_name, chart in self.detailed_charts.items():
            # Use different metrics for different charts
            if 'processing' in chart_name:
                chart.add_data_point(metrics['processing_time'])
            elif 'memory' in chart_name:
                chart.add_data_point(metrics['memory_usage'])
            elif 'gpu' in chart_name:
                chart.add_data_point(metrics['gpu_usage'])
            else:
                chart.add_data_point(random.uniform(10, 90))
        
        # Check for threshold alerts
        if self.alerts_enabled:
            self._check_thresholds(metrics)
    
    def _check_thresholds(self, metrics):
        """Check if any metrics exceed thresholds"""
        thresholds = {
            'cpu_usage': 80,
            'memory_usage': 85,
            'gpu_usage': 90,
            'processing_time': 100,
            'temperature': 75
        }
        
        for metric, value in metrics.items():
            if metric in thresholds and value > thresholds[metric]:
                alert_msg = f"{metric.replace('_', ' ').title()} is high: {value:.1f}"
                self.performance_alert.emit("warning", alert_msg)
    
    def _toggle_alerts(self, enabled):
        """Toggle performance alerts"""
        self.alerts_enabled = enabled
        status = "enabled" if enabled else "disabled"
        print(f"🔥 Performance alerts {status}")
    
    def _export_performance_data(self):
        """Export performance data"""
        self.export_requested.emit("performance_metrics")
        print("🔥 Performance data exported")
    
    def _refresh_metrics(self):
        """Refresh all metrics manually"""
        print("🔥 Refreshing performance metrics")
        # Force immediate update
        self._update_metrics()
    
    def add_custom_metric(self, name, value, unit=""):
        """Add a custom performance metric"""
        if hasattr(self, 'metric_cards'):
            # Add new metric card if space available
            print(f"🔥 Added custom metric: {name} = {value} {unit}")
    
    def set_alert_threshold(self, metric_name, threshold_value):
        """Set alert threshold for a specific metric"""
        if metric_name in self.threshold_controls:
            self.threshold_controls[metric_name].setValue(int(threshold_value))
            print(f"🔥 Alert threshold set: {metric_name} = {threshold_value}")
    
    def get_performance_summary(self):
        """Get current performance summary"""
        return {
            'status': self.system_health_label.text(),
            'cpu': self.cpu_progress.value(),
            'memory': self.memory_progress.value(),
            'gpu': self.gpu_progress.value(),
            'alerts_enabled': self.alerts_enabled
        }
