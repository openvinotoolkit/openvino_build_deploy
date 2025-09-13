from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget, QTableWidgetItem,
    QTabWidget, QScrollArea, QFrame, QHeaderView, QPushButton
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QColor
import re
from datetime import datetime


class AnalyticsTablesWidget(QWidget):
    """Analytics widget with structured tables for debug data"""
    
    def __init__(self):
        super().__init__()
        self.debug_data = {
            'detections': [],
            'traffic_lights': [],
            'violations': [],
            'performance': [],
            'tracking': []
        }
        self.init_ui()
        
    def init_ui(self):
        """Initialize the UI with tabbed tables"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Header
        header = QLabel("📊 Analytics Tables")
        header.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #03DAC5;
                padding: 8px;
                background: #1E1E1E;
                border-radius: 6px;
                margin-bottom: 5px;
            }
        """)
        layout.addWidget(header)
        
        # Create tab widget
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #333;
                background: #1E1E1E;
            }
            QTabWidget::tab-bar {
                alignment: left;
            }
            QTabBar::tab {
                background: #2D2D2D;
                color: #fff;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
            }
            QTabBar::tab:selected {
                background: #03DAC5;
                color: #000;
            }
            QTabBar::tab:hover {
                background: #018786;
            }
        """)
        
        # Create individual tables
        self._create_detection_table()
        self._create_traffic_light_table()
        self._create_violation_table()
        self._create_performance_table()
        self._create_tracking_table()
        
        layout.addWidget(self.tabs)
        
        # Control buttons
        controls = self._create_controls()
        layout.addWidget(controls)
        
    def _create_controls(self):
        """Create control buttons"""
        frame = QFrame()
        frame.setStyleSheet("""
            QFrame {
                background: #2D2D2D;
                border-radius: 6px;
                padding: 5px;
            }
        """)
        layout = QHBoxLayout(frame)
        
        clear_btn = QPushButton("🗑️ Clear All")
        clear_btn.setStyleSheet(self._button_style())
        clear_btn.clicked.connect(self.clear_all_tables)
        
        export_btn = QPushButton("📤 Export CSV")
        export_btn.setStyleSheet(self._button_style())
        export_btn.clicked.connect(self.export_to_csv)
        
        layout.addWidget(clear_btn)
        layout.addWidget(export_btn)
        layout.addStretch()
        
        return frame
        
    def _button_style(self):
        return """
            QPushButton {
                background: #333;
                color: #fff;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
                font-size: 12px;
            }
            QPushButton:hover {
                background: #03DAC5;
                color: #000;
            }
        """
        
    def _create_detection_table(self):
        """Create detection matching table"""
        table = QTableWidget()
        table.setColumnCount(5)
        table.setHorizontalHeaderLabels([
            "Detection (x,y)", "Track ID", "IoU", "Distance", "Match Status"
        ])
        
        # Style the table
        self._style_table(table)
        
        # Add to tabs
        scroll = QScrollArea()
        scroll.setWidget(table)
        scroll.setWidgetResizable(True)
        self.tabs.addTab(scroll, "🎯 Detection Matching")
        
        # Store reference
        self.detection_table = table
        
    def _create_traffic_light_table(self):
        """Create traffic light status table"""
        table = QTableWidget()
        table.setColumnCount(5)
        table.setHorizontalHeaderLabels([
            "Detection", "Red Ratio", "Yellow Ratio", "Green Ratio", "Status"
        ])
        
        self._style_table(table)
        
        scroll = QScrollArea()
        scroll.setWidget(table)
        scroll.setWidgetResizable(True)
        self.tabs.addTab(scroll, "🚦 Traffic Lights")
        
        self.traffic_light_table = table
        
    def _create_violation_table(self):
        """Create violation summary table"""
        table = QTableWidget()
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels([
            "Track ID", "Violation Type", "Timestamp", "Details"
        ])
        
        self._style_table(table)
        
        scroll = QScrollArea()
        scroll.setWidget(table)
        scroll.setWidgetResizable(True)
        self.tabs.addTab(scroll, "🚨 Violations")
        
        self.violation_table = table
        
    def _create_performance_table(self):
        """Create performance metrics table"""
        table = QTableWidget()
        table.setColumnCount(6)
        table.setHorizontalHeaderLabels([
            "Frame", "FPS", "Inference (ms)", "Device", "Objects", "Timestamp"
        ])
        
        self._style_table(table)
        
        scroll = QScrollArea()
        scroll.setWidget(table)
        scroll.setWidgetResizable(True)
        self.tabs.addTab(scroll, "⚡ Performance")
        
        self.performance_table = table
        
    def _create_tracking_table(self):
        """Create vehicle tracking table"""
        table = QTableWidget()
        table.setColumnCount(6)
        table.setHorizontalHeaderLabels([
            "Track ID", "Position (x,y)", "Moving", "Violating", "Confidence", "Status"
        ])
        
        self._style_table(table)
        
        scroll = QScrollArea()
        scroll.setWidget(table)
        scroll.setWidgetResizable(True)
        self.tabs.addTab(scroll, "🚗 Vehicle Tracking")
        
        self.tracking_table = table
        
    def _style_table(self, table):
        """Apply consistent styling to tables"""
        table.setStyleSheet("""
            QTableWidget {
                background-color: #1E1E1E;
                color: #fff;
                gridline-color: #333;
                selection-background-color: #03DAC5;
                selection-color: #000;
                border: none;
                font-size: 12px;
            }
            QTableWidget::item {
                padding: 8px;
                border-bottom: 1px solid #333;
            }
            QHeaderView::section {
                background-color: #2D2D2D;
                color: #03DAC5;
                padding: 8px;
                border: none;
                font-weight: bold;
                font-size: 13px;
            }
        """)
        
        # Set header resize mode
        header = table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(QHeaderView.Interactive)
        
        # Enable sorting
        table.setSortingEnabled(True)
        
    def process_debug_logs(self, debug_text):
        """Process debug logs and update tables"""
        if not debug_text:
            return
            
        # Parse different types of debug data
        self._parse_detection_matches(debug_text)
        self._parse_traffic_lights(debug_text)
        self._parse_violations(debug_text)
        self._parse_performance(debug_text)
        self._parse_tracking(debug_text)
        
    def _parse_detection_matches(self, text):
        """Parse detection matching logs"""
        # Pattern for successful matches
        success_pattern = r'\[MATCH SUCCESS\] Detection at \(([\d.]+),([\d.]+)\) matched with track ID=(\d+).*IoU=([\d.]+), distance=([\d.]+)'
        # Pattern for failed matches
        fail_pattern = r'\[MATCH FAILED\] No suitable match found for car detection at \(([\d.]+), ([\d.]+)\)'
        
        # Find successful matches
        for match in re.finditer(success_pattern, text):
            x, y, track_id, iou, distance = match.groups()
            self._add_detection_row(f"({x},{y})", track_id, iou, distance, "✅ Matched")
            
        # Find failed matches
        for match in re.finditer(fail_pattern, text):
            x, y = match.groups()
            self._add_detection_row(f"({x},{y})", "—", "—", "—", "❌ Unmatched")
            
    def _parse_traffic_lights(self, text):
        """Parse traffic light status logs"""
        # Pattern for traffic light ratios
        ratio_pattern = r'\[DEBUG\] ratios: red=([\d.]+), yellow=([\d.]+), green=([\d.]+)'
        status_pattern = r'📝 Drawing traffic light status: (\w+) at bbox \[(\d+), (\d+), (\d+), (\d+)\]'
        
        ratios = re.findall(ratio_pattern, text)
        statuses = re.findall(status_pattern, text)
        
        # Combine ratios with statuses
        for i, ((red, yellow, green), (status, x1, y1, x2, y2)) in enumerate(zip(ratios, statuses)):
            detection_id = f"Traffic Light {i+1}"
            status_emoji = "🔴" if status == "red" else "🟡" if status == "yellow" else "🟢"
            self._add_traffic_light_row(detection_id, red, yellow, green, f"{status_emoji} {status.title()}")
            
    def _parse_violations(self, text):
        """Parse violation logs"""
        violation_pattern = r'🚨 Emitting RED LIGHT VIOLATION: Track ID (\d+)'
        improper_stop_pattern = r'\[VIOLATION\] Improper stop on crosswalk: Vehicle ID=(\d+) stopped on crosswalk during red light \(overlap=([\d.]+), speed=([\d.]+)\)'
        
        # Red light violations
        for match in re.finditer(violation_pattern, text):
            track_id = match.group(1)
            timestamp = datetime.now().strftime("%H:%M:%S")
            self._add_violation_row(track_id, "🚨 Red Light Violation", timestamp, "Vehicle crossed during red light")
            
        # Improper stops
        for match in re.finditer(improper_stop_pattern, text):
            track_id, overlap, speed = match.groups()
            timestamp = datetime.now().strftime("%H:%M:%S")
            details = f"Overlap: {overlap}, Speed: {speed}"
            self._add_violation_row(track_id, "🚧 Improper Stop", timestamp, details)
            
    def _parse_performance(self, text):
        """Parse performance metrics"""
        perf_pattern = r'\[PERF\] Emitting performance_stats_ready: {.*\'frame_idx\': (\d+).*\'fps\': ([\d.]+).*\'inference_time\': ([\d.]+).*\'device\': \'(\w+)\'.*}'
        stats_pattern = r'🟢 Stats Updated: FPS=([\d.]+), Inference=([\d.]+)ms, Traffic Light=(\w+)'
        
        # Performance stats
        for match in re.finditer(perf_pattern, text):
            frame_idx, fps, inference_time, device = match.groups()
            timestamp = datetime.now().strftime("%H:%M:%S")
            self._add_performance_row(frame_idx, fps, inference_time, device, "—", timestamp)
            
        # Basic stats
        for match in re.finditer(stats_pattern, text):
            fps, inference_time, traffic_light = match.groups()
            timestamp = datetime.now().strftime("%H:%M:%S")
            self._add_performance_row("—", fps, inference_time, "—", "—", timestamp)
            
    def _parse_tracking(self, text):
        """Parse vehicle tracking data"""
        track_pattern = r'Vehicle (\d+): ID=(\d+), center_y=([\d.]+), moving=(\w+), violating=(\w+)'
        bbox_pattern = r'ID=(\d+) bbox=\[([\d.]+), ([\d.]+), ([\d.]+), ([\d.]+)\] center_y=([\d.]+)'
        
        # Track vehicles
        for match in re.finditer(track_pattern, text):
            vehicle_num, track_id, center_y, moving, violating = match.groups()
            position = f"(—, {center_y})"
            status = "🟢 Active"
            confidence = "—"
            self._add_tracking_row(track_id, position, moving, violating, confidence, status)
            
        # Vehicle bboxes
        for match in re.finditer(bbox_pattern, text):
            track_id, x1, y1, x2, y2, center_y = match.groups()
            center_x = (float(x1) + float(x2)) / 2
            position = f"({center_x:.1f}, {center_y})"
            # Update existing row or add new one
            
    def _add_detection_row(self, detection, track_id, iou, distance, status):
        """Add row to detection table"""
        table = self.detection_table
        row = table.rowCount()
        table.insertRow(row)
        
        items = [detection, str(track_id), str(iou), str(distance), status]
        for col, item in enumerate(items):
            table_item = QTableWidgetItem(item)
            if status == "✅ Matched":
                table_item.setBackground(QColor(3, 218, 197, 50))
            elif status == "❌ Unmatched":
                table_item.setBackground(QColor(255, 87, 87, 50))
            table.setItem(row, col, table_item)
            
    def _add_traffic_light_row(self, detection, red, yellow, green, status):
        """Add row to traffic light table"""
        table = self.traffic_light_table
        row = table.rowCount()
        table.insertRow(row)
        
        items = [detection, red, yellow, green, status]
        for col, item in enumerate(items):
            table_item = QTableWidgetItem(item)
            if "🔴" in status:
                table_item.setBackground(QColor(255, 87, 87, 50))
            elif "🟡" in status:
                table_item.setBackground(QColor(255, 193, 7, 50))
            elif "🟢" in status:
                table_item.setBackground(QColor(76, 175, 80, 50))
            table.setItem(row, col, table_item)
            
    def _add_violation_row(self, track_id, violation_type, timestamp, details):
        """Add row to violation table"""
        table = self.violation_table
        row = table.rowCount()
        table.insertRow(row)
        
        items = [track_id, violation_type, timestamp, details]
        for col, item in enumerate(items):
            table_item = QTableWidgetItem(item)
            table_item.setBackground(QColor(255, 87, 87, 30))
            table.setItem(row, col, table_item)
            
    def _add_performance_row(self, frame, fps, inference_time, device, objects, timestamp):
        """Add row to performance table"""
        table = self.performance_table
        row = table.rowCount()
        table.insertRow(row)
        
        items = [str(frame), str(fps), str(inference_time), str(device), str(objects), timestamp]
        for col, item in enumerate(items):
            table_item = QTableWidgetItem(item)
            # Color code by device
            if device == "GPU":
                table_item.setBackground(QColor(3, 218, 197, 30))
            elif device == "CPU":
                table_item.setBackground(QColor(255, 193, 7, 30))
            table.setItem(row, col, table_item)
            
    def _add_tracking_row(self, track_id, position, moving, violating, confidence, status):
        """Add row to tracking table"""
        table = self.tracking_table
        row = table.rowCount()
        table.insertRow(row)
        
        items = [track_id, position, moving, violating, confidence, status]
        for col, item in enumerate(items):
            table_item = QTableWidgetItem(item)
            if violating == "True":
                table_item.setBackground(QColor(255, 87, 87, 50))
            elif moving == "True":
                table_item.setBackground(QColor(3, 218, 197, 30))
            table.setItem(row, col, table_item)
            
    def clear_all_tables(self):
        """Clear all tables"""
        for table in [self.detection_table, self.traffic_light_table, 
                     self.violation_table, self.performance_table, self.tracking_table]:
            table.setRowCount(0)
            
    def export_to_csv(self):
        """Export tables to CSV files"""
        # This would implement CSV export functionality
        print("CSV export functionality would be implemented here")
        
    def update_from_debug_text(self, debug_text):
        """Main method to update tables from debug text"""
        if debug_text:
            self.process_debug_logs(debug_text)
