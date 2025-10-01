from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget, QTableWidgetItem,
    QTabWidget, QScrollArea, QFrame, QGridLayout, QPushButton, QTextEdit
)
from PySide6.QtCore import Signal, Qt, QTimer
from PySide6.QtGui import QFont, QColor, QPalette
import re
import json
from datetime import datetime
from collections import defaultdict, deque


class AnalyticsTable(QTableWidget):
    """Enhanced table widget for analytics data"""
    def __init__(self, headers):
        super().__init__()
        self.setColumnCount(len(headers))
        self.setHorizontalHeaderLabels(headers)
        self.setAlternatingRowColors(True)
        self.setStyleSheet("""
            QTableWidget {
                background-color: #1e1e1e;
                color: #ffffff;
                gridline-color: #404040;
                border: 1px solid #404040;
                border-radius: 8px;
            }
            QTableWidget::item {
                padding: 8px;
                border-bottom: 1px solid #404040;
            }
            QTableWidget::item:selected {
                background-color: #03DAC5;
                color: #000000;
            }
            QHeaderView::section {
                background-color: #2d2d2d;
                color: #ffffff;
                padding: 10px;
                border: none;
                font-weight: bold;
            }
        """)
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setStretchLastSection(True)

    def add_row_data(self, data):
        row = self.rowCount()
        self.insertRow(row)
        for col, value in enumerate(data):
            item = QTableWidgetItem(str(value))
            item.setTextAlignment(Qt.AlignCenter)
            self.setItem(row, col, item)


class StatsCard(QFrame):
    """Statistics card widget"""
    def __init__(self, title, value, color="#03DAC5"):
        super().__init__()
        self.setFixedHeight(120)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: #2d2d2d;
                border: 2px solid {color};
                border-radius: 12px;
                padding: 16px;
            }}
        """)
        
        layout = QVBoxLayout(self)
        
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet(f"color: {color}; font-size: 14px; font-weight: bold;")
        self.title_label.setAlignment(Qt.AlignCenter)
        
        self.value_label = QLabel(str(value))
        self.value_label.setStyleSheet("color: #ffffff; font-size: 24px; font-weight: bold;")
        self.value_label.setAlignment(Qt.AlignCenter)
        
        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)

    def update_value(self, value):
        self.value_label.setText(str(value))


class LogAnalyzer:
    """Analyzes debug logs and extracts structured data"""
    
    @staticmethod
    def parse_match_debug(log_text):
        """Parse vehicle matching debug logs"""
        matches = []
        pattern = r'\[MATCH SUCCESS\] Detection at \(([^)]+)\) matched with track ID=(\d+)\s*-> STATUS: moving=(\w+), violating=(\w+), IoU=([^,]+), distance=([^)]+)'
        
        for match in re.finditer(pattern, log_text):
            position, track_id, moving, violating, iou, distance = match.groups()
            matches.append({
                'position': f"({position})",
                'track_id': track_id,
                'iou': float(iou),
                'distance': float(distance),
                'status': '✅ Matched',
                'moving': moving == 'True',
                'violating': violating == 'True'
            })
        
        # Parse failed matches
        failed_pattern = r'\[MATCH FAILED\] No suitable match found for car detection at \(([^)]+)\)'
        for match in re.finditer(failed_pattern, log_text):
            position = match.group(1)
            matches.append({
                'position': f"({position})",
                'track_id': '—',
                'iou': '—',
                'distance': '—',
                'status': '❌ Unmatched',
                'moving': False,
                'violating': False
            })
        
        return matches

    @staticmethod
    def parse_traffic_lights(log_text):
        """Parse traffic light detection logs"""
        lights = []
        pattern = r'\[DEBUG\] ratios: red=([^,]+), yellow=([^,]+), green=([^,]+).*?📝 Drawing traffic light status: (\w+) at bbox \[([^\]]+)\]'
        
        for i, match in enumerate(re.finditer(pattern, log_text, re.DOTALL)):
            red_ratio, yellow_ratio, green_ratio, status, bbox = match.groups()
            lights.append({
                'detection_id': f"Traffic Light {i+1}",
                'red_ratio': float(red_ratio),
                'yellow_ratio': float(yellow_ratio),
                'green_ratio': float(green_ratio),
                'status': f"🔴 {status.title()}" if status == 'red' else f"🟡 {status.title()}" if status == 'yellow' else f"🟢 {status.title()}",
                'bbox': bbox
            })
        
        return lights

    @staticmethod
    def parse_violations(log_text):
        """Parse violation logs"""
        violations = []
        pattern = r'🚨 Emitting RED LIGHT VIOLATION: Track ID (\d+)'
        
        for match in re.finditer(pattern, log_text):
            track_id = match.group(1)
            violations.append({
                'track_id': track_id,
                'violation_type': '🚨 Red Light Violation',
                'timestamp': datetime.now().strftime("%H:%M:%S")
            })
        
        # Parse crosswalk violations
        crosswalk_pattern = r'\[VIOLATION\] Improper stop on crosswalk: Vehicle ID=(\d+) stopped on crosswalk during red light \(overlap=([^,]+), speed=([^)]+)\)'
        for match in re.finditer(crosswalk_pattern, log_text):
            track_id, overlap, speed = match.groups()
            violations.append({
                'track_id': track_id,
                'violation_type': '🚧 Crosswalk Violation',
                'timestamp': datetime.now().strftime("%H:%M:%S"),
                'details': f"Overlap: {overlap}, Speed: {speed}"
            })
        
        return violations

    @staticmethod
    def parse_performance_stats(log_text):
        """Parse performance statistics"""
        stats = {}
        
        # Extract FPS
        fps_match = re.search(r'FPS: ([\d.]+)', log_text)
        if fps_match:
            stats['fps'] = float(fps_match.group(1))
        
        # Extract inference time
        inference_match = re.search(r'Inference: ([\d.]+)ms', log_text)
        if inference_match:
            stats['inference_time'] = float(inference_match.group(1))
        
        # Extract device
        device_match = re.search(r"'device': '(\w+)'", log_text)
        if device_match:
            stats['device'] = device_match.group(1)
        
        # Extract vehicle counts
        vehicle_match = re.search(r'Vehicles: (\d+) with IDs, (\d+) without IDs', log_text)
        if vehicle_match:
            stats['tracked_vehicles'] = int(vehicle_match.group(1))
            stats['untracked_vehicles'] = int(vehicle_match.group(2))
        
        # Extract detection counts
        detection_match = re.search(r'Detections count: (\d+)', log_text)
        if detection_match:
            stats['total_detections'] = int(detection_match.group(1))
        
        return stats

    @staticmethod
    def parse_bytetrack_stats(log_text):
        """Parse ByteTrack performance statistics"""
        stats = {}
        
        # Extract tracking states
        state_match = re.search(r'Current state: (\d+) tracked, (\d+) lost', log_text)
        if state_match:
            stats['tracked_count'] = int(state_match.group(1))
            stats['lost_count'] = int(state_match.group(2))
        
        # Extract matching results
        matched_match = re.search(r'Matched (\d+) tracks, created (\d+) new tracks, removed (\d+) expired tracks', log_text)
        if matched_match:
            stats['matched_tracks'] = int(matched_match.group(1))
            stats['new_tracks'] = int(matched_match.group(2))
            stats['removed_tracks'] = int(matched_match.group(3))
        
        return stats


class AnalyticsDashboard(QWidget):
    """Comprehensive analytics dashboard for video detection data"""
    
    def __init__(self):
        super().__init__()
        self.log_buffer = deque(maxlen=1000)  # Store recent logs
        self.setup_ui()
        self.setup_update_timer()
        
    def setup_ui(self):
        """Setup the dashboard UI"""
        main_layout = QVBoxLayout(self)
        
        # Header
        header = QLabel("Video Detection Analytics Dashboard")
        header.setStyleSheet("font-size: 24px; font-weight: bold; color: #03DAC5; padding: 16px;")
        header.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(header)
        
        # Stats cards
        stats_layout = QHBoxLayout()
        self.fps_card = StatsCard("FPS", "0.0", "#27ae60")
        self.inference_card = StatsCard("Inference (ms)", "0.0", "#3498db")
        self.vehicles_card = StatsCard("Tracked Vehicles", "0", "#e74c3c")
        self.violations_card = StatsCard("Violations", "0", "#f39c12")
        
        for card in [self.fps_card, self.inference_card, self.vehicles_card, self.violations_card]:
            stats_layout.addWidget(card)
        
        main_layout.addLayout(stats_layout)
        
        # Tabs for different analytics
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #404040;
                background-color: #1e1e1e;
            }
            QTabBar::tab {
                background-color: #2d2d2d;
                color: #ffffff;
                padding: 10px 20px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #03DAC5;
                color: #000000;
            }
        """)
        
        # Vehicle Matching Tab
        self.matching_tab = self.create_matching_tab()
        self.tab_widget.addTab(self.matching_tab, "🚗 Vehicle Matching")
        
        # Traffic Light Tab
        self.traffic_light_tab = self.create_traffic_light_tab()
        self.tab_widget.addTab(self.traffic_light_tab, "🚦 Traffic Lights")
        
        # Violations Tab
        self.violations_tab = self.create_violations_tab()
        self.tab_widget.addTab(self.violations_tab, "⚠️ Violations")
        
        # Performance Tab
        self.performance_tab = self.create_performance_tab()
        self.tab_widget.addTab(self.performance_tab, "📊 Performance")
        
        # Raw Logs Tab
        self.logs_tab = self.create_logs_tab()
        self.tab_widget.addTab(self.logs_tab, "📝 Raw Logs")
        
        main_layout.addWidget(self.tab_widget)
        
    def create_matching_tab(self):
        """Create vehicle matching analytics tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Matching table
        self.matching_table = AnalyticsTable([
            "Detection Position", "Track ID", "IoU", "Distance", "Match Status", "Moving", "Violating"
        ])
        layout.addWidget(self.matching_table)
        
        return widget
        
    def create_traffic_light_tab(self):
        """Create traffic light analytics tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Traffic light table
        self.traffic_light_table = AnalyticsTable([
            "Detection ID", "Red Ratio", "Yellow Ratio", "Green Ratio", "Status", "Bounding Box"
        ])
        layout.addWidget(self.traffic_light_table)
        
        return widget
        
    def create_violations_tab(self):
        """Create violations analytics tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Violations table
        self.violations_table = AnalyticsTable([
            "Track ID", "Violation Type", "Timestamp", "Details"
        ])
        layout.addWidget(self.violations_table)
        
        return widget
        
    def create_performance_tab(self):
        """Create performance analytics tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Performance table
        self.performance_table = AnalyticsTable([
            "Metric", "Current Value", "Unit", "Status"
        ])
        layout.addWidget(self.performance_table)
        
        # ByteTrack stats
        self.bytetrack_table = AnalyticsTable([
            "Metric", "Value", "Description"
        ])
        layout.addWidget(self.bytetrack_table)
        
        return widget
        
    def create_logs_tab(self):
        """Create raw logs display tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Control buttons
        controls = QHBoxLayout()
        clear_btn = QPushButton("Clear Logs")
        clear_btn.clicked.connect(self.clear_logs)
        clear_btn.setStyleSheet("padding: 8px 16px; background: #e74c3c; color: white; border-radius: 4px;")
        
        export_btn = QPushButton("Export Logs")
        export_btn.clicked.connect(self.export_logs)
        export_btn.setStyleSheet("padding: 8px 16px; background: #27ae60; color: white; border-radius: 4px;")
        
        controls.addWidget(clear_btn)
        controls.addWidget(export_btn)
        controls.addStretch()
        layout.addLayout(controls)
        
        # Logs display
        self.logs_display = QTextEdit()
        self.logs_display.setReadOnly(True)
        self.logs_display.setStyleSheet("""
            QTextEdit {
                background-color: #0d1117;
                color: #c9d1d9;
                font-family: 'Consolas', 'SF Mono', monospace;
                font-size: 12px;
                border: 1px solid #404040;
                border-radius: 8px;
                padding: 8px;
            }
        """)
        layout.addWidget(self.logs_display)
        
        return widget
        
    def setup_update_timer(self):
        """Setup timer for periodic updates"""
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.process_logs)
        self.update_timer.start(1000)  # Update every second
        
    def add_log_data(self, log_text):
        """Add new log data to the buffer"""
        self.log_buffer.append({
            'timestamp': datetime.now(),
            'content': log_text
        })
        
        # Update raw logs display
        self.logs_display.append(f"[{datetime.now().strftime('%H:%M:%S')}] {log_text}")
        
    def process_logs(self):
        """Process buffered logs and update analytics"""
        if not self.log_buffer:
            return
            
        # Combine recent logs
        recent_logs = '\n'.join([log['content'] for log in list(self.log_buffer)[-50:]])
        
        # Update matching analytics
        self.update_matching_analytics(recent_logs)
        
        # Update traffic light analytics
        self.update_traffic_light_analytics(recent_logs)
        
        # Update violations analytics
        self.update_violations_analytics(recent_logs)
        
        # Update performance analytics
        self.update_performance_analytics(recent_logs)
        
    def update_matching_analytics(self, log_text):
        """Update vehicle matching analytics"""
        matches = LogAnalyzer.parse_match_debug(log_text)
        
        # Clear and repopulate table
        self.matching_table.setRowCount(0)
        
        for match in matches[-20:]:  # Show last 20 matches
            row_data = [
                match['position'],
                match['track_id'],
                match['iou'],
                match['distance'],
                match['status'],
                '✅' if match['moving'] else '❌',
                '⚠️' if match['violating'] else '✅'
            ]
            self.matching_table.add_row_data(row_data)
            
    def update_traffic_light_analytics(self, log_text):
        """Update traffic light analytics"""
        lights = LogAnalyzer.parse_traffic_lights(log_text)
        
        # Clear and repopulate table
        self.traffic_light_table.setRowCount(0)
        
        for light in lights[-10:]:  # Show last 10 detections
            row_data = [
                light['detection_id'],
                f"{light['red_ratio']:.3f}",
                f"{light['yellow_ratio']:.3f}",
                f"{light['green_ratio']:.3f}",
                light['status'],
                light['bbox']
            ]
            self.traffic_light_table.add_row_data(row_data)
            
    def update_violations_analytics(self, log_text):
        """Update violations analytics"""
        violations = LogAnalyzer.parse_violations(log_text)
        
        # Update violations count card
        total_violations = len(violations)
        self.violations_card.update_value(total_violations)
        
        # Clear and repopulate table
        self.violations_table.setRowCount(0)
        
        for violation in violations[-20:]:  # Show last 20 violations
            row_data = [
                violation['track_id'],
                violation['violation_type'],
                violation['timestamp'],
                violation.get('details', '—')
            ]
            self.violations_table.add_row_data(row_data)
            
    def update_performance_analytics(self, log_text):
        """Update performance analytics"""
        stats = LogAnalyzer.parse_performance_stats(log_text)
        bytetrack_stats = LogAnalyzer.parse_bytetrack_stats(log_text)
        
        # Update stats cards
        if 'fps' in stats:
            self.fps_card.update_value(f"{stats['fps']:.2f}")
        if 'inference_time' in stats:
            self.inference_card.update_value(f"{stats['inference_time']:.1f}")
        if 'tracked_vehicles' in stats:
            self.vehicles_card.update_value(stats['tracked_vehicles'])
            
        # Update performance table
        self.performance_table.setRowCount(0)
        
        performance_metrics = [
            ("FPS", f"{stats.get('fps', 0):.2f}", "frames/sec", "🟢 Good" if stats.get('fps', 0) > 5 else "🔴 Low"),
            ("Inference Time", f"{stats.get('inference_time', 0):.1f}", "ms", "🟢 Fast" if stats.get('inference_time', 0) < 100 else "🟡 Slow"),
            ("Device", stats.get('device', 'Unknown'), "—", "🟢 GPU" if stats.get('device') == 'GPU' else "🟡 CPU"),
            ("Total Detections", stats.get('total_detections', 0), "objects", "🟢 Active"),
            ("Tracked Vehicles", stats.get('tracked_vehicles', 0), "vehicles", "🟢 Active"),
            ("Untracked Vehicles", stats.get('untracked_vehicles', 0), "vehicles", "🟡 Unmatched" if stats.get('untracked_vehicles', 0) > 0 else "🟢 All Tracked")
        ]
        
        for metric in performance_metrics:
            self.performance_table.add_row_data(metric)
            
        # Update ByteTrack table
        self.bytetrack_table.setRowCount(0)
        
        bytetrack_metrics = [
            ("Tracked Objects", bytetrack_stats.get('tracked_count', 0), "Currently being tracked"),
            ("Lost Objects", bytetrack_stats.get('lost_count', 0), "Lost tracking but may recover"),
            ("Matched Tracks", bytetrack_stats.get('matched_tracks', 0), "Successfully matched this frame"),
            ("New Tracks", bytetrack_stats.get('new_tracks', 0), "New objects started tracking"),
            ("Removed Tracks", bytetrack_stats.get('removed_tracks', 0), "Expired tracks removed")
        ]
        
        for metric in bytetrack_metrics:
            self.bytetrack_table.add_row_data(metric)
            
    def clear_logs(self):
        """Clear all logs"""
        self.log_buffer.clear()
        self.logs_display.clear()
        
    def export_logs(self):
        """Export logs to file"""
        from PySide6.QtWidgets import QFileDialog
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Logs", "analytics_logs.txt", "Text Files (*.txt)"
        )
        
        if file_path:
            with open(file_path, 'w') as f:
                for log in self.log_buffer:
                    f.write(f"[{log['timestamp']}] {log['content']}\n")
