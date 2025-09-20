"""
Advanced UI Design for Traffic Intersection Monitoring System
============================================================

This module implements a modern, dark-themed UI with Material Design principles
featuring tabbed navigation, live statistics, violation logs, and animated transitions.

Design Language:
- Dark theme (#121212, #1E1E1E backgrounds)
- Material Design with accent colors (green, red, yellow)
- Rounded corners, subtle shadows, elevation
- Animated transitions and responsive interactions
- Consistent typography (Segoe UI/Inter/Roboto)
- Icon-based navigation and controls

Author: Traffic Monitoring System
Date: July 2025
"""

import sys
from datetime import datetime
from typing import Optional, Dict, List, Any
import json

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QPushButton, QSlider, QCheckBox, QComboBox,
    QTableWidget, QTableWidgetItem, QFrame, QProgressBar, QTextEdit,
    QSplitter, QGroupBox, QGridLayout, QSpacerItem, QSizePolicy,
    QScrollArea, QStackedWidget, QToolBar, QStatusBar, QMenuBar,
    QMenu, QAction, QFileDialog, QMessageBox, QDialog, QDialogButtonBox,
    QFormLayout, QLineEdit, QSpinBox, QDoubleSpinBox, QHeaderView
)

from PySide6.QtCore import (
    Qt, QTimer, QPropertyAnimation, QEasingCurve, QRect, QSize,
    QThread, Signal, QObject, QParallelAnimationGroup, QSequentialAnimationGroup
)

from PySide6.QtGui import (
    QFont, QPixmap, QPainter, QPalette, QColor, QBrush, QLinearGradient,
    QIcon, QAction, QKeySequence, QPen, QFontMetrics
)

try:
    import pyqtgraph as pg
    PYQTGRAPH_AVAILABLE = True
except ImportError:
    PYQTGRAPH_AVAILABLE = False
    print("PyQtGraph not available. Charts will be disabled.")


class MaterialColors:
    """Material Design color palette for dark theme"""
    
    # Background colors
    BACKGROUND_PRIMARY = "#121212"
    BACKGROUND_SECONDARY = "#1E1E1E"
    BACKGROUND_TERTIARY = "#2D2D2D"
    
    # Surface colors
    SURFACE = "#1E1E1E"
    SURFACE_VARIANT = "#323232"
    
    # Accent colors
    PRIMARY = "#00BCD4"  # Cyan
    PRIMARY_VARIANT = "#00ACC1"
    SECONDARY = "#FFC107"  # Amber
    SECONDARY_VARIANT = "#FFB300"
    
    # Status colors
    SUCCESS = "#4CAF50"  # Green
    WARNING = "#FF9800"  # Orange
    ERROR = "#F44336"    # Red
    INFO = "#2196F3"     # Blue
    
    # Text colors
    TEXT_PRIMARY = "#FFFFFF"
    TEXT_SECONDARY = "#B0B0B0"
    TEXT_DISABLED = "#666666"
    
    # Border colors
    BORDER = "#404040"
    BORDER_LIGHT = "#606060"


class AnimationHelper:
    """Helper class for creating smooth animations"""
    
    @staticmethod
    def create_fade_animation(widget, duration=300, start_opacity=0.0, end_opacity=1.0):
        """Create fade in/out animation"""
        animation = QPropertyAnimation(widget, b"windowOpacity")
        animation.setDuration(duration)
        animation.setStartValue(start_opacity)
        animation.setEndValue(end_opacity)
        animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        return animation
    
    @staticmethod
    def create_slide_animation(widget, duration=300, start_pos=None, end_pos=None):
        """Create slide animation"""
        animation = QPropertyAnimation(widget, b"geometry")
        animation.setDuration(duration)
        if start_pos:
            animation.setStartValue(QRect(*start_pos))
        if end_pos:
            animation.setEndValue(QRect(*end_pos))
        animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        return animation


class ModernButton(QPushButton):
    """Custom button with modern styling and animations"""
    
    def __init__(self, text="", icon=None, button_type="primary", parent=None):
        super().__init__(text, parent)
        self.button_type = button_type
        self.setup_style()
        
        if icon:
            self.setIcon(icon)
            self.setIconSize(QSize(16, 16))
    
    def setup_style(self):
        """Apply modern button styling"""
        if self.button_type == "primary":
            bg_color = MaterialColors.PRIMARY
            hover_color = MaterialColors.PRIMARY_VARIANT
        elif self.button_type == "success":
            bg_color = MaterialColors.SUCCESS
            hover_color = "#45A049"
        elif self.button_type == "warning":
            bg_color = MaterialColors.WARNING
            hover_color = "#E68900"
        elif self.button_type == "error":
            bg_color = MaterialColors.ERROR
            hover_color = "#D32F2F"
        else:  # secondary
            bg_color = MaterialColors.SURFACE_VARIANT
            hover_color = MaterialColors.BORDER_LIGHT
        
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {bg_color};
                border: none;
                border-radius: 8px;
                color: {MaterialColors.TEXT_PRIMARY};
                font-weight: 500;
                padding: 8px 16px;
                min-height: 24px;
                font-size: 13px;
            }}
            QPushButton:hover {{
                background-color: {hover_color};
            }}
            QPushButton:pressed {{
                background-color: {bg_color};
            }}
            QPushButton:disabled {{
                background-color: {MaterialColors.SURFACE_VARIANT};
                color: {MaterialColors.TEXT_DISABLED};
            }}
        """)


class ModernCard(QFrame):
    """Modern card widget with shadow and rounded corners"""
    
    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.setup_style()
        self.setup_layout(title)
    
    def setup_style(self):
        """Apply card styling"""
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {MaterialColors.SURFACE};
                border-radius: 12px;
                border: 1px solid {MaterialColors.BORDER};
            }}
        """)
    
    def setup_layout(self, title):
        """Setup card layout with optional title"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        
        if title:
            title_label = QLabel(title)
            title_label.setStyleSheet(f"""
                QLabel {{
                    color: {MaterialColors.TEXT_PRIMARY};
                    font-size: 16px;
                    font-weight: 600;
                    margin-bottom: 8px;
                }}
            """)
            layout.addWidget(title_label)


class LiveStatsWidget(ModernCard):
    """Widget for displaying live statistics"""
    
    def __init__(self, parent=None):
        super().__init__("Live Statistics", parent)
        self.setup_stats_ui()
        
        # Initialize counters
        self.stats = {
            'vehicles_detected': 0,
            'pedestrians_detected': 0,
            'bicycles_detected': 0,
            'violations_total': 0,
            'violations_today': 0,
            'fps': 0.0
        }
        
        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.start(1000)  # Update every second
    
    def setup_stats_ui(self):
        """Setup the statistics display"""
        layout = self.layout()
        
        # Create stats grid
        stats_grid = QGridLayout()
        
        # Vehicle counts
        self.vehicle_label = self.create_stat_widget("Vehicles", "0", MaterialColors.SUCCESS)
        self.pedestrian_label = self.create_stat_widget("Pedestrians", "0", MaterialColors.INFO)
        self.bicycle_label = self.create_stat_widget("Bicycles", "0", MaterialColors.WARNING)
        
        # Violation counts
        self.violations_total_label = self.create_stat_widget("Total Violations", "0", MaterialColors.ERROR)
        self.violations_today_label = self.create_stat_widget("Today's Violations", "0", MaterialColors.ERROR)
        
        # Performance
        self.fps_label = self.create_stat_widget("FPS", "0.0", MaterialColors.PRIMARY)
        
        # Add to grid
        stats_grid.addWidget(self.vehicle_label, 0, 0)
        stats_grid.addWidget(self.pedestrian_label, 0, 1)
        stats_grid.addWidget(self.bicycle_label, 0, 2)
        stats_grid.addWidget(self.violations_total_label, 1, 0)
        stats_grid.addWidget(self.violations_today_label, 1, 1)
        stats_grid.addWidget(self.fps_label, 1, 2)
        
        layout.addLayout(stats_grid)
    
    def create_stat_widget(self, title, value, color):
        """Create a single stat display widget"""
        container = QFrame()
        container.setStyleSheet(f"""
            QFrame {{
                background-color: {MaterialColors.BACKGROUND_SECONDARY};
                border-radius: 8px;
                padding: 12px;
                margin: 4px;
            }}
        """)
        
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)
        
        title_label = QLabel(title)
        title_label.setStyleSheet(f"""
            QLabel {{
                color: {MaterialColors.TEXT_SECONDARY};
                font-size: 12px;
                font-weight: 500;
            }}
        """)
        
        value_label = QLabel(value)
        value_label.setStyleSheet(f"""
            QLabel {{
                color: {color};
                font-size: 24px;
                font-weight: 700;
            }}
        """)
        
        layout.addWidget(title_label)
        layout.addWidget(value_label)
        
        # Store reference to value label for updates
        container.value_label = value_label
        
        return container
    
    def update_stats(self, new_stats):
        """Update statistics with new data"""
        self.stats.update(new_stats)
    
    def update_display(self):
        """Update the display with current stats"""
        self.vehicle_label.value_label.setText(str(self.stats['vehicles_detected']))
        self.pedestrian_label.value_label.setText(str(self.stats['pedestrians_detected']))
        self.bicycle_label.value_label.setText(str(self.stats['bicycles_detected']))
        self.violations_total_label.value_label.setText(str(self.stats['violations_total']))
        self.violations_today_label.value_label.setText(str(self.stats['violations_today']))
        self.fps_label.value_label.setText(f"{self.stats['fps']:.1f}")


class ViolationLogWidget(ModernCard):
    """Advanced violation log table with search and filtering"""
    
    def __init__(self, parent=None):
        super().__init__("Violation Logs", parent)
        self.setup_log_ui()
        self.violations = []
    
    def setup_log_ui(self):
        """Setup the violation log interface"""
        layout = self.layout()
        
        # Controls header
        controls_layout = QHBoxLayout()
        
        # Search box
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search violations...")
        self.search_box.setStyleSheet(f"""
            QLineEdit {{
                background-color: {MaterialColors.BACKGROUND_SECONDARY};
                border: 1px solid {MaterialColors.BORDER};
                border-radius: 6px;
                padding: 8px 12px;
                color: {MaterialColors.TEXT_PRIMARY};
                font-size: 13px;
            }}
            QLineEdit:focus {{
                border-color: {MaterialColors.PRIMARY};
            }}
        """)
        self.search_box.textChanged.connect(self.filter_violations)
        
        # Filter dropdown
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["All Violations", "Red Light", "Crosswalk", "Speed"])
        self.filter_combo.setStyleSheet(f"""
            QComboBox {{
                background-color: {MaterialColors.BACKGROUND_SECONDARY};
                border: 1px solid {MaterialColors.BORDER};
                border-radius: 6px;
                padding: 8px 12px;
                color: {MaterialColors.TEXT_PRIMARY};
                min-width: 120px;
            }}
            QComboBox::drop-down {{
                border: none;
            }}
            QComboBox::down-arrow {{
                image: none;
                border: none;
            }}
        """)
        self.filter_combo.currentTextChanged.connect(self.filter_violations)
        
        # Export button
        self.export_btn = ModernButton("Export Report", button_type="secondary")
        self.export_btn.clicked.connect(self.export_violations)
        
        # Clear button
        self.clear_btn = ModernButton("Clear Logs", button_type="error")
        self.clear_btn.clicked.connect(self.clear_violations)
        
        controls_layout.addWidget(QLabel("Search:"))
        controls_layout.addWidget(self.search_box)
        controls_layout.addWidget(QLabel("Filter:"))
        controls_layout.addWidget(self.filter_combo)
        controls_layout.addStretch()
        controls_layout.addWidget(self.export_btn)
        controls_layout.addWidget(self.clear_btn)
        
        layout.addLayout(controls_layout)
        
        # Violation table
        self.violation_table = QTableWidget()
        self.violation_table.setColumnCount(6)
        self.violation_table.setHorizontalHeaderLabels([
            "ID", "Type", "Timestamp", "Object ID", "Confidence", "Actions"
        ])
        
        # Style the table
        self.violation_table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {MaterialColors.BACKGROUND_SECONDARY};
                border: 1px solid {MaterialColors.BORDER};
                border-radius: 8px;
                gridline-color: {MaterialColors.BORDER};
                color: {MaterialColors.TEXT_PRIMARY};
                selection-background-color: {MaterialColors.PRIMARY};
            }}
            QTableWidget::item {{
                padding: 8px;
                border-bottom: 1px solid {MaterialColors.BORDER};
            }}
            QTableWidget::item:selected {{
                background-color: {MaterialColors.PRIMARY};
            }}
            QHeaderView::section {{
                background-color: {MaterialColors.SURFACE_VARIANT};
                color: {MaterialColors.TEXT_PRIMARY};
                padding: 8px;
                border: none;
                font-weight: 600;
            }}
        """)
        
        # Configure table
        self.violation_table.horizontalHeader().setStretchLastSection(True)
        self.violation_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.violation_table.setAlternatingRowColors(True)
        
        layout.addWidget(self.violation_table)
    
    def add_violation(self, violation_data):
        """Add a new violation to the log"""
        violation = {
            'id': len(self.violations) + 1,
            'type': violation_data.get('type', 'Unknown'),
            'timestamp': violation_data.get('timestamp', datetime.now()),
            'object_id': violation_data.get('object_id', 'N/A'),
            'confidence': violation_data.get('confidence', 0.0),
            'snapshot_path': violation_data.get('snapshot_path', None)
        }
        
        self.violations.append(violation)
        self.update_table()
    
    def update_table(self):
        """Update the violation table display"""
        self.violation_table.setRowCount(len(self.violations))
        
        for row, violation in enumerate(self.violations):
            # ID
            self.violation_table.setItem(row, 0, QTableWidgetItem(str(violation['id'])))
            
            # Type
            type_item = QTableWidgetItem(violation['type'])
            if violation['type'] == 'Red Light':
                type_item.setForeground(QColor(MaterialColors.ERROR))
            elif violation['type'] == 'Crosswalk':
                type_item.setForeground(QColor(MaterialColors.WARNING))
            self.violation_table.setItem(row, 1, type_item)
            
            # Timestamp
            if isinstance(violation['timestamp'], datetime):
                timestamp_str = violation['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
            else:
                timestamp_str = str(violation['timestamp'])
            self.violation_table.setItem(row, 2, QTableWidgetItem(timestamp_str))
            
            # Object ID
            self.violation_table.setItem(row, 3, QTableWidgetItem(str(violation['object_id'])))
            
            # Confidence
            confidence_str = f"{violation['confidence']:.2f}" if isinstance(violation['confidence'], float) else str(violation['confidence'])
            self.violation_table.setItem(row, 4, QTableWidgetItem(confidence_str))
            
            # Actions (View Snapshot button)
            if violation['snapshot_path']:
                view_btn = ModernButton("View", button_type="primary")
                view_btn.clicked.connect(lambda checked, path=violation['snapshot_path']: self.view_snapshot(path))
                self.violation_table.setCellWidget(row, 5, view_btn)
    
    def filter_violations(self):
        """Filter violations based on search and filter criteria"""
        search_text = self.search_box.text().lower()
        filter_type = self.filter_combo.currentText()
        
        for row in range(self.violation_table.rowCount()):
            show_row = True
            
            # Search filter
            if search_text:
                row_text = ""
                for col in range(self.violation_table.columnCount() - 1):  # Exclude actions column
                    item = self.violation_table.item(row, col)
                    if item:
                        row_text += item.text().lower() + " "
                
                if search_text not in row_text:
                    show_row = False
            
            # Type filter
            if filter_type != "All Violations":
                type_item = self.violation_table.item(row, 1)
                if type_item and type_item.text() != filter_type:
                    show_row = False
            
            self.violation_table.setRowHidden(row, not show_row)
    
    def view_snapshot(self, snapshot_path):
        """View violation snapshot in a popup"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Violation Snapshot")
        dialog.setModal(True)
        dialog.resize(600, 400)
        
        layout = QVBoxLayout(dialog)
        
        try:
            pixmap = QPixmap(snapshot_path)
            if not pixmap.isNull():
                label = QLabel()
                label.setPixmap(pixmap.scaled(580, 380, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                layout.addWidget(label)
            else:
                layout.addWidget(QLabel("Error: Could not load snapshot"))
        except Exception as e:
            layout.addWidget(QLabel(f"Error loading snapshot: {str(e)}"))
        
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(dialog.accept)
        layout.addWidget(buttons)
        
        dialog.exec()
    
    def export_violations(self):
        """Export violations to CSV file"""
        if not self.violations:
            QMessageBox.information(self, "Export", "No violations to export.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Violations", "violations_report.csv", "CSV Files (*.csv)"
        )
        
        if file_path:
            try:
                import csv
                with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['ID', 'Type', 'Timestamp', 'Object ID', 'Confidence', 'Snapshot Path'])
                    
                    for violation in self.violations:
                        timestamp_str = violation['timestamp'].strftime("%Y-%m-%d %H:%M:%S") if isinstance(violation['timestamp'], datetime) else str(violation['timestamp'])
                        writer.writerow([
                            violation['id'],
                            violation['type'],
                            timestamp_str,
                            violation['object_id'],
                            violation['confidence'],
                            violation.get('snapshot_path', '')
                        ])
                
                QMessageBox.information(self, "Export", f"Violations exported to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export violations:\n{str(e)}")
    
    def clear_violations(self):
        """Clear all violation logs"""
        reply = QMessageBox.question(
            self, "Clear Logs", "Are you sure you want to clear all violation logs?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.violations.clear()
            self.violation_table.setRowCount(0)


class VideoControlsWidget(QFrame):
    """Modern video control toolbar"""
    
    # Signals
    load_video_requested = Signal()
    play_requested = Signal()
    pause_requested = Signal()
    stop_requested = Signal()
    snapshot_requested = Signal()
    fullscreen_requested = Signal()
    position_changed = Signal(int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_controls()
        self.is_playing = False
        self.video_duration = 0
    
    def setup_controls(self):
        """Setup video control interface"""
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {MaterialColors.SURFACE};
                border-top: 1px solid {MaterialColors.BORDER};
                padding: 8px;
            }}
        """)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 8, 16, 8)
        layout.setSpacing(12)
        
        # Load video button
        self.load_btn = ModernButton("📂 Load Video", button_type="secondary")
        self.load_btn.clicked.connect(self.load_video_requested.emit)
        layout.addWidget(self.load_btn)
        
        layout.addWidget(self.create_separator())
        
        # Playback controls
        self.play_btn = ModernButton("▶️ Play", button_type="success")
        self.play_btn.clicked.connect(self.toggle_playback)
        layout.addWidget(self.play_btn)
        
        self.stop_btn = ModernButton("⏹️ Stop", button_type="error")
        self.stop_btn.clicked.connect(self.stop_video)
        layout.addWidget(self.stop_btn)
        
        layout.addWidget(self.create_separator())
        
        # Progress slider
        self.progress_slider = QSlider(Qt.Orientation.Horizontal)
        self.progress_slider.setMinimum(0)
        self.progress_slider.setMaximum(100)
        self.progress_slider.setValue(0)
        self.progress_slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                background: {MaterialColors.BACKGROUND_SECONDARY};
                height: 6px;
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: {MaterialColors.PRIMARY};
                width: 16px;
                height: 16px;
                border-radius: 8px;
                margin: -5px 0;
            }}
            QSlider::sub-page:horizontal {{
                background: {MaterialColors.PRIMARY};
                border-radius: 3px;
            }}
        """)
        self.progress_slider.sliderPressed.connect(self.on_slider_pressed)
        self.progress_slider.sliderReleased.connect(self.on_slider_released)
        layout.addWidget(self.progress_slider, 1)
        
        # Time display
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setStyleSheet(f"""
            QLabel {{
                color: {MaterialColors.TEXT_SECONDARY};
                font-family: 'Consolas', monospace;
                font-size: 12px;
                min-width: 80px;
            }}
        """)
        layout.addWidget(self.time_label)
        
        layout.addWidget(self.create_separator())
        
        # Additional controls
        self.snapshot_btn = ModernButton("📸 Snapshot", button_type="secondary")
        self.snapshot_btn.clicked.connect(self.snapshot_requested.emit)
        layout.addWidget(self.snapshot_btn)
        
        self.fullscreen_btn = ModernButton("⛶ Fullscreen", button_type="secondary")
        self.fullscreen_btn.clicked.connect(self.fullscreen_requested.emit)
        layout.addWidget(self.fullscreen_btn)
    
    def create_separator(self):
        """Create a visual separator"""
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.VLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setStyleSheet(f"""
            QFrame {{
                color: {MaterialColors.BORDER};
                max-width: 1px;
            }}
        """)
        return separator
    
    def toggle_playback(self):
        """Toggle between play and pause"""
        if self.is_playing:
            self.pause_video()
        else:
            self.play_video()
    
    def play_video(self):
        """Start video playback"""
        self.is_playing = True
        self.play_btn.setText("⏸️ Pause")
        self.play_requested.emit()
    
    def pause_video(self):
        """Pause video playback"""
        self.is_playing = False
        self.play_btn.setText("▶️ Play")
        self.pause_requested.emit()
    
    def stop_video(self):
        """Stop video playback"""
        self.is_playing = False
        self.play_btn.setText("▶️ Play")
        self.progress_slider.setValue(0)
        self.update_time_display(0, self.video_duration)
        self.stop_requested.emit()
    
    def update_progress(self, position, duration):
        """Update progress slider and time display"""
        self.video_duration = duration
        if duration > 0:
            progress = int((position / duration) * 100)
            self.progress_slider.setValue(progress)
        self.update_time_display(position, duration)
    
    def update_time_display(self, position, duration):
        """Update time display label"""
        pos_time = self.format_time(position)
        dur_time = self.format_time(duration)
        self.time_label.setText(f"{pos_time} / {dur_time}")
    
    def format_time(self, seconds):
        """Format time in MM:SS format"""
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    def on_slider_pressed(self):
        """Handle slider press - pause during seeking"""
        self.seeking = True
    
    def on_slider_released(self):
        """Handle slider release - emit position change"""
        self.seeking = False
        position = (self.progress_slider.value() / 100.0) * self.video_duration
        self.position_changed.emit(int(position))


class DetectionControlsWidget(ModernCard):
    """Controls for detection and tracking settings"""
    
    # Signals
    detection_toggled = Signal(bool)
    tracking_toggled = Signal(bool)
    confidence_changed = Signal(float)
    class_visibility_changed = Signal(str, bool)
    
    def __init__(self, parent=None):
        super().__init__("Detection Controls", parent)
        self.setup_controls()
    
    def setup_controls(self):
        """Setup detection control interface"""
        layout = self.layout()
        
        # Main toggle switches
        toggles_layout = QHBoxLayout()
        
        self.detection_checkbox = QCheckBox("Enable Detection")
        self.detection_checkbox.setChecked(True)
        self.detection_checkbox.toggled.connect(self.detection_toggled.emit)
        self.detection_checkbox.setStyleSheet(f"""
            QCheckBox {{
                color: {MaterialColors.TEXT_PRIMARY};
                font-size: 14px;
                font-weight: 500;
            }}
            QCheckBox::indicator {{
                width: 20px;
                height: 20px;
                border-radius: 3px;
                border: 2px solid {MaterialColors.BORDER};
            }}
            QCheckBox::indicator:checked {{
                background-color: {MaterialColors.SUCCESS};
                border-color: {MaterialColors.SUCCESS};
            }}
        """)
        
        self.tracking_checkbox = QCheckBox("Enable Tracking")
        self.tracking_checkbox.setChecked(True)
        self.tracking_checkbox.toggled.connect(self.tracking_toggled.emit)
        self.tracking_checkbox.setStyleSheet(self.detection_checkbox.styleSheet())
        
        toggles_layout.addWidget(self.detection_checkbox)
        toggles_layout.addWidget(self.tracking_checkbox)
        toggles_layout.addStretch()
        
        layout.addLayout(toggles_layout)
        
        # Confidence threshold
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("Confidence Threshold:"))
        
        self.confidence_slider = QSlider(Qt.Orientation.Horizontal)
        self.confidence_slider.setMinimum(1)
        self.confidence_slider.setMaximum(100)
        self.confidence_slider.setValue(50)
        self.confidence_slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                background: {MaterialColors.BACKGROUND_SECONDARY};
                height: 4px;
                border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                background: {MaterialColors.PRIMARY};
                width: 14px;
                height: 14px;
                border-radius: 7px;
                margin: -5px 0;
            }}
            QSlider::sub-page:horizontal {{
                background: {MaterialColors.PRIMARY};
                border-radius: 2px;
            }}
        """)
        self.confidence_slider.valueChanged.connect(self.on_confidence_changed)
        
        self.confidence_label = QLabel("0.50")
        self.confidence_label.setMinimumWidth(40)
        self.confidence_label.setStyleSheet(f"""
            QLabel {{
                color: {MaterialColors.TEXT_PRIMARY};
                font-family: 'Consolas', monospace;
                font-size: 12px;
            }}
        """)
        
        conf_layout.addWidget(self.confidence_slider)
        conf_layout.addWidget(self.confidence_label)
        
        layout.addLayout(conf_layout)
        
        # Class visibility toggles
        class_layout = QVBoxLayout()
        class_layout.addWidget(QLabel("Object Classes:"))
        
        class_grid = QGridLayout()
        
        self.class_checkboxes = {}
        classes = [
            ("Vehicles", MaterialColors.SUCCESS),
            ("Pedestrians", MaterialColors.INFO),
            ("Bicycles", MaterialColors.WARNING)
        ]
        
        for i, (class_name, color) in enumerate(classes):
            checkbox = QCheckBox(class_name)
            checkbox.setChecked(True)
            checkbox.setStyleSheet(f"""
                QCheckBox {{
                    color: {color};
                    font-size: 13px;
                    font-weight: 500;
                }}
                QCheckBox::indicator {{
                    width: 16px;
                    height: 16px;
                    border-radius: 3px;
                    border: 2px solid {color};
                }}
                QCheckBox::indicator:checked {{
                    background-color: {color};
                }}
            """)
            checkbox.toggled.connect(lambda checked, name=class_name: self.class_visibility_changed.emit(name, checked))
            
            self.class_checkboxes[class_name] = checkbox
            class_grid.addWidget(checkbox, i // 2, i % 2)
        
        class_layout.addLayout(class_grid)
        layout.addLayout(class_layout)
    
    def on_confidence_changed(self, value):
        """Handle confidence slider change"""
        confidence = value / 100.0
        self.confidence_label.setText(f"{confidence:.2f}")
        self.confidence_changed.emit(confidence)


class AnalyticsWidget(QWidget):
    """Analytics dashboard with charts and statistics"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_analytics()
        
        # Initialize data
        self.traffic_data = []
        self.violation_data = []
        
        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_charts)
        self.update_timer.start(5000)  # Update every 5 seconds
    
    def setup_analytics(self):
        """Setup analytics dashboard"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)
        
        # Title
        title = QLabel("Analytics Dashboard")
        title.setStyleSheet(f"""
            QLabel {{
                color: {MaterialColors.TEXT_PRIMARY};
                font-size: 24px;
                font-weight: 700;
                margin-bottom: 16px;
            }}
        """)
        layout.addWidget(title)
        
        if PYQTGRAPH_AVAILABLE:
            self.setup_charts(layout)
        else:
            # Fallback when PyQtGraph is not available
            fallback_label = QLabel("PyQtGraph not available. Charts disabled.")
            fallback_label.setStyleSheet(f"""
                QLabel {{
                    color: {MaterialColors.TEXT_SECONDARY};
                    font-size: 14px;
                    text-align: center;
                    padding: 40px;
                    background-color: {MaterialColors.SURFACE};
                    border-radius: 8px;
                }}
            """)
            layout.addWidget(fallback_label)
    
    def setup_charts(self, layout):
        """Setup chart widgets"""
        # Configure PyQtGraph
        pg.setConfigOption('background', MaterialColors.BACKGROUND_SECONDARY)
        pg.setConfigOption('foreground', MaterialColors.TEXT_PRIMARY)
        
        # Charts container
        charts_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Traffic flow chart
        traffic_widget = pg.PlotWidget(title="Traffic Flow (Objects/Minute)")
        traffic_widget.setLabel('left', 'Count')
        traffic_widget.setLabel('bottom', 'Time (minutes)')
        traffic_widget.showGrid(x=True, y=True, alpha=0.3)
        
        self.traffic_curve = traffic_widget.plot(
            pen=pg.mkPen(color=MaterialColors.PRIMARY, width=2),
            name="Traffic Flow"
        )
        
        charts_splitter.addWidget(traffic_widget)
        
        # Violations chart
        violations_widget = pg.PlotWidget(title="Violations Over Time")
        violations_widget.setLabel('left', 'Violations')
        violations_widget.setLabel('bottom', 'Time (minutes)')
        violations_widget.showGrid(x=True, y=True, alpha=0.3)
        
        self.violations_curve = violations_widget.plot(
            pen=pg.mkPen(color=MaterialColors.ERROR, width=2),
            name="Violations"
        )
        
        charts_splitter.addWidget(violations_widget)
        
        layout.addWidget(charts_splitter)
    
    def update_charts(self):
        """Update chart data"""
        if not PYQTGRAPH_AVAILABLE:
            return
        
        # Simulate or get real data
        import time
        current_time = time.time()
        
        # Update traffic data (you would replace this with real data)
        if len(self.traffic_data) > 60:  # Keep last 60 points
            self.traffic_data.pop(0)
        
        # Add new data point (replace with real traffic count)
        self.traffic_data.append((current_time, len(self.traffic_data) % 20 + 10))
        
        # Update violation data
        if len(self.violation_data) > 60:
            self.violation_data.pop(0)
        
        # Add new data point (replace with real violation count)
        self.violation_data.append((current_time, len(self.violation_data) % 5))
        
        # Update curves
        if self.traffic_data:
            x_traffic = [point[0] - self.traffic_data[0][0] for point in self.traffic_data]
            y_traffic = [point[1] for point in self.traffic_data]
            self.traffic_curve.setData(x_traffic, y_traffic)
        
        if self.violation_data:
            x_violations = [point[0] - self.violation_data[0][0] for point in self.violation_data]
            y_violations = [point[1] for point in self.violation_data]
            self.violations_curve.setData(x_violations, y_violations)


class TrafficMonitoringUI(QMainWindow):
    """Main application window with advanced UI design"""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.setup_styling()
        self.setup_shortcuts()
        
        # Initialize components
        self.video_loaded = False
        self.detection_active = False
    
    def setup_ui(self):
        """Setup the main user interface"""
        self.setWindowTitle("Traffic Intersection Monitoring System")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)
        
        # Set application icon (if available)
        # self.setWindowIcon(QIcon("path/to/icon.png"))
        
        # Central widget with tab layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet(f"""
            QTabWidget::pane {{
                border: none;
                background-color: {MaterialColors.BACKGROUND_PRIMARY};
            }}
            QTabBar::tab {{
                background-color: {MaterialColors.SURFACE};
                color: {MaterialColors.TEXT_SECONDARY};
                padding: 12px 24px;
                margin-right: 2px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                font-size: 14px;
                font-weight: 500;
            }}
            QTabBar::tab:selected {{
                background-color: {MaterialColors.BACKGROUND_PRIMARY};
                color: {MaterialColors.TEXT_PRIMARY};
                border-bottom: 2px solid {MaterialColors.PRIMARY};
            }}
            QTabBar::tab:hover {{
                background-color: {MaterialColors.SURFACE_VARIANT};
                color: {MaterialColors.TEXT_PRIMARY};
            }}
        """)
        
        # Create tabs
        self.create_live_monitoring_tab()
        self.create_detection_tab()
        self.create_violations_tab()
        self.create_analytics_tab()
        
        layout.addWidget(self.tab_widget)
        
        # Create status bar
        self.setup_status_bar()
        
        # Create menu bar
        self.setup_menu_bar()
    
    def create_live_monitoring_tab(self):
        """Create the live monitoring tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Main content area
        content_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Video display area
        video_frame = QFrame()
        video_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {MaterialColors.BACKGROUND_SECONDARY};
                border: 2px solid {MaterialColors.BORDER};
                border-radius: 8px;
            }}
        """)
        video_layout = QVBoxLayout(video_frame)
        
        # Video placeholder
        self.video_label = QLabel("Load a video to start monitoring")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet(f"""
            QLabel {{
                color: {MaterialColors.TEXT_SECONDARY};
                font-size: 18px;
                padding: 40px;
            }}
        """)
        video_layout.addWidget(self.video_label)
        
        content_splitter.addWidget(video_frame)
        
        # Side panel with stats
        side_panel = QWidget()
        side_panel.setMaximumWidth(350)
        side_layout = QVBoxLayout(side_panel)
        side_layout.setContentsMargins(16, 16, 16, 16)
        side_layout.setSpacing(16)
        
        # Live stats
        self.live_stats = LiveStatsWidget()
        side_layout.addWidget(self.live_stats)
        
        # Detection controls
        self.detection_controls = DetectionControlsWidget()
        side_layout.addWidget(self.detection_controls)
        
        side_layout.addStretch()
        
        content_splitter.addWidget(side_panel)
        content_splitter.setSizes([800, 350])
        
        layout.addWidget(content_splitter)
        
        # Video controls at bottom
        self.video_controls = VideoControlsWidget()
        layout.addWidget(self.video_controls)
        
        self.tab_widget.addTab(tab, "🎥 Live Monitoring")
    
    def create_detection_tab(self):
        """Create the detection visualization tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)
        
        # Title
        title = QLabel("Detection & Tracking Visualization")
        title.setStyleSheet(f"""
            QLabel {{
                color: {MaterialColors.TEXT_PRIMARY};
                font-size: 20px;
                font-weight: 600;
                margin-bottom: 8px;
            }}
        """)
        layout.addWidget(title)
        
        # Detection display area
        detection_frame = QFrame()
        detection_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {MaterialColors.BACKGROUND_SECONDARY};
                border: 1px solid {MaterialColors.BORDER};
                border-radius: 8px;
                min-height: 400px;
            }}
        """)
        
        detection_layout = QVBoxLayout(detection_frame)
        
        # Detection placeholder
        detection_label = QLabel("Detection visualization will appear here")
        detection_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        detection_label.setStyleSheet(f"""
            QLabel {{
                color: {MaterialColors.TEXT_SECONDARY};
                font-size: 16px;
                padding: 40px;
            }}
        """)
        detection_layout.addWidget(detection_label)
        
        layout.addWidget(detection_frame, 1)
        
        # Detection legend
        legend_frame = ModernCard("Detection Legend")
        legend_layout = QHBoxLayout()
        
        legend_items = [
            ("🚗 Vehicles", MaterialColors.SUCCESS),
            ("🚶 Pedestrians", MaterialColors.INFO),
            ("🚴 Bicycles", MaterialColors.WARNING),
            ("🚨 Violations", MaterialColors.ERROR)
        ]
        
        for text, color in legend_items:
            legend_label = QLabel(text)
            legend_label.setStyleSheet(f"""
                QLabel {{
                    color: {color};
                    font-size: 14px;
                    font-weight: 500;
                    padding: 8px 16px;
                    background-color: {MaterialColors.BACKGROUND_SECONDARY};
                    border-radius: 6px;
                    margin: 4px;
                }}
            """)
            legend_layout.addWidget(legend_label)
        
        legend_layout.addStretch()
        legend_frame.layout().addLayout(legend_layout)
        layout.addWidget(legend_frame)
        
        self.tab_widget.addTab(tab, "🎯 Detection")
    
    def create_violations_tab(self):
        """Create the violations and statistics tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)
        
        # Title
        title = QLabel("Violations & Reports")
        title.setStyleSheet(f"""
            QLabel {{
                color: {MaterialColors.TEXT_PRIMARY};
                font-size: 20px;
                font-weight: 600;
                margin-bottom: 8px;
            }}
        """)
        layout.addWidget(title)
        
        # Violation log widget
        self.violation_log = ViolationLogWidget()
        layout.addWidget(self.violation_log, 1)
        
        self.tab_widget.addTab(tab, "🚨 Violations")
    
    def create_analytics_tab(self):
        """Create the analytics dashboard tab"""
        self.analytics_widget = AnalyticsWidget()
        self.tab_widget.addTab(self.analytics_widget, "📊 Analytics")
    
    def setup_styling(self):
        """Apply global styling to the application"""
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {MaterialColors.BACKGROUND_PRIMARY};
                color: {MaterialColors.TEXT_PRIMARY};
            }}
            QLabel {{
                color: {MaterialColors.TEXT_PRIMARY};
            }}
            QWidget {{
                background-color: {MaterialColors.BACKGROUND_PRIMARY};
                color: {MaterialColors.TEXT_PRIMARY};
            }}
        """)
        
        # Set application font
        font = QFont("Segoe UI", 10)
        font.setHintingPreference(QFont.HintingPreference.PreferDefaultHinting)
        self.setFont(font)
        QApplication.instance().setFont(font)
    
    def setup_menu_bar(self):
        """Setup the application menu bar"""
        menubar = self.menuBar()
        menubar.setStyleSheet(f"""
            QMenuBar {{
                background-color: {MaterialColors.SURFACE};
                color: {MaterialColors.TEXT_PRIMARY};
                border-bottom: 1px solid {MaterialColors.BORDER};
                padding: 4px;
            }}
            QMenuBar::item {{
                background: transparent;
                padding: 6px 12px;
                border-radius: 4px;
            }}
            QMenuBar::item:selected {{
                background-color: {MaterialColors.SURFACE_VARIANT};
            }}
            QMenu {{
                background-color: {MaterialColors.SURFACE};
                color: {MaterialColors.TEXT_PRIMARY};
                border: 1px solid {MaterialColors.BORDER};
                border-radius: 6px;
                padding: 4px;
            }}
            QMenu::item {{
                padding: 6px 16px;
                border-radius: 4px;
            }}
            QMenu::item:selected {{
                background-color: {MaterialColors.PRIMARY};
            }}
        """)
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        load_action = QAction("Load Video...", self)
        load_action.setShortcut(QKeySequence.StandardKey.Open)
        load_action.triggered.connect(self.load_video)
        file_menu.addAction(load_action)
        
        file_menu.addSeparator()
        
        export_action = QAction("Export Report...", self)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self.export_report)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu("View")
        
        fullscreen_action = QAction("Fullscreen", self)
        fullscreen_action.setShortcut("F11")
        fullscreen_action.triggered.connect(self.toggle_fullscreen)
        view_menu.addAction(fullscreen_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def setup_status_bar(self):
        """Setup the status bar"""
        status_bar = self.statusBar()
        status_bar.setStyleSheet(f"""
            QStatusBar {{
                background-color: {MaterialColors.SURFACE};
                color: {MaterialColors.TEXT_SECONDARY};
                border-top: 1px solid {MaterialColors.BORDER};
                font-size: 12px;
            }}
        """)
        
        status_bar.showMessage("Ready - Load a video to start monitoring")
    
    def setup_shortcuts(self):
        """Setup keyboard shortcuts"""
        # Space for play/pause
        play_shortcut = QAction(self)
        play_shortcut.setShortcut("Space")
        play_shortcut.triggered.connect(self.video_controls.toggle_playback)
        self.addAction(play_shortcut)
        
        # S for snapshot
        snapshot_shortcut = QAction(self)
        snapshot_shortcut.setShortcut("S")
        snapshot_shortcut.triggered.connect(self.video_controls.snapshot_requested.emit)
        self.addAction(snapshot_shortcut)
    
    def load_video(self):
        """Load video file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Video", "", 
            "Video Files (*.mp4 *.avi *.mov *.mkv *.flv *.wmv);;All Files (*)"
        )
        
        if file_path:
            self.video_loaded = True
            self.video_label.setText(f"Video loaded: {file_path.split('/')[-1]}")
            self.statusBar().showMessage(f"Video loaded: {file_path}")
            
            # Enable controls
            self.video_controls.load_btn.setText("📂 Change Video")
    
    def export_report(self):
        """Export monitoring report"""
        self.violation_log.export_violations()
    
    def toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()
    
    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self, "About",
            "Traffic Intersection Monitoring System\n\n"
            "An advanced AI-powered system for monitoring traffic\n"
            "intersections and detecting violations.\n\n"
            "Features:\n"
            "• Real-time object detection and tracking\n"
            "• Violation detection and logging\n"
            "• Advanced analytics and reporting\n"
            "• Modern dark theme UI\n\n"
            "Built with PySide6, OpenCV, and YOLO"
        )
    
    def update_stats(self, stats_data):
        """Update live statistics"""
        self.live_stats.update_stats(stats_data)
    
    def add_violation(self, violation_data):
        """Add a new violation to the log"""
        self.violation_log.add_violation(violation_data)
        
        # Update status bar
        self.statusBar().showMessage(
            f"New violation detected: {violation_data.get('type', 'Unknown')}"
        )


class SplashScreen(QWidget):
    """Modern splash screen for application startup"""
    
    def __init__(self):
        super().__init__()
        self.setup_splash()
        
        # Auto-close timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.close)
        self.timer.start(3000)  # Show for 3 seconds
    
    def setup_splash(self):
        """Setup splash screen UI"""
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(400, 300)
        
        # Center on screen
        screen = QApplication.primaryScreen().geometry()
        self.move(
            (screen.width() - self.width()) // 2,
            (screen.height() - self.height()) // 2
        )
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(20)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Main container
        container = QFrame()
        container.setStyleSheet(f"""
            QFrame {{
                background-color: {MaterialColors.SURFACE};
                border-radius: 16px;
                border: 1px solid {MaterialColors.BORDER};
            }}
        """)
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(40, 40, 40, 40)
        container_layout.setSpacing(20)
        container_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Logo/Icon placeholder
        logo_label = QLabel("🚦")
        logo_label.setStyleSheet(f"""
            QLabel {{
                font-size: 48px;
                color: {MaterialColors.PRIMARY};
                margin-bottom: 10px;
            }}
        """)
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        container_layout.addWidget(logo_label)
        
        # Title
        title_label = QLabel("Traffic Monitoring System")
        title_label.setStyleSheet(f"""
            QLabel {{
                color: {MaterialColors.TEXT_PRIMARY};
                font-size: 20px;
                font-weight: 700;
                margin-bottom: 5px;
            }}
        """)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        container_layout.addWidget(title_label)
        
        # Subtitle
        subtitle_label = QLabel("Loading AI-Powered Monitoring...")
        subtitle_label.setStyleSheet(f"""
            QLabel {{
                color: {MaterialColors.TEXT_SECONDARY};
                font-size: 14px;
                margin-bottom: 20px;
            }}
        """)
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        container_layout.addWidget(subtitle_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                border: none;
                border-radius: 6px;
                background-color: {MaterialColors.BACKGROUND_SECONDARY};
                height: 12px;
            }}
            QProgressBar::chunk {{
                background-color: {MaterialColors.PRIMARY};
                border-radius: 6px;
            }}
        """)
        container_layout.addWidget(self.progress_bar)
        
        layout.addWidget(container)


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("Traffic Monitoring System")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("Traffic AI Solutions")
    
    # Set application style
    app.setStyle("Fusion")
    
    # Apply dark palette
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(MaterialColors.BACKGROUND_PRIMARY))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(MaterialColors.TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Base, QColor(MaterialColors.BACKGROUND_SECONDARY))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(MaterialColors.SURFACE_VARIANT))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(MaterialColors.SURFACE))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(MaterialColors.TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Text, QColor(MaterialColors.TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Button, QColor(MaterialColors.SURFACE))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(MaterialColors.TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(MaterialColors.ERROR))
    palette.setColor(QPalette.ColorRole.Link, QColor(MaterialColors.PRIMARY))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(MaterialColors.PRIMARY))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(MaterialColors.TEXT_PRIMARY))
    app.setPalette(palette)
    
    # Show splash screen
    splash = SplashScreen()
    splash.show()
    
    # Process events to show splash
    app.processEvents()
    
    # Create and show main window
    window = TrafficMonitoringUI()
    
    # Close splash and show main window
    splash.close()
    window.show()
    
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
