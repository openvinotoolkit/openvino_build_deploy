from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem, 
    QLineEdit, QLabel, QPushButton, QSplitter, QHeaderView, 
    QComboBox, QGroupBox, QFormLayout
)
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QPixmap, QColor
from datetime import datetime
import os

class ViolationsTab(QWidget):
    """Tab for displaying and managing traffic violations."""
    
    def __init__(self):
        super().__init__()
        self.initUI()
        self.violations_data = []
        
    def initUI(self):
        """Initialize UI components"""        
        layout = QVBoxLayout(self)
        
        # Add status label for violations
        self.status_label = QLabel("🟢 Red Light Violation Detection Active")
        self.status_label.setStyleSheet("font-size: 16px; color: #22AA22; font-weight: bold; padding: 10px;")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        # Search and filter controls
        filter_layout = QHBoxLayout()
        
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search violations...")
        self.search_box.textChanged.connect(self.filter_violations)
        
        self.filter_combo = QComboBox()
        self.filter_combo.addItem("All Types")
        self.filter_combo.addItem("Red Light")
        self.filter_combo.addItem("Stop Sign")
        self.filter_combo.addItem("Speed")
        self.filter_combo.addItem("Lane")
        self.filter_combo.currentTextChanged.connect(self.filter_violations)
        
        filter_layout.addWidget(QLabel("Filter:"))
        filter_layout.addWidget(self.filter_combo)
        filter_layout.addStretch(1)
        filter_layout.addWidget(QLabel("Search:"))
        filter_layout.addWidget(self.search_box)
        
        layout.addLayout(filter_layout)
        
        # Splitter for table and details
        splitter = QSplitter(Qt.Horizontal)
        
        # Violations table
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["ID", "Type", "Timestamp", "Details", "Vehicle"])
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet("alternate-background-color: rgba(240, 240, 240, 100);")
        self.table.selectionModel().selectionChanged.connect(self.on_violation_selected)
        
        splitter.addWidget(self.table)
        
        # Violation details panel
        details_panel = QWidget()
        details_layout = QVBoxLayout(details_panel)
        
        # Violation info
        info_group = QGroupBox("Violation Details")
        info_layout = QFormLayout(info_group)
        
        self.violation_type_label = QLabel("--")
        self.violation_time_label = QLabel("--")
        self.violation_details_label = QLabel("--")
        self.violation_vehicle_label = QLabel("--")
        self.violation_location_label = QLabel("--")
        
        info_layout.addRow("Type:", self.violation_type_label)
        info_layout.addRow("Time:", self.violation_time_label)
        info_layout.addRow("Details:", self.violation_details_label)
        info_layout.addRow("Vehicle ID:", self.violation_vehicle_label)
        info_layout.addRow("Location:", self.violation_location_label)
        
        details_layout.addWidget(info_group)
        
        # Snapshot preview
        snapshot_group = QGroupBox("Violation Snapshot")
        snapshot_layout = QVBoxLayout(snapshot_group)
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(320, 240)
        self.preview_label.setStyleSheet("background-color: #222; border: 1px solid #444;")
        snapshot_layout.addWidget(self.preview_label)
        
        details_layout.addWidget(snapshot_group)
        
        # Actions
        actions_layout = QHBoxLayout()
        self.export_btn = QPushButton("Export Report")
        self.dismiss_btn = QPushButton("Dismiss")
        actions_layout.addWidget(self.export_btn)
        actions_layout.addWidget(self.dismiss_btn)
        
        details_layout.addLayout(actions_layout)
        details_layout.addStretch(1)
        
        splitter.addWidget(details_panel)
        splitter.setSizes([600, 400])  # Initial sizes
        
        layout.addWidget(splitter)
        
        # Status bar
        status_layout = QHBoxLayout()
        self.status_label = QLabel("No violations recorded")
        status_layout.addWidget(self.status_label)
        
        self.clear_btn = QPushButton("Clear All")
        status_layout.addWidget(self.clear_btn)
        
        layout.addLayout(status_layout)
    
    @Slot()
    def filter_violations(self):
        """Filter violations based on search text and type filter"""
        search_text = self.search_box.text().lower()
        filter_type = self.filter_combo.currentText()
        
        self.table.setRowCount(0)
        
        filtered_count = 0
        
        for violation in self.violations_data:
            # Filter by type
            if filter_type != "All Types":
                violation_type = violation.get('type', '').lower()
                filter_match = filter_type.lower() in violation_type
                if not filter_match:
                    continue
            
            # Filter by search text
            if search_text:
                # Search in multiple fields
                searchable_text = (
                    violation.get('type', '').lower() + ' ' +
                    violation.get('details', '').lower() + ' ' +
                    str(violation.get('vehicle_id', '')).lower() + ' ' +
                    str(violation.get('timestamp_str', '')).lower()
                )
                
                if search_text not in searchable_text:
                    continue
            
            # Add row for matching violation
            row_position = self.table.rowCount()
            self.table.insertRow(row_position)
            
            # Create violation ID
            violation_id = violation.get('id', filtered_count + 1)
            self.table.setItem(row_position, 0, QTableWidgetItem(str(violation_id)))
            
            # Format violation type
            violation_type = violation.get('type', '').replace('_', ' ').title()
            type_item = QTableWidgetItem(violation_type)
            
            # Color-code by violation type
            if 'red light' in violation_type.lower():
                type_item.setForeground(QColor(255, 0, 0))
            elif 'stop sign' in violation_type.lower():
                type_item.setForeground(QColor(255, 140, 0))
            elif 'speed' in violation_type.lower():
                type_item.setForeground(QColor(0, 0, 255))
            
            self.table.setItem(row_position, 1, type_item)
            
            # Format timestamp
            timestamp = violation.get('timestamp', 0)
            if isinstance(timestamp, (int, float)):
                timestamp_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
                violation['timestamp_str'] = timestamp_str  # Store for search
            else:
                timestamp_str = str(timestamp)
            
            self.table.setItem(row_position, 2, QTableWidgetItem(timestamp_str))
            
            # Details
            self.table.setItem(row_position, 3, QTableWidgetItem(violation.get('details', '')))
            
            # Vehicle ID
            self.table.setItem(row_position, 4, QTableWidgetItem(str(violation.get('vehicle_id', ''))))
            
            filtered_count += 1
        
        # Update status
        self.status_label.setText(f"Showing {filtered_count} of {len(self.violations_data)} violations")
    
    @Slot()
    def on_violation_selected(self):
        """Handle violation selection in table"""
        selected_items = self.table.selectedItems()
        if not selected_items:
            return
        
        row = selected_items[0].row()
        violation_id = int(self.table.item(row, 0).text())
        
        # Find violation in data
        violation = None
        for v in self.violations_data:
            if v.get('id', -1) == violation_id:
                violation = v
                break
        
        if not violation:
            return
        
        # Update details panel with enhanced information
        violation_type = violation.get('violation_type', 'red_light').replace('_', ' ').title()
        
        # Add traffic light confidence if available
        traffic_light_info = violation.get('traffic_light', {})
        if isinstance(traffic_light_info, dict) and 'confidence' in traffic_light_info:
            tl_color = traffic_light_info.get('color', 'red').upper()
            tl_confidence = traffic_light_info.get('confidence', 0.0)
            violation_type = f"{violation_type} - {tl_color} ({tl_confidence:.2f})"
        
        self.violation_type_label.setText(violation_type)
        
        # Format timestamp
        timestamp = violation.get('timestamp', 0)
        if isinstance(timestamp, (int, float)):
            timestamp_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
        else:
            timestamp_str = str(timestamp)
        self.violation_time_label.setText(timestamp_str)
        
        # Add vehicle details with confidence
        vehicle_type = violation.get('vehicle_type', 'Unknown').capitalize()
        vehicle_confidence = violation.get('confidence', 0.0)
        details = f"{vehicle_type} (Conf: {vehicle_confidence:.2f})"
        
        self.violation_details_label.setText(details)
        self.violation_vehicle_label.setText(str(violation.get('track_id', '--')))
        
        # Format location
        if 'bbox' in violation:
            bbox = violation['bbox']
            loc_str = f"X: {int(bbox[0])}, Y: {int(bbox[1])}"
        else:
            loc_str = "Unknown"
        self.violation_location_label.setText(loc_str)
        
        # Update snapshot if available
        if 'snapshot' in violation and violation['snapshot'] is not None:
            self.preview_label.setPixmap(QPixmap(violation['snapshot']))
        else:
            self.preview_label.setText("No snapshot available")
    
    @Slot(list)
    def update_violations(self, violations):
        """
        Update violations list.
        
        Args:
            violations: List of violation dictionaries
        """
        # Store violations data
        for violation in violations:
            # Check if already in list (by timestamp and vehicle ID)
            is_duplicate = False
            for existing in self.violations_data:
                if (existing.get('timestamp') == violation.get('timestamp') and
                    existing.get('vehicle_id') == violation.get('vehicle_id')):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                # Assign ID
                violation['id'] = len(self.violations_data) + 1
                self.violations_data.append(violation)
        
        # Refresh display
        self.filter_violations()
        
    def clear_all_violations(self):
        """Clear all violation data"""
        self.violations_data = []
        self.table.setRowCount(0)
        self.status_label.setText("No violations recorded")
        
        # Clear details
        self.violation_type_label.setText("--")
        self.violation_time_label.setText("--")
        self.violation_details_label.setText("--")
        self.violation_vehicle_label.setText("--")
        self.violation_location_label.setText("--")
        self.preview_label.clear()
        self.preview_label.setText("No violation selected")
    
    @Slot(object)
    def add_violation(self, violation):
        """
        Add a new violation to the table.
        
        Args:
            violation: Dictionary with violation information
        """
        try:
            # Update status to show active violations
            self.status_label.setText(f"🚨 RED LIGHT VIOLATION DETECTED - Total: {len(self.violations_data) + 1}")
            self.status_label.setStyleSheet("font-size: 16px; color: #FF2222; font-weight: bold; padding: 10px;")
            
            # Add to violations data
            self.violations_data.append(violation)
            
            # Add to table
            row = self.table.rowCount()
            self.table.insertRow(row)
            
            # Format timestamp
            timestamp_str = violation['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
            
            # Set table items with enhanced information
            self.table.setItem(row, 0, QTableWidgetItem(str(violation['id'])))
            
            # Check for traffic light confidence information
            traffic_light_info = violation.get('traffic_light', {})
            if traffic_light_info and isinstance(traffic_light_info, dict):
                tl_confidence = traffic_light_info.get('confidence', 0.0)
                violation_type = f"Red Light ({tl_confidence:.2f})"
            else:
                violation_type = "Red Light"
                
            self.table.setItem(row, 1, QTableWidgetItem(violation_type))
            self.table.setItem(row, 2, QTableWidgetItem(timestamp_str))
            
            # Add vehicle type and detection confidence
            vehicle_type = violation.get('vehicle_type', 'Unknown').capitalize()
            self.table.setItem(row, 3, QTableWidgetItem(f"{vehicle_type}"))
            self.table.setItem(row, 4, QTableWidgetItem(f"{violation.get('confidence', 0.0):.2f}"))
            
            # Highlight new row
            for col in range(5):
                item = self.table.item(row, col)
                if item:
                    item.setBackground(QColor(255, 200, 200))
            
            # Load snapshot if available
            if violation.get('snapshot_path') and os.path.exists(violation['snapshot_path']):
                pixmap = QPixmap(violation['snapshot_path'])
                if not pixmap.isNull():
                    # Store reference to avoid garbage collection
                    violation['pixmap'] = pixmap
        except Exception as e:
            print(f"❌ Error adding violation to UI: {e}")
            import traceback
            traceback.print_exc()
