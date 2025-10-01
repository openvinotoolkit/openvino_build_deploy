"""
Violations Tab - Traffic violation detection and evidence management dashboard
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
                               QTableWidget, QTableWidgetItem, QHeaderView,
                               QGroupBox, QLabel, QPushButton, QComboBox,
                               QDateEdit, QLineEdit, QTextEdit, QFrame,
                               QCheckBox, QSpinBox, QProgressBar, QTabWidget)
from PySide6.QtCore import Qt, Signal, QDateTime, QDate, QTimer
from PySide6.QtGui import QFont, QColor, QPixmap, QIcon

class ViolationItem:
    """Data class for violation items"""
    
    def __init__(self, violation_id, violation_type, timestamp, location, 
                 vehicle_id, evidence_path=None, status="pending"):
        self.violation_id = violation_id
        self.violation_type = violation_type
        self.timestamp = timestamp
        self.location = location
        self.vehicle_id = vehicle_id
        self.evidence_path = evidence_path
        self.status = status
        self.confidence = 0.95
        self.reviewed = False

class ViolationsTab(QWidget):
    """
    Traffic Violations Dashboard with evidence management
    
    Features:
    - Real-time violation detection alerts
    - Evidence gallery with images/videos
    - Violation classification and filtering
    - Manual review and acknowledgment
    - Export capabilities for reporting
    - Statistics and trends analysis
    """
    
    # Signals
    violation_acknowledged = Signal(str)
    violation_exported = Signal(list)
    evidence_viewed = Signal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.violations_data = []
        self.filtered_violations = []
        
        self._setup_ui()
        
        # Sample data for demonstration
        self._add_sample_violations()
        
        print("🚨 Violations Tab initialized")
    
    def _setup_ui(self):
        """Setup the violations dashboard UI"""
        # Main layout
        layout = QVBoxLayout(self)
        
        # Header with summary stats
        header = self._create_header()
        layout.addWidget(header)
        
        # Main content splitter
        main_splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(main_splitter)
        
        # Left panel - Violations list and filters
        left_panel = self._create_left_panel()
        main_splitter.addWidget(left_panel)
        
        # Right panel - Evidence and details
        right_panel = self._create_right_panel()
        main_splitter.addWidget(right_panel)
        
        # Set splitter proportions
        main_splitter.setSizes([600, 400])
    
    def _create_header(self):
        """Create header with violation statistics"""
        header = QFrame()
        header.setFixedHeight(80)
        header.setStyleSheet("""
            QFrame {
                background-color: #34495e;
                border-radius: 8px;
                margin-bottom: 10px;
            }
        """)
        
        layout = QHBoxLayout(header)
        layout.setContentsMargins(20, 10, 20, 10)
        
        # Statistics cards
        stats = [
            ("Total Violations", "127", "#e74c3c"),
            ("Pending Review", "23", "#f39c12"),
            ("Acknowledged", "104", "#27ae60"),
            ("Today's Count", "8", "#3498db")
        ]
        
        for title, value, color in stats:
            card = self._create_stat_card(title, value, color)
            layout.addWidget(card)
        
        layout.addStretch()
        
        # Quick actions
        actions_layout = QVBoxLayout()
        
        export_btn = QPushButton("📊 Export Report")
        export_btn.setFixedSize(120, 30)
        export_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #2980b9;
            }}
        """)
        export_btn.clicked.connect(self._export_violations)
        actions_layout.addWidget(export_btn)
        
        clear_btn = QPushButton("🗑️ Clear Old")
        clear_btn.setFixedSize(120, 30)
        clear_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #95a5a6;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #7f8c8d;
            }}
        """)
        clear_btn.clicked.connect(self._clear_old_violations)
        actions_layout.addWidget(clear_btn)
        
        layout.addLayout(actions_layout)
        
        return header
    
    def _create_stat_card(self, title, value, color):
        """Create a statistics card"""
        card = QFrame()
        card.setFixedSize(140, 50)
        card.setStyleSheet(f"""
            QFrame {{
                background-color: {color};
                border-radius: 6px;
                margin: 5px;
            }}
        """)
        
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 5, 10, 5)
        
        # Value
        value_label = QLabel(value)
        value_label.setFont(QFont("Segoe UI", 16, QFont.Bold))
        value_label.setStyleSheet("color: white;")
        value_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(value_label)
        
        # Title
        title_label = QLabel(title)
        title_label.setFont(QFont("Segoe UI", 8))
        title_label.setStyleSheet("color: white;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        return card
    
    def _create_left_panel(self):
        """Create left panel with violations list and filters"""
        panel = QFrame()
        layout = QVBoxLayout(panel)
        
        # Filters section
        filters = self._create_filters()
        layout.addWidget(filters)
        
        # Violations table
        table_group = QGroupBox("Violations List")
        table_layout = QVBoxLayout(table_group)
        
        self.violations_table = QTableWidget()
        self.violations_table.setColumnCount(7)
        self.violations_table.setHorizontalHeaderLabels([
            "ID", "Type", "Time", "Location", "Vehicle", "Status", "Actions"
        ])
        
        # Configure table
        header = self.violations_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        self.violations_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.violations_table.setAlternatingRowColors(True)
        
        # Connect selection change
        self.violations_table.itemSelectionChanged.connect(self._on_violation_selected)
        
        table_layout.addWidget(self.violations_table)
        layout.addWidget(table_group)
        
        return panel
    
    def _create_filters(self):
        """Create violation filters section"""
        filters = QGroupBox("Filters")
        filters.setFixedHeight(120)
        layout = QVBoxLayout(filters)
        
        # First row of filters
        row1 = QHBoxLayout()
        
        # Violation type filter
        row1.addWidget(QLabel("Type:"))
        self.type_filter = QComboBox()
        self.type_filter.addItems([
            "All Types",
            "Red Light Violation",
            "Speed Violation", 
            "Wrong Direction",
            "Illegal Turn",
            "Lane Violation"
        ])
        self.type_filter.currentTextChanged.connect(self._apply_filters)
        row1.addWidget(self.type_filter)
        
        # Status filter
        row1.addWidget(QLabel("Status:"))
        self.status_filter = QComboBox()
        self.status_filter.addItems(["All Status", "Pending", "Acknowledged", "Dismissed"])
        self.status_filter.currentTextChanged.connect(self._apply_filters)
        row1.addWidget(self.status_filter)
        
        layout.addLayout(row1)
        
        # Second row of filters
        row2 = QHBoxLayout()
        
        # Date range
        row2.addWidget(QLabel("From:"))
        self.date_from = QDateEdit()
        self.date_from.setDate(QDate.currentDate().addDays(-7))
        self.date_from.dateChanged.connect(self._apply_filters)
        row2.addWidget(self.date_from)
        
        row2.addWidget(QLabel("To:"))
        self.date_to = QDateEdit()
        self.date_to.setDate(QDate.currentDate())
        self.date_to.dateChanged.connect(self._apply_filters)
        row2.addWidget(self.date_to)
        
        # Search
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search violations...")
        self.search_input.textChanged.connect(self._apply_filters)
        row2.addWidget(self.search_input)
        
        layout.addLayout(row2)
        
        # Filter controls
        row3 = QHBoxLayout()
        
        self.show_acknowledged_cb = QCheckBox("Show Acknowledged")
        self.show_acknowledged_cb.setChecked(True)
        self.show_acknowledged_cb.toggled.connect(self._apply_filters)
        row3.addWidget(self.show_acknowledged_cb)
        
        row3.addStretch()
        
        clear_filters_btn = QPushButton("Clear Filters")
        clear_filters_btn.clicked.connect(self._clear_filters)
        row3.addWidget(clear_filters_btn)
        
        layout.addLayout(row3)
        
        return filters
    
    def _create_right_panel(self):
        """Create right panel with evidence and details"""
        panel = QFrame()
        layout = QVBoxLayout(panel)
        
        # Evidence section
        evidence_section = self._create_evidence_section()
        layout.addWidget(evidence_section)
        
        # Details section
        details_section = self._create_details_section()
        layout.addWidget(details_section)
        
        # Actions section
        actions_section = self._create_actions_section()
        layout.addWidget(actions_section)
        
        return panel
    
    def _create_evidence_section(self):
        """Create evidence viewing section"""
        section = QGroupBox("Evidence")
        layout = QVBoxLayout(section)
        
        # Evidence tabs
        self.evidence_tabs = QTabWidget()
        
        # Image evidence tab
        image_tab = QWidget()
        image_layout = QVBoxLayout(image_tab)
        
        self.evidence_image = QLabel("Select a violation to view evidence")
        self.evidence_image.setMinimumSize(300, 200)
        self.evidence_image.setAlignment(Qt.AlignCenter)
        self.evidence_image.setStyleSheet("""
            QLabel {
                border: 2px dashed #bdc3c7;
                border-radius: 8px;
                background-color: #ecf0f1;
                color: #7f8c8d;
            }
        """)
        image_layout.addWidget(self.evidence_image)
        
        # Image controls
        image_controls = QHBoxLayout()
        
        zoom_in_btn = QPushButton("🔍+")
        zoom_in_btn.setFixedSize(30, 30)
        image_controls.addWidget(zoom_in_btn)
        
        zoom_out_btn = QPushButton("🔍-")
        zoom_out_btn.setFixedSize(30, 30)
        image_controls.addWidget(zoom_out_btn)
        
        image_controls.addStretch()
        
        save_evidence_btn = QPushButton("💾 Save")
        save_evidence_btn.clicked.connect(self._save_evidence)
        image_controls.addWidget(save_evidence_btn)
        
        image_layout.addLayout(image_controls)
        
        self.evidence_tabs.addTab(image_tab, "📷 Image")
        
        # Video evidence tab
        video_tab = QWidget()
        video_layout = QVBoxLayout(video_tab)
        
        video_placeholder = QLabel("Video evidence player\n(Feature coming soon)")
        video_placeholder.setMinimumSize(300, 200)
        video_placeholder.setAlignment(Qt.AlignCenter)
        video_placeholder.setStyleSheet("""
            QLabel {
                border: 2px dashed #bdc3c7;
                border-radius: 8px;
                background-color: #ecf0f1;
                color: #7f8c8d;
            }
        """)
        video_layout.addWidget(video_placeholder)
        
        self.evidence_tabs.addTab(video_tab, "🎬 Video")
        
        layout.addWidget(self.evidence_tabs)
        
        return section
    
    def _create_details_section(self):
        """Create violation details section"""
        section = QGroupBox("Violation Details")
        layout = QVBoxLayout(section)
        
        # Details text area
        self.details_text = QTextEdit()
        self.details_text.setMaximumHeight(120)
        self.details_text.setPlaceholderText("Select a violation to view details...")
        layout.addWidget(self.details_text)
        
        # Metadata grid
        metadata_layout = QHBoxLayout()
        
        # Left column
        left_metadata = QVBoxLayout()
        self.confidence_label = QLabel("Confidence: --")
        self.camera_label = QLabel("Camera: --")
        self.coordinates_label = QLabel("Coordinates: --")
        
        left_metadata.addWidget(self.confidence_label)
        left_metadata.addWidget(self.camera_label)
        left_metadata.addWidget(self.coordinates_label)
        
        metadata_layout.addLayout(left_metadata)
        
        # Right column
        right_metadata = QVBoxLayout()
        self.weather_label = QLabel("Weather: --")
        self.visibility_label = QLabel("Visibility: --")
        self.reviewed_by_label = QLabel("Reviewed by: --")
        
        right_metadata.addWidget(self.weather_label)
        right_metadata.addWidget(self.visibility_label)
        right_metadata.addWidget(self.reviewed_by_label)
        
        metadata_layout.addLayout(right_metadata)
        
        layout.addLayout(metadata_layout)
        
        return section
    
    def _create_actions_section(self):
        """Create violation actions section"""
        section = QGroupBox("Actions")
        layout = QVBoxLayout(section)
        
        # Primary actions
        primary_actions = QHBoxLayout()
        
        self.acknowledge_btn = QPushButton("✅ Acknowledge")
        self.acknowledge_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #229954;
            }
        """)
        self.acknowledge_btn.clicked.connect(self._acknowledge_violation)
        self.acknowledge_btn.setEnabled(False)
        primary_actions.addWidget(self.acknowledge_btn)
        
        self.dismiss_btn = QPushButton("❌ Dismiss")
        self.dismiss_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        self.dismiss_btn.clicked.connect(self._dismiss_violation)
        self.dismiss_btn.setEnabled(False)
        primary_actions.addWidget(self.dismiss_btn)
        
        layout.addLayout(primary_actions)
        
        # Secondary actions
        secondary_actions = QHBoxLayout()
        
        flag_btn = QPushButton("🏃 Flag for Review")
        flag_btn.clicked.connect(self._flag_violation)
        secondary_actions.addWidget(flag_btn)
        
        notes_btn = QPushButton("📝 Add Notes")
        notes_btn.clicked.connect(self._add_notes)
        secondary_actions.addWidget(notes_btn)
        
        layout.addLayout(secondary_actions)
        
        # Bulk actions
        bulk_actions = QHBoxLayout()
        
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self._select_all_violations)
        bulk_actions.addWidget(select_all_btn)
        
        bulk_acknowledge_btn = QPushButton("Bulk Acknowledge")
        bulk_acknowledge_btn.clicked.connect(self._bulk_acknowledge)
        bulk_actions.addWidget(bulk_acknowledge_btn)
        
        layout.addLayout(bulk_actions)
        
        return section
    
    def _add_sample_violations(self):
        """Add sample violation data for demonstration"""
        sample_violations = [
            ViolationItem("V001", "Red Light Violation", QDateTime.currentDateTime().addSecs(-3600), 
                         "Main St & Oak Ave", "ABC123", "evidence_001.jpg", "pending"),
            ViolationItem("V002", "Speed Violation", QDateTime.currentDateTime().addSecs(-7200),
                         "Highway 101", "XYZ789", "evidence_002.jpg", "acknowledged"),
            ViolationItem("V003", "Wrong Direction", QDateTime.currentDateTime().addSecs(-10800),
                         "5th St & Pine St", "DEF456", "evidence_003.jpg", "pending"),
            ViolationItem("V004", "Illegal Turn", QDateTime.currentDateTime().addSecs(-14400),
                         "Market St", "GHI321", "evidence_004.jpg", "dismissed"),
            ViolationItem("V005", "Lane Violation", QDateTime.currentDateTime().addSecs(-18000),
                         "Broadway & 2nd St", "JKL654", "evidence_005.jpg", "pending"),
        ]
        
        self.violations_data = sample_violations
        self._populate_violations_table()
    
    def _populate_violations_table(self):
        """Populate the violations table with data"""
        self.violations_table.setRowCount(len(self.violations_data))
        
        for row, violation in enumerate(self.violations_data):
            # ID
            self.violations_table.setItem(row, 0, QTableWidgetItem(violation.violation_id))
            
            # Type
            type_item = QTableWidgetItem(violation.violation_type)
            if violation.violation_type == "Red Light Violation":
                type_item.setBackground(QColor(231, 76, 60, 50))  # Light red
            elif violation.violation_type == "Speed Violation":
                type_item.setBackground(QColor(243, 156, 18, 50))  # Light orange
            self.violations_table.setItem(row, 1, type_item)
            
            # Time
            time_str = violation.timestamp.toString("MM/dd hh:mm")
            self.violations_table.setItem(row, 2, QTableWidgetItem(time_str))
            
            # Location
            self.violations_table.setItem(row, 3, QTableWidgetItem(violation.location))
            
            # Vehicle
            self.violations_table.setItem(row, 4, QTableWidgetItem(violation.vehicle_id))
            
            # Status
            status_item = QTableWidgetItem(violation.status.title())
            if violation.status == "pending":
                status_item.setBackground(QColor(243, 156, 18, 50))  # Orange
            elif violation.status == "acknowledged":
                status_item.setBackground(QColor(39, 174, 96, 50))  # Green
            elif violation.status == "dismissed":
                status_item.setBackground(QColor(149, 165, 166, 50))  # Gray
            self.violations_table.setItem(row, 5, status_item)
            
            # Actions (placeholder)
            actions_item = QTableWidgetItem("View")
            self.violations_table.setItem(row, 6, actions_item)
    
    def _apply_filters(self):
        """Apply current filters to violations list"""
        # This would filter the violations based on current filter settings
        self._populate_violations_table()
        print("🚨 Filters applied to violations list")
    
    def _clear_filters(self):
        """Clear all filters"""
        self.type_filter.setCurrentIndex(0)
        self.status_filter.setCurrentIndex(0)
        self.date_from.setDate(QDate.currentDate().addDays(-7))
        self.date_to.setDate(QDate.currentDate())
        self.search_input.clear()
        self.show_acknowledged_cb.setChecked(True)
        print("🚨 Filters cleared")
    
    def _on_violation_selected(self):
        """Handle violation selection"""
        current_row = self.violations_table.currentRow()
        if current_row >= 0 and current_row < len(self.violations_data):
            violation = self.violations_data[current_row]
            self._show_violation_details(violation)
            
            # Enable action buttons
            self.acknowledge_btn.setEnabled(violation.status == "pending")
            self.dismiss_btn.setEnabled(violation.status == "pending")
    
    def _show_violation_details(self, violation):
        """Show details for selected violation"""
        # Update details text
        details = f"""
Violation ID: {violation.violation_id}
Type: {violation.violation_type}
Timestamp: {violation.timestamp.toString("yyyy-MM-dd hh:mm:ss")}
Location: {violation.location}
Vehicle ID: {violation.vehicle_id}
Status: {violation.status.title()}

Description: This violation was automatically detected by the traffic monitoring system.
The evidence has been captured and is available for review.
        """.strip()
        
        self.details_text.setPlainText(details)
        
        # Update metadata labels
        self.confidence_label.setText(f"Confidence: {violation.confidence:.1%}")
        self.camera_label.setText("Camera: Camera 1")
        self.coordinates_label.setText("Coordinates: 37.7749, -122.4194")
        self.weather_label.setText("Weather: Clear")
        self.visibility_label.setText("Visibility: Good")
        self.reviewed_by_label.setText("Reviewed by: --")
        
        # Update evidence display
        self.evidence_image.setText(f"Evidence: {violation.evidence_path or 'No evidence available'}")
        
        print(f"🚨 Showing details for violation: {violation.violation_id}")
    
    def _acknowledge_violation(self):
        """Acknowledge the selected violation"""
        current_row = self.violations_table.currentRow()
        if current_row >= 0:
            violation = self.violations_data[current_row]
            violation.status = "acknowledged"
            violation.reviewed = True
            
            # Update table
            self._populate_violations_table()
            
            # Disable buttons
            self.acknowledge_btn.setEnabled(False)
            self.dismiss_btn.setEnabled(False)
            
            # Emit signal
            self.violation_acknowledged.emit(violation.violation_id)
            
            print(f"🚨 Violation {violation.violation_id} acknowledged")
    
    def _dismiss_violation(self):
        """Dismiss the selected violation"""
        current_row = self.violations_table.currentRow()
        if current_row >= 0:
            violation = self.violations_data[current_row]
            violation.status = "dismissed"
            violation.reviewed = True
            
            # Update table
            self._populate_violations_table()
            
            # Disable buttons
            self.acknowledge_btn.setEnabled(False)
            self.dismiss_btn.setEnabled(False)
            
            print(f"🚨 Violation {violation.violation_id} dismissed")
    
    def _flag_violation(self):
        """Flag violation for manual review"""
        current_row = self.violations_table.currentRow()
        if current_row >= 0:
            violation = self.violations_data[current_row]
            print(f"🚨 Violation {violation.violation_id} flagged for review")
    
    def _add_notes(self):
        """Add notes to violation"""
        current_row = self.violations_table.currentRow()
        if current_row >= 0:
            violation = self.violations_data[current_row]
            print(f"🚨 Adding notes to violation {violation.violation_id}")
    
    def _select_all_violations(self):
        """Select all violations in table"""
        self.violations_table.selectAll()
        print("🚨 All violations selected")
    
    def _bulk_acknowledge(self):
        """Acknowledge all selected violations"""
        selected_rows = set()
        for item in self.violations_table.selectedItems():
            selected_rows.add(item.row())
        
        for row in selected_rows:
            if row < len(self.violations_data):
                violation = self.violations_data[row]
                if violation.status == "pending":
                    violation.status = "acknowledged"
                    violation.reviewed = True
        
        # Update table
        self._populate_violations_table()
        
        print(f"🚨 Bulk acknowledged {len(selected_rows)} violations")
    
    def _export_violations(self):
        """Export violations report"""
        violation_ids = [v.violation_id for v in self.violations_data]
        self.violation_exported.emit(violation_ids)
        print("🚨 Violations exported")
    
    def _clear_old_violations(self):
        """Clear old acknowledged violations"""
        # Remove violations older than 30 days and acknowledged
        cutoff_date = QDateTime.currentDateTime().addDays(-30)
        
        original_count = len(self.violations_data)
        self.violations_data = [
            v for v in self.violations_data 
            if not (v.status == "acknowledged" and v.timestamp < cutoff_date)
        ]
        
        removed_count = original_count - len(self.violations_data)
        
        # Update table
        self._populate_violations_table()
        
        print(f"🚨 Cleared {removed_count} old violations")
    
    def _save_evidence(self):
        """Save current evidence"""
        current_row = self.violations_table.currentRow()
        if current_row >= 0:
            violation = self.violations_data[current_row]
            self.evidence_viewed.emit(violation.violation_id)
            print(f"🚨 Evidence saved for violation {violation.violation_id}")
    
    def add_violation(self, violation_data):
        """Add a new violation to the list"""
        violation = ViolationItem(**violation_data)
        self.violations_data.insert(0, violation)  # Add to beginning
        self._populate_violations_table()
        print(f"🚨 New violation added: {violation.violation_id}")
    
    def get_violation_summary(self):
        """Get summary of violations"""
        total = len(self.violations_data)
        pending = sum(1 for v in self.violations_data if v.status == "pending")
        acknowledged = sum(1 for v in self.violations_data if v.status == "acknowledged")
        dismissed = sum(1 for v in self.violations_data if v.status == "dismissed")
        
        return {
            'total': total,
            'pending': pending,
            'acknowledged': acknowledged,
            'dismissed': dismissed
        }
