from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView
)
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QColor, QFont


class CleanAnalyticsWidget(QWidget):
    """Clean and minimal analytics widget with tabbed interface"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        """Initialize the clean UI with tabs"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Title
        title_label = QLabel("🚦 Traffic Intersection Monitor")
        title_label.setStyleSheet("""
            QLabel {
                font-size: 20px;
                font-weight: bold;
                color: #2C3E50;
                font-family: 'Roboto', Arial, sans-serif;
                padding: 15px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #E8F4FD, stop:1 #F8FBFE);
                border-radius: 8px;
                border: 1px solid #BDC3C7;
            }
        """)
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #BDC3C7;
                border-radius: 8px;
                background-color: white;
            }
            QTabBar::tab {
                background: #ECF0F1;
                color: #2C3E50;
                padding: 12px 20px;
                margin-right: 2px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                font-family: 'Roboto', Arial, sans-serif;
                font-weight: 500;
                min-width: 120px;
            }
            QTabBar::tab:selected {
                background: #3498DB;
                color: white;
            }
            QTabBar::tab:hover:!selected {
                background: #D5DBDB;
            }
        """)
        
        # Create tabs
        self.create_traffic_light_tab()
        self.create_violation_tab()
        self.create_vehicle_tab()
        
        layout.addWidget(self.tab_widget)
        
        # Refresh button
        refresh_btn = QPushButton("🔄 Refresh Data")
        refresh_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3498DB, stop:1 #2980B9);
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 6px;
                font-weight: bold;
                font-family: 'Roboto', Arial, sans-serif;
                font-size: 14px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #5DADE2, stop:1 #3498DB);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2980B9, stop:1 #21618C);
            }
        """)
        refresh_btn.clicked.connect(self.refresh_all_data)
        
        # Center the button
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(refresh_btn)
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
    def create_traffic_light_tab(self):
        """Create traffic light status tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Table
        self.traffic_table = QTableWidget(0, 5)
        self.traffic_table.setHorizontalHeaderLabels([
            "Detection", "Red Ratio", "Yellow Ratio", "Green Ratio", "Status"
        ])
        
        # Apply clean table styling
        self.apply_table_style(self.traffic_table)
        
        # Sample data
        sample_data = [
            ["Traffic Light 1", "0.353", "0.000", "0.000", "🔴 Red"],
            ["Traffic Light 2", "0.399", "0.000", "0.000", "🔴 Red"],
            ["Traffic Light 3", "0.499", "0.000", "0.000", "🔴 Red"],
            ["Traffic Light 4", "0.728", "0.000", "0.000", "🔴 Red"],
            ["Traffic Light 5", "0.958", "0.000", "0.000", "🔴 Red"],
            ["Traffic Light 6", "0.623", "0.000", "0.000", "🔴 Red"],
        ]
        
        self.populate_table(self.traffic_table, sample_data, "traffic_light")
        layout.addWidget(self.traffic_table)
        
        self.tab_widget.addTab(tab, "🚦 Traffic Lights")
        
    def create_violation_tab(self):
        """Create violation summary tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Table
        self.violation_table = QTableWidget(0, 3)
        self.violation_table.setHorizontalHeaderLabels([
            "Track ID", "Violation Type", "Status"
        ])
        
        # Apply clean table styling
        self.apply_table_style(self.violation_table)
        
        # Sample data
        sample_data = [
            ["4", "🚨 Red Light Violation", "Active"],
            ["4", "Improper Stop on Crosswalk", "Detected"],
        ]
        
        self.populate_table(self.violation_table, sample_data, "violation")
        layout.addWidget(self.violation_table)
        
        self.tab_widget.addTab(tab, "🚨 Violations")
        
    def create_vehicle_tab(self):
        """Create vehicle tracking status tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Table
        self.vehicle_table = QTableWidget(0, 6)
        self.vehicle_table.setHorizontalHeaderLabels([
            "Track ID", "Position (x,y)", "Center Y", "Moving", "Violating", "Status"
        ])
        
        # Apply clean table styling
        self.apply_table_style(self.vehicle_table)
        
        # Sample data
        sample_data = [
            ["1", "(387.0, 214.5)", "214.5", "False", "False", "🟢 Tracked"],
            ["2", "(380.5, 135.0)", "135.0", "False", "False", "🟢 Tracked"],
            ["3", "(258.5, 151.5)", "151.5", "False", "False", "🟢 Tracked"],
            ["4", "(519.0, 187.0)", "187.0", "False", "True", "🔴 Violating"],
            ["5", "(520.0, 132.0)", "132.0", "False", "False", "🟢 Tracked"],
            ["6", "(615.0, 172.0)", "172.0", "False", "False", "🟢 Tracked"],
            ["7", "(561.5, 334.0)", "334.0", "False", "False", "🟢 Tracked"],
            ["8", "(401.5, 71.5)", "71.5", "False", "False", "🟢 Tracked"],
        ]
        
        self.populate_table(self.vehicle_table, sample_data, "vehicle")
        layout.addWidget(self.vehicle_table)
        
        self.tab_widget.addTab(tab, "🚗 Vehicles")
        
    def apply_table_style(self, table):
        """Apply consistent styling to tables"""
        # Set font
        font = QFont("Roboto", 10)
        table.setFont(font)
        
        # Header styling
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.horizontalHeader().setStyleSheet("""
            QHeaderView::section {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #34495E, stop:1 #2C3E50);
                color: white;
                padding: 10px;
                border: 1px solid #2C3E50;
                font-weight: bold;
                font-family: 'Roboto', Arial, sans-serif;
            }
        """)
        
        # Table styling
        table.setStyleSheet("""
            QTableWidget {
                gridline-color: #E8E8E8;
                background-color: white;
                alternate-background-color: #F8F9FA;
                selection-background-color: #E3F2FD;
                border: 1px solid #E0E0E0;
                border-radius: 6px;
            }
            QTableWidget::item {
                padding: 8px;
                border-bottom: 1px solid #E8E8E8;
            }
            QTableWidget::item:selected {
                background-color: #E3F2FD;
                color: #1976D2;
            }
        """)
        
        # Enable alternating row colors
        table.setAlternatingRowColors(True)
        
        # Set selection behavior
        table.setSelectionBehavior(QTableWidget.SelectRows)
        
    def populate_table(self, table, data, table_type):
        """Populate table with data and apply color coding"""
        table.setRowCount(len(data))
        
        for i, row in enumerate(data):
            for j, item in enumerate(row):
                cell = QTableWidgetItem(str(item))
                
                # Apply color coding based on content
                if table_type == "traffic_light":
                    if "🔴" in str(item):
                        cell.setBackground(QColor(255, 235, 235))  # Light red
                    elif "🟡" in str(item):
                        cell.setBackground(QColor(255, 255, 235))  # Light yellow
                    elif "🟢" in str(item):
                        cell.setBackground(QColor(235, 255, 235))  # Light green
                        
                elif table_type == "violation":
                    if "Active" in str(item) or "🚨" in str(item):
                        cell.setBackground(QColor(255, 235, 235))  # Light red
                    elif "Detected" in str(item):
                        cell.setBackground(QColor(255, 248, 235))  # Light orange
                        
                elif table_type == "vehicle":
                    if "🔴" in str(item) or "True" in str(item) and j == 4:  # Violating column
                        cell.setBackground(QColor(255, 235, 235))  # Light red
                    elif "🟢" in str(item):
                        cell.setBackground(QColor(235, 255, 235))  # Light green
                
                table.setItem(i, j, cell)
                
    def refresh_all_data(self):
        """Refresh all tables with latest data"""
        print("🔄 Refreshing analytics data...")
        # This would be connected to actual data update logic
        
        
class AnalyticsTab(QWidget):
    """Main analytics tab with clean design"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        """Initialize the main analytics interface"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create the clean analytics widget
        self.analytics_widget = CleanAnalyticsWidget()
        layout.addWidget(self.analytics_widget)
        
    @Slot(dict)
    def update_analytics(self, analytics):
        """Update analytics with new data (placeholder for future implementation)"""
        # This would update the tables with real-time data
        pass
