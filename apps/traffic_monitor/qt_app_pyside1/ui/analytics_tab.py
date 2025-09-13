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
        # Data storage for real-time updates
        self.latest_traffic_lights = []
        self.latest_violations = []
        self.latest_vehicles = []
        self.latest_frame_data = {}
        self.init_ui()
        
        # Add a timer for periodic table updates
        from PySide6.QtCore import QTimer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.refresh_all_data)
        self.update_timer.start(2000)  # Update every 2 seconds
        
    def init_ui(self):
        """Initialize the clean UI with tabs"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Set dark background for the main widget
        self.setStyleSheet("""
            QWidget {
                background-color: #2C3E50;
                color: #FFFFFF;
            }
        """)
        
        # Title
        title_label = QLabel("🚦 Traffic Intersection Monitor")
        title_label.setStyleSheet("""
            QLabel {
                font-size: 20px;
                font-weight: bold;
                color: #FFFFFF;
                font-family: 'Roboto', Arial, sans-serif;
                padding: 15px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #2C3E50, stop:1 #34495E);
                border-radius: 8px;
                border: 1px solid #34495E;
            }
        """)
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #34495E;
                border-radius: 8px;
                background-color: #2C3E50;
            }
            QTabBar::tab {
                background: #34495E;
                color: #FFFFFF;
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
                background: #2C3E50;
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
        
        # Start with empty table - no sample data
        layout.addWidget(self.traffic_table)
        
        self.tab_widget.addTab(tab, "🚦 Traffic Lights")
        
    def create_violation_tab(self):
        """Create violation summary tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Create violations table with more columns
        self.violation_table = QTableWidget(0, 4)
        self.violation_table.setHorizontalHeaderLabels([
            "Track ID", "Violation Type", "Timestamp", "Status"
        ])
        
        # Apply clean table styling
        self.apply_table_style(self.violation_table)
        
        # Start with empty table - no sample data
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
        
        # Start with empty table - no sample data
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
                    stop:0 #1A252F, stop:1 #2C3E50);
                color: #FFFFFF;
                padding: 10px;
                border: 1px solid #2C3E50;
                font-weight: bold;
                font-family: 'Roboto', Arial, sans-serif;
            }
        """)
        
        # Table styling
        table.setStyleSheet("""
            QTableWidget {
                gridline-color: #34495E;
                background-color: #2C3E50;
                alternate-background-color: #34495E;
                selection-background-color: #3498DB;
                border: 1px solid #34495E;
                border-radius: 6px;
                color: #FFFFFF;
            }
            QTableWidget::item {
                padding: 8px;
                border-bottom: 1px solid #34495E;
                color: #FFFFFF;
            }
            QTableWidget::item:selected {
                background-color: #3498DB;
                color: #FFFFFF;
            }
        """)
        
        # Enable alternating row colors
        table.setAlternatingRowColors(True)
        
        # Set selection behavior
        table.setSelectionBehavior(QTableWidget.SelectRows)
        
    def populate_table(self, table, data, table_type):
        """Populate table with data and apply color coding for dark theme"""
        table.setRowCount(len(data))
        
        for i, row in enumerate(data):
            for j, item in enumerate(row):
                cell = QTableWidgetItem(str(item))
                cell.setForeground(QColor(255, 255, 255))  # White text
                
                # Apply color coding based on content for dark theme
                if table_type == "traffic_light":
                    if "🔴" in str(item):
                        cell.setBackground(QColor(139, 69, 19))  # Dark red/brown
                    elif "🟡" in str(item):
                        cell.setBackground(QColor(184, 134, 11))  # Dark yellow
                    elif "🟢" in str(item):
                        cell.setBackground(QColor(34, 139, 34))  # Dark green
                        
                elif table_type == "violation":
                    if "Active" in str(item) or "🚨" in str(item):
                        cell.setBackground(QColor(139, 69, 19))  # Dark red/brown
                        cell.setForeground(QColor(255, 255, 255))  # White text
                    elif "Detected" in str(item):
                        cell.setBackground(QColor(205, 133, 63))  # Dark orange
                        cell.setForeground(QColor(255, 255, 255))  # White text
                        
                elif table_type == "vehicle":
                    if "🔴" in str(item) or ("True" in str(item) and j == 4):  # Violating column
                        cell.setBackground(QColor(139, 69, 19))  # Dark red/brown
                        cell.setForeground(QColor(255, 255, 255))  # White text
                    elif "🟢" in str(item):
                        cell.setBackground(QColor(34, 139, 34))  # Dark green
                        cell.setForeground(QColor(255, 255, 255))  # White text
                
                table.setItem(i, j, cell)
                
    def refresh_all_data(self):
        """Refresh all tables with latest data"""
        print("🔄 Refreshing analytics data...")
        self.update_traffic_lights_table()
        self.update_violations_table()
        self.update_vehicles_table()
    
    @Slot(dict)
    def update_detection_data(self, detection_data):
        """Update analytics with detection data from video tab"""
        try:
            print(f"[ANALYTICS UPDATE] Received detection data with keys: {list(detection_data.keys())}")
            self.latest_frame_data = detection_data
            
            # Extract traffic lights
            detections = detection_data.get('detections', [])
            traffic_lights = []
            vehicles = []
            
            for detection in detections:
                if hasattr(detection, 'label'):
                    label = detection.label
                elif isinstance(detection, dict):
                    label = detection.get('label', detection.get('class', detection.get('class_name', '')))
                else:
                    label = str(detection)
                
                if 'traffic light' in str(label).lower():
                    traffic_lights.append(detection)
                elif any(vehicle_type in str(label).lower() for vehicle_type in ['car', 'truck', 'bus', 'motorcycle']):
                    vehicles.append(detection)
            
            self.latest_traffic_lights = traffic_lights
            
            # Extract vehicle tracking data - Handle the EXACT structure from video controller
            tracked_vehicles = detection_data.get('tracked_vehicles', [])
            print(f"[ANALYTICS UPDATE] Found {len(tracked_vehicles)} tracked vehicles")
            
            # Process tracked vehicles with the correct structure
            processed_vehicles = []
            for vehicle in tracked_vehicles:
                print(f"[ANALYTICS UPDATE] Raw vehicle data: {vehicle}")
                
                # Handle the actual structure: {id, bbox, center_y, is_moving, is_violation}
                if isinstance(vehicle, dict):
                    track_id = vehicle.get('id', 'Unknown')
                    bbox = vehicle.get('bbox', [0, 0, 0, 0])
                    center_y = vehicle.get('center_y', 0)
                    moving = vehicle.get('is_moving', False)
                    violating = vehicle.get('is_violation', False)
                    
                    # Calculate center_x from bbox to match debug output format
                    if len(bbox) >= 4:
                        center_x = int((bbox[0] + bbox[2]) / 2)
                    else:
                        center_x = 0
                    
                else:
                    # Fallback for other object types
                    track_id = getattr(vehicle, 'id', getattr(vehicle, 'track_id', 'Unknown'))
                    bbox = getattr(vehicle, 'bbox', [0, 0, 0, 0])
                    center_y = getattr(vehicle, 'center_y', 0)
                    moving = getattr(vehicle, 'is_moving', getattr(vehicle, 'moving', False))
                    violating = getattr(vehicle, 'is_violation', getattr(vehicle, 'violating', False))
                    
                    if len(bbox) >= 4:
                        center_x = int((bbox[0] + bbox[2]) / 2)
                    else:
                        center_x = 0
                
                processed_vehicles.append({
                    'track_id': track_id,
                    'center': (center_x, int(center_y)),
                    'moving': moving,
                    'violating': violating
                })
                
                print(f"[ANALYTICS UPDATE] Processed vehicle ID={track_id}, center=({center_x:.1f}, {center_y:.1f}), moving={moving}, violating={violating}")
            
            self.latest_vehicles = processed_vehicles
            print(f"[ANALYTICS UPDATE] Stored {len(self.latest_vehicles)} processed vehicles")
            
            # Update tables with new data
            self.update_traffic_lights_table()
            self.update_vehicles_table()
            
        except Exception as e:
            print(f"Error updating detection data: {e}")
            import traceback
            traceback.print_exc()
    
    @Slot(dict)
    def update_violation_data(self, violation_data):
        """Update violations data"""
        try:
            print(f"🚨 [ANALYTICS UPDATE] Received violation data: {violation_data}")
            print(f"🚨 [ANALYTICS UPDATE] Current violations count: {len(self.latest_violations)}")
            
            # Store violation data
            track_id = violation_data.get('track_id')
            # Try both 'violation_type' and 'type' fields
            violation_type = violation_data.get('violation_type', violation_data.get('type', 'Unknown'))
            
            # Format timestamp properly
            timestamp = violation_data.get('timestamp')
            formatted_timestamp = 'N/A'
            if timestamp:
                from datetime import datetime
                try:
                    if hasattr(timestamp, 'strftime'):  # datetime object
                        formatted_timestamp = timestamp.strftime('%H:%M:%S')
                    elif isinstance(timestamp, (int, float)):
                        formatted_timestamp = datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')
                    else:
                        formatted_timestamp = str(timestamp)
                except Exception as e:
                    print(f"[ANALYTICS] Error formatting timestamp: {e}")
                    formatted_timestamp = 'N/A'
            
            print(f"🚨 [ANALYTICS UPDATE] Processing violation - Track ID: {track_id}, Type: {violation_type}, Time: {formatted_timestamp}")
            
            # Add to violations list if not already present
            existing = [v for v in self.latest_violations if v.get('track_id') == track_id and v.get('type') == violation_type]
            if not existing:
                violation_entry = {
                    'track_id': track_id,
                    'type': violation_type,  # Store as 'type' for consistency
                    'status': 'Active',
                    'timestamp': formatted_timestamp
                }
                self.latest_violations.append(violation_entry)
                print(f"🚨 [ANALYTICS UPDATE] Added new violation: {violation_entry}")
                print(f"🚨 [ANALYTICS UPDATE] Total violations now: {len(self.latest_violations)}")
            else:
                print(f"🚨 [ANALYTICS UPDATE] Violation already exists for Track ID {track_id}, Type {violation_type}")

            print(f"🚨 [ANALYTICS UPDATE] Calling update_violations_table()")
            self.update_violations_table()
            print(f"🚨 [ANALYTICS UPDATE] update_violations_table() completed")
            
            # Also force a refresh of the UI to ensure table updates are visible
            try:
                self.violation_table.update()
                self.violation_table.repaint()
                print(f"🚨 [ANALYTICS UPDATE] Forced UI refresh of violation table")
            except Exception as e:
                print(f"🚨 [ANALYTICS UPDATE] Error forcing UI refresh: {e}")
            
        except Exception as e:
            print(f"🚨 [ANALYTICS ERROR] Error updating violation data: {e}")
            import traceback
            traceback.print_exc()
    
    def update_traffic_lights_table(self):
        """Update traffic lights table with latest data"""
        try:
            data = []
            
            # Check if we have traffic light data from frame analysis
            latest_traffic_light = self.latest_frame_data.get('traffic_light', {})
            if latest_traffic_light:
                # Extract traffic light info
                color = latest_traffic_light.get('color', 'unknown')
                confidence = latest_traffic_light.get('confidence', 0.0)
                
                # Create traffic light entries based on the detected signal
                if color == 'red':
                    status = "🔴 Red"
                    red_ratio = confidence
                    yellow_ratio = 0.0
                    green_ratio = 0.0
                elif color == 'yellow':
                    status = "🟡 Yellow"
                    red_ratio = 0.0
                    yellow_ratio = confidence
                    green_ratio = 0.0
                elif color == 'green':
                    status = "🟢 Green"
                    red_ratio = 0.0
                    yellow_ratio = 0.0
                    green_ratio = confidence
                else:
                    status = "❓ Unknown"
                    red_ratio = 0.0
                    yellow_ratio = 0.0
                    green_ratio = 0.0
                
                data.append([
                    "Main Traffic Light",
                    f"{red_ratio:.3f}",
                    f"{yellow_ratio:.3f}",
                    f"{green_ratio:.3f}",
                    status
                ])
            
            # Also check for individual traffic light detections
            for i, tl in enumerate(self.latest_traffic_lights):
                bbox = tl.get('bbox', [0, 0, 0, 0])
                # Extract color ratios from debug data if available
                color_info = tl.get('color_info', {})
                red_ratio = color_info.get('red', 0.0)
                yellow_ratio = color_info.get('yellow', 0.0) 
                green_ratio = color_info.get('green', 0.0)
                
                # Determine status
                if red_ratio > 0.3:
                    status = "🔴 Red"
                elif yellow_ratio > 0.3:
                    status = "🟡 Yellow"
                elif green_ratio > 0.3:
                    status = "🟢 Green"
                else:
                    status = "❓ Unknown"
                
                data.append([
                    f"Traffic Light {i+1}",
                    f"{red_ratio:.3f}",
                    f"{yellow_ratio:.3f}",
                    f"{green_ratio:.3f}",
                    status
                ])
            
            # If no data, show empty table instead of sample data
            if not data:
                data = []
            
            self.populate_table(self.traffic_table, data, "traffic_light")
            
        except Exception as e:
            print(f"Error updating traffic lights table: {e}")
    
    def update_violations_table(self):
        """Update violations table with latest data"""
        try:
            print(f"🚨 [TABLE UPDATE] Starting violations table update with {len(self.latest_violations)} violations")
            data = []
            for i, violation in enumerate(self.latest_violations):
                # Format timestamp if available
                timestamp = violation.get('timestamp', 'N/A')
                if timestamp and timestamp != 'N/A':
                    from datetime import datetime
                    try:
                        if isinstance(timestamp, (int, float)):
                            timestamp = datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')
                        else:
                            timestamp = str(timestamp)
                    except:
                        timestamp = 'N/A'
                
                row_data = [
                    str(violation.get('track_id', 'Unknown')),
                    f"🚨 {violation.get('type', 'Unknown')}",
                    str(timestamp),
                    violation.get('status', 'Active')
                ]
                data.append(row_data)
                print(f"🚨 [TABLE UPDATE] Added violation {i+1}: {row_data}")
            
            # If no violations, show empty table
            if not data:
                print(f"🚨 [TABLE UPDATE] No violations to display")
                data = []
            
            print(f"🚨 [TABLE UPDATE] Calling populate_table with {len(data)} rows")
            self.populate_table(self.violation_table, data, "violation")
            print(f"🚨 [TABLE UPDATE] populate_table completed")
            
        except Exception as e:
            print(f"🚨 [TABLE ERROR] Error updating violations table: {e}")
            import traceback
            traceback.print_exc()
    
    def update_vehicles_table(self):
        """Update vehicles table with latest data"""
        try:
            print(f"[ANALYTICS UPDATE] Updating vehicles table with {len(self.latest_vehicles)} vehicles")
            data = []
            
            for vehicle in self.latest_vehicles:
                track_id = vehicle.get('track_id', 'Unknown')
                center = vehicle.get('center', (0, 0))
                position = f"({center[0]:.1f}, {center[1]:.1f})"
                center_y = center[1] if len(center) > 1 else 0
                moving = vehicle.get('moving', False)
                violating = vehicle.get('violating', False)
                
                if violating:
                    status = "🔴 Violating"
                elif moving:
                    status = "🟡 Moving"
                else:
                    status = "🟢 Stopped"
                
                data.append([
                    str(track_id),
                    position,
                    f"{center_y:.1f}",
                    str(moving),
                    str(violating),
                    status
                ])
                
                print(f"[ANALYTICS UPDATE] Added vehicle row: ID={track_id}, pos={position}, moving={moving}, violating={violating}, status={status}")
            
            print(f"[ANALYTICS UPDATE] Total vehicle rows to display: {len(data)}")
            
            # If no vehicles, show empty table
            if not data:
                data = []
            
            self.populate_table(self.vehicle_table, data, "vehicle")
            
        except Exception as e:
            print(f"Error updating vehicles table: {e}")
            import traceback
            traceback.print_exc()
        
        
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
        """Update analytics with new data"""
        # Forward to the analytics widget
        if hasattr(self.analytics_widget, 'update_detection_data'):
            self.analytics_widget.update_detection_data(analytics)
    
    @Slot(dict)
    def update_detection_data(self, detection_data):
        """Update detection data from video tab"""
        self.analytics_widget.update_detection_data(detection_data)
    
    @Slot(dict) 
    def update_violation_data(self, violation_data):
        """Update violation data"""
        self.analytics_widget.update_violation_data(violation_data)
    
    @Slot(dict)
    def update_smart_intersection_analytics(self, analytics_data):
        """Update smart intersection analytics"""
        # Extract relevant data and forward
        if 'detections' in analytics_data:
            self.analytics_widget.update_detection_data(analytics_data)
        if 'violations' in analytics_data:
            for violation in analytics_data['violations']:
                self.analytics_widget.update_violation_data(violation)
