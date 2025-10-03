"""
Video Analysis Tab - Advanced video analysis with ROI configuration
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
                               QGroupBox, QLabel, QPushButton, QComboBox,
                               QListWidget, QTextEdit, QSlider, QCheckBox,
                               QFrame, QGridLayout, QSpinBox, QProgressBar)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QFont, QPixmap

class VideoAnalysisTab(QWidget):
    """
    Video Analysis Tab with ROI configuration and advanced analytics
    
    Features:
    - ROI (Region of Interest) drawing and management
    - Traffic pattern analysis
    - Speed measurement zones
    - Counting lines configuration
    - Heatmap visualization
    - Advanced analytics reporting
    """
    
    # Signals
    roi_changed = Signal(dict)
    analysis_started = Signal(str)
    export_requested = Signal(str, str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.roi_data = {}
        self.analysis_results = {}
        
        self._setup_ui()
        
        print("🎬 Video Analysis Tab initialized")
    
    def _setup_ui(self):
        """Setup the video analysis UI"""
        # Main splitter
        main_splitter = QSplitter(Qt.Horizontal)
        self.layout = QVBoxLayout(self)
        self.layout.addWidget(main_splitter)
        
        # Left panel - Video and ROI editor
        left_panel = self._create_left_panel()
        main_splitter.addWidget(left_panel)
        
        # Right panel - Analysis controls and results
        right_panel = self._create_right_panel()
        main_splitter.addWidget(right_panel)
        
        # Set splitter proportions
        main_splitter.setSizes([800, 400])
    
    def _create_left_panel(self):
        """Create left panel with video analysis view"""
        panel = QFrame()
        layout = QVBoxLayout(panel)
        
        # Header with controls
        header = self._create_analysis_header()
        layout.addWidget(header)
        
        # Main analysis view (placeholder for now)
        self.analysis_view = QFrame()
        self.analysis_view.setMinimumSize(600, 400)
        self.analysis_view.setStyleSheet("""
            QFrame {
                background-color: #2c3e50;
                border: 2px solid #34495e;
                border-radius: 8px;
            }
        """)
        layout.addWidget(self.analysis_view, 1)
        
        # ROI tools
        roi_tools = self._create_roi_tools()
        layout.addWidget(roi_tools)
        
        return panel
    
    def _create_analysis_header(self):
        """Create analysis header with controls"""
        header = QFrame()
        header.setFixedHeight(50)
        layout = QHBoxLayout(header)
        
        # Source selection
        layout.addWidget(QLabel("Source:"))
        self.source_combo = QComboBox()
        self.source_combo.addItems(["Live Camera 1", "Live Camera 2", "Recorded Video", "Import Video"])
        layout.addWidget(self.source_combo)
        
        layout.addStretch()
        
        # Analysis controls
        self.start_analysis_btn = QPushButton("▶️ Start Analysis")
        self.start_analysis_btn.clicked.connect(self._start_analysis)
        layout.addWidget(self.start_analysis_btn)
        
        self.stop_analysis_btn = QPushButton("⏹️ Stop")
        self.stop_analysis_btn.setEnabled(False)
        self.stop_analysis_btn.clicked.connect(self._stop_analysis)
        layout.addWidget(self.stop_analysis_btn)
        
        # Export button
        export_btn = QPushButton("📊 Export Results")
        export_btn.clicked.connect(self._export_results)
        layout.addWidget(export_btn)
        
        return header
    
    def _create_roi_tools(self):
        """Create ROI drawing tools"""
        tools = QGroupBox("ROI Tools")
        tools.setFixedHeight(80)
        layout = QHBoxLayout(tools)
        
        # ROI type selection
        layout.addWidget(QLabel("ROI Type:"))
        self.roi_type_combo = QComboBox()
        self.roi_type_combo.addItems([
            "Traffic Count Zone",
            "Speed Measurement Zone", 
            "Restricted Area",
            "Parking Detection",
            "Crosswalk Zone"
        ])
        layout.addWidget(self.roi_type_combo)
        
        # ROI drawing buttons
        draw_rect_btn = QPushButton("📐 Rectangle")
        draw_rect_btn.clicked.connect(lambda: self._set_draw_mode("rectangle"))
        layout.addWidget(draw_rect_btn)
        
        draw_poly_btn = QPushButton("🔗 Polygon")
        draw_poly_btn.clicked.connect(lambda: self._set_draw_mode("polygon"))
        layout.addWidget(draw_poly_btn)
        
        draw_line_btn = QPushButton("📏 Line")
        draw_line_btn.clicked.connect(lambda: self._set_draw_mode("line"))
        layout.addWidget(draw_line_btn)
        
        layout.addStretch()
        
        # Clear ROI
        clear_btn = QPushButton("🗑️ Clear All")
        clear_btn.clicked.connect(self._clear_all_roi)
        layout.addWidget(clear_btn)
        
        return tools
    
    def _create_right_panel(self):
        """Create right panel with controls and results"""
        panel = QFrame()
        layout = QVBoxLayout(panel)
        
        # ROI list
        roi_section = self._create_roi_section()
        layout.addWidget(roi_section)
        
        # Analysis settings
        settings_section = self._create_settings_section()
        layout.addWidget(settings_section)
        
        # Results section
        results_section = self._create_results_section()
        layout.addWidget(results_section)
        
        return panel
    
    def _create_roi_section(self):
        """Create ROI management section"""
        section = QGroupBox("ROI Management")
        layout = QVBoxLayout(section)
        
        # ROI list
        self.roi_list = QListWidget()
        self.roi_list.setMaximumHeight(120)
        layout.addWidget(self.roi_list)
        
        # ROI controls
        roi_controls = QHBoxLayout()
        
        edit_btn = QPushButton("✏️ Edit")
        edit_btn.clicked.connect(self._edit_selected_roi)
        roi_controls.addWidget(edit_btn)
        
        delete_btn = QPushButton("🗑️ Delete")
        delete_btn.clicked.connect(self._delete_selected_roi)
        roi_controls.addWidget(delete_btn)
        
        duplicate_btn = QPushButton("📋 Copy")
        duplicate_btn.clicked.connect(self._duplicate_selected_roi)
        roi_controls.addWidget(duplicate_btn)
        
        layout.addLayout(roi_controls)
        
        return section
    
    def _create_settings_section(self):
        """Create analysis settings section"""
        section = QGroupBox("Analysis Settings")
        layout = QGridLayout(section)
        
        # Detection sensitivity
        layout.addWidget(QLabel("Sensitivity:"), 0, 0)
        self.sensitivity_slider = QSlider(Qt.Horizontal)
        self.sensitivity_slider.setRange(1, 10)
        self.sensitivity_slider.setValue(5)
        layout.addWidget(self.sensitivity_slider, 0, 1)
        
        self.sensitivity_label = QLabel("5")
        layout.addWidget(self.sensitivity_label, 0, 2)
        
        # Minimum object size
        layout.addWidget(QLabel("Min Size:"), 1, 0)
        self.min_size_spin = QSpinBox()
        self.min_size_spin.setRange(10, 1000)
        self.min_size_spin.setValue(50)
        self.min_size_spin.setSuffix(" px")
        layout.addWidget(self.min_size_spin, 1, 1, 1, 2)
        
        # Analysis options
        self.track_objects_cb = QCheckBox("Track Objects")
        self.track_objects_cb.setChecked(True)
        layout.addWidget(self.track_objects_cb, 2, 0, 1, 3)
        
        self.speed_analysis_cb = QCheckBox("Speed Analysis")
        self.speed_analysis_cb.setChecked(True)
        layout.addWidget(self.speed_analysis_cb, 3, 0, 1, 3)
        
        self.direction_analysis_cb = QCheckBox("Direction Analysis")
        self.direction_analysis_cb.setChecked(False)
        layout.addWidget(self.direction_analysis_cb, 4, 0, 1, 3)
        
        # Connect sensitivity slider to label
        self.sensitivity_slider.valueChanged.connect(
            lambda v: self.sensitivity_label.setText(str(v))
        )
        
        return section
    
    def _create_results_section(self):
        """Create analysis results section"""
        section = QGroupBox("Analysis Results")
        layout = QVBoxLayout(section)
        
        # Results summary
        summary_layout = QGridLayout()
        
        summary_layout.addWidget(QLabel("Objects Detected:"), 0, 0)
        self.objects_count_label = QLabel("0")
        summary_layout.addWidget(self.objects_count_label, 0, 1)
        
        summary_layout.addWidget(QLabel("Avg Speed:"), 1, 0)
        self.avg_speed_label = QLabel("0.0 km/h")
        summary_layout.addWidget(self.avg_speed_label, 1, 1)
        
        summary_layout.addWidget(QLabel("Violations:"), 2, 0)
        self.violations_count_label = QLabel("0")
        summary_layout.addWidget(self.violations_count_label, 2, 1)
        
        layout.addLayout(summary_layout)
        
        # Progress bar for analysis
        self.analysis_progress = QProgressBar()
        self.analysis_progress.setVisible(False)
        layout.addWidget(self.analysis_progress)
        
        # Detailed results
        self.results_text = QTextEdit()
        self.results_text.setMaximumHeight(150)
        self.results_text.setPlaceholderText("Analysis results will appear here...")
        layout.addWidget(self.results_text)
        
        # Export options
        export_layout = QHBoxLayout()
        
        export_csv_btn = QPushButton("📄 Export CSV")
        export_csv_btn.clicked.connect(lambda: self._export_data("csv"))
        export_layout.addWidget(export_csv_btn)
        
        export_json_btn = QPushButton("📋 Export JSON")
        export_json_btn.clicked.connect(lambda: self._export_data("json"))
        export_layout.addWidget(export_json_btn)
        
        layout.addLayout(export_layout)
        
        return section
    
    def _set_draw_mode(self, mode):
        """Set ROI drawing mode"""
        print(f"🎬 Draw mode set to: {mode}")
        # Implementation for setting drawing mode
    
    def _clear_all_roi(self):
        """Clear all ROI regions"""
        self.roi_list.clear()
        self.roi_data.clear()
        self.roi_changed.emit(self.roi_data)
        print("🎬 All ROI regions cleared")
    
    def _edit_selected_roi(self):
        """Edit the selected ROI"""
        current_item = self.roi_list.currentItem()
        if current_item:
            roi_name = current_item.text()
            print(f"🎬 Editing ROI: {roi_name}")
    
    def _delete_selected_roi(self):
        """Delete the selected ROI"""
        current_item = self.roi_list.currentItem()
        if current_item:
            roi_name = current_item.text()
            self.roi_list.takeItem(self.roi_list.row(current_item))
            if roi_name in self.roi_data:
                del self.roi_data[roi_name]
            self.roi_changed.emit(self.roi_data)
            print(f"🎬 Deleted ROI: {roi_name}")
    
    def _duplicate_selected_roi(self):
        """Duplicate the selected ROI"""
        current_item = self.roi_list.currentItem()
        if current_item:
            roi_name = current_item.text()
            print(f"🎬 Duplicating ROI: {roi_name}")
    
    def _start_analysis(self):
        """Start video analysis"""
        self.start_analysis_btn.setEnabled(False)
        self.stop_analysis_btn.setEnabled(True)
        self.analysis_progress.setVisible(True)
        self.analysis_progress.setRange(0, 0)  # Indeterminate progress
        
        source = self.source_combo.currentText()
        self.analysis_started.emit(source)
        
        # Add sample result
        self.results_text.append(f"Started analysis on: {source}")
        print(f"🎬 Started analysis on: {source}")
    
    def _stop_analysis(self):
        """Stop video analysis"""
        self.start_analysis_btn.setEnabled(True)
        self.stop_analysis_btn.setEnabled(False)
        self.analysis_progress.setVisible(False)
        
        self.results_text.append("Analysis stopped by user")
        print("🎬 Analysis stopped")
    
    def _export_results(self):
        """Export analysis results"""
        print("🎬 Exporting analysis results")
        self.export_requested.emit("analysis_results", "pdf")
    
    def _export_data(self, format_type):
        """Export data in specified format"""
        print(f"🎬 Exporting data as {format_type.upper()}")
        self.export_requested.emit("analysis_data", format_type)
    
    def add_roi(self, roi_name, roi_type, coordinates):
        """Add a new ROI region"""
        self.roi_data[roi_name] = {
            'type': roi_type,
            'coordinates': coordinates
        }
        
        # Add to list
        self.roi_list.addItem(f"{roi_name} ({roi_type})")
        
        # Emit change signal
        self.roi_changed.emit(self.roi_data)
        
        print(f"🎬 Added ROI: {roi_name} ({roi_type})")
    
    def update_analysis_results(self, results):
        """Update analysis results display"""
        self.analysis_results.update(results)
        
        # Update summary labels
        self.objects_count_label.setText(str(results.get('objects_detected', 0)))
        self.avg_speed_label.setText(f"{results.get('average_speed', 0.0):.1f} km/h")
        self.violations_count_label.setText(str(results.get('violations', 0)))
        
        # Add to results text
        if 'message' in results:
            self.results_text.append(results['message'])
    
    def set_analysis_progress(self, value):
        """Set analysis progress"""
        if self.analysis_progress.isVisible():
            if value < 0:
                self.analysis_progress.setRange(0, 0)  # Indeterminate
            else:
                self.analysis_progress.setRange(0, 100)
                self.analysis_progress.setValue(value)
