from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QTextEdit, QComboBox, QLineEdit, QProgressBar, QMessageBox,
    QSplitter, QFrame, QFileDialog, QTabWidget, QApplication
)
from PySide6.QtCore import Qt, Signal, Slot, QTimer
from PySide6.QtGui import QPixmap, QImage, QFont, QColor, QPalette

import sys
import os
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.annotation_utils import convert_cv_to_qimage, convert_cv_to_pixmap

class VLMTab(QWidget):
    """Tab for Vision Language Model interaction (OpenVINO local, user prompt)."""
    process_image_requested = Signal(np.ndarray, str)  # image, prompt

    def __init__(self):
        super().__init__()
        self.setupUI()
        self.current_image = None
        self.result_history = []
    
    def setupUI(self):
        """Set up the user interface."""
        main_layout = QVBoxLayout()
        
        def setupUI(self):
            """Set up the user interface."""
            main_layout = QVBoxLayout()

            # Create splitter for adjustable layout
            splitter = QSplitter(Qt.Horizontal)

            # === Left panel (image display) ===
            left_panel = QWidget()
            left_layout = QVBoxLayout()

            # Image label
            self.image_label = QLabel("No image loaded")
            self.image_label.setAlignment(Qt.AlignCenter)
            self.image_label.setMinimumSize(400, 300)
            self.image_label.setStyleSheet("background-color: #1e1e1e; border: 1px solid #333;")
            left_layout.addWidget(self.image_label)

            # Image controls
            image_controls = QHBoxLayout()
            self.load_image_btn = QPushButton("Load Image")
            self.load_image_btn.clicked.connect(self.load_image)
            image_controls.addWidget(self.load_image_btn)
            self.capture_frame_btn = QPushButton("Capture Frame")
            self.capture_frame_btn.clicked.connect(self.capture_frame)
            image_controls.addWidget(self.capture_frame_btn)
            left_layout.addLayout(image_controls)

            # Prompt input (only prompt, no task selection)
            prompt_layout = QHBoxLayout()
            prompt_label = QLabel("Prompt:")
            prompt_layout.addWidget(prompt_label)
            self.prompt_edit = QLineEdit()
            self.prompt_edit.setPlaceholderText("Enter your prompt/question for the VLM...")
            self.prompt_edit.returnPressed.connect(self.process_query)
            prompt_layout.addWidget(self.prompt_edit)
            self.process_btn = QPushButton("Process")
            self.process_btn.clicked.connect(self.process_query)
            prompt_layout.addWidget(self.process_btn)
            left_layout.addLayout(prompt_layout)

            # Progress bar
            self.progress_bar = QProgressBar()
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            self.progress_bar.setTextVisible(True)
            self.progress_bar.setVisible(False)
            left_layout.addWidget(self.progress_bar)

            left_panel.setLayout(left_layout)

            # === Right panel (results) ===
            right_panel = QWidget()
            right_layout = QVBoxLayout()

            # Results label
            results_label = QLabel("Results:")
            results_label.setFont(QFont("Arial", 12, QFont.Bold))
            right_layout.addWidget(results_label)

            # Results display
            self.results_text = QTextEdit()
            self.results_text.setReadOnly(True)
            self.results_text.setMinimumWidth(350)
            right_layout.addWidget(self.results_text)

            # History controls
            history_layout = QHBoxLayout()
            self.clear_btn = QPushButton("Clear Results")
            self.clear_btn.clicked.connect(self.clear_results)
            history_layout.addWidget(self.clear_btn)
            self.copy_btn = QPushButton("Copy Results")
            self.copy_btn.clicked.connect(self.copy_results)
            history_layout.addWidget(self.copy_btn)
            right_layout.addLayout(history_layout)

            right_panel.setLayout(right_layout)

            # Add panels to splitter
            splitter.addWidget(left_panel)
            splitter.addWidget(right_panel)
            splitter.setSizes([500, 500])  # Initial size distribution

            # Add splitter to main layout
            main_layout.addWidget(splitter)

            # Status bar
            self.status_label = QLabel("Ready")
            main_layout.addWidget(self.status_label)

            self.setLayout(main_layout)
    def load_image(self):
        """Load an image from file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        
        if file_path:
            try:
                self.current_image = cv2.imread(file_path)
                if self.current_image is None:
                    QMessageBox.critical(self, "Error", "Failed to load image")
                    return
                
                # Convert to RGB for display
                self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
                self.update_image_display()
                self.status_label.setText(f"Loaded image: {Path(file_path).name}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error loading image: {str(e)}")
    
    @Slot()
    def capture_frame(self):
        """Capture current frame from video processing."""
        # This should be connected to the video controller
        self.status_label.setText("Waiting for frame from video feed...")
    
    @Slot(np.ndarray)
    def set_frame(self, frame):
        """Set the current frame from video processing."""
        if frame is not None:
            self.current_image = frame.copy()
            # Convert BGR to RGB if needed
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            self.update_image_display()
            self.status_label.setText("Frame captured from video feed")
    
    def update_image_display(self):
        """Update the image display with the current image."""
        if self.current_image is not None:
            # Resize image for display while maintaining aspect ratio
            h, w = self.current_image.shape[:2]
            max_size = 500
            if h > max_size or w > max_size:
                if h > w:
                    new_h = max_size
                    new_w = int(w * (max_size / h))
                else:
                    new_w = max_size
                    new_h = int(h * (max_size / w))
                display_img = cv2.resize(self.current_image, (new_w, new_h))
            else:
                display_img = self.current_image
            
            # Convert to QPixmap and display
            qimage = QImage(
                display_img.data,
                display_img.shape[1],
                display_img.shape[0],
                display_img.shape[1] * 3,
                QImage.Format_RGB888
            )
            pixmap = QPixmap.fromImage(qimage)
            self.image_label.setPixmap(pixmap)
    
    @Slot()
    def process_query(self):
        """Process the current image with the user prompt."""
        if self.current_image is None:
            QMessageBox.warning(self, "Warning", "Please load or capture an image first")
            return
        prompt = self.prompt_edit.text().strip()
        if not prompt:
            QMessageBox.warning(self, "Warning", "Please enter a prompt")
            return
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(10)
        self.status_label.setText("Processing...")
        self.process_btn.setEnabled(False)
        self.process_image_requested.emit(self.current_image, prompt)
    
    @Slot(dict)
    def handle_result(self, result):
        """Handle the result from VLM processing."""
        self.progress_bar.setValue(100)
        self.process_btn.setEnabled(True)
        
        # Hide progress bar after a short delay
        QTimer.singleShot(1000, lambda: self.progress_bar.setVisible(False))
        
        # Format and display result
        task = result.get("task", "unknown")
        query = result.get("query", "")
        answer = result.get("answer", "No answer provided")
        
        # Format timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Add to results text
        result_text = f"[{timestamp}] "
        if task == "vqa":
            result_text += f"Q: {query}\nA: {answer}\n\n"
        else:  # search
            result_text += f"Search: {query}\nResults: {answer}\n\n"
        
        # Add to display
        current_text = self.results_text.toPlainText()
        self.results_text.setText(result_text + current_text)
        
        # Update status
        self.status_label.setText("Processing complete")
        
        # Add to history
        self.result_history.append({
            "timestamp": timestamp,
            "task": task,
            "query": query,
            "answer": answer
        })
    
    @Slot(str)
    def handle_error(self, error_msg):
        """Handle errors from VLM processing."""
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.process_btn.setEnabled(True)
        
        QMessageBox.warning(self, "Processing Error", error_msg)
        self.status_label.setText(f"Error: {error_msg}")
    
    @Slot(int)
    def update_progress(self, value):
        """Update progress bar."""
        self.progress_bar.setValue(value)
    
    @Slot()
    def clear_results(self):
        """Clear the results display."""
        self.results_text.clear()
        self.status_label.setText("Results cleared")
    
    @Slot()
    def copy_results(self):
        """Copy results to clipboard."""
        text = self.results_text.toPlainText()
        clipboard = QApplication.clipboard()
        clipboard.setText(text)
        self.status_label.setText("Results copied to clipboard")
