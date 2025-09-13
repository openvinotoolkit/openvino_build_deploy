from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QTextEdit, QLineEdit, QFrame, QGroupBox, QScrollArea
)
from PySide6.QtCore import Qt, Signal, Slot, QTimer
from PySide6.QtGui import QPixmap, QImage, QFont, QColor

import sys
import os
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.annotation_utils import convert_cv_to_qimage, convert_cv_to_pixmap

class VLMInsightsWidget(QWidget):
    """Widget for Vision Language Model insights in settings panel."""
    analyze_frame_requested = Signal(np.ndarray, str)  # image, prompt

    def __init__(self):
        super().__init__()
        print("[VLM INSIGHTS DEBUG] Initializing VLM Insights Widget")
        self.setupUI()
        self.current_frame = None
        self.detection_data = None  # Store detection data from video controller
        self.is_video_paused = False
        self.setVisible(True)  # Make it visible by default in config panel
        print("[VLM INSIGHTS DEBUG] VLM Insights Widget initialized")
    
    def setupUI(self):
        """Set up the user interface."""
        main_layout = QVBoxLayout()
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # === VLM Insights Group ===
        insights_group = QGroupBox(" Scene Analysis")
        insights_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 14px;
                color: #00d4aa;
                border: 2px solid #00d4aa;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px 0 8px;
                background-color: #2b2b2b;
            }
        """)
        insights_layout = QVBoxLayout()
        
        # Status label
        self.status_label = QLabel("📹 Pause video to analyze current frame")
        self.status_label.setStyleSheet("color: #888; font-style: italic; padding: 5px;")
        insights_layout.addWidget(self.status_label)
        
        # Current frame thumbnail
        self.frame_thumbnail = QLabel("No frame")
        self.frame_thumbnail.setAlignment(Qt.AlignCenter)
        self.frame_thumbnail.setFixedSize(150, 100)
        self.frame_thumbnail.setStyleSheet("""
            background-color: #1e1e1e; 
            border: 1px solid #444; 
            border-radius: 4px;
        """)
        insights_layout.addWidget(self.frame_thumbnail)
        
        # Custom prompt input
        prompt_layout = QHBoxLayout()
        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("Enter your question about the scene...")
        self.prompt_input.setStyleSheet("""
            QLineEdit {
                background-color: #1e1e1e;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 8px;
                color: white;
                font-size: 12px;
            }
            QLineEdit:focus {
                border: 1px solid #00d4aa;
            }
        """)
        self.prompt_input.returnPressed.connect(self._analyze_custom)
        
        self.analyze_custom_btn = QPushButton(" Analyze")
        self.analyze_custom_btn.setStyleSheet(self._get_button_style())
        self.analyze_custom_btn.clicked.connect(self._analyze_custom)
        
        prompt_layout.addWidget(self.prompt_input)
        prompt_layout.addWidget(self.analyze_custom_btn)
        insights_layout.addLayout(prompt_layout)
        
        # Results area with scroll
        results_scroll = QScrollArea()
        results_scroll.setWidgetResizable(True)
        results_scroll.setMaximumHeight(200)
        results_scroll.setStyleSheet("""
            QScrollArea {
                background-color: #1e1e1e;
                border: 1px solid #444;
                border-radius: 4px;
            }
        """)
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                border: none;
                color: #e0e0e0;
                font-size: 11px;
                padding: 8px;
            }
        """)
        self.results_text.setPlaceholderText("AI insights will appear here...")
        results_scroll.setWidget(self.results_text)
        insights_layout.addWidget(results_scroll)
        
        insights_group.setLayout(insights_layout)
        main_layout.addWidget(insights_group)
        
        main_layout.addStretch()
        self.setLayout(main_layout)
        
        # Initially disable analysis buttons
        self._set_analysis_enabled(False)
    
    def _get_button_style(self):
        """Get consistent button styling."""
        return """
            QPushButton {
                background-color: #00d4aa;
                color: #1a1a1a;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #00b89a;
            }
            QPushButton:pressed {
                background-color: #008f7a;
            }
            QPushButton:disabled {
                background-color: #444;
                color: #888;
            }
        """
    
    def _set_analysis_enabled(self, enabled):
        """Enable/disable analysis buttons."""
        self.analyze_custom_btn.setEnabled(enabled)
        self.prompt_input.setEnabled(enabled)
    
    @Slot(bool)
    def on_video_paused(self, is_paused):
        """Called when video is paused/unpaused."""
        print(f"[VLM INSIGHTS DEBUG] Video pause state changed: {is_paused}")
        print(f"[VLM INSIGHTS DEBUG] Current frame available: {self.current_frame is not None}")
        
        self.is_video_paused = is_paused
        if is_paused:
            self.status_label.setText(" Video paused - Frame ready for analysis")
            self.status_label.setStyleSheet("color: #00d4aa; font-weight: bold; padding: 5px;")
            self._set_analysis_enabled(True)
            self.setVisible(True)
            print("[VLM INSIGHTS DEBUG] VLM insights widget made visible and enabled")
            
            # If we have a current frame, update the thumbnail immediately
            if self.current_frame is not None:
                print(f"[VLM INSIGHTS DEBUG] Updating thumbnail with current frame: {self.current_frame.shape}")
                self._update_thumbnail(self.current_frame)
            else:
                print("[VLM INSIGHTS DEBUG] No current frame available for thumbnail")
                
        else:
            self.status_label.setText("📹 Pause video to analyze current frame")
            self.status_label.setStyleSheet("color: #888; font-style: italic; padding: 5px;")
            self._set_analysis_enabled(False)
            print("[VLM INSIGHTS DEBUG] VLM insights widget disabled")
    
    @Slot(np.ndarray)
    def set_current_frame(self, frame):
        """Set the current frame for analysis."""
        print(f"[VLM INSIGHTS DEBUG] Received frame: {frame.shape if frame is not None else 'None'}")
        if frame is not None:
            self.current_frame = frame.copy()
            self._update_thumbnail(self.current_frame)
    
    def set_detection_data(self, detection_data):
        """Set detection data from video controller for rich VLM analysis."""
        print(f"[VLM INSIGHTS DEBUG] Received detection data")
        print(f"[VLM INSIGHTS DEBUG] Data keys: {list(detection_data.keys()) if detection_data else 'None'}")
        
        self.detection_data = detection_data
        
        if detection_data and 'detections' in detection_data:
            detections = detection_data['detections']
            print(f"[VLM INSIGHTS DEBUG] Detections count: {len(detections)}")
            
            # Log some detection info for debugging
            for i, det in enumerate(detections[:3]):  # Show first 3
                if hasattr(det, '__dict__'):
                    print(f"[VLM INSIGHTS DEBUG] Detection {i}: {type(det)} - {getattr(det, 'class_name', 'unknown')}")
                elif isinstance(det, dict):
                    print(f"[VLM INSIGHTS DEBUG] Detection {i}: {det.get('class_name', det.get('label', 'unknown'))}")
                else:
                    print(f"[VLM INSIGHTS DEBUG] Detection {i}: {type(det)}")
        
        print(f"[VLM INSIGHTS DEBUG] Detection data stored successfully")
    
    def _update_thumbnail(self, frame):
        """Update the frame thumbnail display."""
        if frame is None:
            print("[VLM INSIGHTS DEBUG] Cannot update thumbnail - no frame provided")
            return
            
        try:
            # Create thumbnail
            h, w = frame.shape[:2]
            if h > 0 and w > 0:
                # Scale to fit thumbnail
                thumb_h, thumb_w = 100, 150
                scale = min(thumb_w/w, thumb_h/h)
                new_w, new_h = int(w*scale), int(h*scale)
                
                thumbnail = cv2.resize(frame, (new_w, new_h))
                
                # Convert to QPixmap
                if len(thumbnail.shape) == 3:
                    rgb_thumbnail = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_thumbnail.shape
                    bytes_per_line = ch * w
                    qimage = QImage(rgb_thumbnail.data, w, h, bytes_per_line, QImage.Format_RGB888)
                else:
                    h, w = thumbnail.shape
                    bytes_per_line = w
                    qimage = QImage(thumbnail.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
                
                pixmap = QPixmap.fromImage(qimage)
                self.frame_thumbnail.setPixmap(pixmap)
                print(f"[VLM INSIGHTS DEBUG] Frame thumbnail updated successfully")
        except Exception as e:
            print(f"[VLM INSIGHTS DEBUG] Error updating thumbnail: {e}")
    
    def _quick_analyze(self, prompt):
        """Perform quick analysis with predefined prompt."""
        print(f"[VLM INSIGHTS DEBUG] _quick_analyze called")
        print(f"[VLM INSIGHTS DEBUG] Current frame available: {self.current_frame is not None}")
        print(f"[VLM INSIGHTS DEBUG] Detection data available: {self.detection_data is not None}")
        print(f"[VLM INSIGHTS DEBUG] Video paused: {self.is_video_paused}")
        print(f"[VLM INSIGHTS DEBUG] Prompt: {prompt[:50]}...")
        
        if self.current_frame is not None and self.is_video_paused:
            print(f"[VLM INSIGHTS DEBUG] Enhancing prompt with detection data")
            
            # Enhance prompt with detection data
            enhanced_prompt = self._enhance_prompt_with_detections(prompt)
            print(f"[VLM INSIGHTS DEBUG] Enhanced prompt length: {len(enhanced_prompt)} characters")
            
            print(f"[VLM INSIGHTS DEBUG] Emitting analyze_frame_requested signal")
            self.results_text.append(f"\n🔍 Analyzing: {prompt[:50]}...")
            self.analyze_frame_requested.emit(self.current_frame, enhanced_prompt)
            print(f"[VLM INSIGHTS DEBUG] Signal emitted successfully")
        else:
            print(f"[VLM INSIGHTS DEBUG] Cannot analyze - frame: {self.current_frame is not None}, paused: {self.is_video_paused}")
            if self.current_frame is None:
                self.results_text.append("\n❌ No frame available for analysis")
            if not self.is_video_paused:
                self.results_text.append("\n❌ Video must be paused for analysis")

    def _enhance_prompt_with_detections(self, base_prompt):
        """Enhance the analysis prompt with detection data."""
        try:
            enhanced_parts = [base_prompt]
            
            if self.detection_data and 'detections' in self.detection_data:
                detections = self.detection_data['detections']
                
                enhanced_parts.append(f"\n\nDETECTION CONTEXT:")
                enhanced_parts.append(f"Total detections: {len(detections)}")
                
                # Categorize detections
                vehicles = []
                traffic_lights = []
                other_objects = []
                
                for det in detections:
                    if hasattr(det, 'class_name'):
                        class_name = det.class_name
                        bbox = getattr(det, 'bbox', None)
                        track_id = getattr(det, 'track_id', None)
                        confidence = getattr(det, 'confidence', None)
                    elif isinstance(det, dict):
                        class_name = det.get('class_name', det.get('label', 'unknown'))
                        bbox = det.get('bbox', det.get('box', None))
                        track_id = det.get('track_id', det.get('id', None))
                        confidence = det.get('confidence', det.get('conf', None))
                    else:
                        continue
                    
                    detection_info = {
                        'class': class_name,
                        'bbox': bbox,
                        'track_id': track_id,
                        'confidence': confidence
                    }
                    
                    if class_name in ['car', 'truck', 'bus', 'motorcycle', 'vehicle']:
                        vehicles.append(detection_info)
                    elif 'traffic' in class_name.lower() or 'light' in class_name.lower():
                        traffic_lights.append(detection_info)
                    else:
                        other_objects.append(detection_info)
                
                # Add vehicle information
                if vehicles:
                    enhanced_parts.append(f"\nVEHICLES ({len(vehicles)}):")
                    for i, vehicle in enumerate(vehicles):
                        track_info = f" (ID: {vehicle['track_id']})" if vehicle['track_id'] else ""
                        conf_info = f" (conf: {vehicle['confidence']:.2f})" if vehicle['confidence'] else ""
                        bbox_info = f" at {vehicle['bbox']}" if vehicle['bbox'] else ""
                        enhanced_parts.append(f"  - {vehicle['class']}{track_info}{conf_info}{bbox_info}")
                
                # Add traffic light information  
                if traffic_lights:
                    enhanced_parts.append(f"\nTRAFFIC LIGHTS ({len(traffic_lights)}):")
                    for tl in traffic_lights:
                        conf_info = f" (conf: {tl['confidence']:.2f})" if tl['confidence'] else ""
                        bbox_info = f" at {tl['bbox']}" if tl['bbox'] else ""
                        enhanced_parts.append(f"  - {tl['class']}{conf_info}{bbox_info}")
                
                # Add other objects
                if other_objects:
                    enhanced_parts.append(f"\nOTHER OBJECTS ({len(other_objects)}):")
                    for obj in other_objects:
                        conf_info = f" (conf: {obj['confidence']:.2f})" if obj['confidence'] else ""
                        enhanced_parts.append(f"  - {obj['class']}{conf_info}")
            
            # Add additional context from detection data
            if self.detection_data:
                fps = self.detection_data.get('fps', 0)
                if fps > 0:
                    enhanced_parts.append(f"\nVIDEO INFO: FPS: {fps:.1f}")
                
                # Add crosswalk information
                crosswalk_detected = self.detection_data.get('crosswalk_detected', False)
                crosswalk_bbox = self.detection_data.get('crosswalk_bbox', None)
                violation_line_y = self.detection_data.get('violation_line_y', None)
                
                if crosswalk_detected:
                    enhanced_parts.append(f"\nCROSSWALK INFO:")
                    enhanced_parts.append(f"  - Crosswalk detected: YES")
                    if crosswalk_bbox:
                        enhanced_parts.append(f"  - Crosswalk location: {crosswalk_bbox}")
                    if violation_line_y:
                        enhanced_parts.append(f"  - Violation line at y={violation_line_y}")
                else:
                    enhanced_parts.append(f"\nCROSSWALK INFO:")
                    enhanced_parts.append(f"  - Crosswalk detected: NO")
                
                # Add traffic light information
                traffic_light = self.detection_data.get('traffic_light', {})
                if traffic_light:
                    color = traffic_light.get('color', 'unknown')
                    confidence = traffic_light.get('confidence', 0)
                    enhanced_parts.append(f"\nTRAFFIC LIGHT STATUS:")
                    enhanced_parts.append(f"  - Current color: {color.upper()}")
                    enhanced_parts.append(f"  - Confidence: {confidence:.2f}")
            
            # Special instructions for color analysis
            if "color" in base_prompt.lower() or "colour" in base_prompt.lower():
                enhanced_parts.append(f"\nSPECIAL INSTRUCTIONS:")
                enhanced_parts.append(f"- Carefully examine the image for vehicle colors")
                enhanced_parts.append(f"- Look at each car's body color (red, blue, white, black, silver, etc.)")
                enhanced_parts.append(f"- Ignore detection data for color questions - analyze the image visually")
                enhanced_parts.append(f"- List the prominent colors you can identify in the vehicles")
            
            enhanced_parts.append(f"\nAnswer the question directly based on visual analysis of the image. Be concise and specific:")
            enhanced_parts.append(f"Question: {base_prompt}")
            enhanced_parts.append(f"Answer:")
            
            enhanced_prompt = "\n".join(enhanced_parts)
            print(f"[VLM INSIGHTS DEBUG] Enhanced prompt created: {len(enhanced_prompt)} chars")
            return enhanced_prompt
            
        except Exception as e:
            print(f"[VLM INSIGHTS DEBUG] Error enhancing prompt: {e}")
            return base_prompt
    
    def _analyze_custom(self):
        """Perform analysis with custom prompt."""
        print(f"[VLM INSIGHTS DEBUG] _analyze_custom called")
        prompt = self.prompt_input.text().strip()
        print(f"[VLM INSIGHTS DEBUG] Custom prompt: '{prompt}'")
        print(f"[VLM INSIGHTS DEBUG] Current frame available: {self.current_frame is not None}")
        print(f"[VLM INSIGHTS DEBUG] Detection data available: {self.detection_data is not None}")
        print(f"[VLM INSIGHTS DEBUG] Video paused: {self.is_video_paused}")
        
        if prompt and self.current_frame is not None and self.is_video_paused:
            print(f"[VLM INSIGHTS DEBUG] Enhancing custom prompt with detection data")
            
            # Enhance prompt with detection data
            enhanced_prompt = self._enhance_prompt_with_detections(prompt)
            
            print(f"[VLM INSIGHTS DEBUG] Emitting analyze_frame_requested signal for custom prompt")
            self.results_text.append(f"\n Question: {prompt}")
            self.analyze_frame_requested.emit(self.current_frame, enhanced_prompt)
            self.prompt_input.clear()
            print(f"[VLM INSIGHTS DEBUG] Custom analysis signal emitted successfully")
        else:
            print(f"[VLM INSIGHTS DEBUG] Cannot analyze custom - prompt: '{prompt}', frame: {self.current_frame is not None}, paused: {self.is_video_paused}")
            if not prompt:
                self.results_text.append("\n❌ Please enter a prompt for analysis")
            elif self.current_frame is None:
                self.results_text.append("\n❌ No frame available for analysis")
            elif not self.is_video_paused:
                self.results_text.append("\n❌ Video must be paused for analysis")
    
    @Slot(object)
    def on_analysis_result(self, result):
        """Display analysis result."""
        print(f"[VLM INSIGHTS DEBUG] Received analysis result: {type(result)}")
        print(f"[VLM INSIGHTS DEBUG] Result content: {str(result)[:200]}...")
        
        # Extract the actual response text from the result
        response_text = ""
        
        try:
            if isinstance(result, dict):
                print(f"[VLM INSIGHTS DEBUG] Result is dict with keys: {list(result.keys())}")
                
                # Check if it's the OpenVINO response format
                if 'response' in result:
                    response_text = str(result['response'])
                    print(f"[VLM INSIGHTS DEBUG] Extracted response from dict")
                elif 'message' in result:
                    response_text = str(result['message'])
                    print(f"[VLM INSIGHTS DEBUG] Extracted message from dict")
                else:
                    response_text = str(result)
                    print(f"[VLM INSIGHTS DEBUG] Using dict as string")
            elif isinstance(result, str):
                response_text = result
                print(f"[VLM INSIGHTS DEBUG] Result is already string")
            else:
                # Try to convert any other type to string
                response_text = str(result)
                print(f"[VLM INSIGHTS DEBUG] Converted to string")
            
            # Clean up the response text
            response_text = response_text.strip()
            
            if not response_text:
                response_text = "No response text found in result."
                
        except Exception as e:
            print(f"[VLM INSIGHTS DEBUG] Error extracting response: {e}")
            response_text = f"Error extracting response: {str(e)}"
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_response = f" {response_text}"
        
        print(f"[VLM INSIGHTS DEBUG] Final response: {response_text[:100]}...")
        self.results_text.append(f" Answer: {formatted_response}\n")
        
        # Auto-scroll to bottom
        scrollbar = self.results_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
        print(f"[VLM INSIGHTS DEBUG] Analysis result displayed successfully")
