"""
VLM AI Insights Tab - ChatGPT-like interface for Vision Language Model interactions
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
                               QLineEdit, QPushButton, QScrollArea, QFrame,
                               QLabel, QSplitter, QComboBox, QSlider, QGroupBox,
                               QCheckBox, QSpinBox)
from PySide6.QtCore import Qt, Signal, QTimer, QDateTime
from PySide6.QtGui import QFont, QTextCharFormat, QColor, QPixmap

class ChatMessage(QFrame):
    """Individual chat message widget"""
    
    def __init__(self, message, is_user=True, timestamp=None, parent=None):
        super().__init__(parent)
        
        self.message = message
        self.is_user = is_user
        self.timestamp = timestamp or QDateTime.currentDateTime()
        
        self._setup_ui()
        self._apply_style()
    
    def _setup_ui(self):
        """Setup message UI"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        
        if self.is_user:
            layout.addStretch()
        
        # Message bubble
        bubble = QFrame()
        bubble.setMaximumWidth(400)
        bubble_layout = QVBoxLayout(bubble)
        bubble_layout.setContentsMargins(12, 8, 12, 8)
        
        # Message text
        message_label = QLabel(self.message)
        message_label.setWordWrap(True)
        message_label.setFont(QFont("Segoe UI", 9))
        bubble_layout.addWidget(message_label)
        
        # Timestamp
        time_label = QLabel(self.timestamp.toString("hh:mm"))
        time_label.setFont(QFont("Segoe UI", 7))
        time_label.setAlignment(Qt.AlignRight if self.is_user else Qt.AlignLeft)
        bubble_layout.addWidget(time_label)
        
        layout.addWidget(bubble)
        
        if not self.is_user:
            layout.addStretch()
    
    def _apply_style(self):
        """Apply message styling"""
        if self.is_user:
            # User message (blue, right-aligned)
            self.setStyleSheet("""
                QFrame {
                    background-color: #3498db;
                    border-radius: 12px;
                    margin-left: 50px;
                }
                QLabel {
                    color: white;
                }
            """)
        else:
            # AI message (gray, left-aligned)
            self.setStyleSheet("""
                QFrame {
                    background-color: #ecf0f1;
                    border-radius: 12px;
                    margin-right: 50px;
                }
                QLabel {
                    color: #2c3e50;
                }
            """)

class VLMInsightsTab(QWidget):
    """
    VLM AI Insights Tab with ChatGPT-like interface
    
    Features:
    - Chat-style interface for VLM interactions
    - Image context from traffic cameras
    - Predefined prompts for traffic analysis
    - Conversation history
    - AI insights and recommendations
    - Export conversation functionality
    """
    
    # Signals
    insight_generated = Signal(str)
    vlm_query_sent = Signal(str, dict)
    conversation_exported = Signal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.conversation_history = []
        self.current_image_context = None
        
        self._setup_ui()
        
        print("🤖 VLM AI Insights Tab initialized")
    
    def _setup_ui(self):
        """Setup the VLM insights UI"""
        # Main splitter
        main_splitter = QSplitter(Qt.Horizontal)
        layout = QVBoxLayout(self)
        layout.addWidget(main_splitter)
        
        # Left panel - Chat interface
        left_panel = self._create_chat_panel()
        main_splitter.addWidget(left_panel)
        
        # Right panel - Settings and context
        right_panel = self._create_settings_panel()
        main_splitter.addWidget(right_panel)
        
        # Set splitter proportions (70% chat, 30% settings)
        main_splitter.setSizes([700, 300])
    
    def _create_chat_panel(self):
        """Create chat interface panel"""
        panel = QFrame()
        layout = QVBoxLayout(panel)
        
        # Chat header
        header = self._create_chat_header()
        layout.addWidget(header)
        
        # Conversation area
        self.conversation_scroll = QScrollArea()
        self.conversation_scroll.setWidgetResizable(True)
        self.conversation_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarNever)
        
        self.conversation_widget = QWidget()
        self.conversation_layout = QVBoxLayout(self.conversation_widget)
        self.conversation_layout.setAlignment(Qt.AlignTop)
        self.conversation_layout.setSpacing(5)
        
        self.conversation_scroll.setWidget(self.conversation_widget)
        layout.addWidget(self.conversation_scroll, 1)
        
        # Input area
        input_area = self._create_input_area()
        layout.addWidget(input_area)
        
        return panel
    
    def _create_chat_header(self):
        """Create chat header with title and controls"""
        header = QFrame()
        header.setFixedHeight(50)
        header.setStyleSheet("background-color: #34495e; border-radius: 8px; margin-bottom: 5px;")
        
        layout = QHBoxLayout(header)
        layout.setContentsMargins(15, 10, 15, 10)
        
        # Title and status
        title_layout = QVBoxLayout()
        
        title = QLabel("🤖 VLM AI Assistant")
        title.setFont(QFont("Segoe UI", 12, QFont.Bold))
        title.setStyleSheet("color: white;")
        title_layout.addWidget(title)
        
        self.status_label = QLabel("Ready to analyze traffic scenes")
        self.status_label.setFont(QFont("Segoe UI", 8))
        self.status_label.setStyleSheet("color: #bdc3c7;")
        title_layout.addWidget(self.status_label)
        
        layout.addLayout(title_layout)
        
        layout.addStretch()
        
        # Action buttons
        clear_btn = QPushButton("🗑️ Clear")
        clear_btn.setFixedSize(60, 30)
        clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 8pt;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        clear_btn.clicked.connect(self._clear_conversation)
        layout.addWidget(clear_btn)
        
        export_btn = QPushButton("📤 Export")
        export_btn.setFixedSize(60, 30)
        export_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 8pt;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        export_btn.clicked.connect(self._export_conversation)
        layout.addWidget(export_btn)
        
        return header
    
    def _create_input_area(self):
        """Create message input area"""
        input_frame = QFrame()
        input_frame.setFixedHeight(100)
        layout = QVBoxLayout(input_frame)
        
        # Quick prompts
        prompts_layout = QHBoxLayout()
        
        prompts = [
            "Analyze current traffic",
            "Count vehicles",
            "Detect violations",
            "Safety assessment"
        ]
        
        for prompt in prompts:
            btn = QPushButton(prompt)
            btn.setMaximumHeight(25)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #ecf0f1;
                    border: 1px solid #bdc3c7;
                    border-radius: 12px;
                    padding: 4px 8px;
                    font-size: 8pt;
                }
                QPushButton:hover {
                    background-color: #d5dbdb;
                }
            """)
            btn.clicked.connect(lambda checked, p=prompt: self._send_quick_prompt(p))
            prompts_layout.addWidget(btn)
        
        prompts_layout.addStretch()
        layout.addLayout(prompts_layout)
        
        # Message input
        input_layout = QHBoxLayout()
        
        self.message_input = QLineEdit()
        self.message_input.setPlaceholderText("Ask the AI about traffic conditions, violations, or safety...")
        self.message_input.setFont(QFont("Segoe UI", 9))
        self.message_input.setStyleSheet("""
            QLineEdit {
                border: 2px solid #bdc3c7;
                border-radius: 20px;
                padding: 8px 15px;
                font-size: 9pt;
            }
            QLineEdit:focus {
                border-color: #3498db;
            }
        """)
        self.message_input.returnPressed.connect(self._send_message)
        input_layout.addWidget(self.message_input)
        
        self.send_btn = QPushButton("➤")
        self.send_btn.setFixedSize(40, 40)
        self.send_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 20px;
                font-size: 16pt;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #21618c;
            }
        """)
        self.send_btn.clicked.connect(self._send_message)
        input_layout.addWidget(self.send_btn)
        
        layout.addLayout(input_layout)
        
        return input_frame
    
    def _create_settings_panel(self):
        """Create settings and context panel"""
        panel = QFrame()
        layout = QVBoxLayout(panel)
        
        # Image context section
        context_section = self._create_context_section()
        layout.addWidget(context_section)
        
        # VLM settings
        settings_section = self._create_vlm_settings()
        layout.addWidget(settings_section)
        
        # Conversation stats
        stats_section = self._create_stats_section()
        layout.addWidget(stats_section)
        
        layout.addStretch()
        
        return panel
    
    def _create_context_section(self):
        """Create image context section"""
        section = QGroupBox("Image Context")
        layout = QVBoxLayout(section)
        
        # Current image preview
        self.image_preview = QLabel("No image selected")
        self.image_preview.setFixedSize(200, 150)
        self.image_preview.setAlignment(Qt.AlignCenter)
        self.image_preview.setStyleSheet("""
            QLabel {
                border: 2px dashed #bdc3c7;
                border-radius: 8px;
                background-color: #ecf0f1;
                color: #7f8c8d;
            }
        """)
        layout.addWidget(self.image_preview)
        
        # Image source selection
        source_layout = QHBoxLayout()
        source_layout.addWidget(QLabel("Source:"))
        
        self.image_source_combo = QComboBox()
        self.image_source_combo.addItems([
            "Live Camera 1",
            "Live Camera 2", 
            "Current Frame",
            "Upload Image"
        ])
        self.image_source_combo.currentTextChanged.connect(self._change_image_source)
        source_layout.addWidget(self.image_source_combo)
        
        layout.addLayout(source_layout)
        
        # Capture button
        capture_btn = QPushButton("📸 Capture Current")
        capture_btn.clicked.connect(self._capture_current_frame)
        layout.addWidget(capture_btn)
        
        return section
    
    def _create_vlm_settings(self):
        """Create VLM model settings"""
        section = QGroupBox("AI Settings")
        layout = QVBoxLayout(section)
        
        # Model selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model:"))
        
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "LLaVA-Next-Video",
            "GPT-4 Vision",
            "Claude Vision"
        ])
        model_layout.addWidget(self.model_combo)
        
        layout.addLayout(model_layout)
        
        # Temperature setting
        temp_layout = QHBoxLayout()
        temp_layout.addWidget(QLabel("Creativity:"))
        
        self.temperature_slider = QSlider(Qt.Horizontal)
        self.temperature_slider.setRange(1, 10)
        self.temperature_slider.setValue(5)
        temp_layout.addWidget(self.temperature_slider)
        
        self.temp_label = QLabel("0.5")
        temp_layout.addWidget(self.temp_label)
        
        layout.addLayout(temp_layout)
        
        # Max response length
        length_layout = QHBoxLayout()
        length_layout.addWidget(QLabel("Max Length:"))
        
        self.max_length_spin = QSpinBox()
        self.max_length_spin.setRange(50, 1000)
        self.max_length_spin.setValue(300)
        self.max_length_spin.setSuffix(" words")
        length_layout.addWidget(self.max_length_spin)
        
        layout.addLayout(length_layout)
        
        # Analysis options
        self.detailed_analysis_cb = QCheckBox("Detailed Analysis")
        self.detailed_analysis_cb.setChecked(True)
        layout.addWidget(self.detailed_analysis_cb)
        
        self.safety_focus_cb = QCheckBox("Safety Focus")
        self.safety_focus_cb.setChecked(True)
        layout.addWidget(self.safety_focus_cb)
        
        # Connect temperature slider to label
        self.temperature_slider.valueChanged.connect(
            lambda v: self.temp_label.setText(f"{v/10:.1f}")
        )
        
        return section
    
    def _create_stats_section(self):
        """Create conversation statistics"""
        section = QGroupBox("Conversation Stats")
        layout = QVBoxLayout(section)
        
        # Message count
        self.message_count_label = QLabel("Messages: 0")
        layout.addWidget(self.message_count_label)
        
        # Insights generated
        self.insights_count_label = QLabel("Insights: 0")
        layout.addWidget(self.insights_count_label)
        
        # Session time
        self.session_time_label = QLabel("Session: 0 min")
        layout.addWidget(self.session_time_label)
        
        return section
    
    def _send_message(self):
        """Send user message"""
        message = self.message_input.text().strip()
        if not message:
            return
        
        # Add user message
        self._add_message(message, is_user=True)
        
        # Clear input
        self.message_input.clear()
        
        # Send to VLM (simulate response for now)
        self._simulate_vlm_response(message)
        
        # Emit signal
        context = {
            'image': self.current_image_context,
            'settings': self._get_current_settings()
        }
        self.vlm_query_sent.emit(message, context)
    
    def _send_quick_prompt(self, prompt):
        """Send a quick prompt"""
        self.message_input.setText(prompt)
        self._send_message()
    
    def _add_message(self, message, is_user=True):
        """Add message to conversation"""
        chat_message = ChatMessage(message, is_user)
        self.conversation_layout.addWidget(chat_message)
        
        # Scroll to bottom
        QTimer.singleShot(100, lambda: self.conversation_scroll.verticalScrollBar().setValue(
            self.conversation_scroll.verticalScrollBar().maximum()
        ))
        
        # Update stats
        self.conversation_history.append({
            'message': message,
            'is_user': is_user,
            'timestamp': QDateTime.currentDateTime()
        })
        
        self._update_stats()
    
    def _simulate_vlm_response(self, user_message):
        """Simulate VLM response (replace with actual VLM integration)"""
        # Simulate processing delay
        self.status_label.setText("AI is analyzing...")
        
        QTimer.singleShot(2000, lambda: self._generate_response(user_message))
    
    def _generate_response(self, user_message):
        """Generate AI response"""
        # Simple response simulation
        responses = {
            "analyze current traffic": "I can see moderate traffic flow with 8 vehicles currently in view. Traffic appears to be flowing smoothly with no apparent congestion. Most vehicles are maintaining safe following distances.",
            "count vehicles": "I count 5 cars, 2 trucks, and 1 motorcycle currently visible in the intersection. Traffic density appears normal for this time of day.",
            "detect violations": "I don't detect any obvious traffic violations at this moment. All vehicles appear to be following traffic signals and maintaining proper lanes.",
            "safety assessment": "Overall safety conditions look good. Visibility is clear, traffic signals are functioning properly, and vehicle speeds appear appropriate for the intersection."
        }
        
        # Find best matching response
        response = None
        for key, value in responses.items():
            if key.lower() in user_message.lower():
                response = value
                break
        
        if not response:
            response = f"I understand you're asking about '{user_message}'. Based on the current traffic scene, I can provide analysis of vehicle movements, count objects, assess safety conditions, and identify potential violations. Could you be more specific about what aspect you'd like me to focus on?"
        
        # Add AI response
        self._add_message(response, is_user=False)
        
        # Update status
        self.status_label.setText("Ready to analyze traffic scenes")
        
        # Emit insight signal
        self.insight_generated.emit(response)
    
    def _clear_conversation(self):
        """Clear conversation history"""
        # Remove all message widgets
        for i in reversed(range(self.conversation_layout.count())):
            child = self.conversation_layout.itemAt(i).widget()
            if child:
                child.setParent(None)
        
        # Clear history
        self.conversation_history.clear()
        
        # Update stats
        self._update_stats()
        
        print("🤖 Conversation cleared")
    
    def _export_conversation(self):
        """Export conversation history"""
        self.conversation_exported.emit("conversation_history")
        print("🤖 Conversation exported")
    
    def _change_image_source(self, source):
        """Change image context source"""
        self.image_preview.setText(f"Source: {source}")
        print(f"🤖 Image source changed to: {source}")
    
    def _capture_current_frame(self):
        """Capture current frame for analysis"""
        # Simulate frame capture
        self.image_preview.setText("Current frame\ncaptured")
        self.current_image_context = "current_frame"
        print("🤖 Current frame captured for analysis")
    
    def _get_current_settings(self):
        """Get current VLM settings"""
        return {
            'model': self.model_combo.currentText(),
            'temperature': self.temperature_slider.value() / 10.0,
            'max_length': self.max_length_spin.value(),
            'detailed_analysis': self.detailed_analysis_cb.isChecked(),
            'safety_focus': self.safety_focus_cb.isChecked()
        }
    
    def _update_stats(self):
        """Update conversation statistics"""
        total_messages = len(self.conversation_history)
        ai_messages = sum(1 for msg in self.conversation_history if not msg['is_user'])
        
        self.message_count_label.setText(f"Messages: {total_messages}")
        self.insights_count_label.setText(f"Insights: {ai_messages}")
    
    def add_ai_insight(self, insight_text):
        """Add an AI insight to the conversation"""
        self._add_message(insight_text, is_user=False)
    
    def set_image_context(self, image_data):
        """Set image context for VLM analysis"""
        self.current_image_context = image_data
        # Update image preview if needed
        print("🤖 Image context updated for VLM analysis")
