import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QStatusBar, 
                            QGroupBox, QComboBox, QSlider, QCheckBox, QSpinBox,
                            QScrollArea, QGridLayout, QMessageBox, QSplitter,
                            QTabWidget, QDoubleSpinBox, QListWidget, QStackedWidget,
                            QListWidgetItem, QFrame, QLineEdit, QStyle, QTextEdit, 
                            QDialog, QInputDialog, QButtonGroup, QSizePolicy)
from PyQt6.QtGui import QPixmap, QImage, QFont, QIcon, QColor, QKeySequence
from PyQt6.QtCore import Qt, QSize, QThread, QPoint, QRect, QMargins, pyqtSignal
from typing import Optional
from gui_worker import GestureEngineWorker
from config_manager import config_manager, ApplicationModeConfig, ApplicationModeGesture
from benchmark_dialog import BenchmarkDialog

def get_gesture_display_name(gesture_key):
    """Returns a user-friendly display name for a given gesture key."""
    gesture_names = {
        'right_index_bent': 'Right Hand: Bend INDEX finger down',
        'left_index_bent': 'Left Hand: Bend INDEX finger down', 
        'right_index_middle_bent': 'Right Hand: Bend INDEX + MIDDLE fingers down',
        'left_index_middle_bent': 'Left Hand: Bend INDEX + MIDDLE fingers down',
        'fist_gesture': 'Either Hand: Make a FIST (close all fingers)',
        'open_palm_gesture': 'Either Hand: Show OPEN PALM (all fingers extended)',
        'iloveyou_gesture': 'Either Hand: I LOVE YOU sign (thumb + index + pinky up)',
    }
    return gesture_names.get(gesture_key, gesture_key.replace('_', ' ').title())

class FlowLayout(QGridLayout):
    """A layout that arranges widgets in a flowing manner."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._item_list = []

    def addItem(self, item):
        self._item_list.append(item)

    def count(self):
        return len(self._item_list)

    def itemAt(self, index):
        if 0 <= index < len(self._item_list):
            return self._item_list[index]
        return None

    def takeAt(self, index):
        if 0 <= index < len(self._item_list):
            return self._item_list.pop(index)
        return None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._do_layout(QRect(0, 0, width, 0), True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._do_layout(rect, False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for item in self._item_list:
            size = size.expandedTo(item.minimumSize())
        size += QSize(2 * self.contentsMargins().top(), 2 * self.contentsMargins().top())
        return size

    def _do_layout(self, rect, test_only):
        x = rect.x()
        y = rect.y()
        line_height = 0
        spacing = self.spacing()

        for item in self._item_list:
            wid = item.widget()
            space_x = spacing + wid.style().layoutSpacing(
                QSizePolicy.ControlType.PushButton, QSizePolicy.ControlType.PushButton, Qt.Orientation.Horizontal
            )
            space_y = spacing + wid.style().layoutSpacing(
                QSizePolicy.ControlType.PushButton, QSizePolicy.ControlType.PushButton, Qt.Orientation.Vertical
            )
            next_x = x + item.sizeHint().width() + space_x
            if next_x - space_x > rect.right() and line_height > 0:
                x = rect.x()
                y = y + line_height + space_y
                next_x = x + item.sizeHint().width() + space_x
                line_height = 0

            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), item.sizeHint()))

            x = next_x
            line_height = max(line_height, item.sizeHint().height())

        return y + line_height - rect.y()


class KeyCaptureButton(QPushButton):
    """A button that captures a key combination of any length when clicked."""
    binding_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__("Click to bind", parent)
        self.is_recording = False
        self.binding_keys = []  # The final, saved list of keys
        self.temp_keys = set()  # A temporary set for recording to avoid duplicates
        self.setToolTip("Double-left-click to bind keys.\nDouble-right-click to clear.\nPress Enter to save, Esc to cancel.")
        self.setStyleSheet("""
            QPushButton {
                text-align: left;
                padding-left: 10px;
                background-color: #3e3e42;
                border: 1px solid #5a5a5a;
                color: #ffffff;
                min-height: 28px;
            }
            QPushButton:hover {
                border-color: #0078d4;
            }
            QPushButton:focus {
                border: 2px solid #0078d4;
                background-color: #4a4a4e;
            }
        """)

    def getBindingString(self) -> str:
        """Returns the binding as a sorted, lowercase, '+'-separated string."""
        modifiers = sorted([k for k in self.binding_keys if k in ['ctrl', 'alt', 'shift', 'win']])
        primary = sorted([k for k in self.binding_keys if k not in ['ctrl', 'alt', 'shift', 'win']])
        return "+".join(modifiers + primary).lower()

    def setBinding(self, key_string: Optional[str]):
        """Sets the binding from a string like 'ctrl+s'."""
        if key_string and isinstance(key_string, str):
            self.binding_keys = [key for key in key_string.lower().split('+') if key]
        else:
            self.binding_keys = []
        self._update_text()

    def _update_text(self):
        """Updates the button's display text based on its state."""
        keys_to_display = self.temp_keys if self.is_recording else self.binding_keys
        
        if self.is_recording and not keys_to_display:
            self.setText("Press keys...")
        elif not keys_to_display:
            self.setText("Click to bind")
        else:
            modifiers = sorted([k for k in keys_to_display if k in ['ctrl', 'alt', 'shift', 'win']])
            primary = sorted([k for k in keys_to_display if k not in ['ctrl', 'alt', 'shift', 'win']])
            display_keys = [k.title() for k in (modifiers + primary)]
            self.setText(" + ".join(display_keys))

    def mouseDoubleClickEvent(self, event):
        """Handle double-click events for starting recording or resetting."""
        if event.button() == Qt.MouseButton.LeftButton:
            if not self.is_recording:
                self._start_recording()
        elif event.button() == Qt.MouseButton.RightButton:
            self._reset_binding()
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event: QKeySequence):
        if self.is_recording:
            if event.isAutoRepeat():
                return

            key = event.key()

            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self._stop_recording(save_changes=True)
                return
            if key == Qt.Key.Key_Escape:
                self._stop_recording(save_changes=False)
                return
            
            key_name = ""
            if key == Qt.Key.Key_Control: key_name = 'ctrl'
            elif key == Qt.Key.Key_Shift: key_name = 'shift'
            elif key == Qt.Key.Key_Alt: key_name = 'alt'
            elif key == Qt.Key.Key_Meta: key_name = 'win'
            elif key != Qt.Key.Key_unknown:
                key_name = QKeySequence(key).toString().lower()

            if key_name:
                self.temp_keys.add(key_name)
            
            self._update_text()
        else:
            super().keyPressEvent(event)

    def focusOutEvent(self, event):
        if self.is_recording:
            self._stop_recording(save_changes=True)
        super().focusOutEvent(event)

    def _start_recording(self):
        self.is_recording = True
        self.temp_keys = set(self.binding_keys)
        self.grabKeyboard()
        self._update_text()

    def _stop_recording(self, save_changes=True):
        if not self.is_recording:
            return
            
        self.is_recording = False
        self.releaseKeyboard()
        
        if save_changes:
            self.binding_keys = list(self.temp_keys)
            self.binding_changed.emit(self.getBindingString())

        self.temp_keys.clear()
        self._update_text()

    def _reset_binding(self):
        """Resets the current binding to empty."""
        if self.is_recording:
            self._stop_recording(save_changes=False)
        self.binding_keys = []
        self.temp_keys.clear()
        self.binding_changed.emit("")
        self._update_text()




class CustomModeDialog(QDialog):
    """Modal dialog for creating/editing custom gesture modes."""
    
    def __init__(self, parent=None, edit_mode=None, mode_data=None):
        super().__init__(parent)
        self.edit_mode = edit_mode
        self.mode_data = mode_data
        
        # Define valid keys and buttons separately
        self.valid_mouse_buttons = ["left", "right", "middle"]
        # NEW: Separate keys for better UI
        all_keys = sorted(config_manager.gesture_mapping.available_keys)
        self.modifier_keys = sorted([k for k in all_keys if k in ['ctrl', 'alt', 'shift', 'win', 'cmd', 'option']])
        self.primary_keys = sorted([k for k in all_keys if k not in self.modifier_keys])
        self.all_keys_sorted = self.modifier_keys + self.primary_keys # Modifiers first
        
        self.setWindowTitle("Custom Mode Builder" if not edit_mode else f"Edit {edit_mode.replace('_', ' ').title()}")
        self.setMinimumSize(800, 700) # Increased height for better spacing
        self.setModal(True)
        if parent and hasattr(parent, 'styleSheet'):
            self.setStyleSheet(parent.styleSheet())
        
        self.setup_ui()
        if edit_mode and mode_data:
            self.load_existing_mode()

    def setup_ui(self):
        """Setup the custom mode dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(20, 20, 20, 20)

        # Header
        if self.edit_mode:
            header = QLabel(f"✏️ Editing {self.edit_mode.replace('_', ' ').title()}")
        else:
            header = QLabel("🛠️ Create Your Custom Gesture Mode")
        header.setStyleSheet("font-size: 18px; font-weight: bold; color: #00ffff; margin-bottom: 10px;")
        layout.addWidget(header)

        # Mode Name (only for new modes)
        if not self.edit_mode:
            name_frame = QFrame()
            name_layout = QHBoxLayout(name_frame)
            name_layout.addWidget(QLabel("Mode Name:"))
            self.mode_name_edit = QLineEdit("My Custom Mode")
            name_layout.addWidget(self.mode_name_edit)
            layout.addWidget(name_frame)

        # Gestures Section
        gestures_group = QGroupBox("Configure Gestures")
        gestures_layout = QVBoxLayout(gestures_group)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        
        available_gestures = [
            'right_index_bent', 'left_index_bent', 
            'right_index_middle_bent', 'left_index_middle_bent', 
            'fist_gesture',
            'open_palm_gesture',     # ADD NEW GESTURES
            'iloveyou_gesture'       # ADD NEW GESTURES
        ]
        
        self.gesture_widgets = {}
        for gesture_key in available_gestures:
            gesture_card = self.create_gesture_card(gesture_key)
            scroll_layout.addWidget(gesture_card)
        
        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        gestures_layout.addWidget(scroll)
        layout.addWidget(gestures_group)

        # Action Buttons
        button_layout = QHBoxLayout()
        
        cancel_btn = QPushButton(" Cancel")
        cancel_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogCancelButton))
        save_btn = QPushButton(" Save Mode" if not self.edit_mode else " Update Mode")
        save_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton))
        
        cancel_btn.clicked.connect(self.reject)
        save_btn.clicked.connect(self.save_mode)
        
        button_layout.addStretch()
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(save_btn)
        layout.addLayout(button_layout)

    def _on_action_type_changed(self, gesture_key, is_key_press):
        """Called when action type changes - switches between key/mouse UI."""
        widgets = self.gesture_widgets[gesture_key]
        widgets['action_stack'].setCurrentIndex(0 if is_key_press else 1)
        
        # if is_key_press:
        #     # Switched to Key Press - default to 'space'
        #     widgets['current_key'] = 'space'
        #     self.update_key_display(gesture_key, 'space')
        # else:
        #     # Switched to Mouse Click - default to 'left'
        #     widgets['current_key'] = 'left'
        #     self.update_key_display(gesture_key, 'left')  
          
    def create_gesture_card(self, gesture_key):
        """Create a card for configuring a single gesture."""
        card = QFrame()
        card.setFrameStyle(QFrame.Shape.StyledPanel)
        card.setStyleSheet("""
            QFrame {
                background-color: #2d2d30;
                border: 1px solid #3e3e42;
                border-radius: 8px;
                margin: 4px;
                padding: 12px;
            }
        """)
        
        layout = QVBoxLayout(card)
        
        gesture_name = get_gesture_display_name(gesture_key)
        name_label = QLabel(gesture_name)
        name_label.setStyleSheet("""
            font-weight: 700;
            font-size: 14px;
            color: #ffffff;
            padding: 6px;
            background-color: #1e1e1e;
            border-radius: 4px;
            margin-bottom: 8px;
        """)
        layout.addWidget(name_label)
        
        controls_layout = QGridLayout()        
        enable_cb = QCheckBox("Enable this gesture")
        enable_cb.setStyleSheet("""
            QCheckBox {
                color: #ffffff;
                font-weight: bold;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 2px solid #5a5a5a;
                border-radius: 3px;
                background-color: #1e1e1e;
            }
            QCheckBox::indicator:checked {
                background-color: #0078d4;
                border-color: #0078d4;
            }
            QCheckBox::indicator:hover {
                border-color: #0078d4;
            }
        """)
        enable_cb.toggled.connect(lambda checked, key=gesture_key: self.toggle_gesture_controls(key, checked))
        controls_layout.addWidget(enable_cb, 0, 0, 1, 2)
        controls_layout.addWidget(QLabel("Description:"), 1, 0)
        desc_edit = QLineEdit()
        controls_layout.addWidget(desc_edit, 1, 1)
        
        controls_layout.addWidget(QLabel("Action Type:"), 2, 0)
        action_frame = QFrame()
        action_layout = QHBoxLayout(action_frame)
        action_layout.setContentsMargins(0, 0, 0, 0)        
        key_press_btn = QPushButton(" Key Press")
        mouse_click_btn = QPushButton(" Mouse Click")
        
        key_press_btn.setCheckable(True)
        mouse_click_btn.setCheckable(True)
        key_press_btn.setChecked(True)
        
        key_press_btn.toggled.connect(lambda checked, key=gesture_key: self._on_action_type_changed(key, checked))
        
        action_button_group = QButtonGroup(action_frame)
        action_button_group.addButton(key_press_btn)
        action_button_group.addButton(mouse_click_btn)
        action_button_group.setExclusive(True)
        
        action_layout.addWidget(key_press_btn)
        action_layout.addWidget(mouse_click_btn)
        controls_layout.addWidget(action_frame, 2, 1)
        
        # --- NEW: Stacked widget for Key vs Mouse selection ---
        controls_layout.addWidget(QLabel("Action Target:"), 3, 0)
        action_stack = QStackedWidget()
        
        # --- Page 0: Key Combination Builder ---
        key_capture_button = KeyCaptureButton()
        action_stack.addWidget(key_capture_button)

        # --- Page 1: Mouse Button Selector ---
        mouse_button_combo = QComboBox()
        mouse_button_combo.addItems(self.valid_mouse_buttons)
        action_stack.addWidget(mouse_button_combo)
        
        controls_layout.addWidget(action_stack, 3, 1)
        
        controls_layout.addWidget(QLabel("Cooldown (s):"), 4, 0)
        cooldown_spin = QDoubleSpinBox()
        cooldown_spin.setRange(0.1, 5.0)
        cooldown_spin.setSingleStep(0.1)
        cooldown_spin.setValue(0.8)
        controls_layout.addWidget(cooldown_spin, 4, 1)
        
        # --- FIX: Add the controls layout and store the widgets ---
        layout.addLayout(controls_layout)
        
        self.gesture_widgets[gesture_key] = {
            'enable': enable_cb,
            'description': desc_edit,
            'key_press_btn': key_press_btn,
            'mouse_click_btn': mouse_click_btn,
            'action_stack': action_stack,
            'key_capture_button': key_capture_button,
            'mouse_button_combo': mouse_button_combo,
            'cooldown': cooldown_spin,
        }
        
        # Disable all controls by default until 'Enable' is checked
        self.toggle_gesture_controls(gesture_key, False)
        
        # --- FIX: Return the created card widget ---
        return card

    def _add_key_combo(self, gesture_key):
        widgets = self.gesture_widgets[gesture_key]
        key_combos = widgets['key_combos']
        
        for i, combo in enumerate(key_combos):
            if not combo.isVisible():
                combo.setVisible(True)
                widgets['remove_key_btn'].setVisible(True)
                if i == len(key_combos) - 1:
                    widgets['add_key_btn'].setVisible(False)
                return

    def _remove_key_combo(self, gesture_key):
        widgets = self.gesture_widgets[gesture_key]
        key_combos = widgets['key_combos']
        
        for i in range(len(key_combos) - 1, -1, -1):
            if key_combos[i].isVisible():
                key_combos[i].setVisible(False)
                widgets['add_key_btn'].setVisible(True)
                if i == 1:
                    widgets['remove_key_btn'].setVisible(False)
                return
    
    

    def toggle_gesture_controls(self, gesture_key, enabled):
        widgets = self.gesture_widgets[gesture_key]
        widgets['description'].setEnabled(enabled)
        widgets['key_press_btn'].parent().setEnabled(enabled)
        widgets['cooldown'].setEnabled(enabled)
        widgets['action_stack'].setEnabled(enabled)



    def load_existing_mode(self):
        """Load existing mode data into the dialog."""
        if not self.mode_data:
            return
            
        if hasattr(self, 'mode_name_edit'):
            self.mode_name_edit.setText(self.mode_data.name)
            if self.edit_mode:
                self.mode_name_edit.setEnabled(False)
            
        for gesture_key, gesture_data in self.mode_data.gestures.items():
            if gesture_key in self.gesture_widgets:
                widgets = self.gesture_widgets[gesture_key]
                
                widgets['enable'].setChecked(True)
                widgets['description'].setText(gesture_data.description)
                widgets['cooldown'].setValue(gesture_data.cooldown)
                
                if gesture_data.action == 'key_press':
                    widgets['key_press_btn'].setChecked(True)
                    widgets['action_stack'].setCurrentIndex(0)
                    # --- FIX: Use the new key capture button ---
                    widgets['key_capture_button'].setBinding(gesture_data.key)

                elif gesture_data.action == 'mouse_click':
                    widgets['mouse_click_btn'].setChecked(True)
                    widgets['action_stack'].setCurrentIndex(1)
                    widgets['mouse_button_combo'].setCurrentText(gesture_data.button)

    def save_mode(self):
        """Save the custom mode."""
        if self.edit_mode and self.mode_data:
            mode_name = self.mode_data.name
            mode_key = self.edit_mode
        else:
            mode_name = self.mode_name_edit.text().strip()
            if not mode_name:
                QMessageBox.warning(self, "Input Error", "Mode name cannot be empty.")
                return
            mode_key = f"{mode_name.lower().replace(' ', '_')}_mode"
            if hasattr(config_manager.app_modes, mode_key) and not self.edit_mode:
                QMessageBox.warning(self, "Input Error", f"A mode with key '{mode_key}' already exists.")
                return
        
        gestures = {}
        enabled_count = 0
        
        for gesture_key, widgets in self.gesture_widgets.items():
            if widgets['enable'].isChecked():
                enabled_count += 1
                description = widgets['description'].text()
                cooldown = widgets['cooldown'].value()
                
                action = 'key_press' if widgets['key_press_btn'].isChecked() else 'mouse_click'
                key = None
                button = None
                
                if action == 'key_press':
                    # --- FIX: Use the new key capture button ---
                    key = widgets['key_capture_button'].getBindingString()
                    if not key: # Ensure a key is bound
                        QMessageBox.warning(self, "Input Error", f"No key bound for gesture: {get_gesture_display_name(gesture_key)}")
                        return
                else: # mouse_click
                    button = widgets['mouse_button_combo'].currentText()

                gestures[gesture_key] = ApplicationModeGesture(
                    action=action,
                    key=key,
                    button=button,
                    description=description,
                    cooldown=cooldown
                )
        
        if enabled_count == 0:
            QMessageBox.warning(self, "Input Error", "You must enable and configure at least one gesture.")
            return
        
        if self.edit_mode and self.mode_data:
            mode_config = self.mode_data
            mode_config.name = mode_name
            mode_config.gestures = gestures
        else:
            mode_config = ApplicationModeConfig(name=mode_name, gestures=gestures)
            setattr(config_manager.app_modes, mode_key, mode_config)
        
        config_manager.save_config()
        
        action_str = "updated" if self.edit_mode else "created"
        QMessageBox.information(self, "Success", f"Mode '{mode_name}' {action_str} with {enabled_count} gestures!")
        
        self.accept()

class SettingsDialog(QDialog):
    """A dedicated dialog for all application settings."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configuration Settings")
        self.setWindowIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon))
        self.setMinimumSize(800, 700)
        self.setModal(True)
        if parent and hasattr(parent, 'styleSheet'):
            self.setStyleSheet(parent.styleSheet()) # Inherit stylesheet

        self.setup_ui()
        self.load_settings()

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # Tabs for organization
        tabs = QTabWidget()
        main_layout.addWidget(tabs)

        # Create tabs
        tabs.addTab(self.create_detection_tab(), "Detection")
        

        # Set tab icons
        tabs.setTabIcon(0, self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogDetailedView))
        


 # Create tabs
        tabs.addTab(self.create_detection_tab(), "Detection")
        tabs.addTab(self.create_control_tab(), "Control System")
        tabs.addTab(self.create_smart_palm_tab(), "Smart Palm")
        tabs.addTab(self.create_device_tab(), "Device Selection")
        
        # Set tab icons
        tabs.setTabIcon(0, self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogDetailedView))
        tabs.setTabIcon(1, self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        tabs.setTabIcon(2, self.style().standardIcon(QStyle.StandardPixmap.SP_DesktopIcon))
        tabs.setTabIcon(3, self.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon))


        # Action Buttons
        button_layout = QHBoxLayout()
        reset_btn = QPushButton(" Reset to Defaults")
        reset_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
        reset_btn.clicked.connect(self.reset_settings)
        
        self.save_btn = QPushButton(" Save & Apply")
        self.save_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton))
        self.save_btn.clicked.connect(self.accept)
        
        cancel_btn = QPushButton(" Cancel")
        cancel_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogCancelButton))
        cancel_btn.clicked.connect(self.reject)

        button_layout.addWidget(reset_btn)
        button_layout.addStretch()
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(self.save_btn)
        main_layout.addLayout(button_layout)

    def _add_widget(self, layout, row, label, widget):
        """Helper to add a labeled widget to a grid layout."""
        label_widget = QLabel(label)
        layout.addWidget(label_widget, row, 0, Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(widget, row, 1, Qt.AlignmentFlag.AlignRight)

    def create_detection_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(20)

        # Core Detection Group
        group = QGroupBox("Core Detection Parameters")
        grid = QGridLayout(group)
        self.input_size_spin = QSpinBox()
        self.input_size_spin.setRange(64, 512)
        self._add_widget(grid, 0, "Input Size (px):", self.input_size_spin)

        self.score_threshold_spin = QDoubleSpinBox()
        self.score_threshold_spin.setRange(0.1, 1.0)
        self.score_threshold_spin.setSingleStep(0.05)
        self._add_widget(grid, 1, "Score Threshold:", self.score_threshold_spin)

        self.nms_threshold_spin = QDoubleSpinBox()
        self.nms_threshold_spin.setRange(0.1, 1.0)
        self.nms_threshold_spin.setSingleStep(0.05)
        self._add_widget(grid, 2, "NMS Threshold:", self.nms_threshold_spin)

        self.iou_match_threshold_spin = QDoubleSpinBox()
        self.iou_match_threshold_spin.setRange(0.1, 1.0)
        self.iou_match_threshold_spin.setSingleStep(0.05)
        self._add_widget(grid, 3, "IOU Match Threshold:", self.iou_match_threshold_spin)
        
        self.detection_smoothing_alpha_spin = QDoubleSpinBox()
        self.detection_smoothing_alpha_spin.setRange(0.0, 1.0)
        self.detection_smoothing_alpha_spin.setSingleStep(0.05)
        self._add_widget(grid, 4, "Detection Smoothing Alpha:", self.detection_smoothing_alpha_spin)

        self.landmark_score_threshold_spin = QDoubleSpinBox()
        self.landmark_score_threshold_spin.setRange(0.1, 1.0)
        self.landmark_score_threshold_spin.setSingleStep(0.05)
        self._add_widget(grid, 5, "Landmark Redetection Threshold:", self.landmark_score_threshold_spin)

        self.always_run_palm_cb = QCheckBox("Always Run Palm Detection")
        grid.addWidget(self.always_run_palm_cb, 6, 0, 1, 2)
        
        self.enable_finger_detection_cb = QCheckBox("Enable Finger Angle Detection")
        grid.addWidget(self.enable_finger_detection_cb, 7, 0, 1, 2)
        layout.addWidget(group)

        # Visuals Group
        group_vis = QGroupBox("Visuals & Overlays")
        grid_vis = QGridLayout(group_vis)
        self.show_landmarks_cb = QCheckBox("Show Hand Landmarks")
        grid_vis.addWidget(self.show_landmarks_cb, 0, 0)
        self.show_static_gestures_cb = QCheckBox("Show Static Gestures (e.g., Fist)")
        grid_vis.addWidget(self.show_static_gestures_cb, 1, 0)
        layout.addWidget(group_vis)

        layout.addStretch()
        return tab

    def create_device_tab(self):
        """Create a Device Selection tab allowing per-model device selection."""
        from openvino_models import model_manager

        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(20)

        group = QGroupBox("Device Selection for Models")
        grid = QGridLayout(group)

        # Get device data and extract description strings
        devices_data = model_manager.get_available_devices_with_descriptions()
        if devices_data:
            devices = [d['description'] if isinstance(d, dict) else str(d) for d in devices_data]
        else:
            devices = ['CPU', 'AUTO']

        self.enable_device_switching_cb = QCheckBox("Enable Device Switching")
        grid.addWidget(self.enable_device_switching_cb, 0, 0, 1, 2)

        self.model_device_combos = {}
        model_items = [
            ("palm_detection", "Palm Detector Device:"),
            ("hand_landmarks", "Hand Landmarks Device:"),
            ("gesture_embedder", "Gesture Embedder Device:"),
            ("gesture_classifier", "Gesture Classifier Device:")
        ]

        for i, (key, label) in enumerate(model_items, start=1):
            combo = QComboBox()
            combo.addItems(devices)
            grid.addWidget(QLabel(label), i, 0)
            grid.addWidget(combo, i, 1)
            self.model_device_combos[key] = combo

        layout.addWidget(group)
        layout.addStretch()
        return tab


    

    def create_control_tab(self):
        tab = QWidget()
        main_layout = QVBoxLayout(tab)
        main_layout.setSpacing(10)

        # Create a QTabWidget for sub-sections
        sub_tabs = QTabWidget()
        main_layout.addWidget(sub_tabs)

        # --- Cursor Control Tab ---
        cursor_tab = QWidget()
        cursor_layout = QVBoxLayout(cursor_tab)
        cursor_layout.setContentsMargins(10, 10, 10, 10)
        cursor_layout.setSpacing(20)
        group_cursor = QGroupBox("Cursor Control")
        group_cursor.setStyleSheet("QGroupBox { padding: 18px; }")
        grid_cursor = QGridLayout(group_cursor)
        grid_cursor.setHorizontalSpacing(20)
        grid_cursor.setVerticalSpacing(12)
        self.enable_cursor_cb = QCheckBox("Enable Cursor Control")
        grid_cursor.addWidget(self.enable_cursor_cb, 0, 0, 1, 2)
        self.cursor_smoothing_spin = QDoubleSpinBox()
        self.cursor_smoothing_spin.setRange(0.0, 1.0)
        self.cursor_smoothing_spin.setSingleStep(0.05)
        self._add_widget(grid_cursor, 1, "Cursor Smoothing:", self.cursor_smoothing_spin)
        self.cursor_sensitivity_spin = QDoubleSpinBox()
        self.cursor_sensitivity_spin.setRange(0.5, 10.0)
        self.cursor_sensitivity_spin.setSingleStep(0.1)
        self._add_widget(grid_cursor, 2, "Cursor Sensitivity:", self.cursor_sensitivity_spin)
        cursor_layout.addWidget(group_cursor)
        cursor_layout.addStretch()
        cursor_scroll = QScrollArea()
        cursor_scroll.setWidgetResizable(True)
        cursor_scroll.setWidget(cursor_tab)
        sub_tabs.addTab(cursor_scroll, "Cursor")

        # --- Scroll Control Tab ---
        scroll_tab = QWidget()
        scroll_layout = QVBoxLayout(scroll_tab)
        scroll_layout.setContentsMargins(10, 10, 10, 10)
        scroll_layout.setSpacing(20)
        group_scroll = QGroupBox("Scroll Control")
        group_scroll.setStyleSheet("QGroupBox { padding: 18px; }")
        grid_scroll = QGridLayout(group_scroll)
        grid_scroll.setHorizontalSpacing(20)
        grid_scroll.setVerticalSpacing(12)
        self.enable_scroll_cb = QCheckBox("Enable Scroll Control")
        grid_scroll.addWidget(self.enable_scroll_cb, 0, 0, 1, 2)
        self.scroll_sensitivity_spin = QSpinBox()
        self.scroll_sensitivity_spin.setRange(1, 50)
        self._add_widget(grid_scroll, 1, "Scroll Sensitivity:", self.scroll_sensitivity_spin)
        self.scroll_threshold_spin = QDoubleSpinBox()
        self.scroll_threshold_spin.setRange(0.01, 0.2)
        self.scroll_threshold_spin.setSingleStep(0.01)
        self._add_widget(grid_scroll, 2, "Scroll Activation Threshold:", self.scroll_threshold_spin)
        self.scroll_smoothing_spin = QDoubleSpinBox()
        self.scroll_smoothing_spin.setRange(0.0, 1.0)
        self.scroll_smoothing_spin.setSingleStep(0.05)
        self._add_widget(grid_scroll, 3, "Scroll Smoothing:", self.scroll_smoothing_spin)
        self.scroll_hand_pref_combo = QComboBox()
        self.scroll_hand_pref_combo.addItems(['any', 'left', 'right'])
        self._add_widget(grid_scroll, 4, "Scroll Hand Preference:", self.scroll_hand_pref_combo)
        scroll_layout.addWidget(group_scroll)
        scroll_layout.addStretch()
        scroll_scroll = QScrollArea()
        scroll_scroll.setWidgetResizable(True)
        scroll_scroll.setWidget(scroll_tab)
        sub_tabs.addTab(scroll_scroll, "Scroll")

        # --- Keyboard Control Tab ---
        key_tab = QWidget()
        key_layout = QVBoxLayout(key_tab)
        key_layout.setContentsMargins(10, 10, 10, 10)
        key_layout.setSpacing(20)
        group_key = QGroupBox("Keyboard Control")
        group_key.setStyleSheet("QGroupBox { padding: 18px; }")
        grid_key = QGridLayout(group_key)
        grid_key.setHorizontalSpacing(20)
        grid_key.setVerticalSpacing(12)
        self.enable_key_control_cb = QCheckBox("Enable General Key Control")
        grid_key.addWidget(self.enable_key_control_cb, 0, 0, 1, 2)
        self.key_cooldown_spin = QDoubleSpinBox()
        self.key_cooldown_spin.setRange(0.1, 5.0)
        self.key_cooldown_spin.setSingleStep(0.1)
        self.key_cooldown_spin.setSuffix(" s")
        self._add_widget(grid_key, 1, "Key Press Cooldown:", self.key_cooldown_spin)
        key_layout.addWidget(group_key)
        key_layout.addStretch()
        key_scroll = QScrollArea()
        key_scroll.setWidgetResizable(True)
        key_scroll.setWidget(key_tab)
        sub_tabs.addTab(key_scroll, "Keyboard")

        # --- Game Control Tab ---
        game_tab = QWidget()
        game_layout = QVBoxLayout(game_tab)
        game_layout.setContentsMargins(10, 10, 10, 10)
        game_layout.setSpacing(20)
        group_game = QGroupBox("Game Control (Racing)")
        group_game.setStyleSheet("QGroupBox { padding: 18px; }")
        grid_game = QGridLayout(group_game)
        grid_game.setHorizontalSpacing(20)
        grid_game.setVerticalSpacing(12)
        self.enable_game_cb = QCheckBox("Enable Game Control Mode")
        grid_game.addWidget(self.enable_game_cb, 0, 0, 1, 2)
        self.game_control_type_combo = QComboBox()
        self.game_control_type_combo.addItems(['keyboard', 'directinput'])
        self._add_widget(grid_game, 1, "Input Method:", self.game_control_type_combo)
        self.steering_sensitivity_spin = QDoubleSpinBox()
        self.steering_sensitivity_spin.setRange(0.1, 3.0)
        self.steering_sensitivity_spin.setSingleStep(0.1)
        self._add_widget(grid_game, 2, "Steering Sensitivity:", self.steering_sensitivity_spin)
        self.steering_deadzone_spin = QDoubleSpinBox()
        self.steering_deadzone_spin.setRange(0.0, 0.3)
        self.steering_deadzone_spin.setSingleStep(0.01)
        self._add_widget(grid_game, 3, "Steering Deadzone:", self.steering_deadzone_spin)
        self.open_palm_threshold_spin = QDoubleSpinBox()
        self.open_palm_threshold_spin.setRange(0.3, 1.0)
        self.open_palm_threshold_spin.setSingleStep(0.05)
        self._add_widget(grid_game, 4, "Open Palm Threshold:", self.open_palm_threshold_spin)

        # Add the missing parameters:

        self.steering_box_width_spin = QDoubleSpinBox()
        self.steering_box_width_spin.setRange(0.05, 1.0)
        self.steering_box_width_spin.setSingleStep(0.01)
        self._add_widget(grid_game, 5, "Steering Box Width:", self.steering_box_width_spin)

        self.steering_box_height_spin = QDoubleSpinBox()
        self.steering_box_height_spin.setRange(0.05, 1.0)
        self.steering_box_height_spin.setSingleStep(0.01)
        self._add_widget(grid_game, 6, "Steering Box Height:", self.steering_box_height_spin)

        self.steering_box_x_spin = QDoubleSpinBox()
        self.steering_box_x_spin.setRange(0.0, 1.0)
        self.steering_box_x_spin.setSingleStep(0.01)
        self._add_widget(grid_game, 7, "Steering Box X:", self.steering_box_x_spin)

        self.steering_box_y_spin = QDoubleSpinBox()
        self.steering_box_y_spin.setRange(0.0, 1.0)
        self.steering_box_y_spin.setSingleStep(0.01)
        self._add_widget(grid_game, 8, "Steering Box Y:", self.steering_box_y_spin)

        self.steering_smoothing_spin = QDoubleSpinBox()
        self.steering_smoothing_spin.setRange(0.0, 1.0)
        self.steering_smoothing_spin.setSingleStep(0.01)
        self._add_widget(grid_game, 9, "Steering Smoothing:", self.steering_smoothing_spin)

        self.steering_exponent_spin = QDoubleSpinBox()
        self.steering_exponent_spin.setRange(0.1, 3.0)
        self.steering_exponent_spin.setSingleStep(0.1)
        self._add_widget(grid_game, 10, "Steering Exponent:", self.steering_exponent_spin)

        self.steering_displacement_amplification_spin = QDoubleSpinBox()
        self.steering_displacement_amplification_spin.setRange(0.1, 10.0)
        self.steering_displacement_amplification_spin.setSingleStep(0.1)
        self._add_widget(grid_game, 11, "Steering Displacement Amplification:", self.steering_displacement_amplification_spin)

        self.game_gesture_cooldown_spin = QDoubleSpinBox()
        self.game_gesture_cooldown_spin.setRange(0.05, 2.0)
        self.game_gesture_cooldown_spin.setSingleStep(0.01)
        self._add_widget(grid_game, 12, "Game Gesture Cooldown:", self.game_gesture_cooldown_spin)

        # Add hand label offset controls
        self.hand_label_offset_x_spin = QDoubleSpinBox()
        self.hand_label_offset_x_spin.setRange(-200.0, 200.0)
        self.hand_label_offset_x_spin.setSingleStep(5.0)
        self._add_widget(grid_game, 13, "Global Label X Offset:", self.hand_label_offset_x_spin)

        self.hand_label_offset_y_spin = QDoubleSpinBox()
        self.hand_label_offset_y_spin.setRange(-200.0, 200.0)
        self.hand_label_offset_y_spin.setSingleStep(5.0)
        self._add_widget(grid_game, 14, "Global Label Y Offset:", self.hand_label_offset_y_spin)

        self.left_hand_offset_x_spin = QDoubleSpinBox()
        self.left_hand_offset_x_spin.setRange(-200.0, 200.0)
        self.left_hand_offset_x_spin.setSingleStep(5.0)
        self._add_widget(grid_game, 15, "Left Hand X Offset:", self.left_hand_offset_x_spin)

        self.left_hand_offset_y_spin = QDoubleSpinBox()
        self.left_hand_offset_y_spin.setRange(-200.0, 200.0)
        self.left_hand_offset_y_spin.setSingleStep(5.0)
        self._add_widget(grid_game, 16, "Left Hand Y Offset:", self.left_hand_offset_y_spin)

        self.right_hand_offset_x_spin = QDoubleSpinBox()
        self.right_hand_offset_x_spin.setRange(-200.0, 200.0)
        self.right_hand_offset_x_spin.setSingleStep(5.0)
        self._add_widget(grid_game, 17, "Right Hand X Offset:", self.right_hand_offset_x_spin)

        self.right_hand_offset_y_spin = QDoubleSpinBox()
        self.right_hand_offset_y_spin.setRange(-200.0, 200.0)
        self.right_hand_offset_y_spin.setSingleStep(5.0)
        self._add_widget(grid_game, 18, "Right Hand Y Offset:", self.right_hand_offset_y_spin)

        game_layout.addWidget(group_game)
        game_layout.addStretch()
        game_scroll = QScrollArea()
        game_scroll.setWidgetResizable(True)
        game_scroll.setWidget(game_tab)
        sub_tabs.addTab(game_scroll, "Game")

        main_layout.addStretch()
        return tab

    def create_smart_palm_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(20)
        
        group = QGroupBox("Smart Palm State Machine")
        grid = QGridLayout(group)
        
        self.grace_period_spin = QDoubleSpinBox()
        self.grace_period_spin.setRange(0.1, 5.0)
        self.grace_period_spin.setSingleStep(0.1)
        self.grace_period_spin.setSuffix(" s")
        self._add_widget(grid, 0, "Grace Period Duration:", self.grace_period_spin)

        self.periodic_check_spin = QSpinBox()
        self.periodic_check_spin.setRange(5, 100)
        self.periodic_check_spin.setSuffix(" frames")
        self._add_widget(grid, 1, "Periodic Check Interval:", self.periodic_check_spin)

        self.state_debug_cb = QCheckBox("Enable State Transition Debugging")
        grid.addWidget(self.state_debug_cb, 2, 0, 1, 2)
        
        layout.addWidget(group)
        layout.addStretch()
        return tab

    # def create_gesture_mapping_tab(self):
    #     tab = QWidget()
    #     layout = QVBoxLayout(tab)
    #     layout.setSpacing(20)

    #     group = QGroupBox("Gesture Mapping Engine")
    #     grid = QGridLayout(group)

    #     self.enable_gesture_mapping_cb = QCheckBox("Enable Gesture Mapping System")
    #     grid.addWidget(self.enable_gesture_mapping_cb, 0, 0, 1, 2)

    #     self.require_simultaneous_cb = QCheckBox("Require Simultaneous Detection for Combos")
    #     grid.addWidget(self.require_simultaneous_cb, 1, 0, 1, 2)

    #     self.global_cooldown_spin = QDoubleSpinBox()
    #     self.global_cooldown_spin.setRange(0.0, 2.0)
    #     self.global_cooldown_spin.setSingleStep(0.05)
    #     self.global_cooldown_spin.setSuffix(" s")
    #     self._add_widget(grid, 2, "Global Cooldown:", self.global_cooldown_spin)

    #     self.gesture_stability_frames_spin = QSpinBox()
    #     self.gesture_stability_frames_spin.setRange(1, 20)
    #     self.gesture_stability_frames_spin.setSuffix(" frames")
    #     self._add_widget(grid, 3, "Gesture Stability Frames:", self.gesture_stability_frames_spin)
    #     layout.addWidget(group)

    #     group_thresh = QGroupBox("Finger Angle Thresholds")
    #     grid_thresh = QGridLayout(group_thresh)
    #     self.relaxed_thresh_spin = QSpinBox()
    #     self.relaxed_thresh_spin.setRange(30, 100)
    #     self.relaxed_thresh_spin.setSuffix("°")
    #     self._add_widget(grid_thresh, 0, "Relaxed Angle Max:", self.relaxed_thresh_spin)

    #     self.bent_thresh_spin = QSpinBox()
    #     self.bent_thresh_spin.setRange(100, 180)
    #     self.bent_thresh_spin.setSuffix("°")
    #     self._add_widget(grid_thresh, 1, "Bent Angle Min:", self.bent_thresh_spin)
    #     layout.addWidget(group_thresh)

    #     layout.addStretch()
    #     return tab

    def load_settings(self):
        """Load all settings from config manager into the UI."""
        # Detection
        self.input_size_spin.setValue(config_manager.detection.input_size)
        self.score_threshold_spin.setValue(config_manager.detection.score_threshold)
        self.nms_threshold_spin.setValue(config_manager.detection.nms_threshold)
        self.iou_match_threshold_spin.setValue(config_manager.detection.iou_match_threshold)
        self.detection_smoothing_alpha_spin.setValue(config_manager.detection.detection_smoothing_alpha)
        self.landmark_score_threshold_spin.setValue(config_manager.detection.landmark_score_for_palm_redetection_threshold)
        self.always_run_palm_cb.setChecked(config_manager.detection.always_run_palm_detection)
        self.enable_finger_detection_cb.setChecked(config_manager.detection.enable_finger_detection)
        self.show_landmarks_cb.setChecked(config_manager.detection.show_landmarks)
        self.show_static_gestures_cb.setChecked(config_manager.detection.show_static_gestures)

        # Control System
        self.enable_cursor_cb.setChecked(config_manager.control_system.enable_cursor_control)
        self.cursor_smoothing_spin.setValue(config_manager.control_system.cursor_smoothing)
        self.cursor_sensitivity_spin.setValue(config_manager.control_system.cursor_sensitivity)
        self.enable_scroll_cb.setChecked(config_manager.control_system.enable_scroll_control)
        self.scroll_sensitivity_spin.setValue(config_manager.control_system.scroll_sensitivity)
        self.scroll_threshold_spin.setValue(config_manager.control_system.scroll_threshold)
        self.scroll_smoothing_spin.setValue(config_manager.control_system.scroll_smoothing)
        self.scroll_hand_pref_combo.setCurrentText(config_manager.control_system.scroll_hand_preference)
        self.enable_key_control_cb.setChecked(config_manager.control_system.enable_key_control)
        self.key_cooldown_spin.setValue(config_manager.control_system.key_press_cooldown)

        # Smart Palm
        self.grace_period_spin.setValue(config_manager.smart_palm.grace_period_duration)
        self.periodic_check_spin.setValue(config_manager.smart_palm.periodic_check_interval)
        self.state_debug_cb.setChecked(config_manager.smart_palm.state_transition_debug)

        # Game Mode 
        self.enable_game_cb.setChecked(config_manager.control_system.enable_game_control)
        self.game_control_type_combo.setCurrentText(config_manager.control_system.game_control_type)
        self.steering_sensitivity_spin.setValue(config_manager.control_system.steering_sensitivity)
        self.steering_deadzone_spin.setValue(config_manager.control_system.steering_deadzone)
        self.open_palm_threshold_spin.setValue(config_manager.control_system.open_palm_threshold)
        self.steering_box_width_spin.setValue(config_manager.control_system.steering_box_width)
        self.steering_box_height_spin.setValue(config_manager.control_system.steering_box_height)
        self.steering_box_x_spin.setValue(config_manager.control_system.steering_box_x)
        self.steering_box_y_spin.setValue(config_manager.control_system.steering_box_y)
        self.steering_smoothing_spin.setValue(config_manager.control_system.steering_smoothing)
        self.steering_exponent_spin.setValue(config_manager.control_system.steering_exponent)
        self.steering_displacement_amplification_spin.setValue(config_manager.control_system.steering_displacement_amplification)
        self.game_gesture_cooldown_spin.setValue(config_manager.control_system.game_gesture_cooldown)

        # Load hand label offset settings
        self.hand_label_offset_x_spin.setValue(config_manager.control_system.hand_label_offset_x)
        self.hand_label_offset_y_spin.setValue(config_manager.control_system.hand_label_offset_y)
        self.left_hand_offset_x_spin.setValue(config_manager.control_system.left_hand_offset_x)
        self.left_hand_offset_y_spin.setValue(config_manager.control_system.left_hand_offset_y)
        self.right_hand_offset_x_spin.setValue(config_manager.control_system.right_hand_offset_x)
        self.right_hand_offset_y_spin.setValue(config_manager.control_system.right_hand_offset_y)

        # Device Selection (if UI created)
        try:
            from openvino_models import model_manager
            devices = model_manager.get_available_devices_with_descriptions()
            # Ensure devices list exists
            if hasattr(self, 'enable_device_switching_cb'):
                self.enable_device_switching_cb.setChecked(config_manager.detection.enable_device_switching)
            if hasattr(self, 'model_device_combos'):
                for model_key, combo in self.model_device_combos.items():
                    sel = config_manager.detection.model_devices.get(model_key, None)
                    if sel in devices:
                        combo.setCurrentIndex(devices.index(sel))
            if hasattr(self, 'fallback_device_combo'):
                fb = config_manager.detection.fallback_device
                if fb in devices:
                    self.fallback_device_combo.setCurrentIndex(devices.index(fb))
        except Exception:
            # Non-fatal: device tab may not be present in older builds
            pass

    def save_settings(self):
        """Save all settings from UI to config manager."""
        # Detection
        config_manager.detection.input_size = self.input_size_spin.value()
        config_manager.detection.score_threshold = self.score_threshold_spin.value()
        config_manager.detection.nms_threshold = self.nms_threshold_spin.value()
        config_manager.detection.iou_match_threshold = self.iou_match_threshold_spin.value()
        config_manager.detection.detection_smoothing_alpha = self.detection_smoothing_alpha_spin.value()
        config_manager.detection.landmark_score_for_palm_redetection_threshold = self.landmark_score_threshold_spin.value()
        config_manager.detection.always_run_palm_detection = self.always_run_palm_cb.isChecked()
        config_manager.detection.enable_finger_detection = self.enable_finger_detection_cb.isChecked()
        config_manager.detection.show_landmarks = self.show_landmarks_cb.isChecked()
        config_manager.detection.show_static_gestures = self.show_static_gestures_cb.isChecked()

        # Control System
        config_manager.control_system.enable_cursor_control = self.enable_cursor_cb.isChecked()
        config_manager.control_system.cursor_smoothing = self.cursor_smoothing_spin.value()
        config_manager.control_system.cursor_sensitivity = self.cursor_sensitivity_spin.value()
        config_manager.control_system.enable_scroll_control = self.enable_scroll_cb.isChecked()
        config_manager.control_system.scroll_sensitivity = self.scroll_sensitivity_spin.value()
        config_manager.control_system.scroll_threshold = self.scroll_threshold_spin.value()
        config_manager.control_system.scroll_smoothing = self.scroll_smoothing_spin.value()
        config_manager.control_system.scroll_hand_preference = self.scroll_hand_pref_combo.currentText()
        config_manager.control_system.enable_key_control = self.enable_key_control_cb.isChecked()
        config_manager.control_system.key_press_cooldown = self.key_cooldown_spin.value()

        # Smart Palm
        config_manager.smart_palm.grace_period_duration = self.grace_period_spin.value()
        config_manager.smart_palm.periodic_check_interval = self.periodic_check_spin.value()
        config_manager.smart_palm.state_transition_debug = self.state_debug_cb.isChecked()


        # Game Control Settings
        config_manager.control_system.enable_game_control = self.enable_game_cb.isChecked()
        config_manager.control_system.game_control_type = self.game_control_type_combo.currentText()
        config_manager.control_system.steering_sensitivity = self.steering_sensitivity_spin.value()
        config_manager.control_system.steering_deadzone = self.steering_deadzone_spin.value()
        config_manager.control_system.open_palm_threshold = self.open_palm_threshold_spin.value()
        config_manager.control_system.steering_box_width = self.steering_box_width_spin.value()
        config_manager.control_system.steering_box_height = self.steering_box_height_spin.value()
        config_manager.control_system.steering_box_x = self.steering_box_x_spin.value()
        config_manager.control_system.steering_box_y = self.steering_box_y_spin.value()
        config_manager.control_system.steering_smoothing = self.steering_smoothing_spin.value()
        config_manager.control_system.steering_exponent = self.steering_exponent_spin.value()
        config_manager.control_system.steering_displacement_amplification = self.steering_displacement_amplification_spin.value()
        config_manager.control_system.game_gesture_cooldown = self.game_gesture_cooldown_spin.value()

        # Save hand label offset settings
        config_manager.control_system.hand_label_offset_x = self.hand_label_offset_x_spin.value()
        config_manager.control_system.hand_label_offset_y = self.hand_label_offset_y_spin.value()
        config_manager.control_system.left_hand_offset_x = self.left_hand_offset_x_spin.value()
        config_manager.control_system.left_hand_offset_y = self.left_hand_offset_y_spin.value()
        config_manager.control_system.right_hand_offset_x = self.right_hand_offset_x_spin.value()
        config_manager.control_system.right_hand_offset_y = self.right_hand_offset_y_spin.value()
        
        # Device Selection (if UI present)
        try:
            from openvino_models import model_manager
            devices = model_manager.get_available_devices_with_descriptions()
            if hasattr(self, 'enable_device_switching_cb'):
                config_manager.detection.enable_device_switching = self.enable_device_switching_cb.isChecked()
            if hasattr(self, 'model_device_combos'):
                for model_key, combo in self.model_device_combos.items():
                    idx = combo.currentIndex()
                    if 0 <= idx < len(devices):
                        config_manager.detection.model_devices[model_key] = devices[idx]
            if hasattr(self, 'fallback_device_combo'):
                idx = self.fallback_device_combo.currentIndex()
                if 0 <= idx < len(devices):
                    config_manager.detection.fallback_device = devices[idx]
        except Exception:
            pass
            
        config_manager.save_config()
        QMessageBox.information(self, "Settings Saved", "All settings have been applied and saved successfully!")


    def reset_settings(self):
        reply = QMessageBox.question(self, "Reset Settings", 
            "Are you sure you want to reset all settings to their defaults?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            config_manager.reset_to_defaults()
            self.load_settings()
            # The main window needs to be refreshed too
            main_window = self.parent()
            if main_window and isinstance(main_window, GestureDashboard):
                main_window.refresh_mode_list()
                main_window.update_mode_combo()
            QMessageBox.information(self, "Settings Reset", "Settings have been reset to their default values.")

    def accept(self):
        self.save_settings()
        super().accept()

class NoArrowKeyComboBox(QComboBox):
    """QComboBox that ignores arrow key events for mode switching."""
    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down):
            # Ignore arrow keys to prevent accidental mode switching
            event.ignore()
            return
        super().keyPressEvent(event)


class PopoutWindow(QWidget):
    stop_engine_requested = pyqtSignal()
    pause_engine_requested = pyqtSignal()
    mode_changed_requested = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Gesture Control Mini")
        self.setMinimumSize(320, 240)
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Window)
        self.is_docked = False
        self.popout_window = None

        layout = QVBoxLayout(self)
        self.video_label = QLabel("Waiting for camera...")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: white; border-radius: 4px;")
        self.video_label.setMinimumSize(400, 300)  # Increased size (width, height)
        self.video_label.setMaximumSize(800, 600)
        layout.addWidget(self.video_label, 1)

        controls = QHBoxLayout()
        self.mode_combo = QComboBox()
        self.pause_button = QPushButton("Pause")
        self.stop_button = QPushButton("Stop")
        self.dock_button = QPushButton("📌 Dock")
        self.dock_button.setCheckable(True)
        controls.addWidget(self.mode_combo)
        controls.addWidget(self.pause_button)
        controls.addWidget(self.stop_button)
        controls.addWidget(self.dock_button)
        layout.addLayout(controls)

        self.stop_button.clicked.connect(self.stop_engine_requested.emit)
        self.pause_button.clicked.connect(self.pause_engine_requested.emit)
        self.mode_combo.currentTextChanged.connect(self.mode_changed_requested.emit)
        self.dock_button.toggled.connect(self.toggle_dock)

    def update_video(self, rgb_frame):
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.video_label.setPixmap(pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))

    def populate_modes(self, modes: list, current_mode: str):
        self.mode_combo.blockSignals(True)
        self.mode_combo.clear()
        self.mode_combo.addItems(modes)
        self.mode_combo.setCurrentText(current_mode)
        self.mode_combo.blockSignals(False)

    def toggle_dock(self, docked: bool):
        self.is_docked = docked
        if docked:
            self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool)
            screen_geometry = QApplication.primaryScreen().geometry()
            self.setGeometry(
                screen_geometry.width() - self.width() - 10,
                screen_geometry.height() - self.height() - 45,
                self.width(),
                self.height()
            )
            self.dock_button.setText("↩️ Undock")
        else:
            self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.WindowStaysOnTopHint)
            self.dock_button.setText("📌 Dock")
        self.show()

    def closeEvent(self, event):
        self.stop_engine_requested.emit()
        super().closeEvent(event)

class GestureDashboard(QMainWindow):
    """
    Modern gesture control dashboard with custom mode functionality.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gesture Control System - Neon Dashboard")
        self.setGeometry(100, 100, 1400, 800)
        self.setMinimumSize(1200, 700)
        self.worker = None
        self.worker_thread = None
        self.gesture_widgets = {}
        self.mode_tags_group = QButtonGroup()
        self.popout_window = None
        

        self.apply_stylesheet()
        self.setup_ui()

    def apply_stylesheet(self):
        """Complete black + neon theme."""
        neon_color = "#00ffff"  # Neon Cyan
        neon_hover = "#39ff14"  # Neon Green
        bg_color = "#000000"
        border_color = "#222222"
        text_color = "#ffffff"
        
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background-color: {bg_color};
                color: {text_color};
                font-family: 'Segoe UI', 'Roboto', Arial, sans-serif;
                font-size: 12px;
                border: none;
            }}
            
            /* Video Area */
            #VideoFrame {{
                background-color: {bg_color};
                border: 2px solid {neon_color};
                border-radius: 12px;
            }}
            #VideoLabel {{
                background-color: rgba(0, 0, 0, 0.8);
                color: #888888;
                border-radius: 8px;
                font-size: 20px;
                font-weight: 600;
            }}
            
            /* Group Boxes */
            QGroupBox {{
                font-weight: 700;
                font-size: 13px;
                border: 2px solid {border_color};
                border-radius: 8px;
                margin-top: 12px;
                padding: 20px 15px 15px 15px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 6px 14px;
                margin-left: 8px;
                color: {neon_color};
                background-color: {bg_color};
                border: 1px solid {neon_color};
                border-radius: 6px;
                font-weight: 700;
            }}
            
            /* Mode Tags */
            QPushButton#ModeTag {{
                background-color: transparent;
                color: {text_color};
                border: 2px solid {border_color};
                border-radius: 15px;
                padding: 8px 18px;
                font-weight: 600;
                text-align: center;
            }}
            QPushButton#ModeTag:hover {{
                border-color: {neon_hover};
                color: {neon_hover};
            }}
            QPushButton#ModeTag:checked {{
                background-color: {neon_color};
                color: {bg_color};
                border-color: {neon_color};
            }}

            /* Buttons */
            QPushButton {{
                background-color: {neon_color};
                color: {bg_color};
                border: none;
                border-radius: 6px;
                padding: 12px 20px;
                font-weight: 700;
                font-size: 12px;
            }}
            QPushButton:hover {{ 
                background-color: {neon_hover};
            }}
            QPushButton:pressed {{ 
                background-color: #333;
                color: {neon_hover};
            }}
            QPushButton#StartBtn {{ background-color: #00b300; }}
            QPushButton#StartBtn:hover {{ background-color: #00d900; }}
            QPushButton#StopBtn {{ background-color: #cc0000; }}
            QPushButton#StopBtn:hover {{ background-color: #ff0000; }}
            QPushButton:disabled {{ background-color: #333; color: #777; }}
            
            /* Form Controls */
            QSlider::groove:horizontal {{
                height: 4px;
                background-color: #444;
                border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                background-color: {neon_color};
                width: 20px;
                margin: -8px 0;
                border-radius: 10px;
                border: 2px solid {bg_color};
            }}
            QSlider::handle:horizontal:hover {{
                background-color: {neon_hover};
            }}
            
            QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox, QTextEdit {{
                background-color: #111;
                border: 2px solid {border_color};
                border-radius: 6px;
                padding: 8px 12px;
                font-weight: 500;
            }}
            QComboBox:focus, QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QTextEdit:focus {{
                border-color: {neon_color};
            }}
            QComboBox::drop-down {{
                border: none;
            }}
            QComboBox::down-arrow {{
                image: url(./arrow.png); /* Needs an arrow icon */
            }}
            
            QCheckBox {{
                spacing: 10px;
            }}
            QCheckBox::indicator {{
                width: 18px; height: 18px;
                border: 2px solid {border_color};
                border-radius: 4px;
                background-color: #111;
            }}
            QCheckBox::indicator:checked {{ 
                background-color: {neon_color};
                border-color: {neon_color};
            }}
            
            QScrollArea {{ border: none; background-color: transparent; }}
            QStatusBar {{ 
                font-weight: 600;
                border-top: 1px solid {border_color};
            }}
            QDialog {{ background-color: {bg_color}; }}
        """)

    def setup_ui(self):
        """Setup the main UI."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(20)

        # Left Side: Video + Controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(15)
        left_panel.setFixedWidth(520)

        video_frame = self.create_video_area()
        engine_controls = self.create_engine_controls()
        
        left_layout.addWidget(video_frame)
        left_layout.addWidget(engine_controls)

        # Right Side: Gesture Modes
        right_panel = self.create_gesture_modes_panel()

        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)
        
        self.setup_status_bar()

    def create_video_area(self):
        frame = QFrame()
        frame.setObjectName("VideoFrame")
        layout = QVBoxLayout(frame)
        self.video_label = QLabel("🎥 Gesture Engine Offline")
        self.video_label.setObjectName("VideoLabel")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setFixedSize(500, 375)
        layout.addWidget(self.video_label)
        return frame

    def create_engine_controls(self):
        """Create engine control buttons and mode selector."""
        group = QGroupBox("Engine Controls")
        layout = QGridLayout(group)
        layout.setSpacing(10)

        self.start_btn = QPushButton(" Start Engine")
        self.start_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.start_btn.clicked.connect(self.start_engine)

        self.stop_btn = QPushButton(" Stop Engine")
        self.stop_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop))
        self.stop_btn.clicked.connect(self.stop_engine)
        self.stop_btn.setEnabled(False)

        self.pause_btn = QPushButton(" Pause Engine")
        self.pause_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause))
        self.pause_btn.setCheckable(True)
        self.pause_btn.clicked.connect(self.pause_resume_engine)
        self.pause_btn.setEnabled(False)

        self.settings_btn = QPushButton(" Settings")
        self.settings_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon))
        self.settings_btn.clicked.connect(self.open_settings_dialog)

        # --- ADD THE BENCHMARK BUTTON HERE ---
        self.benchmark_btn = QPushButton("🔬 Benchmark Studio")
        self.benchmark_btn.clicked.connect(self.open_benchmark_studio)
        # --- END OF ADDITION ---

# --- FIX: Mode Selection Combo ---
        mode_label = QLabel("Mode:")
        self.mode_combo = QComboBox()
        self.update_mode_combo()  # Use the method that loads from config
        self.mode_combo.currentTextChanged.connect(self.change_mode)
        # --- END OF FIX ---

            # Hide both the label and combo box
        mode_label.hide()
        self.mode_combo.hide()
        
        layout.addWidget(mode_label, 3, 0)
        layout.addWidget(self.mode_combo, 3, 1)

        layout.addWidget(self.start_btn, 0, 0)
        layout.addWidget(self.stop_btn, 0, 1)
        layout.addWidget(self.pause_btn, 1, 0)
        layout.addWidget(self.settings_btn, 1, 1)
        layout.addWidget(self.benchmark_btn, 2, 0, 1, 2)  # Add button to the grid

        return group

    def open_benchmark_studio(self):
        """Opens the benchmark dialog."""
        # It's good practice to pause the main engine to free up resources
        if hasattr(self, 'worker_thread') and self.worker_thread and self.worker_thread.isRunning() and not self.pause_btn.isChecked():
            self.pause_btn.click()  # Programmatically click the pause button

        dialog = BenchmarkDialog(self)
        dialog.exec()

    def open_settings_dialog(self):
        """Open the settings dialog."""
        dialog = SettingsDialog(self)
        dialog.exec()

    def update_mode_combo(self):
        current_text = self.mode_combo.currentText() if hasattr(self, 'mode_combo') else None
        self.mode_combo.clear()
        self.mode_combo.addItem("Disabled")
        for mode_key in dir(config_manager.app_modes):
            if mode_key.endswith('_mode') and not mode_key.startswith('_'):
                mode_obj = getattr(config_manager.app_modes, mode_key, None)
                if mode_obj and hasattr(mode_obj, 'name'):
                    self.mode_combo.addItem(mode_obj.name)
        if current_text:
            index = self.mode_combo.findText(current_text)
            if index >= 0:
                self.mode_combo.setCurrentIndex(index)

    def create_gesture_modes_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(15)

        nav_group = QGroupBox("🎯 Gesture Modes")
        nav_layout = QVBoxLayout(nav_group)
        
        # Tags container
        self.mode_tags_widget = QWidget()
        self.mode_tags_layout = FlowLayout()
        self.mode_tags_widget.setLayout(self.mode_tags_layout)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.mode_tags_widget)
        
        custom_mode_btn = QPushButton(" Create Custom Mode")
        custom_mode_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogNewFolder))
        custom_mode_btn.clicked.connect(self.create_custom_mode)
        
        nav_layout.addWidget(scroll_area)
        nav_layout.addWidget(custom_mode_btn)
        
        self.mode_content_stack = QStackedWidget()
        
        layout.addWidget(nav_group)
        layout.addWidget(self.mode_content_stack)

        self.refresh_mode_list()
        return panel

    def refresh_mode_list(self):
        # Clear layout but keep widgets
        while self.mode_tags_layout.count():
            item = self.mode_tags_layout.takeAt(0)
            if item and item.widget():
                widget = item.widget()
                if widget:
                    widget.setParent(None)
        
        # Clear stack
        while self.mode_content_stack.count():
            widget = self.mode_content_stack.widget(0)
            self.mode_content_stack.removeWidget(widget)
            if widget:
                widget.setParent(None)

        self.mode_tags_group = QButtonGroup(self)
        self.mode_tags_group.setExclusive(True)

        # Add all modes
        idx = 0
        for mode_key in dir(config_manager.app_modes):
            if mode_key.endswith('_mode') and not mode_key.startswith('_'):
                mode_obj = getattr(config_manager.app_modes, mode_key, None)
                if mode_obj and hasattr(mode_obj, 'name'):
                    icon_map = {
                        "custom": QStyle.StandardPixmap.SP_CustomBase,
                        "ppt_mode": QStyle.StandardPixmap.SP_FileDialogInfoView,
                        "media_mode": QStyle.StandardPixmap.SP_MediaVolume,
                        "browser_mode": QStyle.StandardPixmap.SP_ComputerIcon
                    }
                    icon_pixmap = QStyle.StandardPixmap.SP_ComputerIcon # Default
                    for key_part, pixmap in icon_map.items():
                        if key_part in mode_key.lower():
                            icon_pixmap = pixmap
                            break
                    
                    tag_button = QPushButton(f" {mode_obj.name}")
                    tag_button.setIcon(self.style().standardIcon(icon_pixmap))
                    tag_button.setObjectName("ModeTag")
                    tag_button.setCheckable(True)
                    tag_button.clicked.connect(lambda _, i=idx: self.mode_content_stack.setCurrentIndex(i))
                    
                    self.mode_tags_layout.addWidget(tag_button)
                    self.mode_tags_group.addButton(tag_button)
                    
                    page = self.create_gesture_mode_page(mode_key)
                    self.mode_content_stack.addWidget(page)
                    idx += 1
        
        if self.mode_tags_group.buttons():
            self.mode_tags_group.buttons()[0].setChecked(True)
            self.mode_content_stack.setCurrentIndex(0)

    def create_gesture_mode_page(self, mode_key):
        page_widget = QWidget()
        layout = QVBoxLayout(page_widget)
        layout.setSpacing(15)
        
        header_frame = QFrame()
        header_layout = QHBoxLayout(header_frame)
        
        mode_obj = getattr(config_manager.app_modes, mode_key)
        header_label = QLabel(f"Configure {mode_obj.name}")
        header_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #00ffff;")
        
        edit_btn = QPushButton(" Edit Mode")
        edit_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_FileLinkIcon))
        edit_btn.clicked.connect(lambda: self.edit_mode(mode_key))
        
        header_layout.addWidget(header_label)
        header_layout.addStretch()
        header_layout.addWidget(edit_btn)
        layout.addWidget(header_frame)
        
        gestures_group = QGroupBox("Active Gestures")
        gestures_layout = QVBoxLayout(gestures_group)
        
        # Create scroll area for gestures
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                background-color: #111;
                width: 12px;
                border-radius: 6px;
                margin: 2px;
            }
            QScrollBar::handle:vertical {
                background-color: #00ffff;
                border-radius: 5px;
                min-height: 20px;
                margin: 1px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #39ff14;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                border: none;
                background: none;
                height: 0px;
            }
        """)
        
        # Create container widget for scrollable content
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(16)
        scroll_layout.setContentsMargins(10, 10, 10, 10)
        
        gestures = mode_obj.gestures if hasattr(mode_obj, 'gestures') else {}
        
        if not gestures:
            scroll_layout.addWidget(QLabel("No gestures configured for this mode."))
        else:
            for gesture_key, gesture_data in gestures.items():
                card = self.create_gesture_display_card(gesture_key, gesture_data)
                scroll_layout.addWidget(card)
        
        # Add stretch to push cards to top
        scroll_layout.addStretch()
        
        # Set the scroll widget and add to main layout
        scroll_area.setWidget(scroll_widget)
        gestures_layout.addWidget(scroll_area)
        
        layout.addWidget(gestures_group)
        return page_widget

    def create_gesture_display_card(self, gesture_key, gesture_data):
        # Main card container, applying 8-point grid principles
        card = QFrame()
        card.setFrameStyle(QFrame.Shape.StyledPanel)
        card.setStyleSheet("""
            QFrame {
                background-color: #2d2d30;   
                border: 1px solid #3e3e42;   
                border-radius: 8px;         
                margin: 4px;                
                padding: 18px 8px;
                min-height: 60px;
                max-height: 120px; /* Prevent card from growing too tall */
            }
        """)
        # Remove setMinimumHeight, let max-height control it

        # Main layout with reduced spacing and margins
        layout = QHBoxLayout(card)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # --- Base Style for the inner text boxes ---
        element_style = """
            QLabel {
                background-color: #242424;
                border: 1px solid #3c3c3c;
                border-radius: 8px;
                font-size: 12px;
                font-weight: 500;
                padding: 6px 12px;
                min-height: 20px;
                max-width: 220px; /* Prevent text from overflowing horizontally */
            }
            QLabel:hover {
                border-color: #00ffff;
            }
        """

        # --- 1. Gesture Name Box ---
        name_text = get_gesture_display_name(gesture_key)
        name_label = QLabel(name_text)
        name_label.setStyleSheet(element_style + "QLabel { color: #00ffff; font-weight: bold; }")
        name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        name_label.setWordWrap(True)
        name_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        name_label.setMaximumWidth(220)

        # --- Handle Data ---
        if hasattr(gesture_data, 'description'):
            desc_text = gesture_data.description or "No description"
            key_val = gesture_data.key or gesture_data.button
        else:
            desc_text = gesture_data.get('description', "No description")
            key_val = gesture_data.get('key') or gesture_data.get('button')

        # --- 2. Description Box ---
        desc_label = QLabel(desc_text)
        desc_label.setStyleSheet(element_style + "QLabel { color: #dddddd; }")
        desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc_label.setWordWrap(True)
        desc_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        desc_label.setMaximumWidth(220)

        # --- 3. Key Binding Badge (more prominent) ---
        formatted_key = f"[{key_val.title()}]" if key_val else "[None]"
        key_label = QLabel(formatted_key)
        key_label.setStyleSheet(element_style + """
            QLabel { 
                color: #00ffff; 
                font-family: 'Consolas', monospace; 
                font-weight: bold;
                padding: 8px 16px;
            }
        """)
        key_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        key_label.setWordWrap(False)
        key_label.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)
        key_label.setMaximumWidth(120)

        # --- Add Widgets to Layout ---
        layout.addWidget(name_label, 3)
        layout.addWidget(desc_label, 5)
        layout.addStretch(1)
        layout.addWidget(key_label, 2)

        return card
    

    def create_custom_mode(self):
        dialog = CustomModeDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.refresh_mode_list()
            self.update_mode_combo()

    def edit_mode(self, mode_key):
        mode_obj = getattr(config_manager.app_modes, mode_key)
        dialog = CustomModeDialog(self, edit_mode=mode_key, mode_data=mode_obj)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.refresh_mode_list()

    def setup_status_bar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready. Start the engine to begin gesture control.")

    # Engine control methods
    def start_engine(self):
        if self.worker_thread is not None:
            return
        self.worker = GestureEngineWorker()
        self.worker_thread = QThread()
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.new_frame.connect(self.update_video_display)
        self.worker.status_update.connect(self.status_bar.showMessage)
        self.worker_thread.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.pause_btn.setEnabled(True)
        self.status_bar.showMessage("Engine starting...")

        # --- Popout window logic ---
        self.popout_window = PopoutWindow()
        self.popout_window.stop_engine_requested.connect(self.stop_engine)
        self.popout_window.pause_engine_requested.connect(self.pause_resume_engine)
        self.popout_window.mode_changed_requested.connect(self.change_mode)
        self.worker.new_frame.connect(self.popout_window.update_video)
        all_modes = [self.mode_combo.itemText(i) for i in range(self.mode_combo.count())]
        self.popout_window.populate_modes(all_modes, self.mode_combo.currentText())
        self.popout_window.show()
        self.showMinimized()

    
    def stop_engine(self):
        if self.worker:
            self.worker.stop()
        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait()
        self.worker, self.worker_thread = None, None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setText("⏸️ Pause")
        self.video_label.setText("🎥 Gesture Engine Offline")
        self.status_bar.showMessage("Engine stopped.")
        if self.popout_window:
            self.popout_window.blockSignals(True)
            self.popout_window.close()
            self.popout_window = None
        self.showNormal()
        self.activateWindow()

    def pause_resume_engine(self):
        if self.worker:
            from gesture_engine import complete_engine
            if complete_engine.paused:
                complete_engine.resume()
                self.pause_btn.setText(" Pause")
                self.pause_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause))
                self.status_bar.showMessage("Engine resumed.", 2000)
            else:
                complete_engine.pause()
                self.pause_btn.setText(" Resume")
                self.pause_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
                self.status_bar.showMessage("Engine paused.", 2000)

    def change_mode(self, mode_text):
        """Change application mode"""
        print(f"🎮 GUI changing mode to: {mode_text}")
        
        if mode_text == "Disabled":
            mode_key = 'disabled'
        else:
            # Find the mode key for the display name
            mode_key = None
            for attr_name in dir(config_manager.app_modes):
                if attr_name.endswith('_mode') and not attr_name.startswith('_'):
                    mode_obj = getattr(config_manager.app_modes, attr_name)
                    if hasattr(mode_obj, 'name') and mode_obj.name == mode_text:
                        mode_key = attr_name
                        break
        
        if mode_key:
            print(f"   Mapped to mode key: {mode_key}")
            # Make sure we're calling the engine's switch_mode method
            if hasattr(self, 'worker') and self.worker and hasattr(self.worker, 'engine'):
                success = self.worker.engine.switch_mode(mode_key)
                print(f"   Engine mode switch result: {success}")
            else:
                print("   ❌ No engine worker available")
        else:
            print(f"   ❌ Could not find mode key for: {mode_text}")

    def update_video_display(self, rgb_frame):
        if rgb_frame is not None:
            h, w, ch = rgb_frame.shape
            q_image = QImage(rgb_frame.data, w, h, ch * w, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def closeEvent(self, event):
        self.stop_engine()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = GestureDashboard()
    window.show()
    sys.exit(app.exec())