"""
Notification Center Widget - Modern notification panel with filtering and management
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                               QPushButton, QScrollArea, QFrame, QComboBox,
                               QLineEdit, QCheckBox, QSizePolicy)
from PySide6.QtCore import Qt, Signal, QTimer, QDateTime
from PySide6.QtGui import QFont, QIcon

class NotificationItem(QFrame):
    """Individual notification item with modern styling"""
    
    dismissed = Signal(object)  # Emitted when notification is dismissed
    
    def __init__(self, message, level="info", timestamp=None, parent=None):
        super().__init__(parent)
        
        self.message = message
        self.level = level
        self.timestamp = timestamp or QDateTime.currentDateTime()
        
        self.setObjectName(f"notificationItem_{level}")
        self.setFrameStyle(QFrame.Box)
        self.setMaximumHeight(80)
        
        self._setup_ui()
        self._apply_style()
        
        # Auto-dismiss timer for info notifications
        if level == "info":
            QTimer.singleShot(10000, self._auto_dismiss)  # 10 seconds
    
    def _setup_ui(self):
        """Setup the notification item UI"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        
        # Level icon
        icon_label = QLabel()
        icon_label.setFixedSize(24, 24)
        icon_label.setAlignment(Qt.AlignCenter)
        
        icons = {
            'info': '💡',
            'warning': '⚠️',
            'error': '❌',
            'success': '✅'
        }
        icon_label.setText(icons.get(self.level, '📋'))
        icon_label.setStyleSheet("font-size: 16px;")
        layout.addWidget(icon_label)
        
        # Content area
        content_layout = QVBoxLayout()
        
        # Message
        message_label = QLabel(self.message)
        message_label.setWordWrap(True)
        message_label.setFont(QFont("Segoe UI", 9))
        content_layout.addWidget(message_label)
        
        # Timestamp
        time_label = QLabel(self.timestamp.toString("hh:mm:ss"))
        time_label.setFont(QFont("Segoe UI", 8))
        time_label.setStyleSheet("color: gray;")
        content_layout.addWidget(time_label)
        
        layout.addLayout(content_layout, 1)
        
        # Dismiss button
        dismiss_btn = QPushButton("×")
        dismiss_btn.setFixedSize(20, 20)
        dismiss_btn.setFont(QFont("Arial", 10, QFont.Bold))
        dismiss_btn.clicked.connect(self._dismiss)
        layout.addWidget(dismiss_btn)
    
    def _apply_style(self):
        """Apply level-specific styling"""
        styles = {
            'info': "border-left: 3px solid #3498db; background-color: rgba(52, 152, 219, 0.1);",
            'warning': "border-left: 3px solid #f39c12; background-color: rgba(243, 156, 18, 0.1);",
            'error': "border-left: 3px solid #e74c3c; background-color: rgba(231, 76, 60, 0.1);",
            'success': "border-left: 3px solid #27ae60; background-color: rgba(39, 174, 96, 0.1);"
        }
        
        self.setStyleSheet(styles.get(self.level, styles['info']))
    
    def _dismiss(self):
        """Dismiss this notification"""
        self.dismissed.emit(self)
    
    def _auto_dismiss(self):
        """Auto-dismiss for info notifications"""
        if self.level == "info":
            self._dismiss()

class NotificationCenter(QWidget):
    """
    Modern notification center with filtering and management capabilities
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setFixedWidth(350)
        self.notifications = []
        self.max_notifications = 50
        
        self._setup_ui()
        
        print("✅ Notification Center initialized")
    
    def _setup_ui(self):
        """Setup the notification center UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)
        
        # Header
        header = self._create_header()
        layout.addWidget(header)
        
        # Filter controls
        filters = self._create_filters()
        layout.addWidget(filters)
        
        # Notification list
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarNever)
        
        self.notifications_widget = QWidget()
        self.notifications_layout = QVBoxLayout(self.notifications_widget)
        self.notifications_layout.setAlignment(Qt.AlignTop)
        self.scroll_area.setWidget(self.notifications_widget)
        
        layout.addWidget(self.scroll_area)
        
        # Footer with actions
        footer = self._create_footer()
        layout.addWidget(footer)
    
    def _create_header(self):
        """Create the header with title and controls"""
        header = QFrame()
        header.setFixedHeight(40)
        layout = QHBoxLayout(header)
        
        # Title
        title = QLabel("Notifications")
        title.setFont(QFont("Segoe UI", 11, QFont.Bold))
        layout.addWidget(title)
        
        layout.addStretch()
        
        # Notification count
        self.count_label = QLabel("0")
        self.count_label.setFont(QFont("Segoe UI", 9))
        self.count_label.setStyleSheet("color: gray;")
        layout.addWidget(self.count_label)
        
        return header
    
    def _create_filters(self):
        """Create filter controls"""
        filters = QFrame()
        filters.setFixedHeight(35)
        layout = QHBoxLayout(filters)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Level filter
        self.level_filter = QComboBox()
        self.level_filter.addItems(["All", "Info", "Warning", "Error", "Success"])
        self.level_filter.currentTextChanged.connect(self._apply_filters)
        layout.addWidget(self.level_filter)
        
        # Search filter
        self.search_filter = QLineEdit()
        self.search_filter.setPlaceholderText("Search notifications...")
        self.search_filter.textChanged.connect(self._apply_filters)
        layout.addWidget(self.search_filter)
        
        return filters
    
    def _create_footer(self):
        """Create footer with action buttons"""
        footer = QFrame()
        footer.setFixedHeight(35)
        layout = QHBoxLayout(footer)
        
        # Clear all button
        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self.clear_all)
        layout.addWidget(clear_btn)
        
        layout.addStretch()
        
        # Export button
        export_btn = QPushButton("Export")
        export_btn.clicked.connect(self._export_notifications)
        layout.addWidget(export_btn)
        
        return footer
    
    def add_notification(self, message, level="info"):
        """Add a new notification"""
        # Remove oldest notification if at limit
        if len(self.notifications) >= self.max_notifications:
            oldest = self.notifications[0]
            self._remove_notification(oldest)
        
        # Create new notification
        notification = NotificationItem(message, level)
        notification.dismissed.connect(self._remove_notification)
        
        # Add to list and layout
        self.notifications.append(notification)
        self.notifications_layout.insertWidget(0, notification)  # Add to top
        
        # Update count
        self._update_count()
        
        # Scroll to top to show new notification
        self.scroll_area.verticalScrollBar().setValue(0)
        
        print(f"📝 Notification added: {level.upper()} - {message[:50]}...")
    
    def _remove_notification(self, notification):
        """Remove a notification"""
        if notification in self.notifications:
            self.notifications.remove(notification)
            notification.setParent(None)
            self._update_count()
    
    def _apply_filters(self):
        """Apply current filters to notifications"""
        level_filter = self.level_filter.currentText().lower()
        search_text = self.search_filter.text().lower()
        
        for notification in self.notifications:
            # Check level filter
            level_match = (level_filter == "all" or 
                         notification.level == level_filter)
            
            # Check search filter
            search_match = (not search_text or 
                          search_text in notification.message.lower())
            
            # Show/hide notification
            notification.setVisible(level_match and search_match)
    
    def _update_count(self):
        """Update the notification count"""
        visible_count = sum(1 for n in self.notifications if n.isVisible())
        total_count = len(self.notifications)
        self.count_label.setText(f"{visible_count}/{total_count}")
    
    def clear_all(self):
        """Clear all notifications"""
        for notification in self.notifications[:]:  # Copy list to avoid modification during iteration
            self._remove_notification(notification)
    
    def _export_notifications(self):
        """Export notifications to file"""
        # Implementation for exporting notifications
        print("📤 Exporting notifications...")
    
    def get_notification_summary(self):
        """Get summary of current notifications"""
        summary = {
            'total': len(self.notifications),
            'by_level': {}
        }
        
        for notification in self.notifications:
            level = notification.level
            summary['by_level'][level] = summary['by_level'].get(level, 0) + 1
        
        return summary
