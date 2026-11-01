"""Caption Results View component matching Images 1 and 2.

Handles:
- Empty state (Image 1):
    - Cloud illustration, 'No image yet', and subtext
    - Bottom caption card with 'Your caption will appear here' and disabled action buttons
- Result state (Image 2):
    - Large rounded image preview maintaining aspect ratio
    - Bottom caption card with large bold text and orange trailing accent dot
    - Action buttons: 'Copy caption', 'Regenerate', 'Export', '➔ View attention'
"""

from typing import Optional

from PySide6.QtCore import QSize, Qt, Signal, QTimer
from PySide6.QtGui import (
    QIcon,
    QPainter,
    QPainterPath,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from frontend.icons import (
    create_arrow_right_icon,
    create_copy_icon,
    create_empty_state_pixmap,
    create_export_icon,
    create_regenerate_icon,
)


class RoundedImageLabel(QLabel):
    """Image preview label maintaining aspect ratio with rounded corners."""

    def __init__(self, radius: int = 12, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.radius = radius
        self._pixmap: Optional[QPixmap] = None
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(220, 180)

    def set_pixmap(self, pixmap: Optional[QPixmap]):
        self._pixmap = pixmap
        self._update_display()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_display()

    def _update_display(self):
        if self._pixmap and not self._pixmap.isNull():
            target_w = max(10, self.width())
            target_h = max(10, self.height())
            scaled = self._pixmap.scaled(
                QSize(target_w, target_h),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            # Clip with rounded corners
            rounded = QPixmap(scaled.size())
            rounded.fill(Qt.transparent)
            painter = QPainter(rounded)
            painter.setRenderHint(QPainter.Antialiasing)
            path = QPainterPath()
            path.addRoundedRect(0, 0, scaled.width(), scaled.height(), self.radius, self.radius)
            painter.setClipPath(path)
            painter.drawPixmap(0, 0, scaled)
            painter.end()

            super().setPixmap(rounded)
        else:
            super().setPixmap(QPixmap())


class CaptionResultsView(QWidget):
    """View widget for the 'Caption Results' tab matching Images 1 and 2."""

    copy_clicked = Signal(str)
    regenerate_clicked = Signal()
    export_clicked = Signal()
    view_attention_clicked = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._raw_pixmap: Optional[QPixmap] = None
        self._caption_text: str = ""
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Upper Area: Stack between Empty State and Image Preview
        self.upper_stack = QStackedWidget()

        # Page 0: Empty Illustration (Image 1)
        self.page_empty = QWidget()
        empty_layout = QVBoxLayout(self.page_empty)
        empty_layout.setContentsMargins(20, 30, 20, 30)
        empty_layout.setAlignment(Qt.AlignCenter)
        empty_layout.setSpacing(10)

        cloud_lbl = QLabel()
        cloud_lbl.setPixmap(create_empty_state_pixmap(240, 180))
        cloud_lbl.setAlignment(Qt.AlignCenter)
        empty_layout.addWidget(cloud_lbl)

        empty_layout.addSpacing(4)

        title_lbl = QLabel("No image yet")
        title_lbl.setObjectName("emptyTitle")
        title_lbl.setAlignment(Qt.AlignCenter)
        empty_layout.addWidget(title_lbl)

        sub_lbl = QLabel("Upload an image to get an AI caption\nand attention score")
        sub_lbl.setObjectName("emptySubtitle")
        sub_lbl.setAlignment(Qt.AlignCenter)
        empty_layout.addWidget(sub_lbl)

        self.upper_stack.addWidget(self.page_empty)

        # Page 1: Loaded Image Preview (Image 2)
        self.page_loaded = QWidget()
        loaded_layout = QVBoxLayout(self.page_loaded)
        loaded_layout.setContentsMargins(16, 16, 16, 12)
        loaded_layout.setSpacing(0)

        self.image_preview = RoundedImageLabel(radius=12)
        loaded_layout.addWidget(self.image_preview)

        self.upper_stack.addWidget(self.page_loaded)

        layout.addWidget(self.upper_stack, 1)

        # Divider between image area and bottom caption area
        div = QFrame()
        div.setObjectName("thinDivider")
        layout.addWidget(div)

        # Bottom Caption Area
        self.bottom_panel = QFrame()
        self.bottom_panel.setObjectName("bottomCaptionPanel")
        bottom_layout = QVBoxLayout(self.bottom_panel)
        bottom_layout.setContentsMargins(20, 14, 20, 16)
        bottom_layout.setSpacing(8)

        # "CAPTION"
        cap_title = QLabel("CAPTION")
        cap_title.setObjectName("captionTitleLbl")
        bottom_layout.addWidget(cap_title)

        # Stack for caption text (Empty placeholder vs Loaded text)
        self.caption_stack = QStackedWidget()

        # Empty placeholder
        lbl_placeholder = QLabel("Your caption will appear here")
        lbl_placeholder.setObjectName("captionPlaceholderLbl")
        self.caption_stack.addWidget(lbl_placeholder)

        # Loaded caption text with orange accent dot
        self.lbl_caption = QLabel()
        self.lbl_caption.setObjectName("captionTextLbl")
        self.lbl_caption.setWordWrap(True)
        self.caption_stack.addWidget(self.lbl_caption)

        bottom_layout.addWidget(self.caption_stack)

        bottom_layout.addSpacing(4)

        # Buttons Row
        self.btn_row = QHBoxLayout()
        self.btn_row.setContentsMargins(0, 0, 0, 0)
        self.btn_row.setSpacing(10)

        # Copy caption button
        self.btn_copy = QPushButton(" Copy caption")
        self.btn_copy.setObjectName("primaryBtn")
        self.btn_copy.setCursor(Qt.PointingHandCursor)
        self.btn_copy.setIcon(QIcon(create_copy_icon(14, "#FFFFFF")))
        self.btn_copy.clicked.connect(self._on_copy)
        self.btn_row.addWidget(self.btn_copy)

        # Regenerate button
        self.btn_regen = QPushButton(" Regenerate")
        self.btn_regen.setObjectName("outlineBtn")
        self.btn_regen.setCursor(Qt.PointingHandCursor)
        self.btn_regen.setIcon(QIcon(create_regenerate_icon(14, "#334155")))
        self.btn_regen.clicked.connect(self.regenerate_clicked.emit)
        self.btn_row.addWidget(self.btn_regen)

        # Export button
        self.btn_export = QPushButton(" Export")
        self.btn_export.setObjectName("outlineBtn")
        self.btn_export.setCursor(Qt.PointingHandCursor)
        self.btn_export.setIcon(QIcon(create_export_icon(14, "#334155")))
        self.btn_export.clicked.connect(self.export_clicked.emit)
        self.btn_row.addWidget(self.btn_export)

        # View attention button (Only in Image 2)
        self.btn_view_att = QPushButton(" View attention")
        self.btn_view_att.setObjectName("outlineBtn")
        self.btn_view_att.setCursor(Qt.PointingHandCursor)
        self.btn_view_att.setIcon(QIcon(create_arrow_right_icon(14, "#334155")))
        self.btn_view_att.clicked.connect(self.view_attention_clicked.emit)
        self.btn_row.addWidget(self.btn_view_att)

        self.btn_row.addStretch()
        bottom_layout.addLayout(self.btn_row)

        layout.addWidget(self.bottom_panel, 0)

        # Default to empty state
        self.show_empty()

    def show_empty(self):
        """Show empty state matching Image 1."""
        self._raw_pixmap = None
        self._caption_text = ""
        self.upper_stack.setCurrentIndex(0)
        self.caption_stack.setCurrentIndex(0)

        # In empty state, buttons are disabled
        self.btn_copy.setEnabled(False)
        self.btn_regen.setEnabled(False)
        self.btn_export.setEnabled(False)
        self.btn_view_att.setVisible(False)

    def set_content(self, pixmap: QPixmap, caption: str):
        """Show loaded image and caption matching Image 2."""
        self._raw_pixmap = pixmap
        self._caption_text = caption.strip()
        self.image_preview.set_pixmap(pixmap)

        # Capitalize first letter and format with orange dot
        display_cap = self._caption_text
        if display_cap and not display_cap[0].isupper():
            display_cap = display_cap[0].upper() + display_cap[1:]

        # Strip trailing dot if present so we can add the styled orange dot
        if display_cap.endswith("."):
            base_text = display_cap[:-1].strip()
        else:
            base_text = display_cap

        # Render with styled orange dot matching Image 2
        html_caption = f'{base_text}<span style="color:#EA580C; font-weight:900;">.</span>'
        self.lbl_caption.setText(html_caption)

        self.upper_stack.setCurrentIndex(1)
        self.caption_stack.setCurrentIndex(1)

        # Enable buttons
        self.btn_copy.setEnabled(True)
        self.btn_regen.setEnabled(True)
        self.btn_export.setEnabled(True)
        self.btn_view_att.setVisible(True)
        self.btn_view_att.setEnabled(True)

    def update_caption_text(self, new_caption: str):
        """Update just caption text (e.g. when alternative caption is selected)."""
        self._caption_text = new_caption.strip()
        display_cap = self._caption_text
        if display_cap and not display_cap[0].isupper():
            display_cap = display_cap[0].upper() + display_cap[1:]
        if display_cap.endswith("."):
            base_text = display_cap[:-1].strip()
        else:
            base_text = display_cap
        html_caption = f'{base_text}<span style="color:#EA580C; font-weight:900;">.</span>'
        self.lbl_caption.setText(html_caption)

    def _on_copy(self):
        if self._caption_text:
            QApplication.clipboard().setText(self._caption_text)
            self.copy_clicked.emit(self._caption_text)
            # Brief visual feedback on button text
            old_text = self.btn_copy.text()
            self.btn_copy.setText(" Copied!")
            QTimer.singleShot(1500, lambda: self.btn_copy.setText(old_text))
