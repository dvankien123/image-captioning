"""Caption Results View component matching Image 1 and Image 2.

Handles:
- Empty state (Image 1): Cloud illustration, 'No image yet', and subtext.
- Result state (Image 2): Top image preview card + Bottom centered caption card.
- Micro-interaction: Animated loading ellipsis while waiting for AI caption.
"""

from typing import Optional

from PySide6.QtCore import Qt, QSize, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from frontend.icons import create_empty_state_pixmap


class ScaledImageLabel(QLabel):
    """Image preview label maintaining aspect ratio with smooth scaling."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._pixmap: Optional[QPixmap] = None
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(200, 160)

    def set_pixmap(self, pixmap: Optional[QPixmap]):
        self._pixmap = pixmap
        self._update_display()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_display()

    def _update_display(self):
        if self._pixmap and not self._pixmap.isNull():
            target_w = max(10, self.width() - 20)
            target_h = max(10, self.height() - 20)
            scaled = self._pixmap.scaled(
                QSize(target_w, target_h),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            super().setPixmap(scaled)
        else:
            super().setPixmap(QPixmap())


class CaptionResultsView(QWidget):
    """View widget for the 'Caption Results' tab."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._loading_timer: Optional[QTimer] = None
        self._dot_count: int = 0
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(16)

        self.stack = QStackedWidget()

        # Page 0: Empty state (Image 1) - strictly identical geometry to AttentionsView
        self.empty_page = QWidget()
        empty_layout = QVBoxLayout(self.empty_page)
        empty_layout.setContentsMargins(20, 40, 20, 40)
        empty_layout.setAlignment(Qt.AlignCenter)
        empty_layout.setSpacing(12)

        cloud_lbl = QLabel()
        cloud_lbl.setPixmap(create_empty_state_pixmap(240, 180))
        cloud_lbl.setAlignment(Qt.AlignCenter)
        empty_layout.addWidget(cloud_lbl)

        empty_layout.addSpacing(6)

        title_lbl = QLabel("No image yet")
        title_lbl.setObjectName("emptyTitle")
        title_lbl.setAlignment(Qt.AlignCenter)
        empty_layout.addWidget(title_lbl)

        sub_lbl = QLabel("Upload an image to get an AI caption\nand attention score")
        sub_lbl.setObjectName("emptySubtitle")
        sub_lbl.setAlignment(Qt.AlignCenter)
        empty_layout.addWidget(sub_lbl)

        self.stack.addWidget(self.empty_page)

        # Page 1: Result state (Image 2)
        self.result_page = QWidget()
        result_layout = QVBoxLayout(self.result_page)
        result_layout.setContentsMargins(0, 0, 0, 0)
        result_layout.setSpacing(16)

        # Top Image Preview Card
        self.preview_card = QFrame()
        self.preview_card.setObjectName("imagePreviewCard")
        preview_layout = QVBoxLayout(self.preview_card)
        preview_layout.setContentsMargins(16, 16, 16, 16)
        self.image_label = ScaledImageLabel()
        preview_layout.addWidget(self.image_label)
        result_layout.addWidget(self.preview_card, 7)

        # Bottom Caption Card
        self.caption_card = QFrame()
        self.caption_card.setObjectName("captionCard")
        caption_layout = QHBoxLayout(self.caption_card)
        caption_layout.setContentsMargins(20, 24, 20, 24)
        caption_layout.setAlignment(Qt.AlignCenter)

        self.caption_text = QLabel("A picture about a mountain")
        self.caption_text.setObjectName("captionTextLabel")
        self.caption_text.setAlignment(Qt.AlignCenter)
        self.caption_text.setWordWrap(True)
        caption_layout.addWidget(self.caption_text)
        result_layout.addWidget(self.caption_card, 3)

        self.stack.addWidget(self.result_page)

        layout.addWidget(self.stack)
        self.stack.setCurrentIndex(0)

    def _start_loading_animation(self):
        self._stop_loading_animation()
        self._dot_count = 0
        self._loading_timer = QTimer(self)
        self._loading_timer.setInterval(360)

        def update_dots():
            self._dot_count = (self._dot_count % 3) + 1
            dots = "." * self._dot_count
            self.caption_text.setText(f"Generating caption{dots}")

        self._loading_timer.timeout.connect(update_dots)
        self._loading_timer.start()

    def _stop_loading_animation(self):
        if self._loading_timer and self._loading_timer.isActive():
            self._loading_timer.stop()
            self._loading_timer = None

    def show_empty(self):
        self._stop_loading_animation()
        self.stack.setCurrentIndex(0)

    def set_content(self, pixmap: QPixmap, caption: str):
        self.image_label.set_pixmap(pixmap)
        clean_caption = caption or "A picture about a mountain"

        if "Generating" in clean_caption:
            self.caption_text.setText("Generating caption...")
            self._start_loading_animation()
        else:
            self._stop_loading_animation()
            self.caption_text.setText(clean_caption)

        self.stack.setCurrentIndex(1)

    def clear(self):
        self._stop_loading_animation()
        self.image_label.set_pixmap(None)
        self.caption_text.setText("")
        self.stack.setCurrentIndex(0)
