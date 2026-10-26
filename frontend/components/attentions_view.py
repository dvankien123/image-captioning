"""Attentions View component matching Image 3.

Visual elements:
- Top: Centered caption card ('A picture about a mountain')
- Bottom: Rounded container holding a 4-column grid of attention heatmap thumbnails.
- Each thumbnail has:
  - 'Word: ...' label on top
  - Heatmap image below with rounded corners, hover effect, and click-to-open modal dialog.
- Micro-interactions: Smooth hover zoom on thumbnails and interactive feedback.
"""

import base64
from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QPixmap, QPainter, QPainterPath
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from frontend.icons import create_empty_state_pixmap
from frontend.motion import SmoothThumbnailLabel


def make_rounded_pixmap(pixmap: QPixmap, radius: int = 6) -> QPixmap:
    """Return a new QPixmap with rounded corners."""
    if pixmap.isNull():
        return pixmap
    rounded = QPixmap(pixmap.size())
    rounded.fill(Qt.transparent)

    painter = QPainter(rounded)
    painter.setRenderHint(QPainter.Antialiasing)
    path = QPainterPath()
    path.addRoundedRect(0, 0, pixmap.width(), pixmap.height(), radius, radius)
    painter.setClipPath(path)
    painter.drawPixmap(0, 0, pixmap)
    painter.end()
    return rounded


class AttentionCard(QFrame):
    """Individual attention item with 'Word: ...' header and interactive heatmap thumbnail."""

    clicked = Signal(str, QPixmap, int)

    def __init__(self, word: str, pixmap: QPixmap, index: int, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.word = word
        self.pixmap = pixmap
        self.index = index
        self.setObjectName("attentionItemFrame")
        self.setCursor(Qt.PointingHandCursor)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)
        layout.setAlignment(Qt.AlignCenter)

        # Word label: "Word: ..."
        word_text = f"Word: {self.word}" if self.word and self.word != "..." else "Word: ..."
        self.word_lbl = QLabel(word_text)
        self.word_lbl.setObjectName("attentionWordLabel")
        self.word_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.word_lbl)

        # Heatmap thumbnail with smooth hover zoom & border micro-interaction
        self.img_lbl = SmoothThumbnailLabel(self.pixmap)
        layout.addWidget(self.img_lbl)

    def enterEvent(self, event):
        self.img_lbl.set_hovered(True)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.img_lbl.set_hovered(False)
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.word, self.pixmap, self.index)
        super().mousePressEvent(event)


class AttentionsView(QWidget):
    """View widget for the 'Attentions' tab."""

    heatmap_selected = Signal(str, QPixmap, int, list)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._current_steps: List[Dict[str, Any]] = []
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(16)

        self.stack = QStackedWidget()

        # Page 0: Empty state (matching Image 1) - strictly identical geometry to CaptionResultsView
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

        # Page 1: Attentions content (Image 3)
        self.content_page = QWidget()
        content_layout = QVBoxLayout(self.content_page)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(16)

        # Top Caption Box (Image 3)
        self.caption_card = QFrame()
        self.caption_card.setObjectName("captionCard")
        caption_layout = QHBoxLayout(self.caption_card)
        caption_layout.setContentsMargins(20, 20, 20, 20)
        caption_layout.setAlignment(Qt.AlignCenter)

        self.caption_text = QLabel("A picture about a mountain")
        self.caption_text.setObjectName("captionTextLabel")
        self.caption_text.setAlignment(Qt.AlignCenter)
        self.caption_text.setWordWrap(True)
        caption_layout.addWidget(self.caption_text)
        content_layout.addWidget(self.caption_card, 0)

        # Bottom Grid Box (Image 3)
        self.grid_card = QFrame()
        self.grid_card.setObjectName("attentionsGridBox")
        grid_outer_layout = QVBoxLayout(self.grid_card)
        grid_outer_layout.setContentsMargins(16, 16, 16, 16)

        # Scroll area for the grid
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.grid_layout = QGridLayout(self.scroll_content)
        self.grid_layout.setContentsMargins(8, 8, 8, 8)
        self.grid_layout.setSpacing(16)
        self.grid_layout.setAlignment(Qt.AlignTop | Qt.AlignHCenter)

        self.scroll_area.setWidget(self.scroll_content)
        grid_outer_layout.addWidget(self.scroll_area)
        content_layout.addWidget(self.grid_card, 1)

        self.stack.addWidget(self.content_page)
        layout.addWidget(self.stack)
        self.stack.setCurrentIndex(0)

    def set_content(self, caption: str, steps: List[Dict[str, Any]]):
        """Populate the grid with attention heatmap cards."""
        self.caption_text.setText(caption or "A picture about a mountain")
        self._current_steps = steps or []

        # Clear existing grid items
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        if not self._current_steps:
            self.stack.setCurrentIndex(0)
            return

        cols = 4  # 4 columns across as in Image 3
        for idx, step in enumerate(self._current_steps):
            word = step.get("word", "...")
            img_data = step.get("image", "")

            pixmap = QPixmap()
            if isinstance(img_data, str) and img_data.startswith("data:image"):
                try:
                    b64_part = img_data.split(",", 1)[1]
                    raw_bytes = base64.b64decode(b64_part)
                    pixmap.loadFromData(raw_bytes)
                except Exception:
                    pixmap = QPixmap()
            elif isinstance(img_data, QPixmap):
                pixmap = img_data

            card = AttentionCard(word, pixmap, idx)
            card.clicked.connect(self._on_card_clicked)

            row = idx // cols
            col = idx % cols
            self.grid_layout.addWidget(card, row, col)

        self.stack.setCurrentIndex(1)

    def _on_card_clicked(self, word: str, pixmap: QPixmap, index: int):
        self.heatmap_selected.emit(word, pixmap, index, self._current_steps)

    def clear(self):
        self._current_steps = []
        self.caption_text.setText("")
        self.stack.setCurrentIndex(0)
