"""Attentions View component matching Image 3.

Visual elements:
- Top Main Image View with overlay:
    - Displays selected word heatmap overlay (or clean original image)
    - Top-left floating frosted pill badge: 'Attention for word: "dog"'
    - Bottom-right floating segmented toggle pill: 'Original | Heatmap'
- Bottom Word Attention Thumbnails Carousel:
    - Horizontal row of cards for each token in caption ('a', 'white', 'dog', 'runs', 'through', 'the', 'grass', '.')
    - Each card has a small heatmap thumbnail + dark word pill underneath
    - Selected token ('dog') has an orange border and orange word pill background
    - Clicking any thumbnail changes the selected word!
"""

import base64
from typing import Any, Dict, List, Optional

from PySide6.QtCore import QPoint, QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QCursor,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from frontend.icons import create_empty_state_pixmap


def make_rounded_pixmap(pixmap: QPixmap, radius: int = 8, border_color: Optional[str] = None) -> QPixmap:
    """Return a rounded thumbnail, optionally with an active border."""
    if pixmap.isNull():
        return pixmap
    size = pixmap.size()
    rounded = QPixmap(size)
    rounded.fill(Qt.transparent)

    painter = QPainter(rounded)
    painter.setRenderHint(QPainter.Antialiasing)

    path = QPainterPath()
    path.addRoundedRect(0, 0, size.width(), size.height(), radius, radius)
    painter.setClipPath(path)
    painter.drawPixmap(0, 0, pixmap)

    if border_color:
        painter.setClipping(False)
        pen = QPen(QColor(border_color), 2.0)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(1, 1, size.width() - 2, size.height() - 2, radius, radius)

    painter.end()
    return rounded


class OverlayImageWidget(QWidget):
    """Main image display with top-left badge and bottom-right toggle overlays."""

    toggle_mode_changed = Signal(str)  # 'original' or 'heatmap'

    def __init__(self, radius: int = 12, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.radius = radius
        self._orig_pixmap: Optional[QPixmap] = None
        self._heatmap_pixmap: Optional[QPixmap] = None
        self._mode: str = "heatmap"  # 'original' or 'heatmap'
        self._word_text: str = "dog"

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(240, 200)

        # Internal overlay widgets
        self._init_overlays()

    def _init_overlays(self):
        # 1. Top-Left Badge: Attention for word: "dog"
        self.badge = QLabel(self)
        self.badge.setStyleSheet(
            "background-color: rgba(255, 255, 255, 0.94);"
            "border: 1px solid rgba(226, 232, 240, 0.9);"
            "border-radius: 14px;"
            "padding: 4px 12px;"
            "font-size: 11.5px;"
        )
        self._update_badge_text()

        # 2. Bottom-Right Toggle: Original | Heatmap
        self.toggle_container = QFrame(self)
        self.toggle_container.setStyleSheet(
            "background-color: rgba(255, 255, 255, 0.94);"
            "border: 1px solid rgba(226, 232, 240, 0.9);"
            "border-radius: 15px;"
            "padding: 2px;"
        )
        toggle_layout = QHBoxLayout(self.toggle_container)
        toggle_layout.setContentsMargins(2, 2, 2, 2)
        toggle_layout.setSpacing(2)

        self.btn_orig = QPushButton("Original")
        self.btn_orig.setCursor(Qt.PointingHandCursor)
        self.btn_orig.clicked.connect(lambda: self.set_mode("original"))

        self.btn_heat = QPushButton("Heatmap")
        self.btn_heat.setCursor(Qt.PointingHandCursor)
        self.btn_heat.clicked.connect(lambda: self.set_mode("heatmap"))

        toggle_layout.addWidget(self.btn_orig)
        toggle_layout.addWidget(self.btn_heat)

        self._update_toggle_styles()

    def _update_badge_text(self):
        self.badge.setText(
            f'Attention for word: <b style="color: #EA580C;">"{self._word_text}"</b>'
        )
        self.badge.adjustSize()

    def _update_toggle_styles(self):
        if self._mode == "heatmap":
            self.btn_heat.setStyleSheet(
                "background-color: #2563EB; color: #FFFFFF; font-weight: 600; "
                "font-size: 11px; border: none; border-radius: 12px; padding: 4px 10px;"
            )
            self.btn_orig.setStyleSheet(
                "background-color: transparent; color: #64748B; font-weight: 500; "
                "font-size: 11px; border: none; padding: 4px 8px;"
            )
        else:
            self.btn_orig.setStyleSheet(
                "background-color: #2563EB; color: #FFFFFF; font-weight: 600; "
                "font-size: 11px; border: none; border-radius: 12px; padding: 4px 10px;"
            )
            self.btn_heat.setStyleSheet(
                "background-color: transparent; color: #64748B; font-weight: 500; "
                "font-size: 11px; border: none; padding: 4px 8px;"
            )

    def set_images(self, orig_pixmap: Optional[QPixmap], heatmap_pixmap: Optional[QPixmap], word: str):
        self._orig_pixmap = orig_pixmap
        self._heatmap_pixmap = heatmap_pixmap
        self._word_text = word
        self._update_badge_text()
        self.update()

    def set_mode(self, mode: str):
        self._mode = mode
        self._update_toggle_styles()
        self.toggle_mode_changed.emit(mode)
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Position floating badge at top-left with padding
        self.badge.move(18, 18)

        # Position toggle container at bottom-right with padding
        self.toggle_container.adjustSize()
        tw = self.toggle_container.width()
        th = self.toggle_container.height()
        self.toggle_container.move(max(10, self.width() - tw - 18), max(10, self.height() - th - 18))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Determine which pixmap to draw
        active_pixmap = self._heatmap_pixmap if self._mode == "heatmap" else self._orig_pixmap
        if active_pixmap is None or active_pixmap.isNull():
            active_pixmap = self._orig_pixmap

        if active_pixmap and not active_pixmap.isNull():
            w = self.width()
            h = self.height()
            scaled = active_pixmap.scaled(
                QSize(w, h),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            sw = scaled.width()
            sh = scaled.height()
            x = (w - sw) // 2
            y = (h - sh) // 2

            # Clip rounded rectangle
            path = QPainterPath()
            path.addRoundedRect(x, y, sw, sh, self.radius, self.radius)
            painter.setClipPath(path)
            painter.drawPixmap(x, y, scaled)

            # Move overlays relative to the scaled image
            self.badge.move(x + 14, y + 14)
            tw = self.toggle_container.width()
            th = self.toggle_container.height()
            self.toggle_container.move(x + sw - tw - 14, y + sh - th - 14)
        else:
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor("#F8FAFC")))
            painter.drawRoundedRect(QRectF(0, 0, self.width(), self.height()), self.radius, self.radius)

        painter.end()


class WordThumbnailCard(QFrame):
    """Small card with heatmap thumbnail on top and word label pill at bottom."""

    clicked = Signal(int, str)

    def __init__(self, index: int, word: str, pixmap: QPixmap, selected: bool = False, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.index = index
        self.word = word
        self.pixmap = pixmap
        self.selected = selected
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(58, 70)
        self._press_pos: Optional[QPoint] = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Thumbnail
        self.thumb_lbl = QLabel()
        self.thumb_lbl.setFixedSize(54, 46)
        self.thumb_lbl.setAlignment(Qt.AlignCenter)
        if not self.pixmap.isNull():
            scaled = self.pixmap.scaled(54, 46, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            w = scaled.width()
            h = scaled.height()
            x = max(0, (w - 54) // 2)
            y = max(0, (h - 46) // 2)
            cropped = scaled.copy(x, y, 54, 46)
            self.thumb_lbl.setPixmap(make_rounded_pixmap(cropped, radius=6))
        layout.addWidget(self.thumb_lbl)

        # Word label bar
        self.word_lbl = QLabel(self.word)
        self.word_lbl.setFixedHeight(18)
        self.word_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.word_lbl)

        self.update_selection(self.selected)

    def update_selection(self, selected: bool):
        self.selected = selected
        if selected:
            self.setStyleSheet(
                "WordThumbnailCard {"
                "   border: 2px solid #EA580C;"
                "   border-radius: 8px;"
                "   background-color: #EA580C;"
                "}"
            )
            self.word_lbl.setStyleSheet(
                "background-color: #EA580C; color: #FFFFFF; font-weight: 700; "
                "font-size: 10.5px; border-bottom-left-radius: 6px; border-bottom-right-radius: 6px;"
            )
        else:
            self.setStyleSheet(
                "WordThumbnailCard {"
                "   border: 1px solid #E2E8F0;"
                "   border-radius: 8px;"
                "   background-color: #0F172A;"
                "}"
            )
            self.word_lbl.setStyleSheet(
                "background-color: #0F172A; color: #FFFFFF; font-weight: 600; "
                "font-size: 10.5px; border-bottom-left-radius: 6px; border-bottom-right-radius: 6px;"
            )

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._press_pos = event.globalPosition().toPoint()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._press_pos:
            dist = (event.globalPosition().toPoint() - self._press_pos).manhattanLength()
            if dist <= 4:
                self.clicked.emit(self.index, self.word)
        super().mouseReleaseEvent(event)


class DraggableAttentionScrollArea(QScrollArea):
    """Horizontal scroll area for attention thumbnails without visible scrollbar, supporting drag-to-scroll."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFixedHeight(76)
        self.setStyleSheet(
            "QScrollArea { background: transparent; border: none; }"
            "QScrollBar:horizontal { height: 0px; max-height: 0px; background: transparent; border: none; }"
            "QScrollBar::handle:horizontal { height: 0px; max-height: 0px; background: transparent; border: none; }"
            "QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0px; height: 0px; background: none; }"
        )

        self._dragging = False
        self._drag_start_x = 0
        self._scroll_start_x = 0
        self._moved = False

        self.viewport().installEventFilter(self)

    def eventFilter(self, obj, event):
        if obj == self.viewport():
            if event.type() == event.Type.MouseButtonPress and event.button() == Qt.LeftButton:
                self._dragging = True
                self._drag_start_x = event.globalPosition().toPoint().x()
                self._scroll_start_x = self.horizontalScrollBar().value()
                self._moved = False
            elif event.type() == event.Type.MouseMove and self._dragging:
                delta = event.globalPosition().toPoint().x() - self._drag_start_x
                if abs(delta) > 3:
                    self._moved = True
                    self.horizontalScrollBar().setValue(self._scroll_start_x - delta)
                    return True
            elif event.type() == event.Type.MouseButtonRelease and event.button() == Qt.LeftButton:
                was_dragging = self._dragging and self._moved
                self._dragging = False
                if was_dragging:
                    return True  # Consume release event so thumbnails do not trigger click on drag
        return super().eventFilter(obj, event)

    def wheelEvent(self, event):
        delta = event.angleDelta().y() or event.angleDelta().x()
        self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta // 2)
        event.accept()


class AttentionsView(QWidget):
    """View widget for the 'Attentions' tab matching Image 3."""

    word_selected = Signal(int, str)  # index, word

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._orig_pixmap: Optional[QPixmap] = None
        self._steps: List[Dict[str, Any]] = []
        self._current_index: int = 0
        self._init_ui()

    def _init_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self.stack = QStackedWidget()

        # Page 0: Empty state matching Caption Results
        self.empty_page = QWidget()
        empty_layout = QVBoxLayout(self.empty_page)
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

        self.stack.addWidget(self.empty_page)

        # Page 1: Loaded Attentions View
        self.loaded_page = QWidget()
        loaded_layout = QVBoxLayout(self.loaded_page)
        loaded_layout.setContentsMargins(16, 16, 16, 14)
        loaded_layout.setSpacing(12)

        # 1. Main Overlay Image Widget
        self.overlay_widget = OverlayImageWidget(radius=12)
        loaded_layout.addWidget(self.overlay_widget, 1)

        # 2. Bottom Word Thumbnail Carousel with drag-to-scroll and no visible scrollbar
        self.scroll_area = DraggableAttentionScrollArea()

        self.cards_container = QWidget()
        self.cards_layout = QHBoxLayout(self.cards_container)
        self.cards_layout.setContentsMargins(4, 2, 4, 2)
        self.cards_layout.setSpacing(8)
        self.cards_layout.setAlignment(Qt.AlignCenter)

        self.scroll_area.setWidget(self.cards_container)
        loaded_layout.addWidget(self.scroll_area, 0)

        self.stack.addWidget(self.loaded_page)
        root_layout.addWidget(self.stack)

        self._card_widgets: List[WordThumbnailCard] = []

        # Default to empty state
        self.show_empty()

    def show_empty(self):
        """Show empty state matching Caption Results."""
        self._orig_pixmap = None
        self._steps = []
        self.stack.setCurrentIndex(0)

    def set_content(
        self,
        orig_pixmap: QPixmap,
        steps: List[Dict[str, Any]],
        initial_word: str = "dog",
        initial_index: Optional[int] = None,
    ):
        """Populate the Attentions View with steps and set default active word."""
        self._orig_pixmap = orig_pixmap
        self._steps = steps

        # Clear existing cards
        while self.cards_layout.count():
            item = self.cards_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self._card_widgets.clear()

        if orig_pixmap is None or orig_pixmap.isNull() or not steps:
            self.show_empty()
            return

        self.stack.setCurrentIndex(1)

        # Find initial word index (prefer initial_index, then 'dog' if present, otherwise 0)
        target_idx = 0
        if initial_index is not None and 0 <= initial_index < len(steps):
            target_idx = initial_index
        elif initial_word:
            for i, step in enumerate(steps):
                w = step.get("word", "").strip().lower()
                if w == initial_word.lower():
                    target_idx = i
                    break

        self._current_index = target_idx

        # Create cards for each step
        for i, step in enumerate(steps):
            word = step.get("word", f"word{i}")
            # Decode heatmap image
            img_data = step.get("image", "")
            pix = self._decode_b64_pixmap(img_data)
            if pix.isNull() and orig_pixmap:
                pix = orig_pixmap

            is_sel = (i == target_idx)
            card = WordThumbnailCard(i, word, pix, selected=is_sel)
            card.clicked.connect(self._on_card_clicked)
            self._card_widgets.append(card)
            self.cards_layout.addWidget(card)

        # Update overlay with target index
        self._apply_step(target_idx)

    def _decode_b64_pixmap(self, img_data: str) -> QPixmap:
        pix = QPixmap()
        if not img_data:
            return pix
        try:
            if "," in img_data:
                img_data = img_data.split(",", 1)[1]
            raw = base64.b64decode(img_data)
            pix.loadFromData(raw)
        except Exception:
            pass
        return pix

    def _on_card_clicked(self, index: int, word: str):
        self.select_word_by_index(index, emit_signal=True)

    def select_word_by_index(self, index: int, emit_signal: bool = True):
        if not (0 <= index < len(self._steps)):
            return
        self._current_index = index
        for i, card in enumerate(self._card_widgets):
            card.update_selection(i == index)
        self._apply_step(index)
        if 0 <= index < len(self._card_widgets):
            self.scroll_area.ensureWidgetVisible(self._card_widgets[index], 20, 0)
        word = self._steps[index].get("word", "")
        if emit_signal:
            self.word_selected.emit(index, word)

    def select_word_by_name(self, word_name: str):
        for i, step in enumerate(self._steps):
            if step.get("word", "").strip().lower() == word_name.strip().lower():
                self.select_word_by_index(i)
                return

    def _apply_step(self, index: int):
        if not (0 <= index < len(self._steps)):
            return
        step = self._steps[index]
        word = step.get("word", "")
        img_data = step.get("image", "")
        heat_pix = self._decode_b64_pixmap(img_data)
        self.overlay_widget.set_images(self._orig_pixmap, heat_pix, word)
