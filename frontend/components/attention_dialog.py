"""Attention Heatmap Modal Dialog matching Image 4 strictly.

Visual elements:
- Soft silver/gray rounded dialog container (#D5DCE4, border-radius 18px)
- Clean 'X' close button at top-right
- Centered large view of the selected attention heatmap image with rounded corners
- Centered bottom label: 'Word: ...'
- Micro-interactions: Silky opacity entrance and exit animations.
"""

import base64
from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QKeyEvent, QPainter, QPainterPath, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from frontend.motion import ModalFadeController


class ScaledHeatmapLabel(QLabel):
    """Large heatmap display preserving aspect ratio with smooth scaling and rounded corners."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._pixmap: Optional[QPixmap] = None
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(320, 220)

    def set_heatmap_pixmap(self, pixmap: Optional[QPixmap]):
        self._pixmap = pixmap
        self._update_display()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_display()

    def _update_display(self):
        if self._pixmap and not self._pixmap.isNull():
            target_w = max(10, self.width() - 8)
            target_h = max(10, self.height() - 8)
            scaled = self._pixmap.scaled(
                QSize(target_w, target_h),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )

            # Round the image corners
            rounded = QPixmap(scaled.size())
            rounded.fill(Qt.transparent)
            painter = QPainter(rounded)
            painter.setRenderHint(QPainter.Antialiasing)
            path = QPainterPath()
            path.addRoundedRect(0, 0, scaled.width(), scaled.height(), 8, 8)
            painter.setClipPath(path)
            painter.drawPixmap(0, 0, scaled)
            painter.end()

            super().setPixmap(rounded)
        else:
            super().setPixmap(QPixmap())


class AttentionDialog(QDialog):
    """Modal popup dialog matching Image 4 with smooth entrance and exit animations."""

    def __init__(
        self,
        word: str,
        pixmap: QPixmap,
        current_index: int = 0,
        all_steps: Optional[List[Dict[str, Any]]] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setObjectName("attentionModalDialog")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)

        self.word = word
        self.pixmap = pixmap
        self.current_index = current_index
        self.all_steps = all_steps or []

        self._fade_controller = ModalFadeController(self, duration_in=180, duration_out=130)

        self.setWindowTitle("Attention Detail")
        self.setModal(True)
        self.resize(540, 400)
        self.setMinimumSize(440, 320)

        self._init_ui()
        self._load_current_step()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(18, 14, 18, 18)
        main_layout.setSpacing(10)

        # Top bar with 'X' button aligned right
        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(0, 0, 0, 0)
        top_bar.addStretch()

        self.close_btn = QPushButton("✕")
        self.close_btn.setObjectName("modalCloseXBtn")
        self.close_btn.setCursor(Qt.PointingHandCursor)
        self.close_btn.setFixedSize(30, 30)
        self.close_btn.clicked.connect(self.close_animated)
        top_bar.addWidget(self.close_btn)

        main_layout.addLayout(top_bar)

        # Center large heatmap
        self.heatmap_label = ScaledHeatmapLabel()
        main_layout.addWidget(self.heatmap_label, 1)

        # Bottom label: "Word: ..."
        word_text = f"Word: {self.word}" if self.word and self.word != "..." else "Word: ..."
        self.bottom_word_lbl = QLabel(word_text)
        self.bottom_word_lbl.setObjectName("modalWordLabel")
        self.bottom_word_lbl.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.bottom_word_lbl, 0)

    def showEvent(self, event):
        super().showEvent(event)
        self._fade_controller.animate_open()

    def close_animated(self):
        self._fade_controller.animate_close(on_closed=self.accept)

    def _load_current_step(self):
        if self.all_steps and 0 <= self.current_index < len(self.all_steps):
            step = self.all_steps[self.current_index]
            self.word = step.get("word", "...")
            img_data = step.get("image", "")

            loaded_pix = QPixmap()
            if isinstance(img_data, str) and img_data.startswith("data:image"):
                try:
                    b64_part = img_data.split(",", 1)[1]
                    loaded_pix.loadFromData(base64.b64decode(b64_part))
                except Exception:
                    loaded_pix = self.pixmap
            elif isinstance(img_data, QPixmap):
                loaded_pix = img_data
            else:
                loaded_pix = self.pixmap

            self.pixmap = loaded_pix

        self.heatmap_label.set_heatmap_pixmap(self.pixmap)
        word_text = f"Word: {self.word}" if self.word and self.word != "..." else "Word: ..."
        self.bottom_word_lbl.setText(word_text)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Escape:
            self.close_animated()
            return
        super().keyPressEvent(event)
