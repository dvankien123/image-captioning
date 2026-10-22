"""Main Application Window for Image Captioning matching the 4 design mockups strictly.

Layout structure:
- Top Title: "Image Captioning"
- Left Column:
  - Top Card: Upload Image (DropZone)
  - Bottom Card: Image Information (InfoPanel)
- Right Column:
  - Top segmented tab bar ("Caption Results" vs "Attentions")
  - Tab 1: CaptionResultsView (Empty state or Image preview + Caption)
  - Tab 2: AttentionsView (Caption + 4-column attention heatmap grid)
- Interaction:
  - Drag-and-drop or file picker selection
  - Immediate metadata population in InfoPanel
  - Background asynchronous API inference (or graceful sample fallback)
  - Modal AttentionDialog on heatmap click
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from frontend.api_client import BackendClient
from frontend.components.attention_dialog import AttentionDialog
from frontend.components.attentions_view import AttentionsView
from frontend.components.caption_view import CaptionResultsView
from frontend.components.drop_zone import DropZone
from frontend.components.info_panel import InfoPanel


class InferenceWorker(QThread):
    """Background thread executing backend inference without freezing the UI."""

    finished = Signal(dict)
    failed = Signal(str)

    def __init__(self, client: BackendClient, file_path: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.client = client
        self.file_path = file_path

    def run(self):
        try:
            result = self.client.upload_image(self.file_path, show_attention=True)
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    """Main window faithfully reproducing the 4 design mockups."""

    def __init__(self, backend_url: str = "http://127.0.0.1:8000"):
        super().__init__()
        self.backend_client = BackendClient(backend_url)
        self._active_worker: Optional[InferenceWorker] = None

        self._current_file_path: Optional[str] = None
        self._current_pixmap: Optional[QPixmap] = None
        self._current_caption: str = ""
        self._current_steps: List[Dict[str, Any]] = []

        self.setWindowTitle("Image Captioning")
        self.resize(1060, 720)
        self.setMinimumSize(920, 640)

        self._init_ui()

    def _init_ui(self):
        central_widget = QWidget()
        central_widget.setObjectName("centralWidget")
        self.setCentralWidget(central_widget)

        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(28, 22, 28, 24)
        root_layout.setSpacing(16)

        # 1. Main Title Header: "Image Captioning"
        self.title_lbl = QLabel("Image Captioning")
        self.title_lbl.setObjectName("appTitle")
        root_layout.addWidget(self.title_lbl)

        # 2. Main Content Two-Column Layout
        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(22)

        # ----------------- Left Column -----------------
        left_col = QVBoxLayout()
        left_col.setContentsMargins(0, 0, 0, 0)
        left_col.setSpacing(18)

        # Card 1: Upload Image (Drop Zone)
        self.drop_zone = DropZone()
        self.drop_zone.file_selected.connect(self._on_file_selected)
        left_col.addWidget(self.drop_zone)

        # Card 2: Image Information
        self.info_panel = InfoPanel()
        left_col.addWidget(self.info_panel)

        left_col.addStretch()

        left_container = QWidget()
        left_container.setLayout(left_col)
        left_container.setFixedWidth(310)
        content_layout.addWidget(left_container, 0)

        # ----------------- Right Column -----------------
        right_card = QFrame()
        right_card.setObjectName("modernCard")
        right_card_layout = QVBoxLayout(right_card)
        right_card_layout.setContentsMargins(0, 0, 0, 0)
        right_card_layout.setSpacing(0)

        # Segmented Tab Bar across top of right card
        self.tab_bar_frame = QFrame()
        self.tab_bar_frame.setObjectName("tabBarContainer")
        tab_layout = QHBoxLayout(self.tab_bar_frame)
        tab_layout.setContentsMargins(0, 0, 0, 0)
        tab_layout.setSpacing(0)

        self.btn_caption_tab = QPushButton("Caption Results")
        self.btn_caption_tab.setCursor(Qt.PointingHandCursor)
        self.btn_caption_tab.clicked.connect(lambda: self._set_active_tab(0))

        self.btn_attentions_tab = QPushButton("Attentions")
        self.btn_attentions_tab.setCursor(Qt.PointingHandCursor)
        self.btn_attentions_tab.clicked.connect(lambda: self._set_active_tab(1))

        tab_layout.addWidget(self.btn_caption_tab, 1)
        tab_layout.addWidget(self.btn_attentions_tab, 1)
        right_card_layout.addWidget(self.tab_bar_frame, 0)

        # Stacked Views inside Right Card
        self.right_stack = QStackedWidget()
        self.caption_view = CaptionResultsView()
        self.attentions_view = AttentionsView()
        self.attentions_view.heatmap_selected.connect(self._open_attention_modal)

        self.right_stack.addWidget(self.caption_view)       # Index 0
        self.right_stack.addWidget(self.attentions_view)    # Index 1

        right_card_layout.addWidget(self.right_stack, 1)
        content_layout.addWidget(right_card, 1)

        root_layout.addLayout(content_layout)

        # Default to Caption Results tab
        self._set_active_tab(0)

    def _set_active_tab(self, index: int):
        """Update tab button styles and visible page."""
        self.right_stack.setCurrentIndex(index)
        if index == 0:
            self.btn_caption_tab.setObjectName("tabBtnActive")
            self.btn_attentions_tab.setObjectName("tabBtnInactive")
            self.btn_caption_tab.setStyleSheet(
                "background-color: #FFFFFF; color: #2563EB; font-size: 15px; font-weight: 700; "
                "border: none; padding: 14px 28px; border-top-left-radius: 18px; border-top-right-radius: 0px;"
            )
            self.btn_attentions_tab.setStyleSheet(
                "background-color: #DCE5EE; color: #0F172A; font-size: 15px; font-weight: 700; "
                "border: none; padding: 14px 28px; border-top-left-radius: 0px; border-top-right-radius: 18px;"
            )
        else:
            self.btn_caption_tab.setObjectName("tabBtnInactive")
            self.btn_attentions_tab.setObjectName("tabBtnActive")
            self.btn_caption_tab.setStyleSheet(
                "background-color: #DCE5EE; color: #0F172A; font-size: 15px; font-weight: 700; "
                "border: none; padding: 14px 28px; border-top-left-radius: 18px; border-top-right-radius: 0px;"
            )
            self.btn_attentions_tab.setStyleSheet(
                "background-color: #FFFFFF; color: #2563EB; font-size: 15px; font-weight: 700; "
                "border: none; padding: 14px 28px; border-top-left-radius: 0px; border-top-right-radius: 18px;"
            )

        self.btn_caption_tab.style().unpolish(self.btn_caption_tab)
        self.btn_caption_tab.style().polish(self.btn_caption_tab)
        self.btn_attentions_tab.style().unpolish(self.btn_attentions_tab)
        self.btn_attentions_tab.style().polish(self.btn_attentions_tab)

    def _on_file_selected(self, file_path: str):
        """Handle new image selection: update metadata and launch inference."""
        self._current_file_path = file_path
        self._current_pixmap = QPixmap(file_path)

        # Update metadata card immediately
        self.info_panel.set_file_info(file_path)

        # Temporary preview while waiting for inference
        self.caption_view.set_content(self._current_pixmap, "Generating caption...")

        # Run inference in worker thread
        if self._active_worker and self._active_worker.isRunning():
            self._active_worker.quit()
            self._active_worker.wait()

        self._active_worker = InferenceWorker(self.backend_client, file_path, self)
        self._active_worker.finished.connect(self._on_inference_finished)
        self._active_worker.failed.connect(self._on_inference_failed)
        self._active_worker.start()

    def _on_inference_finished(self, result: dict):
        """Receive caption and attention heatmaps from backend."""
        caption = result.get("caption", "A picture about a mountain")
        attention = result.get("attention") or {}
        steps = attention.get("steps", [])

        self._current_caption = caption
        self._current_steps = steps

        # Update both tabs
        if self._current_pixmap:
            self.caption_view.set_content(self._current_pixmap, caption)
        self.attentions_view.set_content(caption, steps)

    def _on_inference_failed(self, error_message: str):
        """Fallback to demo data matching the design mockups."""
        fallback = self.backend_client.generate_mock_result(self._current_file_path or "")
        self._on_inference_finished(fallback)

    def _open_attention_modal(self, word: str, pixmap: QPixmap, index: int, all_steps: list):
        """Open the Attention modal dialog matching Image 4."""
        dialog = AttentionDialog(
            word=word,
            pixmap=pixmap,
            current_index=index,
            all_steps=all_steps,
            parent=self,
        )
        dialog.exec()
