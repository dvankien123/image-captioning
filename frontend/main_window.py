"""Main Application Window for Image Captioning matching Images 1, 2, and 3 strictly.

Layout Structure:
- Top Header Bar:
    - App Logo (gradient rounded square) + "Image Captioning" title
- 3-Column Layout:
    - Left Column (LeftPanel):
        - Upload Image (Empty dropzone or Loaded image summary + replace dropzone)
        - History row (Empty dashed slots or 4 thumbnails with active blue border)
    - Center Column:
        - Top 50/50 segmented tab bar ("Caption Results" vs "Attentions")
        - Stacked view:
            - Tab 0: CaptionResultsView (Empty state or Loaded image + caption + buttons)
            - Tab 1: AttentionsView (Heatmap overlay on image with Original/Heatmap toggle, + carousel of word thumbnails)
    - Right Column (RightPanel):
        - Tab 0: Alternative Captions with radio options and scores + Processing Info
        - Tab 1: Caption token pills + Why This Word probability distributions + action buttons
"""

from pathlib import Path
import time
from typing import Any, Dict, List, Optional

from PySide6.QtCore import QEvent, QThread, Qt, Signal
from PySide6.QtGui import (
    QDragEnterEvent,
    QDropEvent,
    QIcon,
    QKeySequence,
    QPixmap,
    QShortcut,
)
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from frontend.api_client import BackendClient
from frontend.components.attentions_view import AttentionsView
from frontend.components.caption_view import CaptionResultsView
from frontend.components.right_panel import RightPanel
from frontend.components.sidebar import LeftPanel
from frontend.icons import create_app_logo_icon
from frontend.motion import SlidingTabPill


class InferenceWorker(QThread):
    """Background thread executing backend/model inference without freezing UI."""

    finished = Signal(str, dict, float)  # file_path, result_dict, elapsed_seconds
    failed = Signal(str, str, float)     # file_path, error_message, elapsed_seconds

    def __init__(self, client: BackendClient, file_path: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.client = client
        self.file_path = file_path

    def run(self):
        t0 = time.perf_counter()
        try:
            result = self.client.upload_image(self.file_path, show_attention=True)
            elapsed = time.perf_counter() - t0
            self.finished.emit(self.file_path, result, elapsed)
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            self.failed.emit(self.file_path, str(exc), elapsed)


class ApiWorker(QThread):
    result_ready = Signal(str, object)
    failed = Signal(str, str)

    def __init__(self, task_name: str, callback, parent=None):
        super().__init__(parent)
        self.task_name = task_name
        self.callback = callback

    def run(self):
        try:
            self.result_ready.emit(self.task_name, self.callback())
        except Exception as exc:
            self.failed.emit(self.task_name, str(exc))


class MainWindow(QMainWindow):
    """Main window reproducing Images 1, 2, and 3 faithfully."""

    def __init__(self, backend_url: str = "http://127.0.0.1:8000"):
        super().__init__()
        self.backend_client = BackendClient(backend_url)
        self._active_worker: Optional[InferenceWorker] = None

        self._current_file_path: Optional[str] = None
        self._current_pixmap: Optional[QPixmap] = None
        self._current_caption: str = ""
        self._current_steps: List[Dict[str, Any]] = []
        self._current_result: Dict[str, Any] = {}
        self._current_record_id: Optional[int] = None
        self._requested_history_id: Optional[int] = None
        self._selected_candidate_idx = 0
        self._active_tab_index: int = 0
        # In-memory session cache only (only holds items present in history)
        self._results_cache: Dict[str, Dict[str, Any]] = {}
        self._api_workers = set()

        self.setWindowTitle("Att Captioner")
        self._icon_path = str(Path(__file__).resolve().parent.parent / "icon" / "Att-Captioner-Icon.png")
        self.setWindowIcon(QIcon(self._icon_path))
        self.resize(1024, 640)
        self.setMinimumSize(960, 600)
        self.setAcceptDrops(True)

        self._init_ui()
        self._setup_shortcuts()
        self._load_history()

    def _init_ui(self):
        central_widget = QWidget()
        central_widget.setObjectName("centralWidget")
        self.setCentralWidget(central_widget)

        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # ---------------- 1. Top Header Bar ----------------
        self.header_frame = QFrame()
        self.header_frame.setObjectName("appHeader")
        self.header_frame.setFixedHeight(50)
        header_layout = QHBoxLayout(self.header_frame)
        header_layout.setContentsMargins(20, 0, 20, 0)
        header_layout.setSpacing(12)

        # App Title Text
        self.title_lbl = QLabel("ATT Captioner")
        self.title_lbl.setObjectName("appTitle")
        header_layout.addWidget(self.title_lbl)

        header_layout.addStretch()
        root_layout.addWidget(self.header_frame)

        # ---------------- 2. Main 3-Column Body Layout ----------------
        self.content_frame = QFrame()
        self.content_frame.setObjectName("contentFrame")
        body_layout = QHBoxLayout(self.content_frame)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)

        # Column 1: Left Panel (Sidebar)
        self.left_panel = LeftPanel()
        self.left_panel.file_selected.connect(self._on_file_selected)
        self.left_panel.history_item_selected.connect(self._on_history_item_selected)
        body_layout.addWidget(self.left_panel, 0)

        # Column 2: Center Panel (Main Area with Tabs)
        self.center_panel = QFrame()
        self.center_panel.setObjectName("centerPanel")
        center_layout = QVBoxLayout(self.center_panel)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)

        # Top 50/50 Segmented Tab Bar across center panel
        self.tab_bar_frame = QFrame()
        self.tab_bar_frame.setObjectName("tabBarContainer")
        self.tab_bar_frame.setFixedHeight(44)
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
        center_layout.addWidget(self.tab_bar_frame, 0)

        # Single shared sliding pill indicator behind active tab (spring physics)
        self.tab_pill = SlidingTabPill(self.tab_bar_frame, corner_radius=0.0)
        self.btn_caption_tab.raise_()
        self.btn_attentions_tab.raise_()
        self.tab_bar_frame.installEventFilter(self)

        # Center Stack (CaptionResultsView vs AttentionsView)
        self.center_stack = QStackedWidget()

        self.caption_view = CaptionResultsView()
        self.caption_view.regenerate_clicked.connect(self._on_regenerate)
        self.caption_view.export_clicked.connect(self._on_export)
        self.caption_view.view_attention_clicked.connect(lambda: self._set_active_tab(1))

        self.attentions_view = AttentionsView()
        self.attentions_view.word_selected.connect(self._on_attention_word_selected)

        self.center_stack.addWidget(self.caption_view)     # Page 0
        self.center_stack.addWidget(self.attentions_view)  # Page 1
        center_layout.addWidget(self.center_stack, 1)

        body_layout.addWidget(self.center_panel, 1)

        # Column 3: Right Panel
        self.right_panel = RightPanel()
        self.right_panel.caption_selected.connect(self._on_alt_caption_selected)
        self.right_panel.token_selected.connect(self._on_right_token_selected)
        self.right_panel.export_clicked.connect(self._on_export)
        body_layout.addWidget(self.right_panel, 0)

        root_layout.addWidget(self.content_frame, 1)

        # Set default tab: 0 ("Caption Results")
        self._set_active_tab(0)

    def _setup_shortcuts(self):
        """Setup Ctrl+V shortcut to paste image from clipboard."""
        self.paste_shortcut = QShortcut(QKeySequence.Paste, self)
        self.paste_shortcut.activated.connect(self._on_paste_clipboard)

    def _run_api_task(self, name: str, callback):
        worker = ApiWorker(name, callback, self)
        worker.result_ready.connect(self._on_api_result)
        worker.failed.connect(self._on_api_failed)
        worker.finished.connect(lambda worker=worker: self._api_workers.discard(worker))
        self._api_workers.add(worker)
        worker.start()

    def _load_history(self):
        self._run_api_task("history_list", lambda: self.backend_client.get_history(limit=100))

    def _on_history_item_selected(self, record_id: int):
        self._requested_history_id = record_id
        self._current_file_path = None
        self._current_record_id = None
        self.right_panel.setEnabled(True)
        def load_item():
            record = self.backend_client.get_history_item(record_id)
            image = self.backend_client.get_history_image(record_id) if record.get("has_image") else None
            return record, image
        self._run_api_task(f"history_item:{record_id}", load_item)

    def _on_api_result(self, task_name: str, payload):
        if task_name == "history_list":
            self.left_panel.set_persistent_history(payload)
            return
        if task_name.startswith("history_item:"):
            record, image_data = payload
            if record.get("id") != self._requested_history_id:
                return
            pixmap = QPixmap()
            if image_data:
                pixmap.loadFromData(image_data)
            self._apply_record_result(record, pixmap)
            return
        if task_name.startswith("regenerate:"):
            self.right_panel.setEnabled(True)
            record, image = payload
            if record.get("id") != self._current_record_id:
                return
            pixmap = QPixmap()
            pixmap.loadFromData(image)
            self._apply_record_result(record, pixmap)
            self._load_history()

    def _on_api_failed(self, task_name: str, message: str):
        if task_name == "history_list":
            self.left_panel.set_persistent_history([])
        else:
            if task_name.startswith("history_item:") and int(task_name.split(":", 1)[1]) != self._requested_history_id:
                return
            if task_name.startswith("regenerate:") and int(task_name.split(":", 1)[1]) != self._current_record_id:
                return
            self.right_panel.setEnabled(True)
            QMessageBox.warning(self, "Image captioning failed", message)

    def _on_paste_clipboard(self):
        """Handle pasting image from clipboard (Ctrl+V)."""
        clipboard = QApplication.clipboard()
        mime_data = clipboard.mimeData()

        # Check for image file URLs
        if mime_data.hasUrls():
            for url in mime_data.urls():
                fpath = url.toLocalFile()
                if Path(fpath).is_file():
                    self._on_file_selected(fpath)
                    return

        # Check for raw image in clipboard
        if mime_data.hasImage():
            img = clipboard.image()
            if not img.isNull():
                temp_dir = Path(__file__).resolve().parent.parent / "temp_uploads"
                temp_dir.mkdir(parents=True, exist_ok=True)
                temp_file = temp_dir / "pasted_image.png"
                img.save(str(temp_file))
                self._on_file_selected(str(temp_file))

    def _set_active_tab(self, index: int):
        """Switch active tab between 0 ('Caption Results') and 1 ('Attentions') with spring sliding pill."""
        self._active_tab_index = index
        self.center_stack.setCurrentIndex(index)
        self.right_panel.set_active_tab(index)

        if index == 0:
            self.btn_caption_tab.setObjectName("tabBtnActive")
            self.btn_attentions_tab.setObjectName("tabBtnInactive")
            self.btn_caption_tab.setStyleSheet(
                "background: transparent; color: #2563EB; font-size: 13.5px; font-weight: 700; border: none; padding: 12px 16px;"
            )
            self.btn_attentions_tab.setStyleSheet(
                "background: transparent; color: #64748B; font-size: 13.5px; font-weight: 700; border: none; padding: 12px 16px;"
            )
        else:
            self.btn_caption_tab.setObjectName("tabBtnInactive")
            self.btn_attentions_tab.setObjectName("tabBtnActive")
            self.btn_caption_tab.setStyleSheet(
                "background: transparent; color: #64748B; font-size: 13.5px; font-weight: 700; border: none; padding: 12px 16px;"
            )
            self.btn_attentions_tab.setStyleSheet(
                "background: transparent; color: #2563EB; font-size: 13.5px; font-weight: 700; border: none; padding: 12px 16px;"
            )

        self.btn_caption_tab.style().unpolish(self.btn_caption_tab)
        self.btn_caption_tab.style().polish(self.btn_caption_tab)
        self.btn_attentions_tab.style().unpolish(self.btn_attentions_tab)
        self.btn_attentions_tab.style().polish(self.btn_attentions_tab)

        if hasattr(self, "tab_pill") and hasattr(self, "tab_bar_frame"):
            self.tab_pill.resize(self.tab_bar_frame.size())
            target_btn = self.btn_caption_tab if index == 0 else self.btn_attentions_tab
            if target_btn.width() > 0:
                self.tab_pill.slide_to(target_btn.x(), target_btn.width())

    def showEvent(self, event):
        super().showEvent(event)
        if hasattr(self, "tab_pill") and hasattr(self, "tab_bar_frame"):
            self.tab_pill.resize(self.tab_bar_frame.size())
            idx = getattr(self, "_active_tab_index", 0)
            target_btn = self.btn_caption_tab if idx == 0 else self.btn_attentions_tab
            if target_btn.width() > 0:
                self.tab_pill.snap_to(target_btn.x(), target_btn.width())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "tab_pill") and hasattr(self, "tab_bar_frame"):
            self.tab_pill.resize(self.tab_bar_frame.size())
            idx = getattr(self, "_active_tab_index", 0)
            target_btn = self.btn_caption_tab if idx == 0 else self.btn_attentions_tab
            if target_btn.width() > 0:
                self.tab_pill.snap_to(target_btn.x(), target_btn.width())

    def eventFilter(self, obj, event):
        if hasattr(self, "tab_bar_frame") and obj == self.tab_bar_frame and event.type() == QEvent.Type.Resize:
            if hasattr(self, "tab_pill"):
                self.tab_pill.resize(self.tab_bar_frame.size())
                idx = getattr(self, "_active_tab_index", 0)
                target_btn = self.btn_caption_tab if idx == 0 else self.btn_attentions_tab
                if target_btn.width() > 0:
                    self.tab_pill.snap_to(target_btn.x(), target_btn.width())
        return super().eventFilter(obj, event)

    def _render_result(self, result: Dict[str, Any], candidate_index: int = 0):
        candidates = result.get("candidates") or []
        candidate_index = min(candidate_index, max(0, len(candidates) - 1))
        candidate = candidates[candidate_index] if candidates else {}
        caption = candidate.get("caption") or result.get("caption", "")
        attention = result.get("attention") or {}
        tokens = candidate.get("tokens") or attention.get("steps", [])
        metadata = result.get("metadata") or {}
        self._current_result = result
        self._selected_candidate_idx = candidate_index
        self._current_caption = caption.strip()
        self._current_steps = tokens

        if self._current_pixmap and not self._current_pixmap.isNull():
            self.caption_view.set_content(self._current_pixmap, self._current_caption)
            if attention.get("method") == "vit_cross_attention" and tokens:
                self.attentions_view.set_content(self._current_pixmap, tokens, initial_index=0)
            else:
                self.attentions_view.show_empty()

        elapsed_ms = metadata.get("elapsed_ms")
        proc_info = {
            "words": str(len(tokens) if tokens else len(caption.split())),
            "time": f"{float(elapsed_ms) / 1000:.2f} s" if elapsed_ms is not None else "—",
            "device": metadata.get("device", "—"),
            "model": metadata.get("model", "—"),
            "beam_size": metadata.get("beam_size", "—"),
        }
        self.right_panel.set_loaded_content(
            self._current_caption,
            alternatives=candidates,
            proc_info=proc_info,
            steps=tokens,
            selected_candidate=candidate_index,
        )

    def _apply_record_result(self, record: Dict[str, Any], pixmap: QPixmap):
        self._current_record_id = record.get("id")
        self._current_file_path = None
        self._current_pixmap = pixmap
        self._render_result(record)
        self.left_panel.set_loaded_history_record(
            self._current_record_id,
            record.get("original_filename", "Image"),
            pixmap,
        )

    def _prune_cache_to_history(self):
        history_paths = {item["path"] for item in self.left_panel._history_items if "path" in item}
        self._results_cache = {path: value for path, value in self._results_cache.items() if path in history_paths}

    def _apply_cached_results(self, cached: Dict[str, Any]):
        self._current_result = cached.get("result", {})
        self._current_record_id = self._current_result.get("id")
        self._render_result(self._current_result, cached.get("selected_candidate", 0))

    def _on_file_selected(self, file_path: str):
        """User uploaded or chose an image file."""
        self._requested_history_id = None
        self.right_panel.setEnabled(True)
        self._current_file_path = file_path
        self._current_record_id = None
        self._current_result = {}
        self._selected_candidate_idx = 0
        self._current_pixmap = QPixmap(file_path)

        # Update left panel immediately
        self.left_panel.set_loaded_file(file_path)

        # Fast cache hit: instant restore without lag!
        if file_path in self._results_cache:
            self._apply_cached_results(self._results_cache[file_path])
            return

        # Show temporary placeholder while waiting for inference
        self.caption_view.set_content(self._current_pixmap, "Generating caption...")

        # Start non-blocking inference worker tied to this specific file_path
        worker = InferenceWorker(self.backend_client, file_path, self)
        worker.finished.connect(self._on_inference_finished)
        worker.failed.connect(self._on_inference_failed)
        self._active_worker = worker
        worker.start()

    def _on_inference_finished(self, *args, **kwargs):
        """Receive caption and attention heatmaps from backend/model, bound to its source file."""
        if len(args) == 3:
            file_path, result, elapsed = args
        elif len(args) == 2:
            if isinstance(args[0], str) and isinstance(args[1], dict):
                file_path = args[0]
                result = args[1]
                elapsed = kwargs.get("elapsed", 0.0)
            else:
                file_path = self._current_file_path
                result = args[0]
                elapsed = args[1]
        elif len(args) == 1 and isinstance(args[0], dict):
            file_path = self._current_file_path
            result = args[0]
            elapsed = kwargs.get("elapsed", 0.0)
        else:
            file_path = kwargs.get("file_path", self._current_file_path)
            result = kwargs.get("result", {})
            elapsed = kwargs.get("elapsed", 0.0)

        result.setdefault("metadata", {}).setdefault("elapsed_ms", round(elapsed * 1000, 1))
        if file_path:
            self._results_cache[file_path] = {
                "result": result,
            }
            self._prune_cache_to_history()
        self._load_history()

        # CRITICAL FIX: If the user replaced or navigated away to another image while
        # this one was generating, do NOT overwrite the currently active UI with this old result!
        if file_path != self._current_file_path:
            return

        self._current_record_id = result.get("id")
        self._current_result = result
        self._render_result(result)

    def _on_inference_failed(self, *args, **kwargs):
        """Report backend failures without replacing them with demo output."""
        if len(args) == 3:
            file_path, error_message, elapsed = args
        elif len(args) == 2:
            file_path = self._current_file_path
            error_message = args[0]
            elapsed = args[1]
        else:
            file_path = kwargs.get("file_path", self._current_file_path)
            error_message = kwargs.get("error_message", "Inference error")
            elapsed = kwargs.get("elapsed", 0.0)

        if file_path != self._current_file_path:
            return
        self.caption_view.set_content(self._current_pixmap, "Caption unavailable")
        self.right_panel.set_empty_state()
        self.attentions_view.show_empty()
        QMessageBox.warning(self, "Image captioning failed", error_message)

    def _on_alt_caption_selected(self, index: int, _caption: str):
        """Switch all token-level views to the selected model candidate."""
        if not self._current_result:
            return
        self._render_result(self._current_result, candidate_index=index)
        if self._current_file_path in self._results_cache:
            self._results_cache[self._current_file_path]["selected_candidate"] = index

    def _on_attention_word_selected(self, index: int, word: str):
        """User selected a word thumbnail from bottom carousel in AttentionsView."""
        self.right_panel.set_selected_token_by_index(index)

    def _on_right_token_selected(self, index: int, word: Optional[str] = None):
        """User clicked a caption token pill in RightPanel."""
        if isinstance(index, int):
            self.attentions_view.select_word_by_index(index, emit_signal=False)
        elif isinstance(index, str):
            self.attentions_view.select_word_by_name(index)

    def _on_regenerate(self):
        """Re-run captioning and update the existing saved record when possible."""
        if self._current_record_id is not None:
            record_id = self._current_record_id
            self.right_panel.setEnabled(False)
            self._run_api_task(
                f"regenerate:{record_id}",
                lambda: (
                    (record := self.backend_client.regenerate_history_item(record_id)),
                    self.backend_client.get_history_image(record_id),
                ),
            )
        elif self._current_file_path:
            self._results_cache.pop(self._current_file_path, None)
            self._on_file_selected(self._current_file_path)

    def _on_export(self):
        """Export caption to a text or JSON file."""
        if not self._current_caption:
            return
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Caption",
            "caption.txt",
            "Text File (*.txt);;JSON File (*.json)",
        )
        if save_path:
            try:
                p = Path(save_path)
                if p.suffix.lower() == ".json":
                    import json
                    candidates = self._current_result.get("candidates") or []
                    candidate = candidates[self._selected_candidate_idx] if self._selected_candidate_idx < len(candidates) else {}
                    data = {
                        "file": self._current_result.get("original_filename") or self._current_file_path,
                        "caption": self._current_caption,
                        "created_at": self._current_result.get("created_at"),
                        "beam_score": candidate.get("beam_score"),
                        "tokens": candidate.get("tokens", []),
                        "metadata": self._current_result.get("metadata", {}),
                    }
                    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
                else:
                    p.write_text(self._current_caption, encoding="utf-8")
            except Exception as exc:
                QMessageBox.warning(self, "Export failed", str(exc))

    def closeEvent(self, event):
        """Release in-memory cache and clean up session on window close."""
        self._results_cache.clear()
        temp_dir = Path(__file__).resolve().parent.parent / "temp_uploads"
        if temp_dir.is_dir():
            import shutil
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass
        super().closeEvent(event)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                ext = Path(url.toLocalFile()).suffix.lower()
                if ext in {".jpg", ".jpeg", ".png", ".webp", ".webg"}:
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event: QDropEvent):
        for url in event.mimeData().urls():
            fpath = url.toLocalFile()
            if Path(fpath).is_file():
                self._on_file_selected(fpath)
                event.acceptProposedAction()
                return
        event.ignore()
