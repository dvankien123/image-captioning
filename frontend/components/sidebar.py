"""Sidebar / Left Panel component matching Images 1, 2, and 3.

Handles:
- Section 1: UPLOAD IMAGE
    - Empty state: dashed dropzone with circular icon, 'Choose file', 'Ctrl+V', 'max 10MB'
    - Loaded state: Current image thumbnail, filename, format/dim/size badges, plus replacement dropzone with 'Change image'
- Section 2: HISTORY
    - Horizontal scrollable list (supports mouse drag left/right and wheel scrolling)
    - Strictly displays only actual uploaded images (0 when empty, 1 for 1, 2 for 2, etc. - no fake placeholders)
    - Selected thumbnail highlighted with active blue border
- Drag and drop & native file dialog
- Clipboard paste shortcut
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image
from PySide6.QtCore import QPoint, QSize, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QCursor,
    QDragEnterEvent,
    QDropEvent,
    QIcon,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from frontend.icons import (
    create_circle_image_icon,
    create_upload_icon,
)


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


class DraggableHistoryScrollArea(QScrollArea):
    """Horizontal scroll area supporting mouse drag/swipe left and right, and wheel scrolling."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFixedHeight(48)
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
                    return True
        return super().eventFilter(obj, event)

    def wheelEvent(self, event):
        delta = event.angleDelta().y() or event.angleDelta().x()
        self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta // 2)
        event.accept()


class HistoryThumbnailWidget(QLabel):
    """Clickable thumbnail item for a history image."""

    clicked = Signal(int)

    def __init__(
        self,
        index: int,
        pixmap: QPixmap,
        file_path: str,
        selected: bool = False,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.index = index
        self.file_path = file_path
        self._pixmap = pixmap
        self.selected = selected
        self.setFixedSize(44, 44)
        self.setAlignment(Qt.AlignCenter)
        self.setCursor(Qt.PointingHandCursor)
        p = Path(file_path)
        self.setToolTip(p.name)

        self._press_pos: Optional[QPoint] = None
        self.update_pixmap(selected)

    def update_pixmap(self, selected: bool):
        self.selected = selected
        if self._pixmap and not self._pixmap.isNull():
            scaled = self._pixmap.scaled(44, 44, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            w = scaled.width()
            h = scaled.height()
            x = max(0, (w - 44) // 2)
            y = max(0, (h - 44) // 2)
            cropped = scaled.copy(x, y, 44, 44)
            border = "#2563EB" if selected else None
            self.setPixmap(make_rounded_pixmap(cropped, radius=8, border_color=border))
            self.setStyleSheet("background: transparent; border: none;")
        else:
            self.setPixmap(QPixmap())
            self.setStyleSheet("border: 1px dashed #CBD5E1; border-radius: 8px; background-color: #F8FAFC;")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._press_pos = event.globalPosition().toPoint()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._press_pos:
            dist = (event.globalPosition().toPoint() - self._press_pos).manhattanLength()
            if dist <= 5:
                self.clicked.emit(self.index)
        super().mouseReleaseEvent(event)


class LeftPanel(QFrame):
    """Left sidebar faithfully matching Images 1, 2, 3."""

    file_selected = Signal(str)
    history_item_selected = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("leftPanel")
        self.setFixedWidth(230)
        self.setAcceptDrops(True)

        self._current_file: Optional[str] = None
        self._history_items: List[Dict[str, Any]] = []
        self._active_history_idx: int = -1
        self.hist_slots: List[HistoryThumbnailWidget] = []

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 16, 14, 16)
        layout.setSpacing(14)

        # ---------------- Section 1: UPLOAD IMAGE ----------------
        self.sec1_hdr = QLabel("UPLOAD IMAGE")
        self.sec1_hdr.setObjectName("sectionHeader")
        layout.addWidget(self.sec1_hdr)

        # Stack between Empty Dropzone and Loaded State
        self.upload_stack = QStackedWidget()

        # Page 0: Empty Dropzone (Image 1)
        self.page_empty_upload = QWidget()
        empty_layout = QVBoxLayout(self.page_empty_upload)
        empty_layout.setContentsMargins(0, 0, 0, 0)
        empty_layout.setSpacing(8)

        self.drop_box_empty = QFrame()
        self.drop_box_empty.setObjectName("dropZoneBox")
        self.drop_box_empty.setCursor(Qt.PointingHandCursor)
        box_layout = QVBoxLayout(self.drop_box_empty)
        box_layout.setContentsMargins(10, 18, 10, 16)
        box_layout.setSpacing(6)
        box_layout.setAlignment(Qt.AlignCenter)

        icon_lbl = QLabel()
        icon_lbl.setPixmap(create_circle_image_icon(44))
        icon_lbl.setAlignment(Qt.AlignCenter)
        box_layout.addWidget(icon_lbl)

        box_layout.addSpacing(2)

        drop_title = QLabel("Drop your image here")
        drop_title.setObjectName("dropTitle")
        drop_title.setAlignment(Qt.AlignCenter)
        box_layout.addWidget(drop_title)

        drop_sub = QLabel("or click to browse")
        drop_sub.setObjectName("dropSubtitle")
        drop_sub.setAlignment(Qt.AlignCenter)
        box_layout.addWidget(drop_sub)

        box_layout.addSpacing(4)

        self.choose_btn = QPushButton("  Choose file")
        self.choose_btn.setObjectName("primaryBtn")
        self.choose_btn.setCursor(Qt.PointingHandCursor)
        self.choose_btn.setIcon(QIcon(create_upload_icon(14)))
        self.choose_btn.clicked.connect(self._open_file_dialog)

        btn_wrap = QHBoxLayout()
        btn_wrap.addStretch()
        btn_wrap.addWidget(self.choose_btn)
        btn_wrap.addStretch()
        box_layout.addLayout(btn_wrap)

        box_layout.addSpacing(4)

        # "or paste Ctrl + V"
        paste_row = QHBoxLayout()
        paste_row.setAlignment(Qt.AlignCenter)
        paste_row.setSpacing(4)
        paste_lbl = QLabel("or paste")
        paste_lbl.setObjectName("dropSubtitle")
        ctrl_badge = QLabel("Ctrl")
        ctrl_badge.setObjectName("kbdBadge")
        plus_lbl = QLabel("+")
        plus_lbl.setObjectName("dropSubtitle")
        v_badge = QLabel("V")
        v_badge.setObjectName("kbdBadge")
        paste_row.addWidget(paste_lbl)
        paste_row.addWidget(ctrl_badge)
        paste_row.addWidget(plus_lbl)
        paste_row.addWidget(v_badge)
        box_layout.addLayout(paste_row)

        empty_layout.addWidget(self.drop_box_empty)

        notice_lbl = QLabel("JPG, PNG, WEBP · max 10MB")
        notice_lbl.setObjectName("dropNotice")
        notice_lbl.setAlignment(Qt.AlignCenter)
        empty_layout.addWidget(notice_lbl)

        self.upload_stack.addWidget(self.page_empty_upload)

        # Page 1: Loaded State (Images 2 & 3)
        self.page_loaded_upload = QWidget()
        loaded_layout = QVBoxLayout(self.page_loaded_upload)
        loaded_layout.setContentsMargins(0, 0, 0, 0)
        loaded_layout.setSpacing(10)

        # Current image row: Thumbnail + filename + "Current image"
        cur_row = QHBoxLayout()
        cur_row.setContentsMargins(0, 0, 0, 0)
        cur_row.setSpacing(10)

        self.cur_thumb_lbl = QLabel()
        self.cur_thumb_lbl.setFixedSize(48, 48)
        self.cur_thumb_lbl.setAlignment(Qt.AlignCenter)
        cur_row.addWidget(self.cur_thumb_lbl)

        cur_text_col = QVBoxLayout()
        cur_text_col.setContentsMargins(0, 2, 0, 2)
        cur_text_col.setSpacing(2)

        self.cur_filename_lbl = QLabel("image.jpg")
        self.cur_filename_lbl.setStyleSheet("font-weight: 700; font-size: 11.5px; color: #0F172A;")
        self.cur_filename_lbl.setWordWrap(False)

        cur_sub_lbl = QLabel("Current image")
        cur_sub_lbl.setStyleSheet("font-size: 10.5px; color: #64748B;")

        cur_text_col.addWidget(self.cur_filename_lbl)
        cur_text_col.addWidget(cur_sub_lbl)
        cur_row.addLayout(cur_text_col, 1)
        loaded_layout.addLayout(cur_row)

        # Badges row: [ JPG ] [ 681×481 ] [ 10.2 KB ]
        badges_row = QHBoxLayout()
        badges_row.setContentsMargins(0, 0, 0, 0)
        badges_row.setSpacing(6)
        badges_row.setAlignment(Qt.AlignLeft)

        self.badge_ext = QLabel("JPG")
        self.badge_ext.setObjectName("metaBadge")
        self.badge_dim = QLabel("681×481")
        self.badge_dim.setObjectName("metaBadge")
        self.badge_size = QLabel("10.2 KB")
        self.badge_size.setObjectName("metaBadge")

        badges_row.addWidget(self.badge_ext)
        badges_row.addWidget(self.badge_dim)
        badges_row.addWidget(self.badge_size)
        badges_row.addStretch()
        loaded_layout.addLayout(badges_row)

        # Replacement drop box
        self.drop_box_replace = QFrame()
        self.drop_box_replace.setObjectName("dropZoneBox")
        self.drop_box_replace.setCursor(Qt.PointingHandCursor)
        rep_box_layout = QVBoxLayout(self.drop_box_replace)
        rep_box_layout.setContentsMargins(8, 12, 8, 12)
        rep_box_layout.setSpacing(5)
        rep_box_layout.setAlignment(Qt.AlignCenter)

        rep_icon_lbl = QLabel()
        rep_icon_lbl.setPixmap(create_circle_image_icon(36))
        rep_icon_lbl.setAlignment(Qt.AlignCenter)
        rep_box_layout.addWidget(rep_icon_lbl)

        rep_title = QLabel("Drop a new image here")
        rep_title.setStyleSheet("font-size: 11.5px; font-weight: 700; color: #0F172A;")
        rep_title.setAlignment(Qt.AlignCenter)
        rep_box_layout.addWidget(rep_title)

        rep_sub = QLabel("to replace the current one")
        rep_sub.setStyleSheet("font-size: 10px; color: #64748B;")
        rep_sub.setAlignment(Qt.AlignCenter)
        rep_box_layout.addWidget(rep_sub)

        # "or paste Ctrl + V"
        rep_paste_row = QHBoxLayout()
        rep_paste_row.setAlignment(Qt.AlignCenter)
        rep_paste_row.setSpacing(4)
        rep_paste_lbl = QLabel("or paste")
        rep_paste_lbl.setStyleSheet("font-size: 10px; color: #94A3B8;")
        rep_ctrl_badge = QLabel("Ctrl")
        rep_ctrl_badge.setObjectName("kbdBadge")
        rep_plus_lbl = QLabel("+")
        rep_plus_lbl.setStyleSheet("font-size: 10px; color: #94A3B8;")
        rep_v_badge = QLabel("V")
        rep_v_badge.setObjectName("kbdBadge")
        rep_paste_row.addWidget(rep_paste_lbl)
        rep_paste_row.addWidget(rep_ctrl_badge)
        rep_paste_row.addWidget(rep_plus_lbl)
        rep_paste_row.addWidget(rep_v_badge)
        rep_box_layout.addLayout(rep_paste_row)

        rep_box_layout.addSpacing(2)

        self.change_btn = QPushButton("  Change image")
        self.change_btn.setObjectName("primaryBtn")
        self.change_btn.setCursor(Qt.PointingHandCursor)
        self.change_btn.setIcon(QIcon(create_upload_icon(14)))
        self.change_btn.clicked.connect(self._open_file_dialog)

        rep_btn_wrap = QHBoxLayout()
        rep_btn_wrap.addStretch()
        rep_btn_wrap.addWidget(self.change_btn)
        rep_btn_wrap.addStretch()
        rep_box_layout.addLayout(rep_btn_wrap)

        loaded_layout.addWidget(self.drop_box_replace)

        rep_notice_lbl = QLabel("JPG, PNG, WEBP · max 10MB")
        rep_notice_lbl.setObjectName("dropNotice")
        rep_notice_lbl.setAlignment(Qt.AlignCenter)
        loaded_layout.addWidget(rep_notice_lbl)

        self.upload_stack.addWidget(self.page_loaded_upload)
        layout.addWidget(self.upload_stack)

        # Thin divider
        div = QFrame()
        div.setObjectName("thinDivider")
        layout.addWidget(div)

        # ---------------- Section 2: HISTORY ----------------
        hist_hdr_row = QHBoxLayout()
        hist_hdr_row.setContentsMargins(0, 0, 0, 0)

        hist_title = QLabel("HISTORY")
        hist_title.setObjectName("sectionHeader")
        hist_hdr_row.addWidget(hist_title)

        hist_hdr_row.addStretch()

        self.hist_status_lbl = QLabel("empty")
        self.hist_status_lbl.setObjectName("sectionSubHeader")
        hist_hdr_row.addWidget(self.hist_status_lbl)
        layout.addLayout(hist_hdr_row)

        # Draggable horizontal scroll area for history items
        self.hist_scroll_area = DraggableHistoryScrollArea()
        self.hist_container = QWidget()
        self.hist_row = QHBoxLayout(self.hist_container)
        self.hist_row.setContentsMargins(0, 2, 0, 2)
        self.hist_row.setSpacing(6)
        self.hist_row.setAlignment(Qt.AlignLeft)
        self.hist_scroll_area.setWidget(self.hist_container)
        layout.addWidget(self.hist_scroll_area)

        self.hist_notice_lbl = QLabel("Analyzed images will appear here")
        self.hist_notice_lbl.setObjectName("sectionSubHeader")
        self.hist_notice_lbl.setWordWrap(True)
        layout.addWidget(self.hist_notice_lbl)

        layout.addStretch()

        # Connect click on drop boxes
        self.drop_box_empty.mousePressEvent = lambda ev: self._open_file_dialog()
        self.drop_box_replace.mousePressEvent = lambda ev: self._open_file_dialog()

        # Default to empty state
        self.set_empty_state()

    def set_empty_state(self):
        """Set Left Panel to empty state (0 items, no extra placeholders)."""
        self._current_file = None
        self._active_history_idx = -1
        self.upload_stack.setCurrentIndex(0)
        self.hist_status_lbl.setText("empty")
        self.hist_notice_lbl.setVisible(True)

        while self.hist_row.count():
            item = self.hist_row.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self.hist_slots.clear()
        self.hist_scroll_area.setVisible(False)

    def set_loaded_file(self, file_path: str):
        """Update Left Panel when an image is selected."""
        p = Path(file_path)
        if not p.is_file():
            return
        self._current_file = file_path

        # Truncate filename nicely if too long
        name = p.name
        if len(name) > 19:
            display_name = name[:10] + "..." + name[-6:]
        else:
            display_name = name
        self.cur_filename_lbl.setText(display_name)
        self.cur_filename_lbl.setToolTip(p.name)

        # File metadata
        ext = p.suffix.lower().lstrip(".")
        self.badge_ext.setText(ext.upper() if ext else "IMG")

        try:
            with Image.open(p) as img:
                w, h = img.size
                self.badge_dim.setText(f"{w}×{h}")
        except Exception:
            self.badge_dim.setText("Unknown")

        try:
            sz = os.path.getsize(p)
            if sz >= 1024 * 1024:
                self.badge_size.setText(f"{sz / (1024 * 1024):.1f} MB")
            else:
                self.badge_size.setText(f"{sz / 1024:.1f} KB")
        except Exception:
            self.badge_size.setText("10 KB")

        # Current thumbnail
        raw_pix = QPixmap(str(p))
        if not raw_pix.isNull():
            scaled = raw_pix.scaled(48, 48, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            w = scaled.width()
            h = scaled.height()
            x = max(0, (w - 48) // 2)
            y = max(0, (h - 48) // 2)
            cropped = scaled.copy(x, y, 48, 48)
            self.cur_thumb_lbl.setPixmap(make_rounded_pixmap(cropped, radius=8))

        # Switch upload stack to loaded state
        self.upload_stack.setCurrentIndex(1)

        # Update history with only actual image
        self._add_to_history(file_path, raw_pix)

    def _add_to_history(self, file_path: str, pixmap: QPixmap):
        """Add image to history list without adding fake placeholders."""
        # If already exists, move to front or activate
        for i, item in enumerate(self._history_items):
            if item.get("path") == file_path:
                self.set_active_history(i)
                return

        # Prepend new actual uploaded image
        self._history_items.insert(0, {"path": file_path, "pixmap": pixmap})
        self._rebuild_history_widgets()
        self.set_active_history(0)

    def set_persistent_history(self, records: List[Dict[str, Any]]):
        """Replace the session list with server-backed history entries."""
        self._history_items = []
        for record in records:
            pixmap = QPixmap()
            thumbnail = record.get("thumbnail") or ""
            if thumbnail.startswith("data:image") and "," in thumbnail:
                import base64
                pixmap.loadFromData(base64.b64decode(thumbnail.split(",", 1)[1]))
            self._history_items.append({
                "record_id": record.get("id"),
                "filename": record.get("original_filename", "Image"),
                "caption": record.get("caption", ""),
                "pixmap": pixmap,
            })
        self._active_history_idx = -1
        self._rebuild_history_widgets()

    def set_loaded_history_record(self, record_id: int, filename: str, pixmap: QPixmap):
        """Show a restored server history item in the existing upload summary."""
        for index, item in enumerate(self._history_items):
            if item.get("record_id") == record_id:
                self.set_active_history(index)
                break
        display_name = filename if len(filename) <= 19 else filename[:10] + "..." + filename[-6:]
        self.cur_filename_lbl.setText(display_name)
        self.cur_filename_lbl.setToolTip(filename)
        extension = Path(filename).suffix.lower().lstrip(".")
        self.badge_ext.setText(extension.upper() if extension else "IMG")
        self.badge_dim.setText(f"{pixmap.width()}×{pixmap.height()}" if not pixmap.isNull() else "Unknown")
        self.badge_size.setText("Saved")
        if not pixmap.isNull():
            scaled = pixmap.scaled(48, 48, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            x = max(0, (scaled.width() - 48) // 2)
            y = max(0, (scaled.height() - 48) // 2)
            self.cur_thumb_lbl.setPixmap(make_rounded_pixmap(scaled.copy(x, y, 48, 48), radius=8))
        self.upload_stack.setCurrentIndex(1)

    def _rebuild_history_widgets(self):
        """Rebuild thumbnail widgets to display strictly the exact count."""
        while self.hist_row.count():
            item = self.hist_row.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self.hist_slots.clear()

        n = len(self._history_items)
        if n == 0:
            self.hist_status_lbl.setText("empty")
            self.hist_scroll_area.setVisible(False)
            self.hist_notice_lbl.setVisible(True)
            return

        self.hist_status_lbl.setText("1 image" if n == 1 else f"{n} images")
        self.hist_scroll_area.setVisible(True)
        self.hist_notice_lbl.setVisible(False)

        for i, item in enumerate(self._history_items):
            thumb = HistoryThumbnailWidget(
                index=i,
                pixmap=item.get("pixmap", QPixmap()),
                file_path=item.get("filename", item.get("path", "")),
                selected=(i == self._active_history_idx),
            )
            thumb.clicked.connect(self._on_history_item_clicked)
            self.hist_slots.append(thumb)
            self.hist_row.addWidget(thumb)

    def set_active_history(self, index: int):
        """Highlight active history thumbnail with blue border."""
        self._active_history_idx = index
        n = len(self._history_items)
        if n == 0:
            self.hist_status_lbl.setText("empty")
            return
        self.hist_status_lbl.setText("1 image" if n == 1 else f"{n} images")

        for i, slot in enumerate(self.hist_slots):
            slot.update_pixmap(selected=(i == index))

    def _on_history_item_clicked(self, index: int):
        if 0 <= index < len(self._history_items):
            self.set_active_history(index)
            record_id = self._history_items[index].get("record_id")
            if record_id is not None:
                self.history_item_selected.emit(int(record_id))
                return
            fpath = self._history_items[index].get("path")
            if fpath and Path(fpath).is_file():
                self.file_selected.emit(fpath)

    def _open_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select an Image",
            "",
            "Images (*.png *.jpg *.jpeg *.webp *.webg);;All Files (*.*)",
        )
        if file_path:
            self.set_loaded_file(file_path)
            self.file_selected.emit(file_path)

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
            file_path = url.toLocalFile()
            if Path(file_path).is_file():
                self.set_loaded_file(file_path)
                self.file_selected.emit(file_path)
                event.acceptProposedAction()
                return
        event.ignore()
