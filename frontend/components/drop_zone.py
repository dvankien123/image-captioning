"""Upload Image Drop Zone component matching the design mockups precisely.

Visual elements:
- White modern card container with header (Header photo icon + 'Upload Image' + subtitle)
- Dashed dropzone frame with photo-plus illustration centered
- 'Drop your image here' & 'or click to browse' text
- Blue 'Choose File' button with upload icon
- 'Supports JPG,PNG,WEBG (max 10MB)' footer text
- Drag-and-drop & native file dialog support
"""

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QIcon
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from frontend.icons import (
    create_dropzone_illustration,
    create_header_photo_icon,
    create_upload_icon,
)


class DropZone(QFrame):
    """Card widget for uploading images via drag-and-drop or file picker."""

    file_selected = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("modernCard")
        self.setAcceptDrops(True)
        self._current_file: Optional[str] = None
        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(18, 18, 18, 18)
        main_layout.setSpacing(14)

        # 1. Header: Icon + "Upload Image" + Subtitle
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(10)

        header_icon = QLabel()
        header_icon.setPixmap(create_header_photo_icon(24))
        header_icon.setFixedSize(26, 26)
        header_layout.addWidget(header_icon, 0, Qt.AlignTop)

        title_col = QVBoxLayout()
        title_col.setContentsMargins(0, 0, 0, 0)
        title_col.setSpacing(2)

        title_lbl = QLabel("Upload Image")
        title_lbl.setObjectName("uploadTitle")
        subtitle_lbl = QLabel("Drag and drop an image or choose a file")
        subtitle_lbl.setObjectName("uploadSubtitle")

        title_col.addWidget(title_lbl)
        title_col.addWidget(subtitle_lbl)
        header_layout.addLayout(title_col, 1)

        main_layout.addLayout(header_layout)

        # 2. Dashed Drop Zone Frame
        self.drop_box = QFrame()
        self.drop_box.setObjectName("dropZoneBox")
        self.drop_box.setCursor(Qt.PointingHandCursor)
        self.drop_box.setAcceptDrops(True)

        box_layout = QVBoxLayout(self.drop_box)
        box_layout.setContentsMargins(16, 24, 16, 22)
        box_layout.setSpacing(8)
        box_layout.setAlignment(Qt.AlignCenter)

        # Center photo-plus illustration (properly centered without fixed-size alignment offset)
        self.illus_lbl = QLabel()
        self.illus_lbl.setPixmap(create_dropzone_illustration(64, 64))
        self.illus_lbl.setAlignment(Qt.AlignCenter)
        box_layout.addWidget(self.illus_lbl)

        box_layout.addSpacing(4)

        # "Drop your image here"
        drop_title = QLabel("Drop your image here")
        drop_title.setObjectName("dropTitle")
        drop_title.setAlignment(Qt.AlignCenter)
        box_layout.addWidget(drop_title)

        # "or click to browse"
        drop_sub = QLabel("or click to browse")
        drop_sub.setObjectName("dropSubtitle")
        drop_sub.setAlignment(Qt.AlignCenter)
        box_layout.addWidget(drop_sub)

        box_layout.addSpacing(8)

        # "Choose File" button
        self.choose_btn = QPushButton("  Choose File")
        self.choose_btn.setObjectName("chooseFileBtn")
        self.choose_btn.setCursor(Qt.PointingHandCursor)
        self.choose_btn.setIcon(QIcon(create_upload_icon(16)))
        self.choose_btn.clicked.connect(self._open_file_dialog)

        # Center the button horizontally
        btn_wrapper = QHBoxLayout()
        btn_wrapper.addStretch()
        btn_wrapper.addWidget(self.choose_btn)
        btn_wrapper.addStretch()
        box_layout.addLayout(btn_wrapper)

        box_layout.addSpacing(6)

        # Footer format text
        fmt_lbl = QLabel("Supports JPG,PNG,WEBG (max 10MB)")
        fmt_lbl.setObjectName("dropFormatNotice")
        fmt_lbl.setAlignment(Qt.AlignCenter)
        box_layout.addWidget(fmt_lbl)

        main_layout.addWidget(self.drop_box)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._open_file_dialog()
        super().mousePressEvent(event)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                ext = Path(url.toLocalFile()).suffix.lower()
                if ext in {".jpg", ".jpeg", ".png", ".webp", ".webg"}:
                    event.acceptProposedAction()
                    self.drop_box.setObjectName("dropZoneActive")
                    self.drop_box.style().unpolish(self.drop_box)
                    self.drop_box.style().polish(self.drop_box)
                    return
        event.ignore()

    def dragLeaveEvent(self, event):
        self.drop_box.setObjectName("dropZoneBox")
        self.drop_box.style().unpolish(self.drop_box)
        self.drop_box.style().polish(self.drop_box)
        super().dragLeaveEvent(event)

    def dropEvent(self, event: QDropEvent):
        self.drop_box.setObjectName("dropZoneBox")
        self.drop_box.style().unpolish(self.drop_box)
        self.drop_box.style().polish(self.drop_box)

        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if Path(file_path).is_file():
                self.set_selected_file(file_path)
                event.acceptProposedAction()
                return
        event.ignore()

    def _open_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select an Image",
            "",
            "Images (*.png *.jpg *.jpeg *.webp *.webg);;All Files (*.*)",
        )
        if file_path:
            self.set_selected_file(file_path)

    def set_selected_file(self, file_path: str):
        self._current_file = file_path
        self.file_selected.emit(file_path)

    def clear(self):
        self._current_file = None
