"""Image Information Card component matching the design mockups precisely.

Visual elements:
- White modern card container
- Centered header title: 'Image Information'
- Light blue-gray inner container (#F1F5F9) with 4 bullet items:
  • Name:
  • Item type:
  • Dimension:
  • Size:
"""

import os
from pathlib import Path
from typing import Optional

from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QVBoxLayout,
    QWidget,
)


class InfoPanel(QFrame):
    """Card displaying image metadata (Name, Item type, Dimension, Size)."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("modernCard")
        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(18, 16, 18, 18)
        main_layout.setSpacing(12)

        # Centered Title "Image Information"
        title_lbl = QLabel("Image Information")
        title_lbl.setObjectName("infoPanelTitle")
        title_lbl.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_lbl)

        # Light blue-gray inner box
        self.inner_box = QFrame()
        self.inner_box.setObjectName("infoInnerBox")
        box_layout = QVBoxLayout(self.inner_box)
        box_layout.setContentsMargins(16, 14, 16, 14)
        box_layout.setSpacing(10)

        self.name_lbl = QLabel("• Name:")
        self.name_lbl.setObjectName("infoBulletItem")

        self.type_lbl = QLabel("• Item type")
        self.type_lbl.setObjectName("infoBulletItem")

        self.dim_lbl = QLabel("• Dimension")
        self.dim_lbl.setObjectName("infoBulletItem")

        self.size_lbl = QLabel("• Size:")
        self.size_lbl.setObjectName("infoBulletItem")

        box_layout.addWidget(self.name_lbl)
        box_layout.addWidget(self.type_lbl)
        box_layout.addWidget(self.dim_lbl)
        box_layout.addWidget(self.size_lbl)

        main_layout.addWidget(self.inner_box)

    def set_file_info(self, file_path: str):
        """Extract and display metadata from the given image file."""
        p = Path(file_path)
        if not p.is_file():
            self.clear_info()
            return

        # Name
        self.name_lbl.setText(f"• Name: {p.name}")

        # Item type
        ext = p.suffix.lower().lstrip(".")
        type_str = ext.upper() if ext else "Unknown"
        self.type_lbl.setText(f"• Item type: {type_str}")

        # Dimensions & Size
        try:
            with Image.open(p) as img:
                w, h = img.size
                self.dim_lbl.setText(f"• Dimension: {w} × {h}")
        except Exception:
            self.dim_lbl.setText("• Dimension: Unknown")

        try:
            size_bytes = os.path.getsize(p)
            if size_bytes >= 1024 * 1024:
                size_str = f"{size_bytes / (1024 * 1024):.2f} MB"
            elif size_bytes >= 1024:
                size_str = f"{size_bytes / 1024:.1f} KB"
            else:
                size_str = f"{size_bytes} B"
            self.size_lbl.setText(f"• Size: {size_str}")
        except Exception:
            self.size_lbl.setText("• Size: Unknown")

    def clear_info(self):
        """Reset labels to the empty placeholder state shown in Image 1."""
        self.name_lbl.setText("• Name:")
        self.type_lbl.setText("• Item type")
        self.dim_lbl.setText("• Dimension")
        self.size_lbl.setText("• Size:")
