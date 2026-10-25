"""Application launcher for Image Captioning PySide6 Frontend.

Usage:
    python -m frontend.main
    python frontend/main.py [--backend-url http://127.0.0.1:8000]
"""

import argparse
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from frontend.main_window import MainWindow
from frontend.styles import APP_STYLE


def main():
    parser = argparse.ArgumentParser(description="Image Captioning PySide6 Frontend")
    parser.add_argument(
        "--backend-url",
        default="http://127.0.0.1:8000",
        help="FastAPI backend URL (default: http://127.0.0.1:8000)",
    )
    args = parser.parse_args()

    # High DPI scaling configuration
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("Image Captioning")

    # Set clean system font
    font = QFont("-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica Neue, Arial", 10)
    app.setFont(font)

    # Apply global modern QSS stylesheet
    app.setStyleSheet(APP_STYLE)

    window = MainWindow(backend_url=args.backend_url)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
