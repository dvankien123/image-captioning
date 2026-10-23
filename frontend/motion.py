"""UI Animations and Micro-interactions engine for Image Captioning Frontend.

Provides safe, high-performance micro-interactions built on PySide6:
- SlidingTabPill: Shared-layout / FLIP sliding background pill indicator between tabs
  driven by an exact analytical spring physics solver (stiffness ≈ 500, damping ≈ 40).
- SmoothThumbnailLabel: Micro-interaction thumbnail with smooth easing zoom
  and border highlight on hover.
- ModalFadeController: Smooth entrance and exit transitions for dialogs.
- prefers_reduced_motion: Respects system accessibility preferences for motion.
"""

import math
import os
import time
from typing import Callable, Optional

from PySide6.QtCore import (
    Property,
    QEasingCurve,
    QObject,
    QPointF,
    QPropertyAnimation,
    QRectF,
    QSize,
    QTimer,
    Qt,
)
from PySide6.QtGui import (
    QColor,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QDialog,
    QLabel,
    QWidget,
)


def prefers_reduced_motion() -> bool:
    """Detect if system or user prefers reduced motion (Windows SPI_GETCLIENTAREAANIMATION or env)."""
    if os.environ.get("PREFERS_REDUCED_MOTION", "").lower() in ("1", "true", "yes"):
        return True
    try:
        import ctypes
        val = ctypes.c_int()
        # SPI_GETCLIENTAREAANIMATION = 0x1042
        if ctypes.windll.user32.SystemParametersInfoW(0x1042, 0, ctypes.byref(val), 0):
            return val.value == 0
    except Exception:
        pass
    return False


class SlidingTabPill(QWidget):
    """Shared sliding pill indicator positioned behind active tab.
    
    Uses FLIP / shared-layout technique:
    - Only animates painter transform (translate) during paintEvent to prevent widget reflow.
    - Analytical spring physics: stiffness k ≈ 500, damping c ≈ 40, mass m = 1 (alpha=20, omega_d=10).
    - Respects prefers-reduced-motion by snapping immediately.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._current_x: float = 0.0
        self._current_w: float = 0.0
        self._start_x: float = 0.0
        self._start_w: float = 0.0
        self._target_x: float = 0.0
        self._target_w: float = 0.0
        self._start_time: float = 0.0
        self._initialized: bool = False

        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(16)  # 60fps
        self._anim_timer.timeout.connect(self._step)

    def slide_to(self, target_x: float, target_w: float):
        """Slide pill to target tab position using spring physics."""
        if target_w <= 0:
            return

        if not self._initialized or prefers_reduced_motion():
            self.snap_to(target_x, target_w)
            return

        if abs(self._target_x - target_x) < 0.5 and abs(self._target_w - target_w) < 0.5:
            return

        # FLIP: First = current rendered position, Last = target tab geometry
        self._start_x = self._current_x
        self._start_w = self._current_w
        self._target_x = target_x
        self._target_w = target_w
        self._start_time = time.time()
        self._anim_timer.start()

    def snap_to(self, target_x: float, target_w: float):
        """Immediately snap pill to target tab without animation."""
        self._anim_timer.stop()
        self._current_x = self._start_x = self._target_x = target_x
        self._current_w = self._start_w = self._target_w = target_w
        self._initialized = True
        self.update()

    def _step(self):
        elapsed = time.time() - self._start_time
        # Spring settles completely within ~260ms (error < 0.2%)
        if elapsed >= 0.28:
            self._current_x = self._target_x
            self._current_w = self._target_w
            self._anim_timer.stop()
            self.update()
            return

        # Closed-form analytical solution: m*x'' + c*x' + k*(x - x_target) = 0
        # For k=500, c=40, m=1 -> alpha = c/(2m) = 20, omega_d = sqrt(k/m - alpha^2) = 10
        # p(t) = 1.0 - e^(-alpha*t) * (cos(omega_d*t) + (alpha/omega_d)*sin(omega_d*t))
        decay = math.exp(-20.0 * elapsed)
        p = 1.0 - decay * (math.cos(10.0 * elapsed) + 2.0 * math.sin(10.0 * elapsed))

        self._current_x = self._start_x + (self._target_x - self._start_x) * p
        self._current_w = self._start_w + (self._target_w - self._start_w) * p
        self.update()

    def paintEvent(self, event):
        if self._current_w <= 0:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Only transform painter coordinates - zero widget layout reflow
        painter.translate(self._current_x, 0)

        total_w = self.parent().width() if self.parent() else 1
        max_travel = max(1.0, total_w - self._current_w)
        ratio = max(0.0, min(1.0, self._current_x / max_travel))

        # Smoothly morph corner radii based on tab position
        # Left tab: r_tl = 18px, r_tr = 0px
        # Right tab: r_tl = 0px, r_tr = 18px
        r_tl = 18.0 * (1.0 - ratio)
        r_tr = 18.0 * ratio
        w = self._current_w
        h = self.height()

        path = QPainterPath()
        path.moveTo(0, h)
        path.lineTo(0, r_tl)
        if r_tl > 0.5:
            path.quadTo(0, 0, r_tl, 0)
        else:
            path.lineTo(0, 0)

        path.lineTo(w - r_tr, 0)
        if r_tr > 0.5:
            path.quadTo(w, 0, w, r_tr)
        else:
            path.lineTo(w, 0)

        path.lineTo(w, h)
        path.closeSubpath()

        painter.fillPath(path, QColor("#FFFFFF"))
        painter.end()


class SmoothThumbnailLabel(QLabel):
    """Heatmap thumbnail label with antialiased rounded corners and smooth hover zoom & border."""

    def __init__(self, pixmap: QPixmap, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._pixmap = pixmap
        self._zoom: float = 1.0
        self._is_hovered: bool = False
        self.setObjectName("attentionThumbLabel")
        self.setFixedSize(100, 68)
        self.setAlignment(Qt.AlignCenter)

        self._anim = QPropertyAnimation(self, b"zoom")
        self._anim.setDuration(160)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)

    def get_zoom(self) -> float:
        return self._zoom

    def set_zoom(self, z: float):
        self._zoom = z
        self.update()

    zoom = Property(float, get_zoom, set_zoom)

    def set_pixmap(self, pixmap: QPixmap):
        self._pixmap = pixmap
        self.update()

    def set_hovered(self, hovered: bool):
        self._is_hovered = hovered
        self._anim.stop()
        self._anim.setStartValue(self._zoom)
        self._anim.setEndValue(1.06 if hovered else 1.0)
        self._anim.start()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        rect = self.rect()
        path = QPainterPath()
        path.addRoundedRect(
            rect.x() + 0.5,
            rect.y() + 0.5,
            rect.width() - 1,
            rect.height() - 1,
            6,
            6,
        )
        painter.setClipPath(path)

        if self._pixmap and not self._pixmap.isNull():
            w = rect.width() * self._zoom
            h = rect.height() * self._zoom
            x = (rect.width() - w) / 2.0
            y = (rect.height() - h) / 2.0
            painter.drawPixmap(
                QRectF(x, y, w, h),
                self._pixmap,
                QRectF(self._pixmap.rect()),
            )

        # Draw smooth active border when hovered
        if self._zoom > 1.005:
            painter.setClipping(False)
            pen = QPen(QColor("#2563EB"), 1.8)
            pen.setJoinStyle(Qt.RoundJoin)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), 6, 6)

        painter.end()


class ModalFadeController(QObject):
    """Controls silky entrance and exit opacity animations for QDialog modals."""

    def __init__(self, dialog: QDialog, duration_in: int = 180, duration_out: int = 130):
        super().__init__(dialog)
        self.dialog = dialog
        self.duration_in = duration_in
        self.duration_out = duration_out
        self._is_closing = False
        self._anim: Optional[QPropertyAnimation] = None

    def animate_open(self):
        self.dialog.setWindowOpacity(0.0)
        self._anim = QPropertyAnimation(self.dialog, b"windowOpacity")
        self._anim.setDuration(self.duration_in)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self._anim.start()

    def animate_close(self, on_closed: Optional[Callable[[], None]] = None):
        if self._is_closing:
            return
        self._is_closing = True
        self._anim = QPropertyAnimation(self.dialog, b"windowOpacity")
        self._anim.setDuration(self.duration_out)
        self._anim.setStartValue(self.dialog.windowOpacity())
        self._anim.setEndValue(0.0)
        self._anim.setEasingCurve(QEasingCurve.InCubic)

        def finish():
            if on_closed:
                on_closed()
            else:
                self.dialog.accept()

        self._anim.finished.connect(finish)
        self._anim.start()
