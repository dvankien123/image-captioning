"""Vector / QPainter icon generators matching the design mockups precisely.

Generates:
1. App logo icon (vibrant rounded gradient square: indigo -> pink -> orange).
2. Outlined photo icon with mountain and sun.
3. Dropzone circular blue photo illustration.
4. Upload arrow icon for buttons.
5. Empty-state illustration (soft pastel cloud with sparkles and center photo icon).
6. Custom radio buttons for candidate captions.
7. Utility icons: Copy, Regenerate, Export, Arrow-right.
"""

from pathlib import Path
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)

ASSETS_DIR = Path(__file__).resolve().parent / "assets"


def create_app_logo_icon(size: int = 30) -> QPixmap:
    """Vibrant app logo matching the top-left logo in Images 1, 2, 3."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)

    grad = QLinearGradient(0, 0, size, size)
    grad.setColorAt(0.0, QColor("#6366F1"))  # Indigo
    grad.setColorAt(0.55, QColor("#A855F7")) # Purple
    grad.setColorAt(0.80, QColor("#EC4899")) # Pink
    grad.setColorAt(1.0, QColor("#F97316"))  # Orange

    painter.setBrush(QBrush(grad))
    painter.setPen(Qt.NoPen)
    radius = size * 0.28
    painter.drawRoundedRect(QRectF(1, 1, size - 2, size - 2), radius, radius)

    painter.end()
    return pixmap


def create_circle_image_icon(size: int = 44) -> QPixmap:
    """Light-blue circular container with blue outline photo icon (Dropzone icon)."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)

    # 1. Circle background #EFF6FF
    painter.setBrush(QBrush(QColor("#EFF6FF")))
    painter.setPen(Qt.NoPen)
    painter.drawEllipse(QRectF(1, 1, size - 2, size - 2))

    # 2. Blue image outline inside circle
    pen = QPen(QColor("#3B82F6"), 1.8)
    pen.setJoinStyle(Qt.RoundJoin)
    pen.setCapStyle(Qt.RoundCap)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)

    margin = size * 0.28
    fw = size - 2 * margin
    fh = fw * 0.82
    fy = (size - fh) / 2
    frame_rect = QRectF(margin, fy, fw, fh)
    painter.drawRoundedRect(frame_rect, 3.5, 3.5)

    # Sun dot
    painter.drawEllipse(QPointF(frame_rect.right() - 5.5, frame_rect.top() + 5.5), 1.6, 1.6)

    # Mountain peak
    path = QPainterPath()
    path.moveTo(frame_rect.left() + 3, frame_rect.bottom() - 2.5)
    path.lineTo(frame_rect.left() + fw * 0.38, frame_rect.top() + fh * 0.35)
    path.lineTo(frame_rect.left() + fw * 0.58, frame_rect.bottom() - 2.5)
    painter.drawPath(path)

    path2 = QPainterPath()
    path2.moveTo(frame_rect.left() + fw * 0.46, frame_rect.bottom() - 2.5)
    path2.lineTo(frame_rect.left() + fw * 0.72, frame_rect.top() + fh * 0.45)
    path2.lineTo(frame_rect.right() - 2.5, frame_rect.bottom() - 2.5)
    painter.drawPath(path2)

    painter.end()
    return pixmap


def create_header_photo_icon(size: int = 24) -> QPixmap:
    """Outlined photo icon with mountain and sun."""
    return create_circle_image_icon(size)


def create_dropzone_illustration(w: int = 48, h: int = 48) -> QPixmap:
    """Center dropzone photo icon."""
    return create_circle_image_icon(w)


def create_upload_icon(size: int = 14) -> QPixmap:
    """White upward arrow icon for 'Choose file' / 'Change image' button."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)

    pen = QPen(Qt.white, 2.0)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    painter.setPen(pen)

    # Up arrow shaft
    cx = size * 0.5
    painter.drawLine(QPointF(cx, size * 0.22), QPointF(cx, size * 0.78))

    # Arrow head
    path = QPainterPath()
    path.moveTo(size * 0.22, size * 0.45)
    path.lineTo(cx, size * 0.20)
    path.lineTo(size * 0.78, size * 0.45)
    painter.drawPath(path)

    painter.end()
    return pixmap


def create_radio_icon(checked: bool, size: int = 16) -> QPixmap:
    """Crisp custom radio circle matching Alternative Captions in Images 1 & 2."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)

    cx = size / 2
    cy = size / 2
    r = (size - 3) / 2

    if checked:
        # Outer blue ring
        pen = QPen(QColor("#2563EB"), 1.8)
        painter.setPen(pen)
        painter.setBrush(QBrush(QColor("#FFFFFF")))
        painter.drawEllipse(QPointF(cx, cy), r, r)

        # Center blue dot
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor("#2563EB")))
        painter.drawEllipse(QPointF(cx, cy), r * 0.52, r * 0.52)
    else:
        # Light gray outer ring
        pen = QPen(QColor("#CBD5E1"), 1.8)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(QPointF(cx, cy), r, r)

    painter.end()
    return pixmap


def create_copy_icon(size: int = 14, color: str = "#FFFFFF") -> QPixmap:
    """Copy / Clipboard icon."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)

    pen = QPen(QColor(color), 1.4)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)

    # Back rect
    painter.drawRoundedRect(QRectF(size * 0.12, size * 0.28, size * 0.56, size * 0.60), 2, 2)
    # Front rect
    painter.drawRoundedRect(QRectF(size * 0.32, size * 0.12, size * 0.56, size * 0.60), 2, 2)

    painter.end()
    return pixmap


def create_regenerate_icon(size: int = 14, color: str = "#334155") -> QPixmap:
    """Circular reload arrow icon for 'Regenerate'."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)

    pen = QPen(QColor(color), 1.5)
    pen.setCapStyle(Qt.RoundCap)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)

    # Arc
    rect = QRectF(size * 0.15, size * 0.15, size * 0.70, size * 0.70)
    painter.drawArc(rect, 45 * 16, 270 * 16)

    # Arrow head
    path = QPainterPath()
    path.moveTo(size * 0.70, size * 0.15)
    path.lineTo(size * 0.88, size * 0.28)
    path.lineTo(size * 0.70, size * 0.42)
    painter.drawPath(path)

    painter.end()
    return pixmap


def create_export_icon(size: int = 14, color: str = "#334155") -> QPixmap:
    """Download / Export tray icon."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)

    pen = QPen(QColor(color), 1.5)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)

    cx = size * 0.5
    # Down arrow shaft
    painter.drawLine(QPointF(cx, size * 0.15), QPointF(cx, size * 0.62))
    # Down arrow head
    path = QPainterPath()
    path.moveTo(size * 0.28, size * 0.42)
    path.lineTo(cx, size * 0.64)
    path.lineTo(size * 0.72, size * 0.42)
    painter.drawPath(path)

    # Bottom tray
    tray = QPainterPath()
    tray.moveTo(size * 0.20, size * 0.72)
    tray.lineTo(size * 0.20, size * 0.86)
    tray.lineTo(size * 0.80, size * 0.86)
    tray.lineTo(size * 0.80, size * 0.72)
    painter.drawPath(tray)

    painter.end()
    return pixmap


def create_arrow_right_icon(size: int = 14, color: str = "#334155") -> QPixmap:
    """Arrow right for '➔ View attention' button."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)

    pen = QPen(QColor(color), 1.6)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    painter.setPen(pen)

    cy = size * 0.5
    painter.drawLine(QPointF(size * 0.18, cy), QPointF(size * 0.78, cy))

    path = QPainterPath()
    path.moveTo(size * 0.52, size * 0.25)
    path.lineTo(size * 0.80, cy)
    path.lineTo(size * 0.52, size * 0.75)
    painter.drawPath(path)

    painter.end()
    return pixmap


def create_empty_state_pixmap(w: int = 240, h: int = 180) -> QPixmap:
    """Soft pastel cloud illustration with sparkles and center photo icon."""
    pixmap = QPixmap(w, h)
    pixmap.fill(Qt.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)

    # 1. Cloud shape using intersecting soft rounded ellipses
    cloud_path = QPainterPath()
    cx = w * 0.5
    cy = h * 0.55

    # Cloud puffs
    cloud_path.addEllipse(QPointF(cx - 38, cy + 4), 36, 32)
    cloud_path.addEllipse(QPointF(cx + 38, cy + 6), 34, 30)
    cloud_path.addEllipse(QPointF(cx - 12, cy - 24), 44, 38)
    cloud_path.addEllipse(QPointF(cx + 20, cy - 18), 38, 34)
    cloud_path.addEllipse(QPointF(cx, cy + 12), 48, 28)

    # Soft gradient fill for cloud
    grad = QLinearGradient(0, cy - 60, 0, cy + 40)
    grad.setColorAt(0.0, QColor("#F0F7FF"))
    grad.setColorAt(0.6, QColor("#E0EFFF"))
    grad.setColorAt(1.0, QColor("#D4E8FC"))

    painter.setPen(Qt.NoPen)
    painter.setBrush(QBrush(grad))
    painter.fillPath(cloud_path.simplified(), QBrush(grad))

    # 2. Centered picture frame icon inside cloud
    ic_cx = cx
    ic_cy = cy - 4
    fw = 64
    fh = 52
    frame_rect = QRectF(ic_cx - fw / 2, ic_cy - fh / 2, fw, fh)

    # Picture card background (blue accent)
    painter.setBrush(QBrush(QColor("#70A5F9")))
    painter.drawRoundedRect(frame_rect, 10, 10)

    # White mountain peak inside frame
    peak_path = QPainterPath()
    peak_path.moveTo(frame_rect.left() + 6, frame_rect.bottom() - 4)
    peak_path.lineTo(frame_rect.left() + 24, frame_rect.top() + 18)
    peak_path.lineTo(frame_rect.left() + 38, frame_rect.bottom() - 4)
    painter.fillPath(peak_path, QBrush(QColor("#D6E6FD")))

    peak2_path = QPainterPath()
    peak2_path.moveTo(frame_rect.left() + 28, frame_rect.bottom() - 4)
    peak2_path.lineTo(frame_rect.left() + 44, frame_rect.top() + 24)
    peak2_path.lineTo(frame_rect.right() - 6, frame_rect.bottom() - 4)
    painter.fillPath(peak2_path, QBrush(QColor("#FFFFFF")))

    # Sun in frame
    painter.setBrush(QBrush(QColor("#FFFFFF")))
    painter.drawEllipse(QPointF(frame_rect.right() - 16, frame_rect.top() + 15), 4.5, 4.5)

    # 3. Draw 4-pointed blue sparkles around cloud
    def draw_sparkle(scx: float, scy: float, r: float):
        sparkle = QPainterPath()
        sparkle.moveTo(scx, scy - r)
        sparkle.quadTo(scx, scy, scx + r, scy)
        sparkle.quadTo(scx, scy, scx, scy + r)
        sparkle.quadTo(scx, scy, scx - r, scy)
        sparkle.quadTo(scx, scy, scx, scy - r)
        painter.fillPath(sparkle, QBrush(QColor("#2563EB")))

    draw_sparkle(cx - 72, cy - 30, 7.5)   # Top-left sparkle
    draw_sparkle(cx + 72, cy - 10, 8.5)   # Top-right sparkle
    draw_sparkle(cx - 86, cy + 18, 6.0)   # Mid-left sparkle
    draw_sparkle(cx + 84, cy + 20, 6.5)   # Bottom-right sparkle
    draw_sparkle(cx + 35, cy - 50, 5.0)   # Top accent sparkle

    painter.end()
    return pixmap
