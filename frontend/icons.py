"""Vector / QPainter icon generators matching the design mockups precisely.

Generates:
1. Header photo icon (photo frame with mountain and sun).
2. Dropzone illustration (photo frame with circular '+' badge at bottom-right).
3. Upload arrow icon for the 'Choose File' button.
4. Empty-state illustration (soft pastel cloud with sparkles and center photo icon).
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


def create_header_photo_icon(size: int = 24) -> QPixmap:
    """Outlined photo icon with mountain and sun (top-left of Upload Image card)."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)

    # Frame border
    pen = QPen(QColor("#1E3A8A"), 1.8)
    pen.setJoinStyle(Qt.RoundJoin)
    pen.setCapStyle(Qt.RoundCap)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)

    margin = size * 0.12
    frame_rect = QRectF(margin, margin, size - 2 * margin, size - 2 * margin)
    painter.drawRoundedRect(frame_rect, size * 0.16, size * 0.16)

    # Sun / Circle
    sun_radius = size * 0.09
    sun_center = QPointF(margin + size * 0.22, margin + size * 0.26)
    painter.drawEllipse(sun_center, sun_radius, sun_radius)

    # Mountain peak
    path = QPainterPath()
    path.moveTo(margin + size * 0.12, margin + size * 0.64)
    path.lineTo(margin + size * 0.38, margin + size * 0.38)
    path.lineTo(margin + size * 0.54, margin + size * 0.54)
    path.lineTo(margin + size * 0.64, margin + size * 0.44)
    path.lineTo(margin + size * 0.76, margin + size * 0.64)
    painter.drawPath(path)

    painter.end()
    return pixmap


def create_dropzone_illustration(w: int = 68, h: int = 68) -> QPixmap:
    """Center dropzone photo icon with a circular '+' badge at bottom-right."""
    pixmap = QPixmap(w, h)
    pixmap.fill(Qt.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)

    # Photo outline frame (blue accent)
    pen = QPen(QColor("#3B82F6"), 2.2)
    pen.setJoinStyle(Qt.RoundJoin)
    pen.setCapStyle(Qt.RoundCap)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)

    frame_rect = QRectF(w * 0.14, h * 0.12, w * 0.62, h * 0.62)
    painter.drawRoundedRect(frame_rect, 10, 10)

    # Sun circle
    painter.setPen(pen)
    painter.drawEllipse(QPointF(w * 0.34, h * 0.30), 3.2, 3.2)

    # Mountain peak inside frame
    mpath = QPainterPath()
    mpath.moveTo(w * 0.20, h * 0.62)
    mpath.lineTo(w * 0.42, h * 0.38)
    mpath.lineTo(w * 0.54, h * 0.50)
    mpath.lineTo(w * 0.62, h * 0.42)
    mpath.lineTo(w * 0.70, h * 0.54)
    painter.drawPath(mpath)

    # Plus badge circle at bottom-right
    badge_center = QPointF(w * 0.76, h * 0.72)
    badge_radius = w * 0.17

    painter.setPen(Qt.NoPen)
    painter.setBrush(QBrush(QColor("#2563EB")))
    painter.drawEllipse(badge_center, badge_radius, badge_radius)

    # White '+' cross inside badge
    plus_pen = QPen(Qt.white, 2.4)
    plus_pen.setCapStyle(Qt.RoundCap)
    painter.setPen(plus_pen)
    half_cross = badge_radius * 0.48
    painter.drawLine(
        QPointF(badge_center.x() - half_cross, badge_center.y()),
        QPointF(badge_center.x() + half_cross, badge_center.y()),
    )
    painter.drawLine(
        QPointF(badge_center.x(), badge_center.y() - half_cross),
        QPointF(badge_center.x(), badge_center.y() + half_cross),
    )

    painter.end()
    return pixmap


def create_upload_icon(size: int = 16) -> QPixmap:
    """White upload tray icon for 'Choose File' button."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)

    pen = QPen(Qt.white, 2.0)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    painter.setPen(pen)

    # Up arrow shaft
    painter.drawLine(
        QPointF(size * 0.5, size * 0.2),
        QPointF(size * 0.5, size * 0.65),
    )
    # Arrow head
    path = QPainterPath()
    path.moveTo(size * 0.28, size * 0.4)
    path.lineTo(size * 0.5, size * 0.18)
    path.lineTo(size * 0.72, size * 0.4)
    painter.drawPath(path)

    # Bottom tray line
    tray = QPainterPath()
    tray.moveTo(size * 0.2, size * 0.62)
    tray.lineTo(size * 0.2, size * 0.82)
    tray.lineTo(size * 0.8, size * 0.82)
    tray.lineTo(size * 0.8, size * 0.62)
    painter.drawPath(tray)

    painter.end()
    return pixmap


def create_empty_state_pixmap(w: int = 240, h: int = 180) -> QPixmap:
    """Soft pastel cloud illustration with sparkles and center photo icon."""
    pixmap = QPixmap(w, h)
    pixmap.fill(Qt.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)

    # 1. Draw cloud shape using intersecting soft rounded ellipses
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

    # Picture card background (white with soft blue tint)
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
