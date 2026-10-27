import os
import sys
import time
from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication, QLabel

from frontend.components.attention_dialog import AttentionDialog
from frontend.components.attentions_view import AttentionsView
from frontend.components.caption_view import CaptionResultsView
from frontend.components.drop_zone import DropZone
from frontend.components.info_panel import InfoPanel
from frontend.main_window import MainWindow
from frontend.motion import ModalFadeController, SlidingTabPill, SmoothThumbnailLabel, prefers_reduced_motion


@pytest.fixture(scope="session")
def qapp():
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def test_drop_zone_icon_centering(qapp):
    dz = DropZone()
    dz.resize(310, 360)
    dz.show()
    qapp.processEvents()

    # The drop illustration must be horizontally centered within the drop box
    box_w = dz.drop_box.width()
    icon_w = dz.illus_lbl.width()
    icon_x = dz.illus_lbl.x()
    center_offset = abs((box_w - icon_w) // 2 - icon_x)
    assert center_offset <= 2, f"Drop illustration is not centered: box_w={box_w}, icon_x={icon_x}"


def test_empty_state_alignment_consistency(qapp):
    win = MainWindow()
    win.resize(1060, 720)
    win.show()
    qapp.processEvents()

    # Get empty state cloud and titles in Tab 0 (Caption Results)
    c0_cloud = win.caption_view.empty_page.findChildren(QLabel)[0]
    c0_title = win.caption_view.empty_page.findChildren(QLabel)[1]
    c0_sub = win.caption_view.empty_page.findChildren(QLabel)[2]

    # Switch to Tab 1 (Attentions)
    win._set_active_tab(1)
    qapp.processEvents()

    c1_cloud = win.attentions_view.empty_page.findChildren(QLabel)[0]
    c1_title = win.attentions_view.empty_page.findChildren(QLabel)[1]
    c1_sub = win.attentions_view.empty_page.findChildren(QLabel)[2]

    # Geometries must match identically so there is zero jumping when switching tabs
    assert c0_cloud.geometry() == c1_cloud.geometry(), "Cloud illustration position jumped between tabs!"
    assert c0_title.geometry() == c1_title.geometry(), "Title position jumped between tabs!"
    assert c0_sub.geometry() == c1_sub.geometry(), "Subtitle position jumped between tabs!"


def test_sliding_tab_pill_spring_motion(qapp):
    win = MainWindow()
    win.resize(1060, 720)
    win.show()
    qapp.processEvents()

    pill = win.tab_pill
    b0 = win.btn_caption_tab
    b1 = win.btn_attentions_tab

    # Initial state: pill at Tab 0
    assert abs(pill._current_x - b0.x()) <= 2.0
    assert abs(pill._current_w - b0.width()) <= 2.0

    # Switch to Tab 1: pill slides to Tab 1
    win._set_active_tab(1)
    # Let spring animation play (settles within 280ms)
    for _ in range(20):
        time.sleep(0.016)
        qapp.processEvents()

    assert abs(pill._current_x - b1.x()) <= 2.0
    assert abs(pill._current_w - b1.width()) <= 2.0

    # Switch back to Tab 0: pill slides back to Tab 0
    win._set_active_tab(0)
    for _ in range(20):
        time.sleep(0.016)
        qapp.processEvents()

    assert abs(pill._current_x - b0.x()) <= 2.0
    assert abs(pill._current_w - b0.width()) <= 2.0


def test_sliding_pill_reduced_motion(qapp, monkeypatch):
    win = MainWindow()
    win.resize(1060, 720)
    win.show()
    qapp.processEvents()

    # Mock prefers_reduced_motion to True
    monkeypatch.setenv("PREFERS_REDUCED_MOTION", "1")
    assert prefers_reduced_motion() is True

    b1 = win.btn_attentions_tab
    win._set_active_tab(1)
    qapp.processEvents()

    # Must immediately snap without running timer
    assert abs(win.tab_pill._current_x - b1.x()) <= 1.0
    assert win.tab_pill._anim_timer.isActive() is False


def test_file_drop_generates_caption_and_attentions(qapp):
    win = MainWindow()
    win.show()

    sample_img = Path("frontend/assets/sample_mountain.jpg")
    assert sample_img.is_file(), "sample_mountain.jpg not found"

    # Trigger file selection
    win._on_file_selected(str(sample_img))

    # Wait for inference worker if running
    if win._active_worker:
        win._active_worker.wait(20000)
    qapp.processEvents()

    # Check Caption tab
    assert win.caption_view.stack.currentIndex() == 1, "Caption view did not switch to result page!"
    assert win.caption_view.image_label.isVisible(), "Preview image is not visible!"
    assert win.caption_view.caption_text.text() != "", "Caption text is empty!"
    assert "mountain" in win._current_caption.lower(), "Caption does not match expected result!"

    # Check metadata
    assert "sample_mountain.jpg" in win.info_panel.name_lbl.text()
    assert "JPG" in win.info_panel.type_lbl.text()

    # Check Attentions tab
    win._set_active_tab(1)
    qapp.processEvents()
    assert win.attentions_view.stack.currentIndex() == 1, "Attentions view did not switch to result page!"
    assert win.attentions_view.grid_layout.count() > 0, "Attentions grid has no heatmaps!"


def test_attention_modal(qapp):
    pix = QPixmap(100, 100)
    pix.fill(Qt.green)
    dlg = AttentionDialog(word="mountain", pixmap=pix)
    dlg.show()
    qapp.processEvents()
    assert dlg.bottom_word_lbl.text() == "Word: mountain"
    dlg.close_animated()
    qapp.processEvents()
