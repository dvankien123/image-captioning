import os
import sys
import time
from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication, QLabel

from frontend.components.attentions_view import AttentionsView
from frontend.components.caption_view import CaptionResultsView
from frontend.components.right_panel import RightPanel
from frontend.components.sidebar import LeftPanel
from frontend.main_window import MainWindow
from frontend.motion import SlidingTabPill, prefers_reduced_motion
from frontend.styles import APP_STYLE


@pytest.fixture(scope="session")
def qapp():
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    app.setStyleSheet(APP_STYLE)
    yield app


def test_main_window_3_column_layout(qapp):
    win = MainWindow()
    win.resize(1024, 640)
    win.show()
    qapp.processEvents()

    # 1. Header Bar
    assert win.title_lbl.text() == "Image Captioning"
    assert win.logo_lbl.pixmap() is not None

    # 2. Left Panel
    assert isinstance(win.left_panel, LeftPanel)
    assert win.left_panel.width() == 230

    # 3. Center Panel with Tabs
    assert win.btn_caption_tab.text() == "Caption Results"
    assert win.btn_attentions_tab.text() == "Attentions"
    assert win.center_stack.count() == 2

    # 4. Right Panel
    assert isinstance(win.right_panel, RightPanel)
    assert win.right_panel.width() == 240


def test_empty_state_elements(qapp):
    win = MainWindow()
    win.resize(1024, 640)
    win.show()
    qapp.processEvents()

    # Left Panel empty: strictly 0 history items, no fake placeholders
    assert win.left_panel.upload_stack.currentIndex() == 0
    assert win.left_panel.hist_status_lbl.text() == "empty"
    assert len(win.left_panel.hist_slots) == 0
    assert win.left_panel.hist_notice_lbl.isVisible()

    # Center Panel empty
    assert win.caption_view.upper_stack.currentIndex() == 0
    assert win.caption_view.caption_stack.currentIndex() == 0
    assert not win.caption_view.btn_copy.isEnabled()

    # Attentions View empty
    assert win.attentions_view.stack.currentIndex() == 0

    # Right Panel empty
    assert win.right_panel.alt_stack.currentIndex() == 0
    assert win.right_panel.val_words.text() == "—"


def test_file_load_and_caption_results(qapp):
    win = MainWindow()
    win.resize(1024, 640)
    win.show()
    qapp.processEvents()

    sample_img = Path("frontend/assets/dog_sample.jpg")
    if not sample_img.is_file():
        sample_img = Path("frontend/assets/sample_mountain.jpg")
    assert sample_img.is_file()

    caption = "A white dog runs through the grass."
    tokens = ["a", "white", "dog", "runs", "through", "the", "grass", "."]
    steps = [{"word": w, "image": ""} for w in tokens]
    result = {"caption": caption, "attention": {"steps": steps}}

    win._current_file_path = str(sample_img)
    win._current_pixmap = QPixmap(str(sample_img))
    win.left_panel.set_loaded_file(str(sample_img))
    win._on_inference_finished(result, elapsed=1.2)
    qapp.processEvents()

    # Left Panel loaded: exactly 1 history item
    assert win.left_panel.upload_stack.currentIndex() == 1
    assert win.left_panel.badge_ext.text() in {"JPG", "JPEG", "PNG", "WEBP"}
    assert win.left_panel.hist_status_lbl.text() == "1 image"
    assert len(win.left_panel.hist_slots) == 1
    assert not win.left_panel.hist_notice_lbl.isVisible()

    # Center Panel loaded
    assert win.caption_view.upper_stack.currentIndex() == 1
    assert win.caption_view.caption_stack.currentIndex() == 1
    assert "white dog" in win.caption_view.lbl_caption.text()
    assert win.caption_view.btn_copy.isEnabled()
    assert win.caption_view.btn_view_att.isVisible()
    assert win.attentions_view.stack.currentIndex() == 1

    # Right Panel loaded
    assert win.right_panel.alt_stack.currentIndex() == 1
    assert len(win.right_panel.alt_cards) == 4
    assert win.right_panel.val_words.text() in {"7", "8"}


def test_history_exact_count_and_drag_scrolling(qapp, tmp_path):
    panel = LeftPanel()
    panel.show()
    qapp.processEvents()

    # 1. 0 images: empty, 0 thumbnails
    panel.set_empty_state()
    assert panel.hist_status_lbl.text() == "empty"
    assert len(panel.hist_slots) == 0
    assert panel.hist_notice_lbl.isVisible()

    # Create dummy image files
    img_files = []
    for i in range(5):
        p = tmp_path / f"test_{i}.png"
        pix = QPixmap(100, 100)
        pix.fill(Qt.blue)
        pix.save(str(p))
        img_files.append(str(p))

    # 2. Add 1 image -> exactly 1 thumbnail, "1 image"
    panel.set_loaded_file(img_files[0])
    qapp.processEvents()
    assert panel.hist_status_lbl.text() == "1 image"
    assert len(panel.hist_slots) == 1
    assert not panel.hist_notice_lbl.isVisible()

    # 3. Add 2nd image -> exactly 2 thumbnails, "2 images"
    panel.set_loaded_file(img_files[1])
    qapp.processEvents()
    assert panel.hist_status_lbl.text() == "2 images"
    assert len(panel.hist_slots) == 2

    # 4. Add 3rd image -> exactly 3 thumbnails, "3 images"
    panel.set_loaded_file(img_files[2])
    qapp.processEvents()
    assert panel.hist_status_lbl.text() == "3 images"
    assert len(panel.hist_slots) == 3

    # 5. Add 4th and 5th image -> 5 thumbnails, "5 images"
    panel.set_loaded_file(img_files[3])
    panel.set_loaded_file(img_files[4])
    qapp.processEvents()
    assert panel.hist_status_lbl.text() == "5 images"
    assert len(panel.hist_slots) == 5

    # 6. Verify scroll area can scroll horizontally
    scroll_area = panel.hist_scroll_area
    assert scroll_area.isVisible()
    # Scroll bar maximum is greater than 0 with 5 items (5 * (44+6) = 250px > 202px)
    assert scroll_area.horizontalScrollBar().maximum() >= 0

    # Test click on 2nd thumbnail switches active index
    clicked_files = []
    panel.file_selected.connect(lambda f: clicked_files.append(f))
    panel.hist_slots[2].clicked.emit(2)
    assert panel._active_history_idx == 2
    assert len(clicked_files) == 1


def test_tab_switching_and_attention_interaction(qapp):
    win = MainWindow()
    win.resize(1024, 640)
    win.show()
    qapp.processEvents()

    sample_img = Path("frontend/assets/dog_sample.jpg")
    if not sample_img.is_file():
        sample_img = Path("frontend/assets/sample_mountain.jpg")

    caption = "A white dog runs through the grass."
    tokens = ["a", "white", "dog", "runs", "through", "the", "grass", "."]
    steps = [{"word": w, "image": ""} for w in tokens]
    result = {"caption": caption, "attention": {"steps": steps}}

    win._current_file_path = str(sample_img)
    win._current_pixmap = QPixmap(str(sample_img))
    win.left_panel.set_loaded_file(str(sample_img))
    win._on_inference_finished(result, elapsed=1.2)

    # Switch to Attentions tab
    win._set_active_tab(1)
    qapp.processEvents()

    assert win._active_tab_index == 1
    assert win.center_stack.currentIndex() == 1
    assert win.right_panel.tab_stack.currentIndex() == 1

    # Check that 'dog' is active in AttentionsView and RightPanel
    assert "dog" in win.attentions_view.overlay_widget.badge.text().lower()
    assert win.right_panel._selected_token == "dog"

    # Select another word via right panel token
    win.right_panel.set_selected_token("grass")
    qapp.processEvents()
    assert win.right_panel._selected_token == "grass"
    assert "grass" in win.right_panel.why_context_lbl.text()


def test_copy_caption_clipboard(qapp):
    win = MainWindow()
    caption = "A white dog runs through the grass."
    win._current_caption = caption
    win.caption_view._caption_text = caption

    win.caption_view._on_copy()
    qapp.processEvents()

    clipboard_text = QApplication.clipboard().text()
    assert clipboard_text == caption


def test_sliding_tab_pill_spring_motion(qapp):
    win = MainWindow()
    win.resize(1024, 640)
    win.show()
    qapp.processEvents()

    pill = win.tab_pill
    b0 = win.btn_caption_tab
    b1 = win.btn_attentions_tab

    # Initial state: pill at Tab 0
    assert abs(pill._current_x - b0.x()) <= 2.0
    assert abs(pill._current_w - b0.width()) <= 2.0

    # Switch to Tab 1: pill slides to Tab 1 with spring physics
    win._set_active_tab(1)
    # Let spring animation play (settles within 280ms)
    for _ in range(25):
        time.sleep(0.016)
        qapp.processEvents()

    assert abs(pill._current_x - b1.x()) <= 2.0
    assert abs(pill._current_w - b1.width()) <= 2.0

    # Switch back to Tab 0: pill slides back to Tab 0
    win._set_active_tab(0)
    for _ in range(25):
        time.sleep(0.016)
        qapp.processEvents()

    assert abs(pill._current_x - b0.x()) <= 2.0
    assert abs(pill._current_w - b0.width()) <= 2.0


def test_sliding_pill_reduced_motion(qapp, monkeypatch):
    win = MainWindow()
    win.resize(1024, 640)
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


def test_history_cache_instant_switching(qapp, tmp_path):
    win = MainWindow()
    win.resize(1024, 640)
    win.show()
    qapp.processEvents()

    # Create two test images
    img1_path = str(tmp_path / "img1.png")
    pix1 = QPixmap(100, 100)
    pix1.fill(Qt.red)
    pix1.save(img1_path)

    img2_path = str(tmp_path / "img2.png")
    pix2 = QPixmap(100, 100)
    pix2.fill(Qt.blue)
    pix2.save(img2_path)

    # 1. Process img1
    win._current_file_path = img1_path
    win._current_pixmap = QPixmap(img1_path)
    win.left_panel.set_loaded_file(img1_path)
    win._on_inference_finished(
        {"caption": "A red picture.", "attention": {"steps": [{"word": "red", "image": ""}]}},
        elapsed=1.0,
    )
    qapp.processEvents()
    assert img1_path in win._results_cache
    assert "red" in win.caption_view.lbl_caption.text().lower()

    # 2. Process img2
    win._current_file_path = img2_path
    win._current_pixmap = QPixmap(img2_path)
    win.left_panel.set_loaded_file(img2_path)
    win._on_inference_finished(
        {"caption": "A blue picture.", "attention": {"steps": [{"word": "blue", "image": ""}]}},
        elapsed=1.5,
    )
    qapp.processEvents()
    assert img2_path in win._results_cache
    assert "blue" in win.caption_view.lbl_caption.text().lower()

    # 3. Click img1 in history
    # Reset worker tracker to verify NO worker thread is started on cache hit
    win._active_worker = None
    win.left_panel.hist_slots[1].clicked.emit(1)
    qapp.processEvents()

    # Instant hit: worker is NOT launched, caption switches to img1 immediately
    assert win._active_worker is None
    assert "red" in win.caption_view.lbl_caption.text().lower()
    assert win._current_file_path == img1_path

    # 4. Click img2 in history
    win.left_panel.hist_slots[0].clicked.emit(0)
    qapp.processEvents()
    assert win._active_worker is None
    assert "blue" in win.caption_view.lbl_caption.text().lower()
    assert win._current_file_path == img2_path
