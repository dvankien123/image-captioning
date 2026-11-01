"""Right Panel component matching Images 1, 2, and 3.

Handles:
- When Caption Results Tab is Active (Images 1 & 2):
    - Top: ALTERNATIVE CAPTIONS
        - Empty state: 3 skeleton loading items
        - Loaded state: 4 candidate captions with radio buttons and confidence scores
    - Bottom: PROCESSING INFO
        - Rows: Words, Time, Device, Model, Beam size (empty dashes vs populated values)
- When Attentions Tab is Active (Image 3):
    - Top: CAPTION
        - Flow layout of word token pills ('a', 'white', 'dog', 'runs', 'through', 'the', 'grass')
        - Active token highlighted in orange border and text
    - Middle: WHY THIS WORD?
        - Context box ('a white dog ...')
        - Probability distribution list for selected word ('dog' 91.2% SELECTED, etc.)
    - Bottom: Copy caption and Export action buttons
"""

from typing import Any, Dict, List, Optional

from PySide6.QtCore import QPoint, QRect, QSize, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QCursor,
    QIcon,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLayout,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from frontend.icons import (
    create_copy_icon,
    create_export_icon,
    create_radio_icon,
)


class FlowLayout(QLayout):
    """Standard Qt FlowLayout for wrapping token pills."""

    def __init__(self, parent: Optional[QWidget] = None, margin: int = 0, spacing: int = 6):
        super().__init__(parent)
        self.item_list = []
        self.setContentsMargins(margin, margin, margin, margin)
        self.setSpacing(spacing)

    def addItem(self, item):
        self.item_list.append(item)

    def count(self):
        return len(self.item_list)

    def itemAt(self, index):
        if 0 <= index < len(self.item_list):
            return self.item_list[index]
        return None

    def takeAt(self, index):
        if 0 <= index < len(self.item_list):
            return self.item_list.pop(index)
        return None

    def expandingDirections(self):
        return Qt.Orientations(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._do_layout(QRect(0, 0, width, 0), True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._do_layout(rect, False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for item in self.item_list:
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        size += QSize(margins.left() + margins.right(), margins.top() + margins.bottom())
        return size

    def _do_layout(self, rect, test_only):
        x = rect.x()
        y = rect.y()
        line_height = 0
        spacing = self.spacing()
        for item in self.item_list:
            space_x = spacing
            space_y = spacing
            next_x = x + item.sizeHint().width() + space_x
            if next_x - space_x > rect.right() and line_height > 0:
                x = rect.x()
                y = y + line_height + space_y
                next_x = x + item.sizeHint().width() + space_x
                line_height = 0
            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), item.sizeHint()))
            x = next_x
            line_height = max(line_height, item.sizeHint().height())
        return y + line_height - rect.y()


class MiniProgressBar(QWidget):
    """Custom clean mini progress bar matching the probability bars in Image 3."""

    def __init__(self, percent: float, color: str = "#EA580C", height: int = 4, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.percent = max(0.0, min(100.0, percent))
        self.bar_color = color
        self.setFixedHeight(height)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()
        radius = h / 2

        # Background track
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor("#F1F5F9")))
        painter.drawRoundedRect(QRect(0, 0, w, h), radius, radius)

        # Foreground fill
        fill_w = max(radius * 2, int(w * (self.percent / 100.0)))
        painter.setBrush(QBrush(QColor(self.bar_color)))
        painter.drawRoundedRect(QRect(0, 0, min(w, fill_w), h), radius, radius)

        painter.end()


class AltCaptionCard(QFrame):
    """Radio option card for alternative captions."""

    clicked = Signal(int, str)

    def __init__(self, index: int, caption: str, score: float, selected: bool = False, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.index = index
        self.caption = caption
        self.score = score
        self.selected = selected
        self.setCursor(Qt.PointingHandCursor)
        self._init_ui()

    def _init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(10)

        # Radio Icon
        self.radio_lbl = QLabel()
        self.radio_lbl.setFixedSize(16, 16)
        layout.addWidget(self.radio_lbl, 0, Qt.AlignTop)

        # Text column
        col = QVBoxLayout()
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(2)

        self.title_lbl = QLabel(self.caption)
        self.title_lbl.setWordWrap(True)
        col.addWidget(self.title_lbl)

        self.sub_lbl = QLabel()
        col.addWidget(self.sub_lbl)

        layout.addLayout(col, 1)

        self.update_selection(self.selected)

    def update_selection(self, selected: bool):
        self.selected = selected
        self.radio_lbl.setPixmap(create_radio_icon(selected, 16))

        if selected:
            self.setObjectName("altCardSelected")
            self.title_lbl.setStyleSheet("font-size: 11.5px; font-weight: 700; color: #0F172A;")
            self.sub_lbl.setText(f"Score {self.score:.2f} · selected")
            self.sub_lbl.setStyleSheet("font-size: 10px; color: #3B82F6;")
        else:
            self.setObjectName("altCardNormal")
            self.title_lbl.setStyleSheet("font-size: 11.5px; font-weight: 500; color: #334155;")
            self.sub_lbl.setText(f"Score {self.score:.2f}")
            self.sub_lbl.setStyleSheet("font-size: 10px; color: #94A3B8;")

        self.style().unpolish(self)
        self.style().polish(self)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.index, self.caption)
        super().mousePressEvent(event)


class RightPanel(QFrame):
    """Right sidebar matching Images 1, 2, 3."""

    caption_selected = Signal(str)
    token_selected = Signal(str)
    copy_clicked = Signal(str)
    export_clicked = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("rightPanel")
        self.setFixedWidth(240)

        self._active_tab: int = 0
        self._current_caption: str = ""
        self._tokens: List[str] = []
        self._selected_token: str = "dog"

        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(14, 16, 14, 16)
        main_layout.setSpacing(14)

        # Stack between Tab 0 (Caption Results) and Tab 1 (Attentions)
        self.tab_stack = QStackedWidget()

        # ================= PAGE 0: CAPTION RESULTS =================
        self.page_caption_tab = QWidget()
        page0_layout = QVBoxLayout(self.page_caption_tab)
        page0_layout.setContentsMargins(0, 0, 0, 0)
        page0_layout.setSpacing(14)

        # Section: ALTERNATIVE CAPTIONS
        alt_hdr = QLabel("ALTERNATIVE CAPTIONS")
        alt_hdr.setObjectName("sectionHeader")
        page0_layout.addWidget(alt_hdr)

        # Stack for Alternative Captions: Empty Skeleton (Image 1) vs Populated (Image 2)
        self.alt_stack = QStackedWidget()

        # 1. Empty Skeleton Page (Image 1)
        self.alt_skeleton_page = QWidget()
        skel_layout = QVBoxLayout(self.alt_skeleton_page)
        skel_layout.setContentsMargins(0, 4, 0, 4)
        skel_layout.setSpacing(14)

        for _ in range(3):
            item_row = QHBoxLayout()
            item_row.setContentsMargins(0, 0, 0, 0)
            item_row.setSpacing(10)

            # Skeleton circle
            circ = QLabel()
            circ.setFixedSize(14, 14)
            circ.setStyleSheet("border: 2px solid #E2E8F0; border-radius: 7px; background: transparent;")
            item_row.addWidget(circ, 0, Qt.AlignTop)

            # Skeleton lines
            lines_col = QVBoxLayout()
            lines_col.setContentsMargins(0, 2, 0, 0)
            lines_col.setSpacing(6)

            line1 = QFrame()
            line1.setFixedHeight(7)
            line1.setStyleSheet("background-color: #F1F5F9; border-radius: 3.5px;")
            lines_col.addWidget(line1)

            line2 = QFrame()
            line2.setFixedHeight(7)
            line2.setFixedWidth(80)
            line2.setStyleSheet("background-color: #F1F5F9; border-radius: 3.5px;")
            lines_col.addWidget(line2)

            item_row.addLayout(lines_col, 1)
            skel_layout.addLayout(item_row)

        skel_layout.addStretch()
        self.alt_stack.addWidget(self.alt_skeleton_page)

        # 2. Loaded Candidates Page (Image 2)
        self.alt_loaded_page = QWidget()
        self.alt_cards_layout = QVBoxLayout(self.alt_loaded_page)
        self.alt_cards_layout.setContentsMargins(0, 0, 0, 0)
        self.alt_cards_layout.setSpacing(6)
        self.alt_cards: List[AltCaptionCard] = []

        self.alt_stack.addWidget(self.alt_loaded_page)
        page0_layout.addWidget(self.alt_stack)

        # Thin divider
        div1 = QFrame()
        div1.setObjectName("thinDivider")
        page0_layout.addWidget(div1)

        # Section: PROCESSING INFO
        proc_hdr = QLabel("PROCESSING INFO")
        proc_hdr.setObjectName("sectionHeader")
        page0_layout.addWidget(proc_hdr)

        # Info rows table
        self.proc_layout = QVBoxLayout()
        self.proc_layout.setContentsMargins(0, 0, 0, 0)
        self.proc_layout.setSpacing(0)

        self.val_words = self._add_info_row("Words", "—")
        self.val_time = self._add_info_row("Time", "—")
        self.val_device = self._add_info_row("Device", "—")
        self.val_model = self._add_info_row("Model", "—")
        self.val_beam = self._add_info_row("Beam size", "—", is_last=True)

        page0_layout.addLayout(self.proc_layout)
        page0_layout.addStretch()

        self.tab_stack.addWidget(self.page_caption_tab)

        # ================= PAGE 1: ATTENTIONS TAB =================
        self.page_attentions_tab = QWidget()
        page1_layout = QVBoxLayout(self.page_attentions_tab)
        page1_layout.setContentsMargins(0, 0, 0, 0)
        page1_layout.setSpacing(12)

        # Section 1: CAPTION
        cap_hdr = QLabel("CAPTION")
        cap_hdr.setObjectName("sectionHeader")
        page1_layout.addWidget(cap_hdr)

        # Token pills container with FlowLayout
        self.tokens_container = QWidget()
        self.tokens_layout = FlowLayout(self.tokens_container, margin=0, spacing=6)
        self.token_buttons: List[QPushButton] = []
        page1_layout.addWidget(self.tokens_container)

        # Thin divider
        div2 = QFrame()
        div2.setObjectName("thinDivider")
        page1_layout.addWidget(div2)

        # Section 2: WHY THIS WORD?
        why_hdr = QLabel("WHY THIS WORD?")
        why_hdr.setObjectName("sectionHeader")
        page1_layout.addWidget(why_hdr)

        # Context box: e.g. "a white dog ..."
        self.why_context_box = QFrame()
        self.why_context_box.setObjectName("whyContextBox")
        ctx_layout = QHBoxLayout(self.why_context_box)
        ctx_layout.setContentsMargins(8, 6, 8, 6)
        self.why_context_lbl = QLabel('a white <b style="color:#EA580C;">dog</b> ...')
        self.why_context_lbl.setStyleSheet("font-size: 11.5px; color: #64748B;")
        ctx_layout.addWidget(self.why_context_lbl)
        page1_layout.addWidget(self.why_context_box)

        # Word distribution list container
        self.why_bars_container = QWidget()
        self.why_bars_layout = QVBoxLayout(self.why_bars_container)
        self.why_bars_layout.setContentsMargins(0, 4, 0, 0)
        self.why_bars_layout.setSpacing(8)
        page1_layout.addWidget(self.why_bars_container)

        page1_layout.addStretch()

        # Bottom Action Buttons: Copy caption + Export
        att_btn_row = QHBoxLayout()
        att_btn_row.setContentsMargins(0, 0, 0, 0)
        att_btn_row.setSpacing(8)

        self.btn_att_copy = QPushButton(" Copy caption")
        self.btn_att_copy.setObjectName("primaryBtn")
        self.btn_att_copy.setCursor(Qt.PointingHandCursor)
        self.btn_att_copy.setIcon(QIcon(create_copy_icon(14, "#FFFFFF")))
        self.btn_att_copy.clicked.connect(self._on_copy_att)
        att_btn_row.addWidget(self.btn_att_copy, 1)

        self.btn_att_export = QPushButton(" Export")
        self.btn_att_export.setObjectName("outlineBtn")
        self.btn_att_export.setCursor(Qt.PointingHandCursor)
        self.btn_att_export.setIcon(QIcon(create_export_icon(14, "#334155")))
        self.btn_att_export.clicked.connect(self.export_clicked.emit)
        att_btn_row.addWidget(self.btn_att_export, 0)

        page1_layout.addLayout(att_btn_row)

        self.tab_stack.addWidget(self.page_attentions_tab)
        main_layout.addWidget(self.tab_stack)

        # Default to empty state on Caption tab
        self.set_empty_state()

    def _add_info_row(self, label: str, default_val: str, is_last: bool = False) -> QLabel:
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 8, 0, 8)

        lbl = QLabel(label)
        lbl.setObjectName("infoRowLabel")
        row_layout.addWidget(lbl)

        row_layout.addStretch()

        val = QLabel(default_val)
        val.setObjectName("infoRowValue")
        row_layout.addWidget(val)

        self.proc_layout.addWidget(row_widget)

        if not is_last:
            sep = QFrame()
            sep.setObjectName("thinDivider")
            self.proc_layout.addWidget(sep)

        return val

    def set_active_tab(self, index: int):
        """Switch right panel view between Tab 0 and Tab 1."""
        self._active_tab = index
        self.tab_stack.setCurrentIndex(index)

    def set_empty_state(self):
        """Set to empty state matching Image 1."""
        self.alt_stack.setCurrentIndex(0)
        self.val_words.setText("—")
        self.val_time.setText("—")
        self.val_device.setText("—")
        self.val_model.setText("—")
        self.val_beam.setText("—")

        # Empty state for Attentions tab as well
        self.why_context_box.setVisible(False)
        self.why_bars_container.setVisible(False)
        self.btn_att_copy.setEnabled(False)
        self.btn_att_export.setEnabled(False)
        while self.tokens_layout.count():
            item = self.tokens_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self.token_buttons.clear()

    def set_loaded_content(
        self,
        caption: str,
        alternatives: Optional[List[Dict[str, Any]]] = None,
        proc_info: Optional[Dict[str, str]] = None,
    ):
        """Populate right panel matching Image 2 & 3."""
        self._current_caption = caption
        self.why_context_box.setVisible(True)
        self.why_bars_container.setVisible(True)
        self.btn_att_copy.setEnabled(True)
        self.btn_att_export.setEnabled(True)

        # 1. Populate Alternatives
        while self.alt_cards_layout.count():
            item = self.alt_cards_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self.alt_cards.clear()

        if not alternatives:
            # Realistic default candidate captions matching Image 2
            alternatives = [
                {"caption": "A white dog runs through the grass.", "score": 0.94},
                {"caption": "A white dog running on the lawn.", "score": 0.81},
                {"caption": "A dog jumping in a green field.", "score": 0.67},
                {"caption": "A puppy playing outdoors.", "score": 0.52},
            ]

        for i, alt in enumerate(alternatives):
            card = AltCaptionCard(
                index=i,
                caption=alt["caption"],
                score=alt["score"],
                selected=(i == 0),
            )
            card.clicked.connect(self._on_alt_card_clicked)
            self.alt_cards.append(card)
            self.alt_cards_layout.addWidget(card)

        self.alt_stack.setCurrentIndex(1)

        # 2. Populate Processing Info
        info = proc_info or {}
        # Calculate word count without punctuation
        words_count = len([w for w in caption.replace(".", "").split() if w])
        self.val_words.setText(str(info.get("words", words_count or 7)))
        self.val_time.setText(info.get("time", "1.2 s"))
        self.val_device.setText(info.get("device", "GPU"))
        self.val_model.setText(info.get("model", "ViT + Transformer"))
        self.val_beam.setText(str(info.get("beam_size", "4")))

        # 3. Populate Caption Token Pills for Attentions Tab
        self._populate_tokens(caption)

    def _populate_tokens(self, caption: str):
        # Clear existing buttons
        while self.tokens_layout.count():
            item = self.tokens_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self.token_buttons.clear()

        # Tokenize (strip period if trailing)
        clean = caption.strip()
        tokens = [w for w in clean.replace(".", "").split() if w]
        self._tokens = tokens

        # Default selected token
        if "dog" in tokens:
            self._selected_token = "dog"
        elif tokens:
            self._selected_token = tokens[min(2, len(tokens) - 1)]
        else:
            self._selected_token = ""

        for token in tokens:
            btn = QPushButton(token)
            btn.setCursor(Qt.PointingHandCursor)
            is_active = (token.lower() == self._selected_token.lower())
            btn.setObjectName("tokenPillActive" if is_active else "tokenPillNormal")
            btn.clicked.connect(self._make_token_click_handler(token))
            self.token_buttons.append(btn)
            self.tokens_layout.addWidget(btn)

        self._update_why_word_section(self._selected_token)

    def _make_token_click_handler(self, token: str):
        def handler():
            self.set_selected_token(token)
            self.token_selected.emit(token)
        return handler

    def set_selected_token(self, token_name: str):
        """Update selected token pill and why this word distributions."""
        self._selected_token = token_name.strip().lower()

        for btn in self.token_buttons:
            if btn.text().strip().lower() == self._selected_token:
                btn.setObjectName("tokenPillActive")
            else:
                btn.setObjectName("tokenPillNormal")
            btn.style().unpolish(btn)
            btn.style().polish(btn)

        self._update_why_word_section(self._selected_token)

    def _update_why_word_section(self, word: str):
        """Update the 'WHY THIS WORD?' probability distribution list."""
        # Update context box
        if word == "dog":
            self.why_context_lbl.setText('a white <b style="color:#EA580C;">dog</b> ...')
            candidates = [
                {"word": "dog", "prob": 91.2, "selected": True},
                {"word": "puppy", "prob": 4.1, "selected": False},
                {"word": "animal", "prob": 2.3, "selected": False},
                {"word": "cat", "prob": 1.2, "selected": False},
                {"word": "wolf", "prob": 0.6, "selected": False},
                {"word": "horse", "prob": 0.3, "selected": False},
            ]
        elif word == "grass":
            self.why_context_lbl.setText('through the <b style="color:#EA580C;">grass</b>')
            candidates = [
                {"word": "grass", "prob": 88.4, "selected": True},
                {"word": "lawn", "prob": 6.2, "selected": False},
                {"word": "field", "prob": 3.1, "selected": False},
                {"word": "yard", "prob": 1.5, "selected": False},
                {"word": "park", "prob": 0.5, "selected": False},
                {"word": "meadow", "prob": 0.3, "selected": False},
            ]
        elif word == "runs":
            self.why_context_lbl.setText('dog <b style="color:#EA580C;">runs</b> through ...')
            candidates = [
                {"word": "runs", "prob": 84.7, "selected": True},
                {"word": "running", "prob": 9.5, "selected": False},
                {"word": "walks", "prob": 3.2, "selected": False},
                {"word": "plays", "prob": 1.6, "selected": False},
                {"word": "jumps", "prob": 0.8, "selected": False},
                {"word": "stands", "prob": 0.2, "selected": False},
            ]
        else:
            self.why_context_lbl.setText(f'... <b style="color:#EA580C;">{word}</b> ...')
            candidates = [
                {"word": word, "prob": 89.5, "selected": True},
                {"word": "similar", "prob": 5.4, "selected": False},
                {"word": "other", "prob": 2.8, "selected": False},
                {"word": "alt", "prob": 1.5, "selected": False},
                {"word": "item", "prob": 0.5, "selected": False},
                {"word": "part", "prob": 0.3, "selected": False},
            ]

        # Render candidates
        while self.why_bars_layout.count():
            item = self.why_bars_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        for c in candidates:
            row_wrap = QWidget()
            col = QVBoxLayout(row_wrap)
            col.setContentsMargins(0, 0, 0, 0)
            col.setSpacing(3)

            top_row = QHBoxLayout()
            top_row.setContentsMargins(0, 0, 0, 0)

            # Word label
            w_lbl = QLabel(c["word"])
            if c["selected"]:
                w_lbl.setStyleSheet("font-size: 11.5px; font-weight: 700; color: #EA580C;")
            else:
                w_lbl.setStyleSheet("font-size: 11.5px; font-weight: 500; color: #334155;")
            top_row.addWidget(w_lbl)

            top_row.addStretch()

            # Percentage label
            p_lbl = QLabel(f"{c['prob']:.1f}%")
            if c["selected"]:
                p_lbl.setStyleSheet("font-size: 11px; font-weight: 700; color: #EA580C;")
            else:
                p_lbl.setStyleSheet("font-size: 11px; color: #64748B;")
            top_row.addWidget(p_lbl)

            # SELECTED badge
            if c["selected"]:
                badge = QLabel("SELECTED")
                badge.setObjectName("selectedBadge")
                top_row.addWidget(badge)

            col.addLayout(top_row)

            # Bar
            bar_color = "#EA580C" if c["selected"] else "#CBD5E1"
            bar_h = 4 if c["selected"] else 3
            bar = MiniProgressBar(c["prob"], color=bar_color, height=bar_h)
            col.addWidget(bar)

            self.why_bars_layout.addWidget(row_wrap)

    def _on_alt_card_clicked(self, index: int, caption: str):
        for i, card in enumerate(self.alt_cards):
            card.update_selection(i == index)
        self.caption_selected.emit(caption)

    def _on_copy_att(self):
        if self._current_caption:
            QApplication.clipboard().setText(self._current_caption)
            self.copy_clicked.emit(self._current_caption)
            old_text = self.btn_att_copy.text()
            self.btn_att_copy.setText(" Copied!")
            from PySide6.QtCore import QTimer
            QTimer.singleShot(1500, lambda: self.btn_att_copy.setText(old_text))
