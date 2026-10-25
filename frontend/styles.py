"""Centralized QSS stylesheet matching the 4 design mockups strictly.

No arbitrary extra widgets or badges:
- Clean light background (#F8FAFC)
- Main title "Image Captioning"
- Left column: "Upload Image" card and centered "Image Information" card
- Right column: Segmented tab bar ("Caption Results" vs "Attentions")
- Empty state: Cloud illustration + "No image yet" + subtext
- Caption Results: Image preview frame + centered caption card
- Attentions: Centered caption card + 4-column attention heatmap grid
- Modal dialog: Light silver/gray card with top-right 'X' and bottom 'Word: ...'
"""

APP_STYLE = """
/* Global Reset */
QWidget {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    color: #0F172A;
    font-size: 13px;
}

QMainWindow, QWidget#centralWidget {
    background-color: #F8FAFC;
}

/* App Header Title */
QLabel#appTitle {
    font-size: 24px;
    font-weight: 700;
    color: #0F172A;
    background: transparent;
    padding-left: 4px;
}

/* White Card Containers */
QFrame#modernCard {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 18px;
}

/* Upload Image Card Header */
QLabel#uploadTitle {
    font-size: 15px;
    font-weight: 700;
    color: #0F172A;
    background: transparent;
}

QLabel#uploadSubtitle {
    font-size: 11.5px;
    color: #64748B;
    background: transparent;
}

/* Dashed Drop Zone Container */
QFrame#dropZoneBox {
    background-color: #F8FAFD;
    border: 1.8px dashed #93C5FD;
    border-radius: 16px;
}

QFrame#dropZoneBox:hover {
    background-color: #EFF6FF;
    border-color: #3B82F6;
}

QFrame#dropZoneActive {
    background-color: #EFF6FF;
    border: 2px dashed #2563EB;
    border-radius: 16px;
}

QLabel#dropTitle {
    font-size: 14px;
    font-weight: 700;
    color: #0F172A;
    background: transparent;
}

QLabel#dropSubtitle {
    font-size: 11.5px;
    color: #64748B;
    background: transparent;
}

QLabel#dropFormatNotice {
    font-size: 10.5px;
    color: #94A3B8;
    background: transparent;
}

/* Choose File Button */
QPushButton#chooseFileBtn {
    background-color: #2563EB;
    color: #FFFFFF;
    border: none;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 600;
    padding: 9px 24px;
}

QPushButton#chooseFileBtn:hover {
    background-color: #1D4ED8;
}

QPushButton#chooseFileBtn:pressed {
    background-color: #1E40AF;
}

/* Image Information Card */
QLabel#infoPanelTitle {
    font-size: 15px;
    font-weight: 700;
    color: #0F172A;
    background: transparent;
}

QFrame#infoInnerBox {
    background-color: #F1F6FB;
    border: none;
    border-radius: 14px;
    padding: 14px 16px;
}

QLabel#infoBulletItem {
    font-size: 11.5px;
    font-weight: 500;
    color: #4B617A;
    background: transparent;
    padding: 3px 0px;
}

/* Right Tab Bar */
QFrame#tabBarContainer {
    background-color: #DCE5EE;
    border-top-left-radius: 18px;
    border-top-right-radius: 18px;
    border-bottom: 1px solid #E2E8F0;
}

QPushButton#tabBtnActive {
    background: transparent;
    color: #2563EB;
    font-size: 15px;
    font-weight: 700;
    border: none;
    padding: 14px 28px;
}

QPushButton#tabBtnInactive {
    background: transparent;
    color: #0F172A;
    font-size: 15px;
    font-weight: 700;
    border: none;
    padding: 14px 28px;
}

QPushButton#tabBtnInactive:hover {
    color: #1E40AF;
}

/* Empty State */
QLabel#emptyTitle {
    font-size: 22px;
    font-weight: 700;
    color: #0F172A;
    background: transparent;
}

QLabel#emptySubtitle {
    font-size: 13.5px;
    color: #475569;
    line-height: 1.4;
    background: transparent;
}

/* Image Preview Card in Caption Results */
QFrame#imagePreviewCard {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
}

/* Caption Text Display Box */
QFrame#captionCard {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 16px;
}

QLabel#captionTextLabel {
    font-size: 19px;
    font-weight: 400;
    color: #0F172A;
    background: transparent;
}

/* Attentions View Grid Container */
QFrame#attentionsGridBox {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
}

/* Individual Attention Item */
QFrame#attentionItemFrame {
    background-color: transparent;
    border: none;
}

QLabel#attentionWordLabel {
    font-size: 12.5px;
    font-weight: 500;
    color: #0F172A;
    background: transparent;
}

QLabel#attentionThumbLabel {
    border-radius: 6px;
}

QLabel#attentionThumbLabel:hover {
    border: 1.5px solid #2563EB;
}

/* Attention Modal Dialog (Image 4) */
QDialog#attentionModalDialog {
    background-color: #D5DCE4;
    border: 1px solid #BCC7D3;
    border-radius: 18px;
}

QPushButton#modalCloseXBtn {
    background: transparent;
    color: #0F172A;
    font-size: 20px;
    font-weight: 700;
    border: none;
    padding: 4px 10px;
}

QPushButton#modalCloseXBtn:hover {
    color: #EF4444;
}

QLabel#modalWordLabel {
    font-size: 20px;
    font-weight: 400;
    color: #0F172A;
    background: transparent;
}

/* Scroll Area */
QScrollArea {
    background: transparent;
    border: none;
}

QScrollBar:vertical {
    border: none;
    background: transparent;
    width: 6px;
    margin: 0px;
}

QScrollBar::handle:vertical {
    background: #CBD5E1;
    min-height: 20px;
    border-radius: 3px;
}

QScrollBar::handle:vertical:hover {
    background: #94A3B8;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
"""
