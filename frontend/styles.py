"""Centralized QSS stylesheet matching the 3 design mockups strictly.

Faithfully implements:
- Clean white layout with subtle slate borders (#E2E8F0, #F1F5F9)
- Top header with gradient app icon + 'Image Captioning'
- Left panel: Upload Image card (Empty dropzone or loaded image info + replace dropzone) + History row
- Center panel: Top 50/50 segmented tabs ('Caption Results' & 'Attentions'), main view, and bottom action bar
- Right panel:
    - Tab 0: Alternative Captions with scores + Processing Info
    - Tab 1: Caption token pills + Why This Word probability distributions + action buttons
"""

APP_STYLE = """
/* Global Reset */
QWidget {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    color: #0F172A;
    font-size: 12.5px;
    background: transparent;
}

QMainWindow, QWidget#centralWidget {
    background-color: #FFFFFF;
}

/* ScrollArea */
QScrollArea {
    border: none;
    background: transparent;
}
QScrollBar:horizontal, QScrollBar:vertical {
    border: none;
    background: #F8FAFC;
    height: 6px;
    width: 6px;
    border-radius: 3px;
}
QScrollBar::handle:horizontal, QScrollBar::handle:vertical {
    background: #CBD5E1;
    border-radius: 3px;
    min-width: 20px;
    min-height: 20px;
}
QScrollBar::handle:horizontal:hover, QScrollBar::handle:vertical:hover {
    background: #94A3B8;
}
QScrollBar::add-line, QScrollBar::sub-line {
    border: none;
    background: none;
}

/* App Header */
QFrame#appHeader {
    background-color: #FFFFFF;
    border-bottom: 1px solid #E2E8F0;
    padding: 0px;
}

QLabel#appTitle {
    font-size: 16px;
    font-weight: 700;
    color: #0F172A;
    background: transparent;
}

/* Panel Dividers */
QFrame#leftPanel {
    background-color: #FFFFFF;
    border-right: 1px solid #E2E8F0;
}

QFrame#centerPanel {
    background-color: #FFFFFF;
}

QFrame#rightPanel {
    background-color: #FFFFFF;
    border-left: 1px solid #E2E8F0;
}

/* Section Headers */
QLabel#sectionHeader {
    font-size: 11px;
    font-weight: 700;
    color: #64748B;
    letter-spacing: 0.5px;
    background: transparent;
}

QLabel#sectionSubHeader {
    font-size: 11px;
    color: #94A3B8;
    background: transparent;
}

/* Dashed Drop Zone Container */
QFrame#dropZoneBox {
    background-color: #F8FAFC;
    border: 1.5px dashed #93C5FD;
    border-radius: 12px;
}

QFrame#dropZoneBox:hover {
    background-color: #EFF6FF;
    border-color: #3B82F6;
}

QFrame#dropZoneActive {
    background-color: #EFF6FF;
    border: 2px dashed #2563EB;
    border-radius: 12px;
}

QLabel#dropTitle {
    font-size: 13px;
    font-weight: 700;
    color: #0F172A;
    background: transparent;
}

QLabel#dropSubtitle {
    font-size: 11px;
    color: #94A3B8;
    background: transparent;
}

QLabel#dropNotice {
    font-size: 10px;
    color: #94A3B8;
    background: transparent;
}

/* Keyboard Shortcut Badges */
QLabel#kbdBadge {
    background-color: #F1F5F9;
    border: 1px solid #E2E8F0;
    border-radius: 4px;
    color: #64748B;
    font-size: 10.5px;
    font-weight: 600;
    padding: 1px 5px;
}

/* Image Metadata Badges */
QLabel#metaBadge {
    background-color: #EFF6FF;
    color: #3B82F6;
    border-radius: 6px;
    font-size: 10px;
    font-weight: 600;
    padding: 2px 7px;
}

/* Primary Button (Blue solid) */
QPushButton#primaryBtn {
    background-color: #2563EB;
    color: #FFFFFF;
    border: none;
    border-radius: 8px;
    font-size: 12px;
    font-weight: 600;
    padding: 7px 16px;
}

QPushButton#primaryBtn:hover {
    background-color: #1D4ED8;
}

QPushButton#primaryBtn:pressed {
    background-color: #1E40AF;
}

QPushButton#primaryBtn:disabled {
    background-color: #F1F5F9;
    color: #94A3B8;
    border: 1px solid #E2E8F0;
}

/* Outline Button */
QPushButton#outlineBtn {
    background-color: #FFFFFF;
    color: #334155;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    font-size: 12px;
    font-weight: 600;
    padding: 7px 14px;
}

QPushButton#outlineBtn:hover {
    background-color: #F8FAFC;
    border-color: #CBD5E1;
}

QPushButton#outlineBtn:pressed {
    background-color: #F1F5F9;
}

QPushButton#outlineBtn:disabled {
    background-color: #F8FAFC;
    color: #94A3B8;
    border-color: #E2E8F0;
}

/* Horizontal Dividers */
QFrame#thinDivider {
    background-color: #F1F5F9;
    border: none;
    max-height: 1px;
    min-height: 1px;
}

/* Top Segmented Tab Container & Buttons */
QFrame#tabBarContainer {
    background-color: #DDE6F1;
    border: none;
}

QPushButton#tabBtnActive {
    background: transparent;
    color: #2563EB;
    font-size: 13.5px;
    font-weight: 700;
    border: none;
    padding: 12px 16px;
}

QPushButton#tabBtnInactive {
    background: transparent;
    color: #64748B;
    font-size: 13.5px;
    font-weight: 700;
    border: none;
    padding: 12px 16px;
}

QPushButton#tabBtnInactive:hover {
    color: #334155;
}

/* Empty State */
QLabel#emptyTitle {
    font-size: 20px;
    font-weight: 700;
    color: #0F172A;
    background: transparent;
}

QLabel#emptySubtitle {
    font-size: 12.5px;
    color: #64748B;
    line-height: 1.4;
    background: transparent;
}

/* Caption Text */
QLabel#captionTitleLbl {
    font-size: 11px;
    font-weight: 700;
    color: #64748B;
    letter-spacing: 0.5px;
}

QLabel#captionTextLbl {
    font-size: 22px;
    font-weight: 700;
    color: #0F172A;
    line-height: 1.3;
}

QLabel#captionPlaceholderLbl {
    font-size: 20px;
    font-weight: 600;
    color: #CBD5E1;
}

/* Alternative Caption Card */
QFrame#altCardSelected {
    background-color: #EFF6FF;
    border: 1px solid #BFDBFE;
    border-radius: 8px;
    padding: 6px 10px;
}

QFrame#altCardNormal {
    background-color: transparent;
    border: 1px solid transparent;
    border-radius: 8px;
    padding: 6px 10px;
}

QFrame#altCardNormal:hover {
    background-color: #F8FAFC;
}

/* Processing Info Row */
QLabel#infoRowLabel {
    font-size: 11.5px;
    color: #64748B;
}

QLabel#infoRowValue {
    font-size: 11.5px;
    color: #0F172A;
    font-weight: 600;
}

/* Word Token Pills (Attentions Tab) */
QPushButton#tokenPillNormal {
    background-color: #FFFFFF;
    border: 1px solid #CBD5E1;
    border-radius: 8px;
    color: #1E293B;
    font-size: 12px;
    font-weight: 600;
    padding: 6px 14px;
}

QPushButton#tokenPillNormal:hover {
    border-color: #94A3B8;
    background-color: #F8FAFC;
}

QPushButton#tokenPillActive {
    background-color: #FFFFFF;
    border: 1.5px solid #EA580C;
    border-radius: 8px;
    color: #EA580C;
    font-size: 12px;
    font-weight: 700;
    padding: 6px 14px;
}

/* Why This Word Context Box */
QFrame#whyContextBox {
    background-color: #F1F5F9;
    border-radius: 8px;
    padding: 6px 12px;
}

/* Selected Badge in Why This Word */
QLabel#selectedBadge {
    background-color: #EA580C;
    color: #FFFFFF;
    border-radius: 3px;
    font-size: 9px;
    font-weight: 700;
    padding: 1px 4px;
}
"""
