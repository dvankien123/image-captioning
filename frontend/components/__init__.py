"""Components package for the Image Captioning PySide6 Frontend."""

from frontend.components.sidebar import LeftPanel
from frontend.components.right_panel import RightPanel
from frontend.components.caption_view import CaptionResultsView
from frontend.components.attentions_view import AttentionsView
from frontend.components.drop_zone import DropZone
from frontend.components.info_panel import InfoPanel
from frontend.components.attention_dialog import AttentionDialog

__all__ = [
    "LeftPanel",
    "RightPanel",
    "CaptionResultsView",
    "AttentionsView",
    "DropZone",
    "InfoPanel",
    "AttentionDialog",
]
