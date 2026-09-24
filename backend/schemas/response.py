from pydantic import BaseModel

class CaptionResponse(BaseModel):
    success: bool
    caption: str | None = None
    error: str | None = None
