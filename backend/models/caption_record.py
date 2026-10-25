from sqlalchemy import Column, Integer, String, DateTime
from datetime import datetime, timezone
from backend.core.database import Base

class CaptionRecord(Base):
    __tablename__ = "caption_records"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    caption = Column(String)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
