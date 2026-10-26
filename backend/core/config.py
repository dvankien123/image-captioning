from pathlib import Path
from typing import List, Literal, Union

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    PROJECT_NAME: str = "Image Captioning API"
    MAX_FILE_SIZE_MB: int = Field(default=10, ge=1, le=50)
    MAX_IMAGE_PIXELS: int = Field(default=20_000_000, ge=1_000_000)
    MAX_CAPTION_LENGTH: int = Field(default=30, ge=1, le=100)
    IMAGE_SIZE: int = Field(default=224, ge=32, le=1024)
    MAX_CONCURRENT_INFERENCES: int = Field(default=1, ge=1, le=8)
    INFERENCE_QUEUE_TIMEOUT_SECONDS: int = Field(default=30, ge=1, le=300)
    HISTORY_MAX_LIMIT: int = Field(default=100, ge=1, le=500)
    MODEL_CHECKPOINT: Path = PROJECT_ROOT / "model_vit" / "checkpoints" / "best_model.pth"
    DECODING_METHOD: Literal["greedy", "beam"] = "beam"
    BEAM_SIZE: int = Field(default=3, ge=1, le=10)
    BEAM_LENGTH_PENALTY_ALPHA: float = Field(default=0.6, ge=0.0, le=3.0)
    UPLOAD_DIR: Path = PROJECT_ROOT / "temp_uploads"
    DATABASE_PATH: Path = PROJECT_ROOT / "caption_history.db"
    CORS_ORIGINS: Union[str, List[str]] = ["*"]
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

settings = Settings()
