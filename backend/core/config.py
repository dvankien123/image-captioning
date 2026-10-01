from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Union

class Settings(BaseSettings):
    PROJECT_NAME: str = "FastAPI Project"
    MAX_FILE_SIZE_MB: int = 5
    
    # Cấu hình CORS
    CORS_ORIGINS: Union[str, List[str]] = ["*"]

    # --- BỔ SUNG DÒNG NÀY ĐỂ NHẬN DIỆN API KEY ---
    GOOGLE_API_KEY: str = ""

    # Thêm extra="ignore" để ứng dụng không bị sập nếu file .env có thêm các biến lạ khác
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8", 
        extra="ignore"
    )

# Khởi tạo đối tượng settings để các file khác import vào dùng
settings = Settings()