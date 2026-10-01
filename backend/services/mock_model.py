import time
import asyncio
from backend.core.logger import logger # Import logger vừa tạo

class DummyImageModel:
    def __init__(self):
        self.model_loaded = False

    def load_model(self):
        logger.info("⏳ Bắt đầu load model AI vào RAM/VRAM...")
        time.sleep(3) 
        self.model_loaded = True
        logger.info("✅ Model AI đã được load xong và sẵn sàng!")

    async def predict(self, image_path: str) -> str:
        if not self.model_loaded:
            logger.error("Phát hiện lỗi: Đã gọi hàm predict nhưng model chưa được load.")
            raise RuntimeError("Model chưa được load!")
        
        await asyncio.sleep(1.5) 
        return f"Model đã xử lý thành công ảnh tại: {image_path}"