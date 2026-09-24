import time
import asyncio

class DummyImageModel:
    def __init__(self):
        self.model_loaded = False

    def load_model(self):
        """Giả lập thời gian tốn vài giây để load model vào bộ nhớ"""
        print("⏳ Bắt đầu load model AI vào RAM/VRAM...")
        time.sleep(3) # Cố tình dừng 3 giây để giả lập
        self.model_loaded = True
        print("✅ Model AI đã được load xong và sẵn sàng!")

    async def predict(self, image_path: str) -> str:
        """Giả lập việc model chạy inference để ra caption"""
        if not self.model_loaded:
            raise RuntimeError("Model chưa được load!")
        
        # Giả lập thời gian model suy nghĩ
        await asyncio.sleep(1.5) 
        return f"Model đã xử lý thành công ảnh tại: {image_path}"