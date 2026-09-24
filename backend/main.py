from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routers import upload

# Import biến toàn cục và class Model
from backend.services.model_service import ml_models
from backend.services.mock_model import DummyImageModel

# Định nghĩa vòng đời của ứng dụng
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Code chạy 1 LẦN DUY NHẤT khi khởi động server ---
    # 1. Khởi tạo và load model
    model = DummyImageModel()
    model.load_model()
    
    # 2. Lưu model vào biến toàn cục để các API khác dùng chung
    ml_models["caption_model"] = model
    
    # Bàn giao lại quyền điều khiển cho FastAPI chạy các API
    yield
    
    # --- Code chạy khi tắt server (Ctrl + C) ---
    # 3. Dọn dẹp bộ nhớ (giải phóng RAM/VRAM nếu cần)
    ml_models.clear()
    print("🛑 Đã giải phóng model khỏi bộ nhớ!")

# Truyền lifespan vào FastAPI
app = FastAPI(title="Image Captioning API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router, prefix="/api", tags=["Image Processing"])

@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Backend FastAPI is running"}