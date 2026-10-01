from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routers import upload
from backend.services.model_service import ml_models
from backend.services.mock_model import DummyImageModel

# Import settings vừa tạo
from backend.core.config import settings

from backend.core.database import engine, Base
from backend.models import caption_record
Base.metadata.create_all(bind=engine)

@asynccontextmanager
async def lifespan(app: FastAPI):
    model = DummyImageModel()
    model.load_model()
    ml_models["caption_model"] = model
    yield
    ml_models.clear()

# Sử dụng tên dự án từ biến môi trường
app = FastAPI(title=settings.PROJECT_NAME, lifespan=lifespan)

# Nếu CORS_ORIGINS là chuỗi "*" thì chuyển thành mảng ["*"]
origins = [settings.CORS_ORIGINS] if isinstance(settings.CORS_ORIGINS, str) else settings.CORS_ORIGINS

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router, prefix="/api", tags=["Image Processing"])

@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Backend FastAPI is running"}