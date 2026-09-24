from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routers import upload # Import file upload.py vừa tạo

app = FastAPI(title="Image Captioning API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Đăng ký router upload vào ứng dụng chính với tiền tố /api
app.include_router(upload.router, prefix="/api", tags=["Image Processing"])

@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Backend FastAPI is running"}