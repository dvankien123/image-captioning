import os
import shutil
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from backend.services.model_service import generate_caption 

router = APIRouter()
UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
ALLOWED_EXTENSIONS = {"image/jpeg", "image/png", "image/jpg"}

# Hàm tiện ích để xóa file
def remove_file(path: str):
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception as e:
        print(f"Lỗi khi xóa file {path}: {e}")

@router.post("/upload")
async def upload_image(
    file: UploadFile = File(...), 
    background_tasks: BackgroundTasks = BackgroundTasks() # Thêm tham số này
):
    if file.content_type not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Vui lòng upload ảnh JPG hoặc PNG.")
    
    new_filename = f"{uuid.uuid4()}.{file.filename.split('.')[-1]}"
    file_path = os.path.join(UPLOAD_DIR, new_filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi hệ thống: {str(e)}")
    finally:
        file.file.close()
        
    # Lấy caption từ model
    caption_result = await generate_caption(file_path)
    
    # Đặt lịch xóa file ảnh sau khi API chạy xong
    background_tasks.add_task(remove_file, file_path)
        
    return {
        "status": "success",
        "data": {
            "original_filename": file.filename,
            "caption": caption_result
        }
    }