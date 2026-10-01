import os
import shutil
import uuid
from backend.services.model_service import generate_caption 
from backend.core.config import settings
from backend.core.logger import logger
from sqlalchemy.orm import Session
from backend.core.database import get_db
from backend.models.caption_record import CaptionRecord
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Depends
from backend.services.vector_db import add_caption_to_vector_db, search_similar_images



router = APIRouter()
UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {"image/jpeg", "image/png", "image/jpg"}
MAX_FILE_SIZE = settings.MAX_FILE_SIZE_MB * 1024 * 1024

def remove_file(path: str):
    try:
        if os.path.exists(path):
            os.remove(path)
            logger.info(f"Đã dọn dẹp file tạm thành công: {path}")
    except Exception as e:
        # Ghi nhận lỗi nghiêm trọng bằng logger.error
        logger.error(f"Lỗi hệ thống khi xóa file {path}: {e}")

@router.post("/upload")
async def upload_image(
    file: UploadFile = File(...), 
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db)
):
    logger.info(f"Nhận yêu cầu phân tích ảnh: {file.filename}")
    # 1. Kiểm tra định dạng
    if file.content_type not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Vui lòng upload ảnh JPG hoặc PNG.")
        
    # 2. Kiểm tra dung lượng file
    if file.size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413, 
            detail=f"Dung lượng ảnh quá lớn ({file.size / (1024*1024):.1f}MB). Tối đa chỉ cho phép 5MB."
        )
    
    # 3. Lưu file
    new_filename = f"{uuid.uuid4()}.{file.filename.split('.')[-1]}"
    file_path = os.path.join(UPLOAD_DIR, new_filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi hệ thống: {str(e)}")
    finally:
        file.file.close()
        
    # 4. Gọi Model & Đặt lịch xóa ảnh
    caption_result = await generate_caption(file_path)
    background_tasks.add_task(remove_file, file_path)
        
    # 5. Lưu thông tin vào SQLite
    new_record = CaptionRecord(
        filename=file.filename,
        caption=caption_result
    )
    db.add(new_record)
    db.commit()
    db.refresh(new_record) # Lấy ID mới tạo để trả về
    logger.info(f"Đã lưu kết quả vào database với ID: {new_record.id}")

    add_caption_to_vector_db(new_record.id, new_record.filename, new_record.caption)

    return {
        "status": "success",
        "data": {
            "id": new_record.id,
            "original_filename": new_record.filename,
            "caption": new_record.caption
        }
    }

# --- THÊM MỘT API ĐỂ XEM LẠI LỊCH SỬ ---
@router.get("/history")
def get_upload_history(limit: int = 10, db: Session = Depends(get_db)):
    """API hỗ trợ lấy 10 bức ảnh đã phân tích gần nhất"""
    records = db.query(CaptionRecord).order_by(CaptionRecord.id.desc()).limit(limit).all()
    return {"status": "success", "data": records}

@router.get("/search")
def search_images(query: str, top_k: int = 3):
    """API Tìm kiếm ảnh bằng ngữ nghĩa (Semantic Search)"""
    logger.info(f"Nhận yêu cầu tìm kiếm ảnh với từ khóa: '{query}'")
    try:
        results = search_similar_images(query, top_k)
        return {
            "status": "success", 
            "query": query,
            "data": results
        }
    except Exception as e:
        logger.error(f"Lỗi khi tìm kiếm: {e}")
        raise HTTPException(status_code=500, detail="Lỗi hệ thống khi tìm kiếm vector")