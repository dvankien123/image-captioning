import os
import pytest
from fastapi.testclient import TestClient
from backend.main import app

# Khởi tạo TestClient của FastAPI
client = TestClient(app)

def test_upload_valid_image():
    """Kịch bản 1: Tải lên một file ảnh PNG hợp lệ."""
    
    # 1. Tạo một ảnh giả (chỉ chứa một vài byte dữ liệu để giả lập)
    file_content = b"fake image content"
    files = {
        "file": ("test_image.png", file_content, "image/png")
    }

    # 2. Bắn request POST lên API /api/upload
    response = client.post("/api/upload", files=files)

    # 3. Kiểm tra kết quả
    assert response.status_code == 200 # Phải trả về mã 200 (Thành công)
    
    # Lấy dữ liệu JSON trả về
    data = response.json()
    
    # Kiểm tra xem các trường dữ liệu có tồn tại không
    assert data["status"] == "success"
    assert "data" in data
    assert data["data"]["original_filename"] == "test_image.png"
    assert "caption" in data["data"]

def test_upload_invalid_file_type():
    """Kịch bản 2: Tải lên một file dạng Text, mong đợi API sẽ chặn lại."""
    
    file_content = b"This is a text file, not an image"
    files = {
        "file": ("document.txt", file_content, "text/plain")
    }

    response = client.post("/api/upload", files=files)

    # API phải chặn lại và trả về mã lỗi 400 (Bad Request)
    assert response.status_code == 400 
    
    # Kiểm tra câu thông báo lỗi xem có khớp không
    data = response.json()
    assert data["detail"] == "Vui lòng upload ảnh JPG hoặc PNG."