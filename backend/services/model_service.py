import asyncio

async def generate_caption(image_path: str) -> str:
    """
    Hàm giả lập gọi model AI. 
    Sau này bạn sẽ thay phần ruột của hàm này bằng code gọi model thực tế 
    (ví dụ: load model PyTorch, truyền ảnh qua model và nhận kết quả).
    """
    # Giả lập thời gian model đang suy nghĩ (2 giây)
    await asyncio.sleep(2)
    
    # Trả về một caption giả để frontend có data làm giao diện
    return "Một người đang đi dạo trong công viên vào buổi chiều."