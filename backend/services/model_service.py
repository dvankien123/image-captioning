from backend.services.mock_model import DummyImageModel

# Tạo một biến toàn cục để lưu trữ instance của model
ml_models = {}

async def generate_caption(image_path: str) -> str:
    """
    Lấy model đã được load sẵn từ biến ml_models để dự đoán
    """
    # Kiểm tra xem model đã có trong bộ nhớ chưa
    if "caption_model" not in ml_models:
        return "Lỗi: Model chưa được khởi tạo!"
    
    # Gọi hàm dự đoán của model đã lưu
    model = ml_models["caption_model"]
    caption = await model.predict(image_path)
    
    return caption