# Tài Liệu Tích Hợp Frontend - Backend & Hướng Dẫn Kết Nối Lại

Tài liệu này được biên soạn theo yêu cầu:
> *Lưu ý không được thay đổi logic backend, chỗ nào logic backend chưa đáp ứng được thì không cần kết nối thay vào đấy thì viết tài liệu chỉ ra chỗ cần kết nối lại. Làm đúng như design không được tự ý thêm chi tiết.*

---

## 1. Hiện Trạng Kết Nối Frontend & Backend

Giao diện người dùng (Frontend) được phát triển bằng **PySide6 (Qt for Python)**, tuân thủ chính xác 100% theo 4 ảnh thiết kế (Design Wireframes):
1. **Trạng thái khởi tạo (Empty State):** Header "Image Captioning", thẻ "Upload Image" (vùng kéo thả nét đứt, icon ảnh có dấu cộng, nút "Choose File" màu xanh, định dạng hỗ trợ), thẻ "Image Information" (4 dòng gạch đầu dòng căn giữa), và tab bên phải hiển thị hình minh họa đám mây "No image yet".
2. **Tab "Caption Results":** Khung hiển thị ảnh preview ở trên, khung hiển thị kết quả sinh caption ở dưới (`A picture about a mountain`), thông tin metadata bên trái được điền đầy đủ.
3. **Tab "Attentions":** Khung caption ở trên, bên dưới là lưới 4 cột hiển thị các ảnh Grad-CAM heatmaps với tiêu đề `Word: ...`.
4. **Modal Attention Detail:** Hộp thoại bo góc nền xám bạc, nút đóng `X` ở góc trên bên phải, ảnh heatmap phóng to ở giữa và nhãn `Word: ...` căn giữa bên dưới.

---

## 2. Các Điểm Logic Backend Chưa Đáp Ứng & Giải Pháp Tạm Thời

| STT | Vấn đề trong Backend | Mô tả chi tiết | Xử lý hiện tại ở Frontend |
| :--- | :--- | :--- | :--- |
| **1** | **Chưa có file checkpoint mô hình** (`best_model.pth`) | File `model/checkpoints/best_model.pth` chưa tồn tại trong source code. Khi khởi động backend, `caption_model_service.load()` thất bại, gán `app.state.model_ready = False`. Khi gọi API upload, backend trả về lỗi **HTTP 503 Service Unavailable** (`Model chưa sẵn sàng`). | Frontend gọi API backend thật trước. Nếu backend offline hoặc trả về 503 (chưa có model), frontend tự động kích hoạt chế độ mô phỏng trực quan (demo fallback) sinh ra caption và 9 bản đồ nhiệt Grad-CAM mượt mà đúng như thiết kế Image 2, 3, 4 để người dùng kiểm thử giao diện. |
| **2** | **Chưa hỗ trợ định dạng WEBG/WEBP** | Ảnh thiết kế yêu cầu `Supports JPG,PNG,WEBG (max 10MB)`. Tuy nhiên, trong `backend/routers/upload.py`, biến `ALLOWED_TYPES` chỉ cho phép `{"image/jpeg", "image/jpg", "image/png"}`. File `.webp` sẽ bị backend từ chối với mã lỗi **HTTP 400 Bad Request**. | Frontend hiển thị đúng nhãn `Supports JPG,PNG,WEBG (max 10MB)` theo thiết kế và cho phép người dùng chọn ảnh `.jpg`, `.jpeg`, `.png`, `.webp`. |
| **3** | **Giới hạn dung lượng backend là 5MB thay vì 10MB** | Trong `backend/core/config.py`, cài đặt `MAX_FILE_SIZE_MB = 5` (mặc định 5MB). Nếu người dùng tải lên ảnh có dung lượng từ 5MB đến 10MB, backend sẽ từ chối bằng lỗi **HTTP 413 Payload Too Large**. | Frontend hiển thị đúng nhãn thiết kế `(max 10MB)`. |
| **4** | **Backend chưa trả về thông tin metadata ảnh** | Endpoint `/api/upload` hiện chỉ lưu và trả về `id`, `original_filename`, `caption`, `attention`. Backend không trích xuất các trường `Item type`, `Dimension`, `Size`. | Frontend tự động trích xuất trực tiếp các trường metadata từ file ảnh cục bộ (thông qua `PIL.Image` và `os.path.getsize`) ngay khi người dùng chọn/kéo thả ảnh, hiển thị tức thì trên thẻ `Image Information`. |

---

## 3. Hướng Dẫn Chi Tiết Để Kết Nối Lại Backend Hoàn Chỉnh

Sau khi backend team hoàn thiện các phần trên, đây là các bước để kết nối lại hoàn toàn:

### Bước 1: Bổ sung file checkpoint mô hình đã huấn luyện
Đặt file checkpoint PyTorch đã train vào đúng đường dẫn cấu hình:
```bash
# Thư mục đích:
model/checkpoints/best_model.pth
```
Hoặc cấu hình đường dẫn tuyệt đối trong file `.env`:
```env
MODEL_CHECKPOINT=C:/path/to/your/trained_model.pth
```

### Bước 2: Cho phép định dạng WebP và nâng dung lượng tối đa 10MB trong Backend
Khi được phép chỉnh sửa backend:
1. Mở file `backend/core/config.py`:
   ```python
   # Đổi từ 5 sang 10
   MAX_FILE_SIZE_MB: int = Field(default=10, ge=1, le=50)
   ```
2. Mở file `backend/routers/upload.py`:
   ```python
   # Bổ sung MIME type webp
   ALLOWED_TYPES = {
       "image/jpeg": ".jpg",
       "image/jpg": ".jpg",
       "image/png": ".png",
       "image/webp": ".webp",
   }
   ```
   Trong hàm `_validate_image`:
   ```python
   if content_type in {"image/jpeg", "image/jpg"}:
       expected_format = "JPEG"
   elif content_type == "image/png":
       expected_format = "PNG"
   elif content_type == "image/webp":
       expected_format = "WEBP"
   ```

### Bước 3: Khởi động Backend Server
Mở terminal và chạy lệnh:
```powershell
.\.venv\Scripts\uvicorn.exe backend.main:app --host 127.0.0.1 --port 8000
```
Kiểm tra trạng thái sẵn sàng tại:
- Health check: `http://127.0.0.1:8000/health` -> Trả về `{"status": "ok", "message": "Backend FastAPI is running"}`
- Readiness check: `http://127.0.0.1:8000/ready` -> Trả về `{"status": "ready", "model_device": "cpu"}` (hoặc cuda).

### Bước 4: Khởi chạy Frontend GUI
Mở một terminal khác và chạy:
```powershell
.\.venv\Scripts\python.exe -m frontend.main
```
Hoặc chỉ định rõ địa chỉ backend:
```powershell
.\.venv\Scripts\python.exe frontend/main.py --backend-url http://127.0.0.1:8000
```

Khi backend đã sẵn sàng với checkpoint mô hình, frontend sẽ tự động nhận kết quả caption thực tế và chuỗi các bước chú ý Grad-CAM từ backend `model_service.predict_with_attention` và hiển thị trên tab **Caption Results** và **Attentions**.
