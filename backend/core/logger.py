import logging
import sys
import os

# Tạo thư mục chứa file log nếu chưa tồn tại (nằm ở gốc dự án)
os.makedirs("logs", exist_ok=True)

# Định nghĩa cấu trúc của một dòng log: [Thời gian] - [Tên] - [Mức độ] - [Thông báo]
formatter = logging.Formatter(
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# 1. Handler in log ra màn hình Terminal
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)

# 2. Handler ghi log vào file văn bản
file_handler = logging.FileHandler("logs/app.log", encoding="utf-8")
file_handler.setFormatter(formatter)

# Khởi tạo Logger chính cho toàn bộ ứng dụng
logger = logging.getLogger("CaptionAPI")
logger.setLevel(logging.INFO)  # Ghi nhận từ mức INFO trở lên (INFO, WARNING, ERROR)
logger.addHandler(console_handler)
logger.addHandler(file_handler)