from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Đường dẫn tới file cơ sở dữ liệu SQLite (sẽ tự động tạo file caption_history.db ở gốc dự án)
SQLALCHEMY_DATABASE_URL = "sqlite:///./caption_history.db"

# Khởi tạo engine kết nối
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Class gốc để các bảng dữ liệu kế thừa
Base = declarative_base()

# Hàm tạo phiên làm việc (session) với database cho mỗi request
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()