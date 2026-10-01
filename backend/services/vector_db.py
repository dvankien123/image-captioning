import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from backend.core.logger import logger

# Import cấu hình để lấy biến môi trường
from backend.core.config import settings

# Khởi tạo mô hình nhúng kèm theo API Key từ file .env
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=settings.GOOGLE_API_KEY
)

# Đường dẫn lưu trữ FAISS cục bộ
FAISS_INDEX_PATH = "faiss_index"

def get_vector_store():
    """Load database FAISS từ ổ cứng, nếu chưa có thì tạo mới"""
    if os.path.exists(FAISS_INDEX_PATH):
        return FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    
    dummy_doc = Document(page_content="init", metadata={"id": 0})
    vector_store = FAISS.from_documents([dummy_doc], embeddings)
    vector_store.delete([vector_store.index_to_docstore_id[0]]) 
    return vector_store

def add_caption_to_vector_db(record_id: int, filename: str, caption: str):
    """Chuyển caption thành vector và lưu vào FAISS"""
    try:
        vector_store = get_vector_store()
        doc = Document(
            page_content=caption,
            metadata={"record_id": record_id, "filename": filename}
        )
        vector_store.add_documents([doc])
        vector_store.save_local(FAISS_INDEX_PATH)
        logger.info(f"Đã lưu vector embedding cho ảnh ID: {record_id}")
    except Exception as e:
        logger.error(f"Lỗi khi lưu vector: {e}")

def search_similar_images(query: str, top_k: int = 3):
    """Tìm kiếm ảnh có nội dung tương đồng với câu truy vấn"""
    vector_store = get_vector_store()
    results = vector_store.similarity_search(query, k=top_k)
    
    return [
        {
            "record_id": doc.metadata.get("record_id"),
            "filename": doc.metadata.get("filename"),
            "matched_caption": doc.page_content
        }
        for doc in results
    ]