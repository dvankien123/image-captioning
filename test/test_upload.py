from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import create_engine

import backend.main as main
from backend.core.database import SessionLocal, engine as database_engine
from backend.services.model_service import caption_model_service


@pytest.fixture
def client(tmp_path, monkeypatch):
    test_engine = create_engine(
        f"sqlite:///{(tmp_path / 'captions.db').as_posix()}",
        connect_args={"check_same_thread": False},
    )
    monkeypatch.setattr(main, "engine", test_engine)
    SessionLocal.configure(bind=test_engine)
    monkeypatch.setattr(caption_model_service, "load", lambda: None)
    monkeypatch.setattr(caption_model_service, "predict", lambda _path: "a test caption")

    try:
        with TestClient(main.app) as test_client:
            yield test_client
    finally:
        SessionLocal.configure(bind=database_engine)
        test_engine.dispose()


def test_upload_valid_image(client):
    image = BytesIO()
    Image.new("RGB", (32, 32), color=(80, 120, 160)).save(image, format="PNG")

    response = client.post(
        "/api/upload",
        files={"file": ("test_image.png", image.getvalue(), "image/png")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["data"]["original_filename"] == "test_image.png"
    assert data["data"]["caption"] == "a test caption"


def test_upload_invalid_file_type(client):
    response = client.post(
        "/api/upload",
        files={"file": ("document.txt", b"This is a text file", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Vui lòng upload ảnh JPG hoặc PNG."