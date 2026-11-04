import asyncio
from io import BytesIO
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import create_engine, inspect, text

import backend.main as main
from backend.core import config as backend_config
from backend.core.database import (
    SessionLocal,
    engine as database_engine,
    initialize_database,
)
from backend.routers import upload as upload_router
from backend.services.model_service import CaptionModelService, caption_model_service
from src.beam_search import beam_search


def _result(caption="a test caption"):
    word = caption.split()[0]
    tokens = [{
        "word": word,
        "image": "data:image/jpeg;base64,",
        "next_tokens": [
            {"word": word, "probability": 0.8, "selected": True},
            {"word": "other", "probability": 0.1, "selected": False},
        ],
    }]
    return {
        "caption": caption,
        "candidates": [{"caption": caption, "beam_score": -0.3, "tokens": tokens}],
        "attention": {"method": "vit_cross_attention", "steps": tokens},
        "metadata": {"model": "test model", "device": "cpu", "beam_size": 4},
    }


def _image_bytes():
    image = BytesIO()
    Image.new("RGB", (32, 32), color=(80, 120, 160)).save(image, format="PNG")
    return image.getvalue()


@pytest.fixture
def client(tmp_path, monkeypatch):
    test_engine = create_engine(
        f"sqlite:///{(tmp_path / 'captions.db').as_posix()}",
        connect_args={"check_same_thread": False},
    )
    monkeypatch.setattr(main, "engine", test_engine)
    monkeypatch.setattr(upload_router, "UPLOAD_DIR", tmp_path / "temporary")
    monkeypatch.setattr(upload_router, "HISTORY_IMAGE_DIR", tmp_path / "history_images")
    SessionLocal.configure(bind=test_engine)
    monkeypatch.setattr(caption_model_service, "load", lambda: None)
    monkeypatch.setattr(caption_model_service, "predict_detailed", lambda _path, _attention: _result())

    try:
        with TestClient(main.app) as test_client:
            yield test_client
    finally:
        SessionLocal.configure(bind=database_engine)
        test_engine.dispose()


def test_upload_saves_full_result_and_image(client, tmp_path):
    response = client.post(
        "/api/upload",
        files={"file": ("test_image.png", _image_bytes(), "image/png")},
        data={"show_attention": "true"},
    )

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["original_filename"] == "test_image.png"
    assert data["caption"] == "a test caption"
    assert data["candidates"][0]["beam_score"] == -0.3
    assert data["candidates"][0]["tokens"][0]["next_tokens"][0]["probability"] == 0.8

    listing = client.get("/api/history").json()["data"]
    assert len(listing) == 1
    assert listing[0]["thumbnail"].startswith("data:image/jpeg;base64,")
    detail = client.get(f"/api/history/{data['id']}").json()["data"]
    assert detail["candidates"] == data["candidates"]
    image_response = client.get(f"/api/history/{data['id']}/image")
    assert image_response.status_code == 200
    assert Image.open(BytesIO(image_response.content)).format == "PNG"


def test_regenerate_updates_existing_history_record(client, monkeypatch):
    response = client.post(
        "/api/upload",
        files={"file": ("test_image.png", _image_bytes(), "image/png")},
    )
    record_id = response.json()["data"]["id"]
    monkeypatch.setattr(
        caption_model_service,
        "predict_detailed",
        lambda _path, _attention: _result("new caption"),
    )

    regenerated = client.post(f"/api/history/{record_id}/regenerate")
    assert regenerated.status_code == 200
    assert regenerated.json()["data"]["id"] == record_id
    assert regenerated.json()["data"]["caption"] == "new caption"
    assert len(client.get("/api/history").json()["data"]) == 1


def test_upload_invalid_file_type(client):
    response = client.post(
        "/api/upload",
        files={"file": ("document.txt", b"This is a text file", "text/plain")},
    )
    assert response.status_code == 400


def test_history_keeps_latest_one_hundred_records(client, tmp_path):
    history_dir = upload_router.HISTORY_IMAGE_DIR
    history_dir.mkdir(parents=True, exist_ok=True)
    first_image = None
    for index in range(101):
        image_name = f"{index}.png"
        image_path = history_dir / image_name
        Image.new("RGB", (2, 2), color=(index % 255, 20, 30)).save(image_path)
        if index == 0:
            first_image = image_path
        upload_router._save_record(f"{index}.png", image_name, _result(f"caption {index}"))

    records = client.get("/api/history?limit=100").json()["data"]
    assert len(records) == 100
    assert records[0]["original_filename"] == "100.png"
    assert first_image is not None and not first_image.exists()


def test_migration_adds_history_columns_without_dropping_rows(tmp_path):
    legacy_engine = create_engine(f"sqlite:///{(tmp_path / 'legacy.db').as_posix()}")
    try:
        with legacy_engine.begin() as connection:
            connection.execute(text(
                "CREATE TABLE caption_records "
                "(id INTEGER PRIMARY KEY, filename VARCHAR, caption VARCHAR, created_at DATETIME)"
            ))
            connection.execute(text(
                "INSERT INTO caption_records (id, filename, caption) VALUES (1, 'old.jpg', 'old caption')"
            ))
        initialize_database(legacy_engine)
        initialize_database(legacy_engine)
        columns = {column["name"] for column in inspect(legacy_engine).get_columns("caption_records")}
        assert {"image_path", "result_json"}.issubset(columns)
        with legacy_engine.connect() as connection:
            row = connection.execute(text("SELECT filename, caption FROM caption_records WHERE id=1")).one()
        assert row == ("old.jpg", "old caption")
    finally:
        legacy_engine.dispose()


def test_database_initialization_creates_current_schema(tmp_path):
    fresh_engine = create_engine(f"sqlite:///{(tmp_path / 'fresh.db').as_posix()}")
    try:
        initialize_database(fresh_engine)
        columns = {
            column["name"]
            for column in inspect(fresh_engine).get_columns("caption_records")
        }
        assert {"image_path", "result_json"}.issubset(columns)
    finally:
        fresh_engine.dispose()


def test_model_service_uses_only_the_configured_checkpoint(tmp_path, monkeypatch):
    missing_checkpoint = tmp_path / "missing.pth"
    monkeypatch.setattr(backend_config.settings, "MODEL_CHECKPOINT", missing_checkpoint)
    service = CaptionModelService()

    with pytest.raises(FileNotFoundError, match="Configured model checkpoint"):
        service.load()

    assert not hasattr(service, "_load_legacy")


def test_missing_checkpoint_keeps_api_available_in_degraded_mode(tmp_path, monkeypatch):
    test_engine = create_engine(
        f"sqlite:///{(tmp_path / 'degraded.db').as_posix()}",
        connect_args={"check_same_thread": False},
    )
    monkeypatch.setattr(main, "engine", test_engine)
    monkeypatch.setattr(upload_router, "UPLOAD_DIR", tmp_path / "temporary")
    monkeypatch.setattr(upload_router, "HISTORY_IMAGE_DIR", tmp_path / "history_images")

    def fail_model_load():
        raise FileNotFoundError("Configured model checkpoint not found")

    monkeypatch.setattr(caption_model_service, "load", fail_model_load)
    SessionLocal.configure(bind=test_engine)
    try:
        with TestClient(main.app) as client:
            assert client.get("/health").status_code == 200
            assert client.get("/ready").status_code == 503
            upload = client.post(
                "/api/upload",
                files={"file": ("test.png", _image_bytes(), "image/png")},
            )
            assert upload.status_code == 503
    finally:
        SessionLocal.configure(bind=database_engine)
        test_engine.dispose()


def test_inference_cancellation_keeps_semaphore_until_worker_finishes(monkeypatch):
    inference_started = Event()
    finish_inference = Event()

    def blocking_inference(_path, _attention):
        inference_started.set()
        finish_inference.wait(timeout=5)
        return _result()

    monkeypatch.setattr(caption_model_service, "predict_detailed", blocking_inference)

    async def exercise_cancellation():
        semaphore = asyncio.Semaphore(1)
        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(inference_semaphore=semaphore)
            )
        )
        task = asyncio.create_task(
            upload_router._run_inference(request, Path("unused.png"), False)
        )
        try:
            assert await asyncio.to_thread(inference_started.wait, 2)
            task.cancel()
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(asyncio.shield(task), timeout=0.05)
            assert semaphore.locked()
            finish_inference.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not semaphore.locked()
        finally:
            finish_inference.set()

    asyncio.run(exercise_cancellation())


def test_beam_search_returns_ranked_hypotheses_and_token_probabilities():
    class Decoder:
        def forward_step(self, _features, current_tokens, return_alphas=False):
            import torch

            logits = torch.tensor([[0.0, -5.0, 1.0, 5.0, 4.5, 4.0]])
            attention = torch.full((1, 4), 0.25)
            return logits, attention if return_alphas else None

    vocab = SimpleNamespace(
        word2idx={"<pad>": 0, "<start>": 1, "<end>": 2},
    )
    features = __import__("torch").zeros((1, 4, 3))
    candidates = beam_search(
        Decoder(), features, vocab, beam_size=4, max_len=2,
        return_alphas=True, return_nbest=True, nbest=4, token_top_k=6,
    )

    assert len(candidates) == 4
    assert len({tuple(item["token_ids"]) for item in candidates}) == 4
    assert all(item["beam_score"] <= 0 for item in candidates)
    assert all(item["next_token_distributions"] for item in candidates)
    assert all(len(item["alphas"]) == len(item["token_ids"]) for item in candidates)


def test_beam_search_omits_attention_weights_when_not_requested():
    class Decoder:
        def __init__(self):
            self.requested_attention = []

        def forward_step(self, _features, _current_tokens, return_alphas=False):
            import torch

            self.requested_attention.append(return_alphas)
            logits = torch.tensor([[0.0, -5.0, 1.0, 5.0, 4.5, 4.0]])
            return logits, torch.ones((1, 4)) if return_alphas else None

    import torch

    decoder = Decoder()
    vocab = SimpleNamespace(
        word2idx={"<pad>": 0, "<start>": 1, "<end>": 2},
    )
    features = torch.zeros((1, 4, 3))
    candidates = beam_search(
        decoder,
        features,
        vocab,
        beam_size=2,
        max_len=2,
        return_alphas=False,
        return_nbest=True,
        nbest=2,
    )

    assert candidates
    assert decoder.requested_attention and not any(decoder.requested_attention)
    assert all(item["alphas"].numel() == 0 for item in candidates)


def test_decoder_computes_cross_attention_weights_only_when_requested():
    import torch
    from src.decoder import Decoder

    decoder = Decoder(
        vocab_size=6,
        d_model=4,
        nhead=2,
        num_decoder_layers=1,
        dim_feedforward=8,
        dropout=0,
        max_len=8,
    )
    features = torch.randn(1, 4, 4)
    tokens = torch.tensor([[1, 3]])

    _, without_attention = decoder.forward_step(
        features, tokens, return_alphas=False
    )
    _, with_attention = decoder.forward_step(
        features, tokens, return_alphas=True
    )

    assert without_attention is None
    assert with_attention.shape == (1, 4)


def test_detailed_prediction_obeys_beam_size_and_attention_flag(tmp_path, monkeypatch):
    import torch

    from backend.core.config import settings

    calls = []

    def fake_beam_search(_decoder, features, _vocab, **kwargs):
        calls.append(kwargs)
        alpha = (
            torch.full((1, features.size(1)), 1 / features.size(1))
            if kwargs["return_alphas"]
            else torch.empty(0, features.size(1))
        )
        return [
            {
                "token_ids": [3],
                "beam_score": -0.2,
                "next_token_distributions": [[(3, 0.8)]],
                "alphas": alpha,
            }
            for _ in range(kwargs["nbest"])
        ]

    service = CaptionModelService()
    service.model = SimpleNamespace(
        encoder=lambda _image: torch.zeros((1, 4, 3)),
        decoder=object(),
    )
    service.vocab = SimpleNamespace(
        word2idx={"<pad>": 0, "<start>": 1, "<end>": 2},
        idx2word={3: "dog"},
    )
    service.device = torch.device("cpu")
    service.transform = lambda _image: torch.zeros((3, 4, 4))
    service.beam_search = fake_beam_search
    service.model_name = "test model"
    service.supports_attention = True
    service._attention_overlay = lambda _image, _weights: "data:image/jpeg;base64,heat"
    monkeypatch.setattr(settings, "DECODING_METHOD", "beam")
    monkeypatch.setattr(settings, "BEAM_SIZE", 2)

    image_path = tmp_path / "test.png"
    Image.new("RGB", (32, 32)).save(image_path)

    without_attention = service.predict_detailed(str(image_path), include_attention=False)
    assert calls[-1]["beam_size"] == 2
    assert calls[-1]["nbest"] == 2
    assert calls[-1]["return_alphas"] is False
    assert len(without_attention["candidates"]) == 2
    assert without_attention["metadata"]["beam_size"] == 2
    assert without_attention["attention"]["steps"] == []
    assert "image" not in without_attention["candidates"][0]["tokens"][0]

    with_attention = service.predict_detailed(str(image_path), include_attention=True)
    assert calls[-1]["return_alphas"] is True
    assert with_attention["attention"]["steps"][0]["image"].startswith("data:image/")
