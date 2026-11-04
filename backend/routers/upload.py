import asyncio
import base64
import json
import time
import uuid
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError
from sqlalchemy.orm import Session

from backend.core.config import settings
from backend.core.database import HISTORY_IMAGE_DIR, SessionLocal, get_db
from backend.core.logger import logger
from backend.models.caption_record import CaptionRecord
from backend.services.model_service import caption_model_service

router = APIRouter()
UPLOAD_DIR = Path(settings.UPLOAD_DIR)
ALLOWED_TYPES = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}
MAX_FILE_SIZE = settings.MAX_FILE_SIZE_MB * 1024 * 1024
HISTORY_LIMIT = settings.HISTORY_MAX_LIMIT


def _safe_filename(filename: str | None) -> str:
    name = (filename or "upload").replace("\\", "/").rsplit("/", 1)[-1]
    cleaned = "".join(char for char in name if char.isprintable() and char not in "\r\n\0")
    return cleaned[:255] or "upload"


def _validate_image(content: bytes, content_type: str) -> None:
    with Image.open(BytesIO(content)) as image:
        expected_format = {
            "image/jpeg": "JPEG",
            "image/jpg": "JPEG",
            "image/png": "PNG",
            "image/webp": "WEBP",
        }[content_type]
        if image.format != expected_format:
            raise ValueError("Image format does not match the declared content type.")
        if image.width * image.height > settings.MAX_IMAGE_PIXELS:
            raise OverflowError("Image resolution exceeds the configured limit.")
        image.verify()


def _result_dict(record: CaptionRecord) -> dict:
    fallback = {
        "caption": record.caption,
        "candidates": [],
        "attention": {"method": "unavailable", "steps": []},
    }
    if not record.result_json:
        return fallback
    try:
        return json.loads(record.result_json)
    except (TypeError, json.JSONDecodeError):
        return fallback


def _record_payload(record: CaptionRecord, include_result: bool = False) -> dict:
    payload = {
        "id": record.id,
        "original_filename": record.filename,
        "caption": record.caption,
        "created_at": record.created_at.isoformat() if record.created_at else None,
        "has_image": bool(record.image_path),
    }
    if include_result:
        return {**_result_dict(record), **payload}
    return payload


def _make_thumbnail(image_path: Path) -> str | None:
    try:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image.thumbnail((72, 72))
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=72)
        return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    except (OSError, UnidentifiedImageError):
        return None


def _get_record(db: Session, record_id: int) -> CaptionRecord:
    record = db.query(CaptionRecord).filter(CaptionRecord.id == record_id).first()
    if record is None:
        raise HTTPException(status_code=404, detail="History item not found.")
    return record


def _save_record(filename: str, image_name: str, result: dict) -> dict:
    with SessionLocal() as db:
        record = CaptionRecord(
            filename=filename,
            caption=result["caption"],
            image_path=image_name,
            result_json=json.dumps(result, ensure_ascii=False),
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        payload = _record_payload(record, include_result=True)

        old_records = (
            db.query(CaptionRecord)
            .order_by(CaptionRecord.id.desc())
            .offset(HISTORY_LIMIT)
            .all()
        )
        for old_record in old_records:
            if old_record.image_path:
                (HISTORY_IMAGE_DIR / Path(old_record.image_path).name).unlink(missing_ok=True)
            db.delete(old_record)
        if old_records:
            db.commit()
        return payload


async def _run_inference(
    request: Request, image_path: Path, show_attention: bool
) -> dict:
    semaphore = request.app.state.inference_semaphore
    try:
        await asyncio.wait_for(
            semaphore.acquire(), timeout=settings.INFERENCE_QUEUE_TIMEOUT_SECONDS
        )
    except TimeoutError as exc:
        raise HTTPException(
            status_code=429,
            detail="Model is busy. Please retry.",
            headers={"Retry-After": "5"},
        ) from exc

    try:
        inference = asyncio.create_task(
            run_in_threadpool(
                caption_model_service.predict_detailed,
                str(image_path),
                show_attention,
            )
        )
        try:
            return await asyncio.shield(inference)
        except asyncio.CancelledError:
            while not inference.done():
                try:
                    await asyncio.shield(inference)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            if inference.done() and not inference.cancelled():
                error = inference.exception()
                if error is not None:
                    logger.error(
                        "Inference failed after its request was cancelled",
                        exc_info=(type(error), error, error.__traceback__),
                    )
            raise
    finally:
        semaphore.release()


@router.post("/upload")
async def upload_image(
    request: Request,
    file: UploadFile = File(...),
    show_attention: bool = Form(default=False),
):
    started_at = time.perf_counter()
    if file.content_type not in ALLOWED_TYPES:
        await file.close()
        raise HTTPException(status_code=400, detail="Upload a JPG, PNG, or WebP image.")
    if not getattr(request.app.state, "model_ready", False):
        await file.close()
        raise HTTPException(status_code=503, detail="Caption model is not ready.")
    if file.size is not None and file.size > MAX_FILE_SIZE:
        await file.close()
        raise HTTPException(status_code=413, detail="Image exceeds the allowed file size.")

    temporary_path = None
    history_image_path = None
    try:
        content = await file.read(MAX_FILE_SIZE + 1)
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="Image exceeds the allowed file size.")
        try:
            await run_in_threadpool(_validate_image, content, file.content_type)
        except OverflowError as exc:
            raise HTTPException(status_code=413, detail="Image resolution exceeds the configured limit.") from exc
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
            raise HTTPException(status_code=400, detail="Image file is invalid or corrupted.") from exc

        await run_in_threadpool(UPLOAD_DIR.mkdir, parents=True, exist_ok=True)
        temporary_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{ALLOWED_TYPES[file.content_type]}"
        await run_in_threadpool(temporary_path.write_bytes, content)

        try:
            result = await _run_inference(request, temporary_path, show_attention)
        except (OSError, ValueError) as exc:
            logger.info("Invalid image rejected during preprocessing: %s", exc)
            raise HTTPException(status_code=400, detail="Could not read image contents.") from exc
        except RuntimeError as exc:
            logger.exception("Model inference failed")
            raise HTTPException(status_code=503, detail="Model could not process the image.") from exc

        if not result.get("caption"):
            raise HTTPException(status_code=422, detail="Model did not generate a caption.")
        result.setdefault("metadata", {})["elapsed_ms"] = round((time.perf_counter() - started_at) * 1000, 1)

        await run_in_threadpool(HISTORY_IMAGE_DIR.mkdir, parents=True, exist_ok=True)
        image_name = f"{uuid.uuid4().hex}{ALLOWED_TYPES[file.content_type]}"
        history_image_path = HISTORY_IMAGE_DIR / image_name
        await run_in_threadpool(history_image_path.write_bytes, content)
        record = await run_in_threadpool(
            _save_record, _safe_filename(file.filename), image_name, result
        )
        history_image_path = None
        logger.info("Captioned upload %s in %.2f seconds", record["id"], time.perf_counter() - started_at)
        return {"status": "success", "data": record}
    except HTTPException:
        raise
    except OSError as exc:
        logger.exception("Upload could not be stored")
        raise HTTPException(status_code=500, detail="Could not store uploaded image.") from exc
    except Exception as exc:
        logger.exception("Unexpected upload processing failure")
        raise HTTPException(status_code=500, detail="Unexpected image processing failure.") from exc
    finally:
        if temporary_path is not None:
            await run_in_threadpool(temporary_path.unlink, missing_ok=True)
        if history_image_path is not None:
            await run_in_threadpool(history_image_path.unlink, missing_ok=True)
        await file.close()


@router.get("/history")
def get_upload_history(limit: int = 100, offset: int = 0, db: Session = Depends(get_db)):
    if not 1 <= limit <= settings.HISTORY_MAX_LIMIT:
        raise HTTPException(
            status_code=422,
            detail=f"limit must be between 1 and {settings.HISTORY_MAX_LIMIT}.",
        )
    if offset < 0:
        raise HTTPException(status_code=422, detail="offset must not be negative.")
    records = (
        db.query(CaptionRecord)
        .order_by(CaptionRecord.id.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    items = []
    for record in records:
        item = _record_payload(record)
        image_path = HISTORY_IMAGE_DIR / Path(record.image_path).name if record.image_path else None
        item["thumbnail"] = _make_thumbnail(image_path) if image_path and image_path.is_file() else None
        items.append(item)
    return {"status": "success", "data": items, "limit": limit, "offset": offset}


@router.get("/history/{record_id}")
def get_history_item(record_id: int, db: Session = Depends(get_db)):
    return {"status": "success", "data": _record_payload(_get_record(db, record_id), include_result=True)}


@router.get("/history/{record_id}/image")
def get_history_image(record_id: int, db: Session = Depends(get_db)):
    record = _get_record(db, record_id)
    if not record.image_path:
        raise HTTPException(status_code=404, detail="Image is not available for this history item.")
    image_path = HISTORY_IMAGE_DIR / Path(record.image_path).name
    if not image_path.is_file():
        raise HTTPException(status_code=404, detail="Stored image is missing.")
    return FileResponse(image_path)


@router.post("/history/{record_id}/regenerate")
async def regenerate_history_item(record_id: int, request: Request, db: Session = Depends(get_db)):
    started_at = time.perf_counter()
    record = _get_record(db, record_id)
    if not getattr(request.app.state, "model_ready", False):
        raise HTTPException(status_code=503, detail="Caption model is not ready.")
    if not record.image_path:
        raise HTTPException(status_code=404, detail="Image is not available for this history item.")
    image_path = HISTORY_IMAGE_DIR / Path(record.image_path).name
    if not image_path.is_file():
        raise HTTPException(status_code=404, detail="Stored image is missing.")

    try:
        result = await _run_inference(request, image_path, True)
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail="Model could not process this image.") from exc

    result.setdefault("metadata", {})["elapsed_ms"] = round((time.perf_counter() - started_at) * 1000, 1)
    record.caption = result["caption"]
    record.result_json = json.dumps(result, ensure_ascii=False)
    record.created_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(record)
    return {"status": "success", "data": _record_payload(record, include_result=True)}
