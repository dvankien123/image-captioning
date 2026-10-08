import asyncio
import time
import uuid
from io import BytesIO
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image, UnidentifiedImageError
from sqlalchemy.orm import Session

from backend.core.config import settings
from backend.core.database import SessionLocal, get_db
from backend.core.logger import logger
from backend.models.caption_record import CaptionRecord
from backend.services.model_service import caption_model_service

router = APIRouter()
UPLOAD_DIR = Path(settings.UPLOAD_DIR)
ALLOWED_TYPES = {"image/jpeg": ".jpg", "image/jpg": ".jpg", "image/png": ".png"}
MAX_FILE_SIZE = settings.MAX_FILE_SIZE_MB * 1024 * 1024


def _save_caption(filename: str, caption: str) -> dict:
    with SessionLocal() as db:
        record = CaptionRecord(filename=filename, caption=caption)
        db.add(record)
        db.commit()
        db.refresh(record)
        return {"id": record.id, "original_filename": record.filename, "caption": record.caption}


def _safe_filename(filename: str | None) -> str:
    name = (filename or "upload").replace("\\", "/").rsplit("/", 1)[-1]
    cleaned = "".join(char for char in name if char.isprintable() and char not in "\r\n\0")
    return cleaned[:255] or "upload"


def _validate_image(content: bytes, content_type: str) -> None:
    with Image.open(BytesIO(content)) as image:
        expected_format = "JPEG" if content_type in {"image/jpeg", "image/jpg"} else "PNG"
        if image.format != expected_format:
            raise ValueError("Image format does not match the declared content type.")
        if image.width * image.height > settings.MAX_IMAGE_PIXELS:
            raise OverflowError("Image resolution exceeds the configured limit.")
        image.verify()


@router.post("/upload")
async def upload_image(
    request: Request,
    file: UploadFile = File(...),
):
    started_at = time.perf_counter()

    if file.content_type not in ALLOWED_TYPES:
        await file.close()
        raise HTTPException(status_code=400, detail="Vui lòng upload ảnh JPG hoặc PNG.")
    if not getattr(request.app.state, "model_ready", False):
        await file.close()
        raise HTTPException(status_code=503, detail="Model chưa sẵn sàng. Hãy cấu hình checkpoint đã huấn luyện.")

    if file.size is not None and file.size > MAX_FILE_SIZE:
        await file.close()
        raise HTTPException(status_code=413, detail="Ảnh vượt quá dung lượng cho phép.")

    try:
        content = await file.read(MAX_FILE_SIZE + 1)
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="Ảnh vượt quá dung lượng cho phép.")

        try:
            await run_in_threadpool(_validate_image, content, file.content_type)
        except OverflowError as exc:
            raise HTTPException(status_code=413, detail="Độ phân giải ảnh vượt quá giới hạn cho phép.") from exc
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
            raise HTTPException(status_code=400, detail="File ảnh không hợp lệ hoặc bị hỏng.") from exc

        await run_in_threadpool(UPLOAD_DIR.mkdir, parents=True, exist_ok=True)
        file_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{ALLOWED_TYPES[file.content_type]}"
        await run_in_threadpool(file_path.write_bytes, content)

        semaphore = request.app.state.inference_semaphore
        try:
            await asyncio.wait_for(
                semaphore.acquire(),
                timeout=settings.INFERENCE_QUEUE_TIMEOUT_SECONDS,
            )
        except TimeoutError as exc:
            raise HTTPException(
                status_code=429,
                detail="Model đang bận. Vui lòng thử lại sau.",
                headers={"Retry-After": "5"},
            ) from exc

        try:
            inference = asyncio.create_task(
                run_in_threadpool(caption_model_service.predict, str(file_path))
            )
            try:
                caption = await asyncio.shield(inference)
            except asyncio.CancelledError:
                try:
                    await inference
                finally:
                    raise
        except (OSError, ValueError) as exc:
            logger.info("Invalid image rejected during preprocessing: %s", exc)
            raise HTTPException(status_code=400, detail="Không thể đọc nội dung ảnh.") from exc
        except RuntimeError as exc:
            logger.exception("Model inference failed")
            raise HTTPException(status_code=503, detail="Model hiện chưa thể xử lý ảnh.") from exc
        finally:
            semaphore.release()

        if not caption:
            raise HTTPException(status_code=422, detail="Model không sinh được caption cho ảnh này.")

        result = await run_in_threadpool(
            _save_caption,
            _safe_filename(file.filename),
            caption,
        )
        logger.info(
            "Captioned upload %s in %.2f seconds",
            result["id"],
            time.perf_counter() - started_at,
        )
        return {"status": "success", "data": result}
    except HTTPException:
        raise
    except OSError as exc:
        logger.exception("Upload could not be stored")
        raise HTTPException(status_code=500, detail="Không thể xử lý file upload.") from exc
    except Exception as exc:
        logger.exception("Unexpected upload processing failure")
        raise HTTPException(status_code=500, detail="Lỗi hệ thống khi xử lý ảnh.") from exc
    finally:
        if "file_path" in locals():
            await run_in_threadpool(file_path.unlink, missing_ok=True)
        await file.close()


@router.get("/history")
def get_upload_history(
    limit: int = 10,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    if not 1 <= limit <= settings.HISTORY_MAX_LIMIT:
        raise HTTPException(
            status_code=422,
            detail=f"limit phải nằm trong khoảng 1 đến {settings.HISTORY_MAX_LIMIT}.",
        )
    if offset < 0:
        raise HTTPException(status_code=422, detail="offset không được âm.")
    records = (
        db.query(CaptionRecord)
        .order_by(CaptionRecord.id.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return {"status": "success", "data": records, "limit": limit, "offset": offset}
