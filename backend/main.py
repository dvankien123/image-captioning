import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool

from backend.core.config import settings
from backend.core.database import Base, engine
from backend.core.logger import logger
from backend.models import caption_record
from backend.routers import upload
from backend.services.model_service import caption_model_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    app.state.inference_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_INFERENCES)
    app.state.model_ready = False
    try:
        await run_in_threadpool(caption_model_service.load)
        app.state.model_ready = True
        app.state.model_error = None
        logger.info("Caption model loaded on %s", caption_model_service.device)
    except Exception as exc:  # Keep health/history available if model files are missing.
        logger.exception("Caption model could not be loaded; API will run in degraded mode")
        app.state.model_error = str(exc)
    try:
        yield
    finally:
        caption_model_service.unload()
        engine.dispose()


app = FastAPI(title=settings.PROJECT_NAME, lifespan=lifespan)
origins = [settings.CORS_ORIGINS] if isinstance(settings.CORS_ORIGINS, str) else settings.CORS_ORIGINS

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(upload.router, prefix="/api", tags=["Image Processing"])


@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Backend FastAPI is running"}


@app.get("/ready")
async def readiness_check(request: Request):
    if not getattr(request.app.state, "model_ready", False):
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "reason": "Model checkpoint or model dependencies are unavailable.",
            },
        )
    return {"status": "ready", "model_device": str(caption_model_service.device)}
