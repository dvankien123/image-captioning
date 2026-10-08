# Image Captioning API

FastAPI backend for the trained image captioning model in the model directory.

## Setup

From the repository root:

    python -m venv .venv
    .venv/Scripts/activate
    pip install -r requirements.txt

Train the model using the instructions in model/README.md. The backend expects the checkpoint at model/checkpoints/best_model.pth. Set MODEL_CHECKPOINT in the root .env file to use another trusted checkpoint.

## Run

    uvicorn backend.main:app --host 127.0.0.1 --port 8000

Open http://127.0.0.1:8000/docs to try the API. For a local GPU or CPU server, run one worker so the model is loaded only once:

    uvicorn backend.main:app --host 127.0.0.1 --port 8000 --workers 1

## API

- GET /health checks whether the API process is alive.
- GET /ready checks whether the model is loaded.
- POST /api/upload accepts a JPG or PNG up to 5 MB and returns its generated caption.
- GET /api/history?limit=10&offset=0 returns saved captions.

Image uploads are validated by actual image format and limited to 20 megapixels by default. Inference runs off the event loop and is serialized by default to keep CPU/GPU memory stable; requests waiting more than 30 seconds receive HTTP 429. Tune MAX_FILE_SIZE_MB, MAX_IMAGE_PIXELS, MAX_CONCURRENT_INFERENCES, and INFERENCE_QUEUE_TIMEOUT_SECONDS in the root .env file as needed.

Without a trained checkpoint the API starts in degraded mode. Health remains available, readiness and image uploads return HTTP 503. Inference also requires the packages listed in backend/requirements.txt.
