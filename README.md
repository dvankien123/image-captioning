# Image Captioning API

FastAPI backend for the ViT-B/16 encoder and Transformer decoder.
Model source files are located under `src/`, checkpoints are in `checkpoints/`, backend server code is in `backend/`, and dataset/vocabulary files are in `data/`.

## Setup

From the repository root in PowerShell, create the virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The root `requirements.txt` includes backend, frontend, and training dependencies. Choose a PyTorch build compatible with the machine's CPU or CUDA setup.

The backend loads the ViT-Transformer checkpoint at `checkpoints/best_model_coco.pth` by default. Set `MODEL_CHECKPOINT` in the root `.env` file to use a different checkpoint. If the configured file is missing or invalid, the API starts in degraded mode and reports the problem through `/ready`.

Backend settings are read from environment variables and the root `.env` file. The root `config.yaml` configures the training, prediction, and evaluation scripts.

## Run

To start the backend server:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --workers 1
```

To start the frontend UI:

```powershell
.\run_frontend.bat
```

Open http://127.0.0.1:8000/docs to explore the API documentation.

## API

- GET /health checks whether the API process is alive.
- GET /ready checks whether the model is loaded.
- POST /api/upload accepts JPEG, PNG, or WebP images up to 10 MB and returns a generated caption. Add the optional multipart field `show_attention=true` to include the ViT-Transformer's visual cross-attention overlay for each generated word. Each step contains `word` and a JPEG data URL in `image`.
- GET /api/history?limit=10&offset=0 returns saved captions.

Example response when `show_attention=true`:

```json
{
  "status": "success",
  "data": {
    "id": 1,
    "original_filename": "dog.jpg",
    "caption": "a dog is running",
    "attention": {
      "method": "cross_attention",
      "steps": [
        { "word": "a", "image": "data:image/jpeg;base64,..." }
      ]
    }
  }
}
```

Attention overlays use the decoder's learned cross-attention weights over the
ViT image patches. They are generated only when requested; normal caption
inference does not request or encode attention weights.

Image uploads are validated by actual image format and limited to 20 megapixels by default. Inference runs off the event loop and is serialized by default to keep CPU/GPU memory stable; requests waiting more than 30 seconds receive HTTP 429. The default decoding strategy is beam search with width 3 and returns up to four candidates, limited by the configured beam size. Tune `DECODING_METHOD` (`beam` or `greedy`), `BEAM_SIZE`, `MAX_FILE_SIZE_MB`, `MAX_IMAGE_PIXELS`, `MAX_CONCURRENT_INFERENCES`, and `INFERENCE_QUEUE_TIMEOUT_SECONDS` in the root `.env` file as needed. History is capped at 100 records by default.

Without a usable configured checkpoint the API starts in degraded mode. Health and history remain available; readiness and image uploads return HTTP 503. The frontend launcher starts the local backend automatically when it is not already running.
