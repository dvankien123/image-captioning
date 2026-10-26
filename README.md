# Image Captioning API

FastAPI backend for the ViT-B/16 encoder and Transformer decoder from the
repository's main branch. The original ResNet-LSTM implementation stays in
`model/`; the current model source is isolated under `model_vit/`.

## Setup

From the repository root in PowerShell, create the Python 3.14 environment and install the backend dependencies. The CUDA 12.6 wheels use the NVIDIA GPU when available and fall back to CPU otherwise:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple -r backend\requirements.txt
```

Install the root `requirements.txt` as well if you need the training notebooks.

The GitHub main branch contains the model source and configuration, but does
not include trained weights. Place a checkpoint produced by this ViT-Transformer
model at `model_vit/checkpoints/best_model.pth`. The old file at
`model/checkpoints/best_model.pth` is a ResNet-LSTM checkpoint and is not
compatible. Set `MODEL_CHECKPOINT` in the root `.env` file to use another
compatible checkpoint.

## Run

    .\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

Open http://127.0.0.1:8000/docs to try the API. For a local GPU or CPU server, run one worker so the model is loaded only once:

    .\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --workers 1

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
inference avoids producing and encoding overlays.

Image uploads are validated by actual image format and limited to 20 megapixels by default. Inference runs off the event loop and is serialized by default to keep CPU/GPU memory stable; requests waiting more than 30 seconds receive HTTP 429. The default decoding strategy is beam search with width 3. Tune `DECODING_METHOD` (`beam` or `greedy`), `BEAM_SIZE`, `MAX_FILE_SIZE_MB`, `MAX_IMAGE_PIXELS`, `MAX_CONCURRENT_INFERENCES`, and `INFERENCE_QUEUE_TIMEOUT_SECONDS` in the root `.env` file as needed.

Without a trained checkpoint the API starts in degraded mode. Health remains available, readiness and image uploads return HTTP 503. Inference also requires the packages listed in backend/requirements.txt.
