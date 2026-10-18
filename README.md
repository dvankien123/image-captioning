# Image Captioning API

FastAPI backend for the trained image captioning model in the model directory.

## Setup

From the repository root in PowerShell, create the Python 3.14 environment and install the backend dependencies. The CUDA 12.6 wheels use the NVIDIA GPU when available and fall back to CPU otherwise:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple -r backend\requirements.txt
```

Install the root `requirements.txt` as well if you need the training notebooks.

Train the model using the instructions in model/README.md. The backend expects the checkpoint at model/checkpoints/best_model.pth. Set MODEL_CHECKPOINT in the root .env file to use another trusted checkpoint.

## Run

    .\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

Open http://127.0.0.1:8000/docs to try the API. For a local GPU or CPU server, run one worker so the model is loaded only once:

    .\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --workers 1

## API

- GET /health checks whether the API process is alive.
- GET /ready checks whether the model is loaded.
- POST /api/upload accepts a JPG or PNG up to 5 MB and returns its generated caption. Add the optional multipart field `show_attention=true` to include one token-conditioned Grad-CAM overlay per decoder step. Each attention step contains `word` and a JPEG data URL in `image`; the first step is labelled `<start>`. Attention is computed only when requested, so ordinary caption requests keep the fast inference path.
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
      "method": "gradcam",
      "steps": [
        { "word": "<start>", "image": "data:image/jpeg;base64,..." },
        { "word": "a", "image": "data:image/jpeg;base64,..." }
      ]
    }
  }
}
```

The model uses an LSTM decoder without a learned spatial-attention layer. These overlays are generated post-hoc from ResNet Grad-CAM and do not change the model or checkpoint. Because attention requires gradients and returns an image for each token, it takes more time and bandwidth than the default request.

Image uploads are validated by actual image format and limited to 20 megapixels by default. Inference runs off the event loop and is serialized by default to keep CPU/GPU memory stable; requests waiting more than 30 seconds receive HTTP 429. Tune MAX_FILE_SIZE_MB, MAX_IMAGE_PIXELS, MAX_CONCURRENT_INFERENCES, and INFERENCE_QUEUE_TIMEOUT_SECONDS in the root .env file as needed.

Without a trained checkpoint the API starts in degraded mode. Health remains available, readiness and image uploads return HTTP 503. Inference also requires the packages listed in backend/requirements.txt.
