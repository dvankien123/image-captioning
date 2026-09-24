```python from fastapi import FastAPI

app = FastAPI( title="Image Captioning API", version="1.0.0" )

@app.get("/") def root(): return { "message": "Image Captioning API is running" }

@app.get("/health") def health(): return { "status": "ok" } ```
