"""Client for the FastAPI Image Captioning backend.

Handles:
- Backend health check (/health) and readiness check (/ready)
- Image upload, caption inference, and persistent result retrieval via the API
"""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

import httpx
from PIL import Image, ImageDraw, ImageFilter
import numpy as np


class BackendClient:
    """HTTP client for the FastAPI image captioning backend."""

    _backend_start_lock = threading.Lock()
    _backend_process = None
    _backend_log_handle = None

    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url.rstrip("/")

    def _is_managed_local_url(self) -> bool:
        parsed = urlparse(self.base_url)
        return (
            parsed.scheme == "http"
            and parsed.hostname in {"127.0.0.1", "localhost"}
            and parsed.port == 8000
        )

    def _http_client(self, timeout: float):
        # Localhost requests must bypass system proxies; some proxy settings
        # otherwise turn a refused local connection into a long timeout.
        return httpx.Client(timeout=timeout, trust_env=not self._is_managed_local_url())

    def _ensure_backend(self, timeout: float = 180.0):
        """Start the bundled local API when the default endpoint is not running."""
        try:
            with self._http_client(timeout=1.5) as client:
                response = client.get(f"{self.base_url}/health")
                if response.status_code == 200:
                    return
                raise RuntimeError(f"Backend health check returned HTTP {response.status_code}.")
        except httpx.RequestError as exc:
            if not self._is_managed_local_url():
                raise RuntimeError(f"Cannot reach backend: {exc}") from exc

        if not self._is_managed_local_url():
            raise RuntimeError(f"Backend is not running at {self.base_url}.")

        with BackendClient._backend_start_lock:
            try:
                with self._http_client(timeout=1.5) as client:
                    if client.get(f"{self.base_url}/health").status_code == 200:
                        return
            except httpx.RequestError:
                pass

            process = BackendClient._backend_process
            if process is None or process.poll() is not None:
                project_root = Path(__file__).resolve().parent.parent
                log_path = project_root / "logs" / "backend_autostart.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                command = [
                    sys.executable,
                    "-m", "uvicorn", "backend.main:app",
                    "--host", "127.0.0.1", "--port", "8000", "--workers", "1",
                ]
                creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
                try:
                    BackendClient._backend_log_handle = log_path.open("ab")
                    BackendClient._backend_process = subprocess.Popen(
                        command,
                        cwd=project_root,
                        stdin=subprocess.DEVNULL,
                        stdout=BackendClient._backend_log_handle,
                        stderr=subprocess.STDOUT,
                        creationflags=creation_flags,
                    )
                except OSError as exc:
                    raise RuntimeError(f"Could not start the local backend: {exc}") from exc

            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                process = BackendClient._backend_process
                if process is not None and process.poll() is not None:
                    raise RuntimeError(
                        "The local backend stopped while starting. See logs/backend_autostart.log "
                        "for the startup error."
                    )
                try:
                    with self._http_client(timeout=1.5) as client:
                        if client.get(f"{self.base_url}/health").status_code == 200:
                            return
                except httpx.RequestError:
                    pass
                time.sleep(0.5)
        raise RuntimeError(
            f"The local backend did not start within {timeout / 60:.1f} minutes. "
            "See logs/backend_autostart.log for details."
        )

    def check_health(self) -> Tuple[bool, str]:
        """Check if backend server is responsive."""
        try:
            with self._http_client(timeout=2.0) as client:
                res = client.get(f"{self.base_url}/health")
                if res.status_code == 200:
                    data = res.json()
                    return True, data.get("message", "Backend is running")
                return False, f"HTTP {res.status_code}"
        except Exception as exc:
            return False, str(exc)

    def check_ready(self) -> Tuple[bool, str]:
        """Check if model checkpoint is loaded and ready for inference."""
        try:
            with self._http_client(timeout=2.0) as client:
                res = client.get(f"{self.base_url}/ready")
                if res.status_code == 200:
                    return True, "Model ready"
                data = res.json() if res.headers.get("content-type", "").startswith("application/json") else {}
                reason = data.get("detail", {}).get("reason", f"HTTP {res.status_code}") if isinstance(data.get("detail"), dict) else str(data.get("detail", f"HTTP {res.status_code}"))
                return False, reason
        except Exception as exc:
            return False, str(exc)

    def upload_image(
        self,
        file_path: str,
        show_attention: bool = True,
        timeout: float = 60.0,
    ) -> Dict[str, Any]:
        """Upload an image to the backend and propagate service errors to the UI."""
        p = Path(file_path)
        if not p.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Accurately resolve MIME type (especially on Windows where .webp might not be registered)
        ext = p.suffix.lower()
        if ext in {".webp", ".webg"}:
            mime = "image/webp"
        elif ext == ".png":
            mime = "image/png"
        else:
            mime = "image/jpeg"

        with open(p, "rb") as f:
            files = {"file": (p.name, f.read(), mime)}
        data = {"show_attention": "true" if show_attention else "false"}
        self._ensure_backend(timeout=max(timeout, 180.0))
        with self._http_client(timeout=timeout) as client:
            response = client.post(f"{self.base_url}/api/upload", files=files, data=data)
        self._raise_for_backend_error(response)
        payload = response.json()
        return payload.get("data", payload)

    @staticmethod
    def _raise_for_backend_error(response):
        if response.is_success:
            return
        detail = f"Backend returned HTTP {response.status_code}."
        try:
            body = response.json()
            detail = str(body.get("detail", detail))
        except Exception:
            if response.text:
                detail = response.text
        raise RuntimeError(detail)

    def get_history(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        self._ensure_backend()
        with self._http_client(timeout=15.0) as client:
            response = client.get(f"{self.base_url}/api/history", params={"limit": limit, "offset": offset})
        self._raise_for_backend_error(response)
        return response.json().get("data", [])

    def get_history_item(self, record_id: int) -> Dict[str, Any]:
        self._ensure_backend()
        with self._http_client(timeout=30.0) as client:
            response = client.get(f"{self.base_url}/api/history/{record_id}")
        self._raise_for_backend_error(response)
        return response.json().get("data", {})

    def get_history_image(self, record_id: int) -> bytes:
        self._ensure_backend()
        with self._http_client(timeout=30.0) as client:
            response = client.get(f"{self.base_url}/api/history/{record_id}/image")
        self._raise_for_backend_error(response)
        return response.content

    def regenerate_history_item(self, record_id: int) -> Dict[str, Any]:
        self._ensure_backend(timeout=180.0)
        with self._http_client(timeout=120.0) as client:
            response = client.post(f"{self.base_url}/api/history/{record_id}/regenerate")
        self._raise_for_backend_error(response)
        return response.json().get("data", {})

    @staticmethod
    def _jet_colorize(heatmap: np.ndarray) -> np.ndarray:
        """Convert a normalized [0..1] 2D heatmap into RGB JET colormap."""
        values = np.clip(heatmap, 0.0, 1.0)
        red = np.clip(1.5 - np.abs(4.0 * values - 3.0), 0.0, 1.0)
        green = np.clip(1.5 - np.abs(4.0 * values - 2.0), 0.0, 1.0)
        blue = np.clip(1.5 - np.abs(4.0 * values - 1.0), 0.0, 1.0)
        rgb = np.stack((red, green, blue), axis=-1)
        return (rgb * 255).astype(np.uint8)

    @classmethod
    def generate_mock_result(cls, file_path: str) -> Dict[str, Any]:
        """Generate demo caption and attention overlays matching the design images."""
        p = Path(file_path)
        caption = "A white dog runs through the grass."
        tokens = ["a", "white", "dog", "runs", "through", "the", "grass", "."]

        # Focus points matching the 8 caption tokens
        focus_points = [
            (0.35, 0.40, 70),   # a (general dog background)
            (0.48, 0.38, 90),   # white (body)
            (0.42, 0.36, 100),  # dog (torso + tail hot spot matching Image 3)
            (0.46, 0.58, 85),   # runs (paws)
            (0.48, 0.65, 80),   # through (motion)
            (0.55, 0.70, 75),   # the
            (0.50, 0.80, 110),  # grass (foreground grass)
            (0.50, 0.50, 60),   # .
        ]

        try:
            with Image.open(p) as img:
                base_img = img.convert("RGB")
                orig_w, orig_h = base_img.size
                max_dim = 640
                if max(orig_w, orig_h) > max_dim:
                    scale = max_dim / max(orig_w, orig_h)
                    target_w, target_h = int(orig_w * scale), int(orig_h * scale)
                    resized = base_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
                else:
                    target_w, target_h = orig_w, orig_h
                    resized = base_img.copy()
        except Exception:
            target_w, target_h = 320, 220
            resized = Image.new("RGB", (target_w, target_h), color=(140, 170, 190))

        steps: List[Dict[str, str]] = []
        for i, word in enumerate(tokens):
            fx, fy, r = focus_points[i % len(focus_points)]
            cx = int(fx * target_w)
            cy = int(fy * target_h)

            mask = Image.new("L", (target_w, target_h), 0)
            draw = ImageDraw.Draw(mask)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
            blurred = mask.filter(ImageFilter.GaussianBlur(radius=r * 0.45))

            h_arr = np.array(blurred, dtype=np.float32) / 255.0
            colored_arr = cls._jet_colorize(h_arr)
            colored_img = Image.fromarray(colored_arr)

            overlay = Image.blend(resized, colored_img, alpha=0.50)
            buffer = BytesIO()
            overlay.save(buffer, format="JPEG", quality=85)
            b64_str = base64.b64encode(buffer.getvalue()).decode("ascii")

            steps.append({
                "word": word,
                "image": f"data:image/jpeg;base64,{b64_str}",
            })

        return {
            "id": 1,
            "original_filename": p.name,
            "caption": caption,
            "attention": {
                "method": "vit_cross_attention",
                "steps": steps,
            },
        }
