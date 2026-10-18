import logging
import sys
from pathlib import Path

from backend.core.config import PROJECT_ROOT

logger = logging.getLogger("CaptionAPI")
logger.setLevel(logging.INFO)

if not logger.handlers:
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    for handler in (logging.StreamHandler(sys.stdout),
                    logging.FileHandler(Path(log_dir) / "app.log", encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
