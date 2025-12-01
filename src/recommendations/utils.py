"""
Utility helpers for the recommendation system.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import json
import logging

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RECOMMENDATIONS_DIR = DATA_DIR / "recommendations"
RECOMMENDATIONS_DIR.mkdir(parents=True, exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def save_json(data: Dict[str, Any], file_path: Path) -> None:
    """Save a dictionary as JSON."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def ensure_file_exists(path: Path) -> None:
    """Raise FileNotFoundError if the file is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Expected file does not exist: {path}")


