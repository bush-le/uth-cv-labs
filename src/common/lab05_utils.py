"""Lab 05 utilities shim.

Backward-compatibility shim re-exporting from src.labs.lab05_facenet.visuals and common utilities.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from src.common.config import load_yaml_config
from src.common.device import resolve_device as _resolve_device
from src.common.logger import get_logger
from src.labs.lab05_facenet.visuals import (
    FaceVisualizer,
    FPSMeter,
    RecognitionResult,
)


def resolve_device(device_cfg: str = "auto") -> torch.device:
    """Resolve compute device (returns torch.device)."""
    return torch.device(_resolve_device(device_cfg))


def setup_logger(
    name: str = "lab05",
    level: str = "INFO",
    log_file: str | Path | None = None,
) -> logging.Logger:
    """Setup logger."""
    numeric_level = (
        getattr(logging, level.upper(), logging.INFO)
        if isinstance(level, str)
        else level
    )
    return get_logger(name=name, level=numeric_level, log_file=log_file)


__all__ = [
    "RecognitionResult",
    "load_yaml_config",
    "resolve_device",
    "setup_logger",
    "FPSMeter",
    "FaceVisualizer",
]
