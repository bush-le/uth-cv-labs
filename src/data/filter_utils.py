"""Validation and path helpers for spatial filtering operations."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _validate_image(image: np.ndarray) -> None:
    """Validate that the input is a non-empty uint8 image array.

    Args:
        image: Candidate image array.

    Raises:
        ValueError: If the array is invalid (wrong ndim, empty, or dtype).
    """
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Input must be a non-empty numpy array.")
    if image.ndim not in {2, 3}:
        raise ValueError(f"Expected 2D or 3D image array, got shape {image.shape}")
    if image.ndim == 3 and image.shape[2] not in {1, 3, 4}:
        raise ValueError(f"Expected 1, 3 or 4 channels, got shape {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"Expected dtype uint8, got {image.dtype}")


def _validate_kernel_size(ksize: int) -> None:
    """Validate that a kernel size is a positive odd integer.

    Args:
        ksize: Candidate kernel size.

    Raises:
        ValueError: If ksize is not a positive odd integer.
    """
    if not isinstance(ksize, int) or ksize <= 0 or ksize % 2 == 0:
        raise ValueError(f"Kernel size must be a positive odd integer, got {ksize}")


def find_project_root(start: Path | None = None) -> Path:
    """Locate the repository root by searching for pyproject.toml upward.

    Args:
        start: Starting directory (defaults to this file's location).

    Returns:
        Path: Repository root directory.
    """
    current = Path(start or __file__).resolve()
    if current.is_file():
        current = current.parent
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return current
