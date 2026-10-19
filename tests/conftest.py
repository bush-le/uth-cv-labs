"""Global pytest fixtures for Computer Vision test suites."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def dummy_bgr_image() -> np.ndarray:
    """Fixture returning a deterministic 100x120 3-channel BGR test image.

    Features two distinct color blocks (top half pure Blue, bottom half pure Red)
    useful for testing spatial filters, color transforms, and edge operations.
    """
    img = np.zeros((100, 120, 3), dtype=np.uint8)
    img[:50, :] = [255, 0, 0]  # Blue in BGR
    img[50:, :] = [0, 0, 255]  # Red in BGR
    return img


@pytest.fixture
def dummy_pil_image() -> Image.Image:
    """Fixture returning a deterministic 120x100 RGB Pillow Image."""
    arr = np.zeros((100, 120, 3), dtype=np.uint8)
    arr[:50, :] = [0, 0, 255]  # Blue in RGB
    arr[50:, :] = [255, 0, 0]  # Red in RGB
    return Image.fromarray(arr, mode="RGB")
