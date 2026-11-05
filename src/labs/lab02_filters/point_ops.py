"""Point operations (pixel-wise intensity mappings) for Lab 02."""

from __future__ import annotations

import cv2
import numpy as np

from src.labs.lab02_filters.filter_utils import _validate_image


def adjust_brightness(image: np.ndarray, beta: float) -> np.ndarray:
    """Adjust image brightness by adding a constant to every pixel."""
    _validate_image(image)
    adjusted = image.astype(np.float32) + beta
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def adjust_contrast(image: np.ndarray, alpha: float) -> np.ndarray:
    """Adjust image contrast by multiplying every pixel with a constant."""
    if alpha <= 0:
        raise ValueError(f"Contrast gain alpha must be positive, got {alpha}")
    _validate_image(image)
    adjusted = image.astype(np.float32) * alpha
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def negative_transform(image: np.ndarray) -> np.ndarray:
    """Compute the photographic negative of an 8-bit image."""
    _validate_image(image)
    return (255 - image).astype(np.uint8)


def apply_threshold(
    image: np.ndarray, threshold: int, max_value: int = 255
) -> np.ndarray:
    """Binarize an image with a fixed global threshold."""
    if not (0 <= threshold <= 255):
        raise ValueError(f"Threshold must be in [0, 255], got {threshold}")
    _validate_image(image)
    _, binary = cv2.threshold(image, threshold, max_value, cv2.THRESH_BINARY)
    return binary
