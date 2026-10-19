"""Point operations (pixel-wise intensity mappings) for Computer Vision."""

from __future__ import annotations

import cv2
import numpy as np

from src.data.filter_utils import _validate_image


def adjust_brightness(image: np.ndarray, beta: float) -> np.ndarray:
    """Adjust image brightness by adding a constant to every pixel.

    Formula: g(x, y) = f(x, y) + beta, with saturation clipping to [0, 255].

    Args:
        image: Input image array of shape (H, W, 3) or (H, W), dtype uint8.
        beta: Brightness offset. Positive values brighten, negative darken.

    Returns:
        np.ndarray: Brightness-adjusted image, same shape and dtype uint8.
    """
    _validate_image(image)
    adjusted = image.astype(np.float32) + beta
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def adjust_contrast(image: np.ndarray, alpha: float) -> np.ndarray:
    """Adjust image contrast by multiplying every pixel with a constant.

    Formula: g(x, y) = alpha * f(x, y), with saturation clipping to [0, 255].

    Args:
        image: Input image array of shape (H, W, 3) or (H, W), dtype uint8.
        alpha: Contrast gain. alpha > 1 increases contrast,
            0 < alpha < 1 washes the image out.

    Returns:
        np.ndarray: Contrast-adjusted image, same shape and dtype uint8.

    Raises:
        ValueError: If alpha is non-positive.
    """
    if alpha <= 0:
        raise ValueError(f"Contrast gain alpha must be positive, got {alpha}")
    _validate_image(image)
    adjusted = image.astype(np.float32) * alpha
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def negative_transform(image: np.ndarray) -> np.ndarray:
    """Compute the photographic negative of an 8-bit image.

    Formula: g(x, y) = 255 - f(x, y).

    Args:
        image: Input image array of shape (H, W, 3) or (H, W), dtype uint8.

    Returns:
        np.ndarray: Negative image, same shape and dtype uint8.
    """
    _validate_image(image)
    return (255 - image).astype(np.uint8)


def apply_threshold(
    image: np.ndarray, threshold: int, max_value: int = 255
) -> np.ndarray:
    """Binarize an image with a fixed global threshold.

    Formula: g(x, y) = max_value if f(x, y) > threshold, else 0.

    Args:
        image: Input image array of shape (H, W, 3) or (H, W), dtype uint8.
        threshold: Cutoff intensity in [0, 255].
        max_value: Value assigned to foreground pixels (default 255).

    Returns:
        np.ndarray: Binary image, same shape and dtype uint8.

    Raises:
        ValueError: If threshold is outside [0, 255].
    """
    if not (0 <= threshold <= 255):
        raise ValueError(f"Threshold must be in [0, 255], got {threshold}")
    _validate_image(image)
    _, binary = cv2.threshold(image, threshold, max_value, cv2.THRESH_BINARY)
    return binary
