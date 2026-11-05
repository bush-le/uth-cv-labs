"""First and second derivative spatial filters for edge detection in Lab 02."""

from __future__ import annotations

import cv2
import numpy as np

from src.labs.lab02_filters.filter_utils import _validate_image


def get_sobel_kernels() -> tuple[np.ndarray, np.ndarray]:
    """Return the Sobel kernels for x and y directions."""
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    return sobel_x, sobel_y


def get_prewitt_kernels() -> tuple[np.ndarray, np.ndarray]:
    """Return the Prewitt kernels for x and y directions."""
    prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    return prewitt_x, prewitt_y


def gradient_magnitude(grad_x: np.ndarray, grad_y: np.ndarray) -> np.ndarray:
    """Combine x and y gradients into a magnitude map: G = sqrt(Gx^2 + Gy^2)."""
    if grad_x.shape != grad_y.shape:
        raise ValueError(
            f"Gradient shapes must match, got {grad_x.shape} and {grad_y.shape}"
        )
    magnitude = np.sqrt(
        grad_x.astype(np.float64) ** 2 + grad_y.astype(np.float64) ** 2
    )
    return cv2.convertScaleAbs(magnitude)


def apply_sobel(
    image: np.ndarray, direction: str = "both", ksize: int = 3
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the first derivative with a Sobel operator.

    If direction is 'x' or 'y', returns single gradient array.
    If direction is 'both', returns (grad_x, grad_y, magnitude).
    """
    _validate_image(image)
    if direction not in {"x", "y", "both"}:
        raise ValueError(f"Direction must be 'x', 'y', or 'both', got {direction!r}")

    if direction == "x":
        gradient = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
        return cv2.convertScaleAbs(gradient)
    if direction == "y":
        gradient = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)
        return cv2.convertScaleAbs(gradient)

    gx = cv2.convertScaleAbs(cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize))
    gy = cv2.convertScaleAbs(cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize))
    mag = gradient_magnitude(gx, gy)
    return gx, gy, mag


def apply_prewitt(
    image: np.ndarray, direction: str = "both"
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the first derivative with a Prewitt kernel via filter2D.

    If direction is 'x' or 'y', returns single gradient array.
    If direction is 'both', returns (grad_x, grad_y, magnitude).
    """
    _validate_image(image)
    if direction not in {"x", "y", "both"}:
        raise ValueError(f"Direction must be 'x', 'y', or 'both', got {direction!r}")

    prewitt_x, prewitt_y = get_prewitt_kernels()
    if direction == "x":
        gradient = cv2.filter2D(image.astype(np.float32), cv2.CV_32F, prewitt_x)
        return cv2.convertScaleAbs(gradient)
    if direction == "y":
        gradient = cv2.filter2D(image.astype(np.float32), cv2.CV_32F, prewitt_y)
        return cv2.convertScaleAbs(gradient)

    gx = cv2.convertScaleAbs(cv2.filter2D(image.astype(np.float32), cv2.CV_32F, prewitt_x))
    gy = cv2.convertScaleAbs(cv2.filter2D(image.astype(np.float32), cv2.CV_32F, prewitt_y))
    mag = gradient_magnitude(gx, gy)
    return gx, gy, mag


def apply_laplacian(image: np.ndarray) -> np.ndarray:
    """Compute the second derivative with a Laplacian operator."""
    _validate_image(image)
    response = cv2.Laplacian(image, cv2.CV_64F)
    return cv2.convertScaleAbs(response)
