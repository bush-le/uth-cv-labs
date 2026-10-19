"""First and second derivative spatial filters for edge detection."""

from __future__ import annotations

import cv2
import numpy as np

from src.data.filter_utils import _validate_image


def get_sobel_kernels() -> tuple[np.ndarray, np.ndarray]:
    """Return the Sobel kernels for x and y directions.

    Both kernels are first-derivative operators (kernel sum = 0).
    The middle row/column is emphasized (weight 2) for noise robustness.

    Returns:
        tuple[np.ndarray, np.ndarray]: (sobel_x, sobel_y) kernels.
    """
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    return sobel_x, sobel_y


def get_prewitt_kernels() -> tuple[np.ndarray, np.ndarray]:
    """Return the Prewitt kernels for x and y directions.

    Both kernels are first-derivative operators (kernel sum = 0).
    Unlike Sobel, coefficients are uniform (no emphasis on the middle).

    Returns:
        tuple[np.ndarray, np.ndarray]: (prewitt_x, prewitt_y) kernels.
    """
    prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    return prewitt_x, prewitt_y


def apply_sobel(
    image: np.ndarray, direction: str = "x", ksize: int = 3
) -> np.ndarray:
    """Compute the first derivative with a Sobel operator.

    Args:
        image: Input image array of shape (H, W, 3) or (H, W), dtype uint8.
        direction: Gradient direction, "x" (vertical edges) or "y"
            (horizontal edges).
        ksize: Sobel kernel size (1, 3, 5, or 7).

    Returns:
        np.ndarray: Gradient image rescaled to uint8 via convertScaleAbs.

    Raises:
        ValueError: If direction is not "x" or "y".
    """
    _validate_image(image)
    if direction not in {"x", "y"}:
        raise ValueError(f"Direction must be 'x' or 'y', got {direction!r}")
    dx, dy = (1, 0) if direction == "x" else (0, 1)
    gradient = cv2.Sobel(image, cv2.CV_64F, dx, dy, ksize=ksize)
    return cv2.convertScaleAbs(gradient)


def apply_prewitt(image: np.ndarray, direction: str = "x") -> np.ndarray:
    """Compute the first derivative with a Prewitt kernel via filter2D.

    Args:
        image: Input image array of shape (H, W, 3) or (H, W), dtype uint8.
        direction: Gradient direction, "x" (vertical edges) or "y"
            (horizontal edges).

    Returns:
        np.ndarray: Gradient image rescaled to uint8 via convertScaleAbs.

    Raises:
        ValueError: If direction is not "x" or "y".
    """
    _validate_image(image)
    prewitt_x, prewitt_y = get_prewitt_kernels()
    kernel = prewitt_x if direction == "x" else prewitt_y
    gradient = cv2.filter2D(image.astype(np.float32), cv2.CV_32F, kernel)
    return cv2.convertScaleAbs(gradient)


def apply_laplacian(image: np.ndarray) -> np.ndarray:
    """Compute the second derivative with a Laplacian operator.

    Returns an edge-detection response (kernel sum = 0), not a sharpened
    image. Flat regions map to zero, boundaries produce positive and
    negative lobes with a zero-crossing at the edge.

    Args:
        image: Input image array of shape (H, W, 3) or (H, W), dtype uint8.

    Returns:
        np.ndarray: Laplacian response rescaled to uint8 via convertScaleAbs.
    """
    _validate_image(image)
    response = cv2.Laplacian(image, cv2.CV_64F)
    return cv2.convertScaleAbs(response)


def gradient_magnitude(grad_x: np.ndarray, grad_y: np.ndarray) -> np.ndarray:
    """Combine x and y gradients into a magnitude map: G = sqrt(Gx^2 + Gy^2).

    Args:
        grad_x: Gradient response along x, shape (H, W, C) or (H, W).
        grad_y: Gradient response along y, same shape as grad_x.

    Returns:
        np.ndarray: Magnitude map rescaled to uint8 via convertScaleAbs.

    Raises:
        ValueError: If grad_x and grad_y shapes do not match.
    """
    if grad_x.shape != grad_y.shape:
        raise ValueError(
            f"Gradient shapes must match, got {grad_x.shape} and {grad_y.shape}"
        )
    magnitude = np.sqrt(
        grad_x.astype(np.float64) ** 2 + grad_y.astype(np.float64) ** 2
    )
    return cv2.convertScaleAbs(magnitude)
