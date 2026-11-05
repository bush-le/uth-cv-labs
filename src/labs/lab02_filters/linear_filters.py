"""Linear spatial filtering, convolution kernels, and frequency properties for Lab 02."""

from __future__ import annotations

import cv2
import numpy as np

from src.labs.lab02_filters.filter_utils import _validate_image, _validate_kernel_size


def get_mean_kernel(ksize: int = 3) -> np.ndarray:
    """Build a normalized box (mean) kernel."""
    _validate_kernel_size(ksize)
    return np.full((ksize, ksize), 1.0 / (ksize * ksize), dtype=np.float32)


def get_gaussian_kernel(ksize: int = 3, sigma: float = 1.0) -> np.ndarray:
    """Build a normalized 2D Gaussian kernel."""
    _validate_kernel_size(ksize)
    if sigma <= 0:
        raise ValueError(f"Sigma must be positive, got {sigma}")
    kernel_1d = cv2.getGaussianKernel(ksize, sigma)
    return (kernel_1d @ kernel_1d.T).astype(np.float32)


def get_sharpen_kernel() -> np.ndarray:
    """Build the classic 3x3 sharpening kernel (Identity - Laplacian)."""
    return np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)


def get_laplacian_kernel() -> np.ndarray:
    """Build the 4-neighbor Laplacian kernel (second derivative)."""
    return np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)


def apply_custom_filter(
    image: np.ndarray, kernel: np.ndarray, ddepth: int = -1
) -> np.ndarray:
    """Apply a custom convolution kernel with cv2.filter2D."""
    _validate_image(image)
    if kernel.ndim != 2:
        raise ValueError(f"Kernel must be 2D, got shape {kernel.shape}")
    return cv2.filter2D(image, ddepth, kernel)


def apply_mean_filter(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Blur an image with a normalized mean (box) kernel."""
    return apply_custom_filter(image, get_mean_kernel(ksize))


def apply_gaussian_filter(
    image: np.ndarray, ksize: int = 3, sigma: float = 1.0
) -> np.ndarray:
    """Blur an image with a Gaussian kernel."""
    return apply_custom_filter(image, get_gaussian_kernel(ksize, sigma))


def apply_sharpen_filter(image: np.ndarray) -> np.ndarray:
    """Sharpen an image with the classic 3x3 kernel."""
    return apply_custom_filter(image, get_sharpen_kernel())


def kernel_sum(kernel: np.ndarray) -> float:
    """Return the sum of kernel coefficients."""
    return float(np.sum(kernel))
