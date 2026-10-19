"""Linear spatial filtering, convolution kernels, and frequency properties."""

from __future__ import annotations

import cv2
import numpy as np

from src.data.filter_utils import _validate_image, _validate_kernel_size


def get_mean_kernel(ksize: int = 3) -> np.ndarray:
    """Build a normalized box (mean) kernel.

    All coefficients are positive; the kernel sum equals 1, preserving
    the average brightness (DC response gain = 1).

    Args:
        ksize: Odd kernel size (e.g., 3, 5, 7).

    Returns:
        np.ndarray: Normalized mean kernel of shape (ksize, ksize), float32.

    Raises:
        ValueError: If ksize is not a positive odd integer.
    """
    _validate_kernel_size(ksize)
    return np.full((ksize, ksize), 1.0 / (ksize * ksize), dtype=np.float32)


def get_gaussian_kernel(ksize: int = 3, sigma: float = 1.0) -> np.ndarray:
    """Build a normalized 2D Gaussian kernel.

    Coefficients decrease with distance from the center; the kernel sum
    equals 1, preserving the average brightness.

    Args:
        ksize: Odd kernel size (e.g., 3, 5, 7).
        sigma: Standard deviation of the Gaussian distribution. Must be > 0.

    Returns:
        np.ndarray: Normalized Gaussian kernel of shape (ksize, ksize).

    Raises:
        ValueError: If ksize is not a positive odd integer or sigma <= 0.
    """
    _validate_kernel_size(ksize)
    if sigma <= 0:
        raise ValueError(f"Sigma must be positive, got {sigma}")
    kernel_1d = cv2.getGaussianKernel(ksize, sigma)
    return (kernel_1d @ kernel_1d.T).astype(np.float32)


def get_sharpen_kernel() -> np.ndarray:
    """Build the classic 3x3 sharpening kernel (Identity - Laplacian).

    Because Identity sums to 1 and Laplacian sums to 0, the resulting
    kernel sums to 1. The large positive center (+5) with negative
    neighbors (-1) amplifies center-surround differences.

    Returns:
        np.ndarray: Sharpen kernel of shape (3, 3), float32.
    """
    return np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)


def get_laplacian_kernel() -> np.ndarray:
    """Build the 4-neighbor Laplacian kernel (second derivative).

    The kernel measures curvature (rate of slope change) and sums to 0,
    so flat regions map to zero. This is an EDGE DETECTION kernel,
    not a sharpening kernel.

    Returns:
        np.ndarray: Laplacian kernel of shape (3, 3), float32.
    """
    return np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)


def apply_custom_filter(
    image: np.ndarray, kernel: np.ndarray, ddepth: int = -1
) -> np.ndarray:
    """Apply a custom convolution kernel with cv2.filter2D.

    Args:
        image: Input image array of shape (H, W, 3) or (H, W), dtype uint8.
        kernel: Convolution kernel of shape (kH, kW), float32.
        ddepth: Output depth flag for cv2.filter2D (default -1 = same as input).

    Returns:
        np.ndarray: Filtered image, same shape as input.

    Raises:
        ValueError: If kernel is not a 2D array.
    """
    _validate_image(image)
    if kernel.ndim != 2:
        raise ValueError(f"Kernel must be 2D, got shape {kernel.shape}")
    return cv2.filter2D(image, ddepth, kernel)


def apply_mean_filter(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Blur an image with a normalized mean (box) kernel.

    Args:
        image: Input image array of shape (H, W, 3) or (H, W), dtype uint8.
        ksize: Odd kernel size.

    Returns:
        np.ndarray: Blurred image, same shape as input.
    """
    return apply_custom_filter(image, get_mean_kernel(ksize))


def apply_gaussian_filter(
    image: np.ndarray, ksize: int = 3, sigma: float = 1.0
) -> np.ndarray:
    """Blur an image with a Gaussian kernel (weighted averaging).

    Args:
        image: Input image array of shape (H, W, 3) or (H, W), dtype uint8.
        ksize: Odd kernel size.
        sigma: Gaussian standard deviation (> 0).

    Returns:
        np.ndarray: Blurred image, same shape as input.
    """
    return apply_custom_filter(image, get_gaussian_kernel(ksize, sigma))


def apply_sharpen_filter(image: np.ndarray) -> np.ndarray:
    """Sharpen an image with the classic 3x3 kernel (Identity - Laplacian).

    Args:
        image: Input image array of shape (H, W, 3) or (H, W), dtype uint8.

    Returns:
        np.ndarray: Sharpened image, same shape as input.
    """
    return apply_custom_filter(image, get_sharpen_kernel())


def kernel_sum(kernel: np.ndarray) -> float:
    """Return the sum of kernel coefficients (DC gain indicator).

    A sum of 1 preserves average brightness; a sum of 0 removes flat
    regions and keeps only boundaries (derivative behavior).

    Args:
        kernel: Convolution kernel of shape (kH, kW).

    Returns:
        float: Sum of all kernel coefficients.
    """
    return float(np.sum(kernel))
