"""Standard library wrappers and baseline comparison edge detectors."""

from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import canny as skimage_canny


def run_opencv_canny(
    image: np.ndarray,
    low_threshold: float = 50.0,
    high_threshold: float = 150.0,
    aperture_size: int = 3,
    l2_gradient: bool = False,
) -> np.ndarray:
    """Execute OpenCV Canny edge detector.

    Args:
        image: Grayscale uint8 image of shape (H, W).
        low_threshold: Lower hysteresis threshold.
        high_threshold: Upper hysteresis threshold.
        aperture_size: Sobel aperture size (3, 5, or 7).
        l2_gradient: If True, uses L2 norm sqrt(Gx^2 + Gy^2); else |Gx| + |Gy|.

    Returns:
        np.ndarray: Binary edge map of shape (H, W), dtype uint8 with {0, 255}.
    """
    if image.dtype != np.uint8:
        norm_img = np.clip(image, 0, 255).astype(np.uint8)
    else:
        norm_img = image

    edges = cv2.Canny(
        norm_img,
        threshold1=low_threshold,
        threshold2=high_threshold,
        apertureSize=aperture_size,
        L2gradient=l2_gradient,
    )
    return edges


def run_skimage_canny(
    image: np.ndarray,
    sigma: float = 1.0,
    low_threshold: float | None = 50.0,
    high_threshold: float | None = 150.0,
) -> np.ndarray:
    """Execute Scikit-Image Canny edge detector.

    Args:
        image: Grayscale image of shape (H, W).
        sigma: Standard deviation of Gaussian filter.
        low_threshold: Lower threshold.
        high_threshold: Upper threshold.

    Returns:
        np.ndarray: Binary edge map of shape (H, W), dtype uint8 with {0, 255}.
    """
    if image.dtype == np.uint8:
        img_float = image.astype(np.float32) / 255.0
        low_th_norm = low_threshold / 255.0 if low_threshold is not None else None
        high_th_norm = high_threshold / 255.0 if high_threshold is not None else None
    else:
        max_val = max(1.0, float(np.max(image)))
        img_float = image.astype(np.float32) / max_val
        low_th_norm = low_threshold / max_val if low_threshold is not None else None
        high_th_norm = high_threshold / max_val if high_threshold is not None else None

    edges_bool = skimage_canny(
        img_float,
        sigma=sigma,
        low_threshold=low_th_norm,
        high_threshold=high_th_norm,
    )
    return (edges_bool.astype(np.uint8)) * 255


def run_sobel(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Compute normalized Sobel gradient magnitude for baseline comparison.

    Args:
        image: Grayscale image of shape (H, W).
        ksize: Kernel aperture size.

    Returns:
        np.ndarray: Gradient magnitude normalized to uint8 [0, 255].
    """
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=ksize)
    mag = np.hypot(gx, gy)
    max_val = np.max(mag)
    if max_val > 1e-7:
        norm = (mag / max_val * 255.0).astype(np.uint8)
    else:
        norm = np.zeros_like(image, dtype=np.uint8)
    return norm


def run_laplacian(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Compute normalized Laplacian of Gaussian (LoG) second derivative.

    Args:
        image: Grayscale image of shape (H, W).
        ksize: Kernel aperture size.

    Returns:
        np.ndarray: Absolute second derivative normalized to uint8 [0, 255].
    """
    blurred = cv2.GaussianBlur(image, (5, 5), sigmaX=1.0)
    lap = cv2.Laplacian(blurred, cv2.CV_32F, ksize=ksize)
    abs_lap = np.abs(lap)
    max_val = np.max(abs_lap)
    if max_val > 1e-7:
        norm = (abs_lap / max_val * 255.0).astype(np.uint8)
    else:
        norm = np.zeros_like(image, dtype=np.uint8)
    return norm
