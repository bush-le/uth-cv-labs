"""Adaptive thresholding (Auto-Canny, Otsu) and color gradient tensor detectors."""

from __future__ import annotations

import cv2
import numpy as np


def run_auto_canny(
    image: np.ndarray,
    sigma_factor: float = 0.33,
) -> tuple[np.ndarray, int, int]:
    """Execute Auto-Canny with median-derived adaptive thresholds.

    Calculates thresholds based on the median intensity:
        low_threshold = max(0, (1.0 - sigma_factor) * median)
        high_threshold = min(255, (1.0 + sigma_factor) * median)

    Args:
        image: Grayscale uint8 image of shape (H, W).
        sigma_factor: Percentage deviation from median (standard default 0.33).

    Returns:
        tuple[np.ndarray, int, int]: (edge_map, low_threshold, high_threshold).
    """
    v = float(np.median(image))
    low_thresh = int(max(0, (1.0 - sigma_factor) * v))
    high_thresh = int(min(255, (1.0 + sigma_factor) * v))

    if high_thresh <= low_thresh:
        high_thresh = min(255, low_thresh + 30)

    edges = cv2.Canny(image, low_thresh, high_thresh)
    return edges, low_thresh, high_thresh


def run_otsu_canny(
    image: np.ndarray,
) -> tuple[np.ndarray, int, int]:
    """Compute Canny thresholds adaptively using Otsu's bimodal threshold.

    Args:
        image: Grayscale uint8 image of shape (H, W).

    Returns:
        tuple[np.ndarray, int, int]: (edge_map, low_threshold, high_threshold).
    """
    blurred = cv2.GaussianBlur(image, (5, 5), 1.0)
    otsu_val, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    high_thresh = int(otsu_val)
    low_thresh = int(0.5 * high_thresh)

    edges = cv2.Canny(blurred, low_thresh, high_thresh)
    return edges, low_thresh, high_thresh


def run_multichannel_canny(
    rgb_image: np.ndarray,
    low_threshold: float = 50.0,
    high_threshold: float = 150.0,
) -> np.ndarray:
    """Execute Canny on R, G, and B channels independently, merging via bitwise OR.

    Args:
        rgb_image: Color image of shape (H, W, 3), dtype uint8 RGB.
        low_threshold: Lower threshold.
        high_threshold: Upper threshold.

    Returns:
        np.ndarray: Merged binary edge map (H, W) uint8 {0, 255}.
    """
    edge_r = cv2.Canny(rgb_image[:, :, 0], low_threshold, high_threshold)
    edge_g = cv2.Canny(rgb_image[:, :, 1], low_threshold, high_threshold)
    edge_b = cv2.Canny(rgb_image[:, :, 2], low_threshold, high_threshold)

    merged = cv2.bitwise_or(edge_r, edge_g)
    merged = cv2.bitwise_or(merged, edge_b)
    return merged


def run_di_zenzo_gradient(
    rgb_image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute color gradient magnitude and direction using the Di Zenzo Tensor.

    Args:
        rgb_image: RGB image of shape (H, W, 3), dtype uint8 or float.

    Returns:
        tuple[np.ndarray, np.ndarray]: (magnitude_float32, direction_deg_float32 in [0, 180)).
    """
    img = rgb_image.astype(np.float32)

    rx = cv2.Sobel(img[:, :, 0], cv2.CV_32F, 1, 0, ksize=3)
    ry = cv2.Sobel(img[:, :, 0], cv2.CV_32F, 0, 1, ksize=3)

    gx = cv2.Sobel(img[:, :, 1], cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img[:, :, 1], cv2.CV_32F, 0, 1, ksize=3)

    bx = cv2.Sobel(img[:, :, 2], cv2.CV_32F, 1, 0, ksize=3)
    by = cv2.Sobel(img[:, :, 2], cv2.CV_32F, 0, 1, ksize=3)

    g_xx = rx**2 + gx**2 + bx**2
    g_yy = ry**2 + gy**2 + by**2
    g_xy = rx * ry + gx * gy + bx * by

    delta = np.sqrt(np.maximum(0.0, (g_xx - g_yy) ** 2 + 4.0 * (g_xy**2)))
    lambda_plus = 0.5 * (g_xx + g_yy + delta)

    magnitude = np.sqrt(np.maximum(0.0, lambda_plus))
    angle_rad = 0.5 * np.arctan2(2.0 * g_xy, g_xx - g_yy + 1e-7)
    angle_deg = np.rad2deg(angle_rad) % 180.0

    return magnitude, angle_deg
