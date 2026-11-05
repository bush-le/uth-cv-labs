"""Core 5-stage vectorized implementation of Canny Edge Detector."""

from __future__ import annotations

import cv2
import numpy as np
import scipy.ndimage

from src.common.logger import get_logger

logger = get_logger(__name__)


def custom_canny_pipeline(
    image: np.ndarray,
    sigma: float = 1.0,
    low_threshold: float = 50.0,
    high_threshold: float = 150.0,
) -> dict[str, np.ndarray]:
    """Execute the 5-stage Canny Edge Detector pipeline using pure vectorized NumPy.

    Stages:
        1. Gaussian Smoothing (noise reduction)
        2. Gradient Magnitude & Direction (Sobel operators)
        3. Non-Maximum Suppression (1-pixel thinning along gradient direction)
        4. Double Thresholding (partitioning strong, weak, non-edges)
        5. Edge Tracking by Hysteresis (8-connected component propagation)

    Args:
        image: Grayscale input image of shape (H, W), dtype uint8 or float.
        sigma: Standard deviation for Gaussian smoothing.
        low_threshold: Lower gradient threshold.
        high_threshold: Upper gradient threshold.

    Returns:
        dict[str, np.ndarray]: Intermediate outputs and final edge map:
            - 'smoothed': (H, W) float32
            - 'grad_x': (H, W) float32
            - 'grad_y': (H, W) float32
            - 'magnitude': (H, W) float32
            - 'direction_deg': (H, W) float32 in [0, 180)
            - 'nms': (H, W) float32
            - 'strong_edges': (H, W) bool
            - 'weak_edges': (H, W) bool
            - 'edges': (H, W) uint8 with values 0 and 255
    """
    if image.ndim != 2:
        raise ValueError(f"Input image must be 2D grayscale, got shape {image.shape}")

    img_float = image.astype(np.float32)

    # 1. Gaussian Smoothing
    radius = int(np.ceil(3.0 * sigma))
    ksize = 2 * radius + 1
    smoothed = cv2.GaussianBlur(img_float, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)

    # 2. Sobel Gradients
    grad_x = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)

    magnitude = np.hypot(grad_x, grad_y)
    direction_rad = np.arctan2(grad_y, grad_x)
    direction_deg = np.rad2deg(direction_rad) % 180.0

    # 3. Vectorized Non-Maximum Suppression (NMS)
    pad_m = np.pad(magnitude, 1, mode="constant", constant_values=0.0)

    mask0 = ((direction_deg >= 0.0) & (direction_deg < 22.5)) | (direction_deg >= 157.5)
    mask1 = (direction_deg >= 22.5) & (direction_deg < 67.5)
    mask2 = (direction_deg >= 67.5) & (direction_deg < 112.5)
    mask3 = (direction_deg >= 112.5) & (direction_deg < 157.5)

    is_max0 = mask0 & (magnitude >= pad_m[1:-1, :-2]) & (magnitude >= pad_m[1:-1, 2:])
    is_max1 = mask1 & (magnitude >= pad_m[:-2, 2:]) & (magnitude >= pad_m[2:, :-2])
    is_max2 = mask2 & (magnitude >= pad_m[:-2, 1:-1]) & (magnitude >= pad_m[2:, 1:-1])
    is_max3 = mask3 & (magnitude >= pad_m[:-2, :-2]) & (magnitude >= pad_m[2:, 2:])

    keep_mask = is_max0 | is_max1 | is_max2 | is_max3
    nms = np.where(keep_mask, magnitude, 0.0).astype(np.float32)

    # 4. Double Thresholding
    strong_edges = nms >= high_threshold
    weak_edges = (nms >= low_threshold) & (nms < high_threshold)

    # 5. Edge Tracking by Hysteresis (Vectorized 8-Connectivity)
    candidate_map = strong_edges | weak_edges
    structuring_element = np.ones((3, 3), dtype=bool)
    labeled_map, num_features = scipy.ndimage.label(
        candidate_map, structure=structuring_element
    )

    strong_labels = np.unique(labeled_map[strong_edges])
    strong_labels = strong_labels[strong_labels > 0]

    final_edges_bool = np.isin(labeled_map, strong_labels)
    final_edges = (final_edges_bool.astype(np.uint8)) * 255

    logger.debug(
        f"Custom Canny completed: {num_features} connected components evaluated, "
        f"{len(strong_labels)} retained via strong hysteresis seeds."
    )

    return {
        "smoothed": smoothed,
        "grad_x": grad_x,
        "grad_y": grad_y,
        "magnitude": magnitude,
        "direction_deg": direction_deg,
        "nms": nms,
        "strong_edges": strong_edges,
        "weak_edges": weak_edges,
        "edges": final_edges,
    }
