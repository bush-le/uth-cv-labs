"""Quantitative evaluation metrics for edge detectors (Pratt FOM, F1, Hausdorff)."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial.distance import directed_hausdorff


def calculate_pratt_fom(
    detected_edges: np.ndarray,
    ground_truth_edges: np.ndarray,
    alpha: float = 1.0 / 9.0,
) -> float:
    """Compute Pratt's Figure of Merit (FOM) for edge detection accuracy.

    Args:
        detected_edges: Binary edge map of shape (H, W).
        ground_truth_edges: Binary ground truth edge map of shape (H, W).
        alpha: Penalty scale factor.

    Returns:
        float: Pratt's FOM score in [0.0, 1.0]. Perfect match equals 1.0.
    """
    gt_pts = np.argwhere(ground_truth_edges > 0)
    det_pts = np.argwhere(detected_edges > 0)

    i_n = len(gt_pts)
    i_a = len(det_pts)

    if i_n == 0 and i_a == 0:
        return 1.0
    if i_n == 0 or i_a == 0:
        return 0.0

    gt_binary_inv = (ground_truth_edges == 0).astype(np.uint8)
    dist_map = cv2.distanceTransform(gt_binary_inv, cv2.DIST_L2, 5)

    distances = dist_map[detected_edges > 0]
    penalties = 1.0 / (1.0 + alpha * (distances**2))

    fom = float(np.sum(penalties) / max(i_n, i_a))
    return float(np.clip(fom, 0.0, 1.0))


def calculate_edge_f1_score(
    detected_edges: np.ndarray,
    ground_truth_edges: np.ndarray,
    tolerance: float = 2.0,
) -> dict[str, float]:
    """Calculate Precision, Recall, and F1-Score with distance tolerance.

    Args:
        detected_edges: Binary edge map (H, W).
        ground_truth_edges: Ground truth edge map (H, W).
        tolerance: Maximum pixel distance to count as a true positive.

    Returns:
        dict[str, float]: Precision, Recall, F1 scores in [0.0, 1.0].
    """
    det_mask = detected_edges > 0
    gt_mask = ground_truth_edges > 0

    det_count = int(np.sum(det_mask))
    gt_count = int(np.sum(gt_mask))

    if det_count == 0 and gt_count == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if det_count == 0 or gt_count == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    gt_dist = cv2.distanceTransform((~gt_mask).astype(np.uint8), cv2.DIST_L2, 5)
    det_dist = cv2.distanceTransform((~det_mask).astype(np.uint8), cv2.DIST_L2, 5)

    tp_det = np.sum(gt_dist[det_mask] <= tolerance)
    tp_gt = np.sum(det_dist[gt_mask] <= tolerance)

    precision = float(tp_det / (det_count + 1e-7))
    recall = float(tp_gt / (gt_count + 1e-7))
    f1 = float(2.0 * precision * recall / (precision + recall + 1e-7))

    return {
        "precision": min(1.0, precision),
        "recall": min(1.0, recall),
        "f1": min(1.0, f1),
    }


def calculate_hausdorff_distance(
    detected_edges: np.ndarray,
    ground_truth_edges: np.ndarray,
) -> float:
    """Calculate bidirectional symmetric Hausdorff distance between edge point sets.

    Args:
        detected_edges: Binary edge map (H, W).
        ground_truth_edges: Ground truth edge map (H, W).

    Returns:
        float: Hausdorff distance in pixels.
    """
    pts1 = np.argwhere(detected_edges > 0)
    pts2 = np.argwhere(ground_truth_edges > 0)

    if len(pts1) == 0 or len(pts2) == 0:
        return float("inf")

    d1 = directed_hausdorff(pts1, pts2)[0]
    d2 = directed_hausdorff(pts2, pts1)[0]
    return float(max(d1, d2))
