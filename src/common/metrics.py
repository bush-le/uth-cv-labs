"""Standardized evaluation metrics for Computer Vision tasks.

Consolidates classification, edge detection, and similarity evaluation metrics.
"""

from __future__ import annotations

import numpy as np

from src.models.classical.canny_metrics import (
    calculate_edge_f1_score,
    calculate_hausdorff_distance,
    calculate_pratt_fom,
)
from src.models.classical.wavelet_distance import (
    hamming_distance,
    normalized_hamming_distance,
)
from src.models.classical.wavelet_evaluation import (
    evaluate_ground_truth_pairs,
)


def calculate_binary_confusion_matrix(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
) -> dict[str, int]:
    """Compute True Positives, False Positives, True Negatives, and False Negatives.

    Args:
        predictions: Binary prediction array of shape (N,), dtype int or bool.
        ground_truth: Binary ground truth array of shape (N,), dtype int or bool.

    Returns:
        dict[str, int]: Dictionary containing 'tp', 'fp', 'tn', 'fn'.
    """
    preds = np.asarray(predictions, dtype=np.int32)
    gts = np.asarray(ground_truth, dtype=np.int32)

    tp = int(np.sum((preds == 1) & (gts == 1)))
    fp = int(np.sum((preds == 1) & (gts == 0)))
    tn = int(np.sum((preds == 0) & (gts == 0)))
    fn = int(np.sum((preds == 0) & (gts == 1)))

    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def calculate_classification_metrics(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
) -> dict[str, float]:
    """Compute Accuracy, Precision, Recall, Specificity, and F1-Score.

    Args:
        predictions: Binary prediction array of shape (N,).
        ground_truth: Binary ground truth array of shape (N,).

    Returns:
        dict[str, float]: Standardized classification metrics.
    """
    cm = calculate_binary_confusion_matrix(predictions, ground_truth)
    tp, fp, tn, fn = cm["tp"], cm["fp"], cm["tn"], cm["fn"]

    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-7)
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    specificity = tn / (tn + fp + 1e-7)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-7)

    return {
        "accuracy": float(np.clip(accuracy, 0.0, 1.0)),
        "precision": float(np.clip(precision, 0.0, 1.0)),
        "recall": float(np.clip(recall, 0.0, 1.0)),
        "specificity": float(np.clip(specificity, 0.0, 1.0)),
        "f1": float(np.clip(f1, 0.0, 1.0)),
    }


__all__ = [
    "calculate_binary_confusion_matrix",
    "calculate_classification_metrics",
    "calculate_pratt_fom",
    "calculate_edge_f1_score",
    "calculate_hausdorff_distance",
    "hamming_distance",
    "normalized_hamming_distance",
    "evaluate_ground_truth_pairs",
]
