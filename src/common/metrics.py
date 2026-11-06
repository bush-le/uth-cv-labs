"""Standardized evaluation metrics for Computer Vision tasks.

Provides classification metrics (Accuracy, Precision, Recall, Specificity, F1)
and perceptual/bit distance metrics (Hamming Distance, Normalized Hamming Distance).
Zero dependencies on downstream lab modules.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


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


def hamming_distance(
    hash1: Sequence[int] | np.ndarray,
    hash2: Sequence[int] | np.ndarray,
) -> int:
    """Calculate the exact Hamming distance between two binary hash codes.

    Args:
        hash1: First binary hash code of length B.
        hash2: Second binary hash code of length B.

    Returns:
        int: Total number of bit positions where hash1 and hash2 differ.
    """
    h1 = np.asarray(hash1)
    h2 = np.asarray(hash2)

    if h1.shape != h2.shape:
        raise ValueError(f"Hash lengths mismatch: {h1.shape} vs {h2.shape}")
    if h1.ndim != 1 or not h1.size:
        raise ValueError("Hashes must be nonempty one-dimensional arrays.")
    if not np.isin(h1, [0, 1]).all() or not np.isin(h2, [0, 1]).all():
        raise ValueError("Hashes must contain only binary values.")

    return int(np.count_nonzero(h1 != h2))


def normalized_hamming_distance(
    hash1: Sequence[int] | np.ndarray,
    hash2: Sequence[int] | np.ndarray,
) -> float:
    """Calculate the normalized Hamming distance in range [0.0, 1.0].

    Args:
        hash1: First binary hash code.
        hash2: Second binary hash code.

    Returns:
        float: Normalized distance (Hamming Distance / Bit Length).
    """
    h1 = np.asarray(hash1)
    dist = hamming_distance(h1, hash2)
    return float(dist / len(h1))


__all__ = [
    "calculate_binary_confusion_matrix",
    "calculate_classification_metrics",
    "hamming_distance",
    "normalized_hamming_distance",
]
