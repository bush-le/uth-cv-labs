"""Evaluation metrics, threshold search, and ROC/PR curve analysis for perceptual hashes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def evaluate_ground_truth_pairs(
    distances: Sequence[int | float] | np.ndarray,
    ground_truth_labels: Sequence[int | bool] | np.ndarray,
    thresholds: Sequence[int | float] | np.ndarray | None = None,
    max_distance: int = 64,
) -> dict[str, Any]:
    """Perform comprehensive statistical classification evaluation across distance thresholds.

    Evaluates:
        - Accuracy, Recall (Độ nhạy), Specificity (Độ đặc hiệu), Precision, F1-Score
        - ROC curve coordinates (FPR vs TPR) and AUC score
        - Precision-Recall curve
        - Optimal threshold according to Youden's J statistic

    Args:
        distances: 1D array of pairwise distances between image pairs.
        ground_truth_labels: 1D array of binary labels (1 = Similar/Match, 0 = Dissimilar/Non-match).
        thresholds: Sequence of thresholds to evaluate. If None, sweeps range [0, max_distance].
        max_distance: Maximum possible Hamming distance (bit length).

    Returns:
        dict[str, Any]: Detailed evaluation metrics and curve coordinates.
    """
    dist_arr = np.asarray(distances, dtype=np.float32)
    gt_arr = np.asarray(ground_truth_labels, dtype=np.int32)

    if thresholds is None:
        thresh_list = np.arange(0, max_distance + 1)
    else:
        thresh_list = np.asarray(thresholds)

    tpr_list: list[float] = []
    fpr_list: list[float] = []
    precision_list: list[float] = []
    recall_list: list[float] = []
    specificity_list: list[float] = []
    accuracy_list: list[float] = []
    f1_list: list[float] = []

    pos_count = int(np.sum(gt_arr == 1))
    neg_count = int(np.sum(gt_arr == 0))

    if pos_count == 0 or neg_count == 0:
        raise ValueError(
            "Ground truth labels must contain both positive (1) and negative (0) pairs."
        )

    best_j = -1.0
    best_thresh_j = 0
    best_f1 = -1.0
    best_thresh_f1 = 0

    for tau in thresh_list:
        preds = (dist_arr <= tau).astype(np.int32)

        tp = int(np.sum((preds == 1) & (gt_arr == 1)))
        fp = int(np.sum((preds == 1) & (gt_arr == 0)))
        tn = int(np.sum((preds == 0) & (gt_arr == 0)))
        fn = int(np.sum((preds == 0) & (gt_arr == 1)))

        acc = (tp + tn) / (tp + tn + fp + fn + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        specificity = tn / (tn + fp + 1e-7)
        fpr = fp / (fp + tn + 1e-7)
        precision = tp / (tp + fp + 1e-7)
        f1 = 2.0 * precision * recall / (precision + recall + 1e-7)

        tpr_list.append(recall)
        fpr_list.append(fpr)
        precision_list.append(precision)
        recall_list.append(recall)
        specificity_list.append(specificity)
        accuracy_list.append(acc)
        f1_list.append(f1)

        j_stat = recall - fpr
        if j_stat > best_j:
            best_j = j_stat
            best_thresh_j = int(tau)

        if f1 > best_f1:
            best_f1 = f1
            best_thresh_f1 = int(tau)

    sorted_indices = np.argsort(fpr_list)
    fpr_sorted = np.array(fpr_list)[sorted_indices]
    tpr_sorted = np.array(tpr_list)[sorted_indices]

    fpr_anchored = np.concatenate([[0.0], fpr_sorted, [1.0]])
    tpr_anchored = np.concatenate([[0.0], tpr_sorted, [1.0]])
    auc_val = float(np.trapezoid(tpr_anchored, fpr_anchored))

    opt_idx = int(np.where(thresh_list == best_thresh_f1)[0][0])

    return {
        "thresholds": thresh_list.tolist(),
        "fpr": fpr_list,
        "tpr": tpr_list,
        "auc": float(np.clip(auc_val, 0.0, 1.0)),
        "precision": precision_list,
        "recall": recall_list,
        "specificity": specificity_list,
        "accuracy": accuracy_list,
        "f1": f1_list,
        "optimal_threshold_f1": best_thresh_f1,
        "optimal_threshold_youden": best_thresh_j,
        "optimal_metrics": {
            "threshold": best_thresh_f1,
            "accuracy": accuracy_list[opt_idx],
            "recall": recall_list[opt_idx],
            "specificity": specificity_list[opt_idx],
            "precision": precision_list[opt_idx],
            "f1": f1_list[opt_idx],
        },
    }
