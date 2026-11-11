"""Evaluation metrics, threshold search, and ROC/PR curve analysis for perceptual hashes."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_auc_score

from src.labs.lab04_wavelet.wavelet_distance import (
    find_top_k_similar,
    hamming_distance,
    normalized_hamming_distance,
)
from src.labs.lab04_wavelet.wavelet_distortions import apply_geometric_distortion


def evaluate_ground_truth_pairs(
    distances: Sequence[int | float] | np.ndarray,
    ground_truth_labels: Sequence[int | bool] | np.ndarray,
    thresholds: Sequence[int | float] | np.ndarray | None = None,
    max_distance: int = 64,
) -> dict[str, Any]:
    """Perform comprehensive statistical classification evaluation across distance thresholds."""
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

    for thresh in thresh_list:
        preds = dist_arr <= thresh
        tp = int(np.sum((preds == 1) & (gt_arr == 1)))
        fp = int(np.sum((preds == 1) & (gt_arr == 0)))
        tn = int(np.sum((preds == 0) & (gt_arr == 0)))

        tpr = tp / pos_count if pos_count > 0 else 0.0
        fpr = fp / neg_count if neg_count > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tpr
        specificity = tn / neg_count if neg_count > 0 else 0.0
        accuracy = (tp + tn) / (pos_count + neg_count)
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        tpr_list.append(tpr)
        fpr_list.append(fpr)
        precision_list.append(precision)
        recall_list.append(recall)
        specificity_list.append(specificity)
        accuracy_list.append(accuracy)
        f1_list.append(f1)

        youden_j = tpr + specificity - 1.0
        if youden_j > best_j:
            best_j = youden_j
            best_thresh_j = int(thresh)

        if f1 > best_f1:
            best_f1 = f1
            best_thresh_f1 = int(thresh)

    # Sort FPR for AUC computation
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
        "auc_roc": float(np.clip(auc_val, 0.0, 1.0)),
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


def score_pairs(labels: np.ndarray, distances: np.ndarray, threshold: float) -> dict:
    """Score binary labels and distances (N,) at a fixed threshold."""
    labels = np.asarray(labels)
    distances = np.asarray(distances)
    if labels.ndim != 1 or labels.shape != distances.shape or not labels.size:
        raise ValueError(
            "Labels and distances must be nonempty vectors of equal length."
        )
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Both binary classes are required for this evaluation.")
    if (
        not np.isfinite(distances).all()
        or (distances < 0).any()
        or not np.isfinite(threshold)
    ):
        raise ValueError("Finite threshold and nonnegative finite distances required.")
    tn, fp, fn, tp = confusion_matrix(
        labels, distances <= threshold, labels=[0, 1]
    ).ravel()
    recall = float(tp / (tp + fn))
    specificity = float(tn / (tn + fp))
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    return {
        "threshold": float(threshold),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "accuracy": float((tp + tn) / labels.size),
        "recall": recall,
        "specificity": specificity,
        "precision": precision,
        "f1": (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        ),
        "auc": float(roc_auc_score(labels, -distances)),
    }


def select_threshold(labels: np.ndarray, distances: np.ndarray) -> float:
    """Maximize calibration Youden J, choosing the smallest threshold on ties."""
    distances = np.asarray(distances)
    candidates = np.r_[-1.0, np.unique(distances)]
    scores = [score_pairs(labels, distances, float(t)) for t in candidates]
    index = int(np.argmax([s["recall"] + s["specificity"] - 1 for s in scores]))
    return float(candidates[index])


def compare_hash_methods(
    methods: Sequence[
        tuple[
            str,
            Callable[[np.ndarray], np.ndarray | list[int]],
            Mapping[str, np.ndarray],
        ]
    ],
    pairs: Mapping[str, Sequence[Mapping[str, str]]],
    labels: Mapping[str, np.ndarray],
) -> list[dict]:
    """Compare methods on the same pairs, calibrating each before evaluation."""
    results = []
    for name, hash_fn, images in methods:
        hashes = {
            image_id: np.asarray(hash_fn(image)) for image_id, image in images.items()
        }
        length = len(next(iter(hashes.values())))
        distances = {
            split: np.array(
                [
                    normalized_hamming_distance(
                        hashes[p["left_id"]], hashes[p["right_id"]]
                    )
                    for p in rows
                ]
            )
            for split, rows in pairs.items()
        }
        threshold = select_threshold(labels["calibration"], distances["calibration"])
        results.append(
            {
                "name": name,
                "bits": length,
                "evaluation": score_pairs(
                    labels["evaluation"], distances["evaluation"], threshold
                ),
            }
        )
    return results


def retrieve_top_k(
    query_path: Path,
    database: Sequence[Mapping[str, str]],
    hashes: Mapping[str, np.ndarray],
    root: Path,
    hash_fn: Callable[[np.ndarray], np.ndarray],
    k: int,
) -> tuple[np.ndarray, list[Mapping[str, str]], list[tuple[int, int]]]:
    """Load an RGB query, exclude its own file and rank database hashes."""
    with Image.open(query_path) as image:
        query_rgb = np.array(image.convert("RGB"), dtype=np.uint8)
    candidates = [
        s for s in database if (root / s["path"]).resolve() != query_path.resolve()
    ]
    results = find_top_k_similar(
        hash_fn(query_rgb), [hashes[s["source_id"]] for s in candidates], k=k
    )
    return query_rgb, candidates, results


def measure_robustness(
    sources: Sequence[Mapping[str, str]],
    images: Mapping[str, np.ndarray],
    hashes: Mapping[str, np.ndarray],
    hash_fn: Callable[[np.ndarray], np.ndarray],
    geometric_checks: Sequence[Sequence],
) -> dict[str, list[int]]:
    """Measure existing variants and configured geometric edits on calibration only."""
    source_ids = [s["source_id"] for s in sources if s["split"] == "calibration"]
    result = {
        name: [hamming_distance(hashes[i], hashes[f"{i}__{name}"]) for i in source_ids]
        for name in ("bright", "dark", "noise", "jpeg")
    }
    for name, kind, value in geometric_checks:
        result[name] = [
            hamming_distance(
                hashes[i], hash_fn(apply_geometric_distortion(images[i], kind, value))
            )
            for i in source_ids
        ]
    return result
