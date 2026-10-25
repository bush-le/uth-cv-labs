"""Choose a distance threshold on calibration, then score a separate split."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_auc_score

from src.models.classical.wavelet_hash import (
    apply_geometric_distortion,
    find_top_k_similar,
    hamming_distance,
)


def score_pairs(labels: np.ndarray, distances: np.ndarray, threshold: float) -> dict:
    """Score binary labels and distances (N,) at a fixed threshold.

    Args:
        labels: Binary array (N,), 1 means same source.
        distances: Finite nonnegative distances (N,); smaller means more similar.
        threshold: Fixed decision threshold; distance <= threshold predicts 1.

    Returns:
        Confusion counts, accuracy, recall, specificity, precision, F1 and AUC.
        Precision is defined as zero when no positive prediction exists.
    """
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
    """Maximize calibration Youden J, choosing the smallest threshold on ties.

    Args:
        labels: Calibration labels (N,) with both classes.
        distances: Calibration distances (N,), raw or normalized.

    Returns:
        Threshold chosen entirely from calibration. Includes an all-negative case.
    """
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
    """Compare methods on the same pairs, calibrating each before evaluation.

    Args:
        methods: Name, hash function and image map for each method.
        pairs: Pair manifests keyed by calibration/evaluation split.
        labels: Binary labels for the same manifests and splits.

    Returns:
        Method name, hash length and held-out metrics with its calibrated threshold.
    """
    results = []
    for name, hash_fn, images in methods:
        hashes = {
            image_id: np.asarray(hash_fn(image)) for image_id, image in images.items()
        }
        length = len(next(iter(hashes.values())))
        distances = {
            split: np.array(
                [
                    hamming_distance(hashes[p["left_id"]], hashes[p["right_id"]])
                    / length
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
    """Load an RGB query, exclude its own file and rank database hashes.

    Returns:
        RGB uint8 query (H,W,3), filtered candidates and (index, distance) results.
    """
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
