"""Unit tests for modular classical vision models and evaluation metrics."""

from __future__ import annotations

import numpy as np

from src.common.metrics import (
    calculate_binary_confusion_matrix,
    calculate_classification_metrics,
)
from src.models.classical.canny_advanced import (
    run_auto_canny,
    run_otsu_canny,
)
from src.models.classical.canny_core import custom_canny_pipeline
from src.models.classical.canny_metrics import (
    calculate_edge_f1_score,
    calculate_pratt_fom,
)
from src.models.classical.wavelet_core import (
    average_hash,
    difference_hash,
    phash_dct,
    wavelet_hash,
)
from src.models.classical.wavelet_distance import (
    calculate_pairwise_hamming_matrix,
    hamming_distance,
    normalized_hamming_distance,
)


def test_custom_canny_core() -> None:
    """Verify custom 5-stage Canny edge detector core outputs."""
    img = np.zeros((64, 64), dtype=np.uint8)
    img[20:44, 20:44] = 255
    res = custom_canny_pipeline(img, sigma=1.0, low_threshold=30, high_threshold=90)

    assert "edges" in res
    assert "magnitude" in res
    assert "nms" in res
    assert res["edges"].shape == (64, 64)
    assert np.max(res["edges"]) == 255


def test_canny_advanced_and_metrics() -> None:
    """Verify auto-canny, otsu-canny, and Pratt FOM evaluation."""
    img = np.zeros((64, 64), dtype=np.uint8)
    img[16:48, 16:48] = 200

    auto_edges, low_t, high_t = run_auto_canny(img)
    assert auto_edges.shape == (64, 64)
    assert low_t < high_t

    otsu_edges, o_low, o_high = run_otsu_canny(img)
    assert otsu_edges.shape == (64, 64)

    fom = calculate_pratt_fom(auto_edges, auto_edges)
    assert np.isclose(fom, 1.0)

    f1_dict = calculate_edge_f1_score(auto_edges, auto_edges)
    assert np.isclose(f1_dict["f1"], 1.0)


def test_wavelet_modular_and_distance() -> None:
    """Verify wavelet hashing and distance computation."""
    img = np.zeros((64, 64), dtype=np.uint8)
    img[10:50, 10:50] = 255

    w_code = wavelet_hash(img, bits=8)
    assert len(w_code) == 64

    a_code = average_hash(img, hash_size=8)
    assert len(a_code) == 64

    d_code = difference_hash(img, hash_size=8)
    assert len(d_code) == 64

    p_code = phash_dct(img, hash_size=8)
    assert len(p_code) == 64

    dist = hamming_distance(w_code, w_code)
    assert dist == 0

    norm_dist = normalized_hamming_distance(w_code, w_code)
    assert norm_dist == 0.0

    pair_mat = calculate_pairwise_hamming_matrix([w_code, a_code])
    assert pair_mat.shape == (2, 2)
    assert pair_mat[0, 0] == 0


def test_metrics_evaluation() -> None:
    """Verify classification metrics calculation."""
    preds = np.array([1, 1, 0, 0])
    gts = np.array([1, 0, 0, 1])

    cm = calculate_binary_confusion_matrix(preds, gts)
    assert cm == {"tp": 1, "fp": 1, "tn": 1, "fn": 1}

    metrics = calculate_classification_metrics(preds, gts)
    assert np.isclose(metrics["accuracy"], 0.5)
    assert np.isclose(metrics["precision"], 0.5)
    assert np.isclose(metrics["recall"], 0.5)
    assert np.isclose(metrics["specificity"], 0.5)
