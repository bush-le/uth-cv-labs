"""Unit tests for Lab 04 Wavelet Hash and Image Similarity Suite.

Covers:
- Smoke tests (@pytest.mark.smoke) for zero-crash instantiation and hashing.
- Normal tests for mathematical properties, distance invariants, ROC/AUC, and robustness.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.models.classical.wavelet_hash import (
    apply_geometric_distortion,
    apply_photometric_distortion,
    average_hash,
    calculate_pairwise_hamming_matrix,
    difference_hash,
    evaluate_ground_truth_pairs,
    find_top_k_similar,
    hamming_distance,
    normalized_hamming_distance,
    phash_dct,
    wavelet_hash,
    wavelet_hash_slide17,
)

# ==============================================================================
# SMOKE TEST SUITE (< 5s Sanity Checks)
# ==============================================================================


@pytest.mark.smoke
def test_wavelet_hash_smoke() -> None:
    """Ensure standard wavelet_hash generates binary array of expected length."""
    dummy_img = (np.random.rand(64, 64) * 255).astype(np.uint8)
    h = wavelet_hash(dummy_img, wavelet="db4", level=1, bits=8)

    assert isinstance(h, np.ndarray)
    assert h.shape == (64,)
    assert set(np.unique(h)).issubset({0, 1})


@pytest.mark.smoke
def test_slide17_variant_smoke() -> None:
    """Ensure Slide 17 literal variant executes without error."""
    dummy_img = (np.random.rand(64, 64) * 255).astype(np.uint8)
    h17 = wavelet_hash_slide17(dummy_img, wavelet="db4", level=3, step=2.0)

    assert isinstance(h17, list)
    assert len(h17) > 0
    assert set(np.unique(h17)).issubset({0, 1})


@pytest.mark.smoke
def test_baseline_hashes_smoke() -> None:
    """Check aHash, dHash, and pHash generation."""
    dummy_img = (np.random.rand(64, 64) * 255).astype(np.uint8)
    ah = average_hash(dummy_img, hash_size=8)
    dh = difference_hash(dummy_img, hash_size=8)
    ph = phash_dct(dummy_img, hash_size=8)

    assert ah.shape == (64,)
    assert dh.shape == (64,)
    assert ph.shape == (64,)


# ==============================================================================
# NORMAL TEST SUITE (Logic, Invariants, Edge Cases)
# ==============================================================================


def test_hamming_distance_metric_properties() -> None:
    """Validate metric axioms of Hamming distance."""
    h1 = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
    h2 = np.array([1, 1, 1, 0, 0, 0, 1, 1], dtype=np.uint8)
    h3 = np.array([0, 1, 0, 0, 1, 1, 0, 1], dtype=np.uint8)

    # 1. Identity: D_H(x, x) == 0
    assert hamming_distance(h1, h1) == 0

    # 2. Symmetry: D_H(x, y) == D_H(y, x)
    assert hamming_distance(h1, h2) == hamming_distance(h2, h1)
    assert hamming_distance(h1, h2) == 3

    # 3. Normalized distance in [0.0, 1.0]
    norm_d = normalized_hamming_distance(h1, h2)
    assert 0.0 <= norm_d <= 1.0
    assert norm_d == pytest.approx(3.0 / 8.0)

    # 4. Triangle inequality: D_H(x, z) <= D_H(x, y) + D_H(y, z)
    d13 = hamming_distance(h1, h3)
    d12 = hamming_distance(h1, h2)
    d23 = hamming_distance(h2, h3)
    assert d13 <= d12 + d23


def test_pairwise_matrix_symmetry_and_diagonal() -> None:
    """Ensure calculate_pairwise_hamming_matrix produces symmetric matrix with zero diagonal."""
    hashes = [
        np.array([1, 0, 1, 0], dtype=np.uint8),
        np.array([1, 1, 0, 0], dtype=np.uint8),
        np.array([0, 0, 1, 1], dtype=np.uint8),
    ]
    matrix = calculate_pairwise_hamming_matrix(hashes)

    assert matrix.shape == (3, 3)
    assert np.all(np.diag(matrix) == 0)
    assert np.all(matrix == matrix.T)
    assert matrix[0, 1] == 2
    assert matrix[0, 2] == 2
    assert matrix[1, 2] == 4


def test_ground_truth_roc_auc_evaluation() -> None:
    """Validate statistical evaluation, Confusion Matrix, and ROC/AUC."""
    # Synthetic scenario:
    # 5 similar pairs (distance <= 4)
    # 5 dissimilar pairs (distance >= 20)
    distances = [0, 2, 3, 4, 5, 22, 28, 30, 35, 40]
    labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

    eval_res = evaluate_ground_truth_pairs(distances, labels, max_distance=64)

    assert "auc" in eval_res
    assert eval_res["auc"] >= 0.95
    assert "optimal_metrics" in eval_res

    opt = eval_res["optimal_metrics"]
    assert opt["accuracy"] >= 0.90
    assert opt["recall"] >= 0.90
    assert opt["specificity"] >= 0.90
    assert opt["precision"] >= 0.90
    assert opt["f1"] >= 0.90


def test_top_k_retrieval() -> None:
    """Verify top-k retrieval sorting order."""
    query = np.array([0, 0, 0, 0], dtype=np.uint8)
    database = [
        np.array([1, 1, 1, 1], dtype=np.uint8),  # dist = 4
        np.array([0, 0, 0, 1], dtype=np.uint8),  # dist = 1
        np.array([0, 0, 1, 1], dtype=np.uint8),  # dist = 2
    ]

    top2 = find_top_k_similar(query, database, k=2)
    assert len(top2) == 2
    assert top2[0] == (1, 1)  # Index 1 has distance 1
    assert top2[1] == (2, 2)  # Index 2 has distance 2


def test_distortion_suite_invariants() -> None:
    """Ensure distortion suite preserves spatial dimensions and dtype."""
    img = np.ones((60, 60), dtype=np.uint8) * 120

    # Photometric distortions
    bright = apply_photometric_distortion(img, "brightness", 1.2)
    contrast = apply_photometric_distortion(img, "contrast", 1.4)
    blur = apply_photometric_distortion(img, "blur", 1.0)
    jpeg = apply_photometric_distortion(img, "jpeg", 50)
    noise = apply_photometric_distortion(img, "gaussian_noise", 15)

    for distorted in [bright, contrast, blur, jpeg, noise]:
        assert distorted.shape == (60, 60)
        assert distorted.dtype == np.uint8

    # Geometric distortions
    scaled = apply_geometric_distortion(img, "scale", 1.5)
    cropped = apply_geometric_distortion(img, "crop", 0.8)
    rotated = apply_geometric_distortion(img, "rotate", 5.0)
    flipped = apply_geometric_distortion(img, "flip", True)

    for transformed in [scaled, cropped, rotated, flipped]:
        assert transformed.shape == (60, 60)
        assert transformed.dtype == np.uint8
