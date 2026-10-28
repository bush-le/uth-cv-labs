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
    """Ensure wavelet hash produces expected 64-bit binary output."""
    dummy_img = (np.random.rand(64, 64) * 255).astype(np.uint8)
    h64 = wavelet_hash(dummy_img, wavelet="db4", level=1, bits=8)

    assert h64.shape == (64,)
    assert h64.dtype == np.uint8
    assert set(np.unique(h64)).issubset({0, 1})


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

    # 1. Identity: d(x, x) = 0
    assert hamming_distance(h1, h1) == 0

    # 2. Non-negativity & Positivity: d(x, y) >= 0 and d(x, y) = 0 iff x == y
    dist_12 = hamming_distance(h1, h2)
    assert dist_12 > 0

    # 3. Symmetry: d(x, y) = d(y, x)
    assert dist_12 == hamming_distance(h2, h1)

    # 4. Triangle inequality: d(x, z) <= d(x, y) + d(y, z)
    d_13 = hamming_distance(h1, h3)
    d_23 = hamming_distance(h2, h3)
    assert d_13 <= dist_12 + d_23

    # Normalized distance bounds
    norm_dist = normalized_hamming_distance(h1, h2)
    assert 0.0 <= norm_dist <= 1.0
    assert norm_dist == dist_12 / 8.0


def test_pairwise_matrix_symmetry_and_diagonal() -> None:
    """Ensure pairwise distance matrix is symmetric with 0 diagonal."""
    hashes = np.array(
        [
            [1, 0, 0, 1],
            [1, 1, 0, 1],  # dist to 0 is 1
            [0, 0, 1, 0],  # dist to 0 is 3, to 1 is 4
        ],
        dtype=np.uint8,
    )

    matrix = calculate_pairwise_hamming_matrix(hashes)

    assert matrix.shape == (3, 3)
    np.testing.assert_array_equal(np.diag(matrix), np.zeros(3))
    np.testing.assert_array_equal(matrix, matrix.T)
    assert matrix[0, 1] == 1
    assert matrix[0, 2] == 3
    assert matrix[1, 2] == 4


def test_ground_truth_roc_auc_evaluation() -> None:
    """Validate statistical evaluation, Confusion Matrix, and ROC/AUC."""
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
        np.array([1, 1, 1, 1], dtype=np.uint8),  # dist 4
        np.array([0, 0, 0, 1], dtype=np.uint8),  # dist 1
        np.array([0, 0, 1, 1], dtype=np.uint8),  # dist 2
        np.array([0, 1, 1, 1], dtype=np.uint8),  # dist 3
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


def test_geometric_distortions_preserve_shape_and_dtype() -> None:
    """The transformations used by robustness checks preserve input dimensions."""
    image = np.full((60, 60, 3), 120, dtype=np.uint8)
    for kind, value in [("crop", 0.8), ("rotate", 5.0), ("flip", True)]:
        result = apply_geometric_distortion(image, kind, value)
        assert result.shape == image.shape
        assert result.dtype == np.uint8
