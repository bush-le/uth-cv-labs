"""Smoke and mathematical invariants for the Wavelet functions used by Lab04."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.classical.wavelet_hash import (
    apply_geometric_distortion,
    calculate_pairwise_hamming_matrix,
    find_top_k_similar,
    hamming_distance,
    normalized_hamming_distance,
    wavelet_hash,
    wavelet_hash_slide17,
)


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
    """Ensure the step-quantized slide-inspired variant executes without error."""
    dummy_img = (np.random.rand(64, 64) * 255).astype(np.uint8)
    h17 = wavelet_hash_slide17(dummy_img, wavelet="db4", level=3, step=2.0)

    assert isinstance(h17, list)
    assert len(h17) > 0
    assert set(np.unique(h17)).issubset({0, 1})


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


def test_geometric_distortions_preserve_shape_and_dtype() -> None:
    """The transformations used by robustness checks preserve input dimensions."""
    image = np.full((60, 60, 3), 120, dtype=np.uint8)
    for kind, value in [("crop", 0.8), ("rotate", 5.0), ("flip", True)]:
        result = apply_geometric_distortion(image, kind, value)
        assert result.shape == image.shape
        assert result.dtype == np.uint8
