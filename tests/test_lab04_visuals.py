"""Unit tests for Lab 04 Wavelet Hash spectral, statistical, and retrieval visual functions."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.labs.lab04_wavelet.visuals import (
    plot_annotated_distance_matrix,
    plot_bit_quantization_and_diff,
    plot_cbir_rank_dropoff,
    plot_comprehensive_metric_sweep,
    plot_dual_distance_distribution,
    plot_dwt_energy_analysis,
    plot_roc_pr_isof1,
)


@pytest.fixture
def sample_rgb_image() -> np.ndarray:
    """Generate a synthetic 128x128 RGB test image with smooth gradient and square feature."""
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    y, x = np.mgrid[:128, :128]
    img[:, :, 0] = (x * 2).astype(np.uint8)
    img[:, :, 1] = (y * 2).astype(np.uint8)
    img[:, :, 2] = 128
    # Center bright spot
    img[48:80, 48:80, :] = 240
    return img


@pytest.fixture
def sample_hash_quantization_data() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Generate synthetic 8x8 LL coefficients and corresponding 64-bit hash variations."""
    rng = np.random.default_rng(42)
    ll_small = rng.standard_normal((8, 8)).astype(np.float32)
    # Median partition
    h_orig = (ll_small > np.median(ll_small)).astype(np.uint8).ravel()

    # Distorted: 2 random bit flips
    h_distorted = h_orig.copy()
    flip_indices = rng.choice(64, size=2, replace=False)
    h_distorted[flip_indices] = 1 - h_distorted[flip_indices]

    # Diff class: ~32 bit flips
    h_diff_class = rng.integers(0, 2, size=64, dtype=np.uint8)
    return ll_small, h_orig, h_distorted, h_diff_class


@pytest.fixture
def sample_evaluation_metrics_data() -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic binary labels and Hamming distances for 100 image pairs."""
    rng = np.random.default_rng(42)
    # 50 positives (same source, Hamming dist ~ [0, 18])
    pos_dists = rng.integers(0, 19, size=50)
    pos_labels = np.ones(50, dtype=int)

    # 50 negatives (different sources, Hamming dist ~ [70, 120])
    neg_dists = rng.integers(70, 121, size=50)
    neg_labels = np.zeros(50, dtype=int)

    labels = np.concatenate([pos_labels, neg_labels])
    distances = np.concatenate([pos_dists, neg_dists])
    return labels, distances


def test_plot_dwt_energy_analysis(sample_rgb_image):
    """Verify Mallat 4-quadrant layout and subband Frobenius energy bar chart generation."""
    fig = plot_dwt_energy_analysis(sample_rgb_image, wavelet="haar")
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) >= 2  # Mallat quadrant axis + Energy bar axis
    plt.close(fig)


def test_plot_bit_quantization_and_diff(sample_hash_quantization_data):
    """Verify 64-coefficient median split histogram and 2D bit-flip heatmaps."""
    ll_small, h_orig, h_distorted, h_diff_class = sample_hash_quantization_data
    fig = plot_bit_quantization_and_diff(ll_small, h_orig, h_distorted, h_diff_class)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) >= 3  # Histogram + Distorted diff + Diff class diff
    plt.close(fig)


def test_plot_comprehensive_metric_sweep(sample_evaluation_metrics_data):
    """Verify multi-metric sweep curves across thresholds with optimal zero-error plateau."""
    labels, distances = sample_evaluation_metrics_data
    fig = plot_comprehensive_metric_sweep(labels, distances, selected_tau=33)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) >= 1
    plt.close(fig)


def test_plot_dual_distance_distribution(sample_evaluation_metrics_data):
    """Verify dual distance KDE/histogram, random string Gaussian curve, and margin separation."""
    labels, distances = sample_evaluation_metrics_data
    fig = plot_dual_distance_distribution(labels, distances, threshold=33)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) >= 1
    plt.close(fig)


def test_plot_roc_pr_isof1(sample_evaluation_metrics_data):
    """Verify side-by-side ROC curve (AUC=1.0) and PR curve with Iso-F1 contours."""
    labels, distances = sample_evaluation_metrics_data
    fig = plot_roc_pr_isof1(labels, distances, threshold=33)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) >= 2  # ROC panel + PR panel
    plt.close(fig)


def test_plot_cbir_rank_dropoff():
    """Verify CBIR top-k rank-ordered step-bar chart showing the distance cliff."""
    # Test with default fallback arguments
    fig_default = plot_cbir_rank_dropoff()
    assert isinstance(fig_default, plt.Figure)
    assert len(fig_default.axes) >= 1
    plt.close(fig_default)

    # Test with explicit candidate metadata and top results
    candidates = [
        {"source_id": "chihuahua_01", "category": "chihuahua"},
        {"source_id": "muffin_03", "category": "muffin"},
        {"source_id": "chihuahua_04", "category": "chihuahua"},
    ]
    top_results = [(0, 2), (1, 84), (2, 88)]
    fig_custom = plot_cbir_rank_dropoff(
        candidates=candidates,
        top_results=top_results,
        query_source="chihuahua_01.jpg",
        threshold=33,
    )
    assert isinstance(fig_custom, plt.Figure)
    assert len(fig_custom.axes) >= 1
    plt.close(fig_custom)


def test_plot_annotated_distance_matrix():
    """Verify pairwise distance matrix with category partition boundaries."""
    sources = [
        {"source_id": f"chihuahua_{i:02d}", "category": "chihuahua"}
        for i in range(1, 5)
    ] + [{"source_id": f"muffin_{i:02d}", "category": "muffin"} for i in range(1, 5)]
    n = len(sources)
    rng = np.random.default_rng(42)
    # Symmetric distance matrix
    mat = rng.integers(60, 100, size=(n, n))
    mat = (mat + mat.T) // 2
    np.fill_diagonal(mat, 0)
    # Low intra-class distances
    mat[:4, :4] = np.clip(mat[:4, :4] // 4, 0, 20)
    mat[4:, 4:] = np.clip(mat[4:, 4:] // 4, 0, 20)
    np.fill_diagonal(mat, 0)

    fig = plot_annotated_distance_matrix(mat, sources, bits=192)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) >= 1
    plt.close(fig)
