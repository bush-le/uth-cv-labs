"""Unit tests for Lab 03 Canny multi-stage mechanics and geometric visual functions."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.labs.lab03_canny.visuals import (
    plot_clahe_tile_mechanism,
    plot_gaussian_tradeoff_1d,
    plot_gradient_quiver_roi,
    plot_hysteresis_tricolor,
    plot_magnitude_cdf_histogram,
    plot_nms_ridge_cross_section,
    plot_sector_directional_map,
)


@pytest.fixture
def sample_synthetic_edge_image() -> np.ndarray:
    """Generate a synthetic 64x64 grayscale image with clean horizontal and vertical edges."""
    img = np.zeros((64, 64), dtype=np.uint8)
    # Background 40, circle / square high intensity in the middle
    img[16:48, 16:48] = 200
    # Add an inclined edge
    rr, cc = np.ogrid[:64, :64]
    mask = (rr + cc) > 70
    img[mask] = 160
    return img


def test_plot_gaussian_tradeoff_1d():
    """Verify 1D Gaussian tradeoff edge smoothing and derivative profile generation."""
    fig = plot_gaussian_tradeoff_1d(noise_sigma=8.0, sigmas=(0.6, 1.8, 5.0))
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) >= 2  # Signal and derivative subplots
    plt.close(fig)


def test_plot_gradient_quiver_roi(sample_synthetic_edge_image):
    """Verify gradient quiver vector field generation on an edge ROI."""
    fig = plot_gradient_quiver_roi(
        sample_synthetic_edge_image,
        y_slice=(10, 26),
        x_slice=(10, 26),
        step=1,
    )
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) >= 1
    plt.close(fig)


def test_plot_sector_directional_map(sample_synthetic_edge_image):
    """Verify 4-sector quantized directional map figure generation."""
    fig = plot_sector_directional_map(sample_synthetic_edge_image, threshold=20.0)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) >= 1
    plt.close(fig)


def test_plot_nms_ridge_cross_section(sample_synthetic_edge_image):
    """Verify 1D NMS ridge cross-section comparison figure generation."""
    fig = plot_nms_ridge_cross_section(
        sample_synthetic_edge_image,
        row=20,
        col_start=5,
        col_end=55,
    )
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) >= 1
    plt.close(fig)


def test_plot_hysteresis_tricolor(sample_synthetic_edge_image):
    """Verify tri-color hysteresis classification map generation."""
    fig = plot_hysteresis_tricolor(
        sample_synthetic_edge_image,
        low_thresh=30.0,
        high_thresh=90.0,
    )
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) >= 1
    plt.close(fig)


def test_plot_magnitude_cdf_histogram(sample_synthetic_edge_image):
    """Verify gradient magnitude histogram with dual-axis CDF figure generation."""
    fig = plot_magnitude_cdf_histogram(
        sample_synthetic_edge_image,
        low_thresh=30.0,
        high_thresh=90.0,
    )
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) >= 2  # Histogram axis + twinx CDF axis
    plt.close(fig)


def test_plot_clahe_tile_mechanism(sample_synthetic_edge_image):
    """Verify CLAHE tile mechanism and contrast bounding figure generation."""
    fig = plot_clahe_tile_mechanism(
        sample_synthetic_edge_image,
        clip_limit=2.0,
        tile_grid_size=(8, 8),
    )
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) >= 3  # Tile overlay + tile histogram/CDF + gradient contrast
    plt.close(fig)
