"""Unit tests for Lab 02 visual analytics and transformation curves."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.labs.lab02_filters.visuals import (
    plot_bilateral_decomposition_profile,
    plot_edge_step_response_profile,
    plot_fft_filter_spectra,
    plot_gradient_3d_surface,
    plot_gradient_vector_field,
    plot_kernel_sum_dc_characteristics,
    plot_point_op_transfer_curves,
    plot_threshold_cdf_analysis,
)


@pytest.fixture
def sample_gray_image() -> np.ndarray:
    """Generate a synthetic 64x64 grayscale step-edge image for testing."""
    img = np.zeros((64, 64), dtype=np.uint8)
    img[:, 32:] = 180
    # Add minor noise / structure
    img[16:48, 16:48] = 120
    return img


def test_plot_point_op_transfer_curves():
    """Verify point op transfer curves figure generation."""
    fig = plot_point_op_transfer_curves()
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) >= 4
    plt.close(fig)


def test_plot_threshold_cdf_analysis(sample_gray_image):
    """Verify threshold CDF dual-axis plot generation."""
    fig = plot_threshold_cdf_analysis(sample_gray_image, thresholds=[80, 127, 180])
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) >= 2  # Dual-axis plot
    plt.close(fig)


def test_plot_edge_step_response_profile(sample_gray_image):
    """Verify 1D edge step response profile generation."""
    img_orig = sample_gray_image
    img_mean = np.clip(img_orig.astype(float) * 0.9, 0, 255).astype(np.uint8)
    img_gauss = np.clip(img_orig.astype(float) * 0.95, 0, 255).astype(np.uint8)
    img_sharp5 = np.clip(img_orig.astype(float) * 1.1, 0, 255).astype(np.uint8)
    img_sharp21 = np.clip(img_orig.astype(float) * 1.25, 0, 255).astype(np.uint8)

    fig = plot_edge_step_response_profile(
        img_orig,
        img_mean,
        img_gauss,
        img_sharp5,
        img_sharp21,
        row=32,
        col_range=(10, 54),
    )
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) >= 1
    plt.close(fig)


def test_plot_fft_filter_spectra(sample_gray_image):
    """Verify 2D FFT spectra and 1D radial power profile generation."""
    img_orig = sample_gray_image
    img_blur = np.roll(sample_gray_image, 1, axis=0)
    images = [img_orig, img_blur]
    titles = ["Original", "Blurred"]

    fig = plot_fft_filter_spectra(images, titles)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) >= 3  # 2 spectra subplots + 1 radial profile plot
    plt.close(fig)


def test_plot_gradient_vector_field(sample_gray_image):
    """Verify gradient vector field quiver and angle map generation."""
    fig = plot_gradient_vector_field(sample_gray_image, roi=(16, 48, 16, 48), step=2)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) >= 2
    plt.close(fig)


def test_plot_gradient_3d_surface(sample_gray_image):
    """Verify 3D topographic surface generation."""
    patch = sample_gray_image[16:48, 16:48].astype(np.float32)
    fig = plot_gradient_3d_surface(patch)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) >= 1
    plt.close(fig)


def test_plot_kernel_sum_dc_characteristics():
    """Verify DC transmission line characteristics curve generation."""
    fig = plot_kernel_sum_dc_characteristics()
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) >= 1
    plt.close(fig)


def test_plot_bilateral_decomposition_profile():
    """Verify bilateral decomposition and 1D impulse denoising figure generation."""
    fig = plot_bilateral_decomposition_profile()
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) >= 2
    plt.close(fig)
