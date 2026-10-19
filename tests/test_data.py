"""Unit tests for modular data packages (IO, color spaces, geometric, drawing, filters)."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.color_spaces import bgr_to_grayscale, bgr_to_hsv, bgr_to_lab, bgr_to_rgb
from src.data.drawing import draw_circle, draw_line, draw_rectangle, draw_text
from src.data.edge_filters import apply_sobel, get_sobel_kernels
from src.data.geometric import crop_roi, resize_by_dimensions, resize_by_scale
from src.data.linear_filters import (
    apply_gaussian_filter,
    apply_mean_filter,
    get_gaussian_kernel,
    get_mean_kernel,
    kernel_sum,
)
from src.data.nonlinear_filters import add_salt_pepper_noise, apply_median_filter
from src.data.point_ops import (
    adjust_brightness,
    adjust_contrast,
    apply_threshold,
    negative_transform,
)


@pytest.fixture
def sample_bgr() -> np.ndarray:
    """Fixture providing a deterministic 64x64 BGR test image."""
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    arr[:32, :32] = [255, 0, 0]
    arr[:32, 32:] = [0, 255, 0]
    arr[32:, :] = [0, 0, 255]
    return arr


def test_modular_color_spaces(sample_bgr: np.ndarray) -> None:
    """Verify modular color space conversions."""
    rgb = bgr_to_rgb(sample_bgr)
    assert rgb.shape == (64, 64, 3)
    assert np.array_equal(rgb[0, 0], [0, 0, 255])

    gray = bgr_to_grayscale(sample_bgr)
    assert gray.shape == (64, 64)
    assert gray.dtype == np.uint8

    hsv = bgr_to_hsv(sample_bgr)
    assert hsv.shape == (64, 64, 3)

    lab = bgr_to_lab(sample_bgr)
    assert lab.shape == (64, 64, 3)


def test_modular_point_ops(sample_bgr: np.ndarray) -> None:
    """Verify modular point operations."""
    bright = adjust_brightness(sample_bgr, 30)
    assert bright.dtype == np.uint8

    contrast = adjust_contrast(sample_bgr, 1.5)
    assert contrast.dtype == np.uint8

    neg = negative_transform(sample_bgr)
    assert np.array_equal(neg, 255 - sample_bgr)

    bin_img = apply_threshold(bgr_to_grayscale(sample_bgr), 100)
    assert set(np.unique(bin_img)).issubset({0, 255})


def test_modular_filters(sample_bgr: np.ndarray) -> None:
    """Verify modular linear and nonlinear filters."""
    k_mean = get_mean_kernel(3)
    assert np.isclose(kernel_sum(k_mean), 1.0)

    k_gauss = get_gaussian_kernel(5, 1.0)
    assert np.isclose(kernel_sum(k_gauss), 1.0)

    mean_filtered = apply_mean_filter(sample_bgr, 3)
    assert mean_filtered.shape == sample_bgr.shape

    gauss_filtered = apply_gaussian_filter(sample_bgr, 3, 1.0)
    assert gauss_filtered.shape == sample_bgr.shape

    sx, sy = get_sobel_kernels()
    assert np.isclose(kernel_sum(sx), 0.0)
    assert np.isclose(kernel_sum(sy), 0.0)

    sobel_edge = apply_sobel(sample_bgr, "x")
    assert sobel_edge.shape == sample_bgr.shape

    noisy = add_salt_pepper_noise(sample_bgr, amount=0.05, seed=42)
    denoised = apply_median_filter(noisy, 3)
    assert denoised.shape == sample_bgr.shape


def test_modular_geometric_and_drawing(sample_bgr: np.ndarray) -> None:
    """Verify modular geometric transforms and annotations."""
    cropped = crop_roi(sample_bgr, (10, 10, 40, 40))
    assert cropped.shape == (30, 30, 3)

    resized_dim = resize_by_dimensions(sample_bgr, 32, 32)
    assert resized_dim.shape == (32, 32, 3)

    resized_scale = resize_by_scale(sample_bgr, fx=0.5, fy=0.5)
    assert resized_scale.shape == (32, 32, 3)

    drawn = draw_rectangle(sample_bgr, (5, 5), (20, 20))
    drawn = draw_circle(drawn, (30, 30), 10)
    drawn = draw_line(drawn, (0, 0), (63, 63))
    drawn = draw_text(drawn, "Test", (10, 50))
    assert drawn.shape == sample_bgr.shape
