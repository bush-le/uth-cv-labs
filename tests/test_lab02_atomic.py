"""Tests for Lab 02 atomic filtering modules."""

import numpy as np
import pytest

from src.labs.lab02_filters import (
    edge_filters,
    filter_utils,
    linear_filters,
    nonlinear_filters,
    point_ops,
    visuals,
)


def test_point_operations():
    img = np.full((20, 20, 3), 100, dtype=np.uint8)
    # Brightness
    bright = point_ops.adjust_brightness(img, 50)
    assert bright[0, 0, 0] == 150
    # Contrast
    contrast = point_ops.adjust_contrast(img, 1.5)
    assert contrast[0, 0, 0] == 150
    # Negative
    neg = point_ops.negative_transform(img)
    assert neg[0, 0, 0] == 155
    # Threshold
    gray = np.array([[50, 150], [200, 100]], dtype=np.uint8)
    thresh = point_ops.apply_threshold(gray, 127)
    assert thresh[0, 0] == 0
    assert thresh[1, 0] == 255


def test_linear_filters_and_kernels():
    k_mean = linear_filters.get_mean_kernel(3)
    assert k_mean.shape == (3, 3)
    assert np.isclose(linear_filters.kernel_sum(k_mean), 1.0)

    k_gauss = linear_filters.get_gaussian_kernel(5, 1.0)
    assert k_gauss.shape == (5, 5)
    assert np.isclose(linear_filters.kernel_sum(k_gauss), 1.0)

    img = np.ones((20, 20), dtype=np.uint8) * 100
    mean_filtered = linear_filters.apply_mean_filter(img, 3)
    assert mean_filtered.shape == (20, 20)


def test_nonlinear_filters_and_noise():
    img = np.full((30, 30), 128, dtype=np.uint8)
    noisy = nonlinear_filters.add_salt_pepper_noise(img, amount=0.1, seed=42)
    assert noisy.shape == img.shape
    denoised = nonlinear_filters.apply_median_filter(noisy, 3)
    assert denoised.shape == img.shape


def test_edge_filters():
    sobel_x, sobel_y = edge_filters.get_sobel_kernels()
    assert sobel_x.shape == (3, 3)
    assert sobel_y.shape == (3, 3)
    assert np.isclose(linear_filters.kernel_sum(sobel_x), 0.0)

    img = np.zeros((30, 30), dtype=np.uint8)
    img[:, 15:] = 255
    gx, gy, mag = edge_filters.apply_sobel(img)
    assert mag.shape == (30, 30)
    assert mag[:, 15].mean() > 0


def test_visuals_generation():
    dummy = np.zeros((20, 20, 3), dtype=np.uint8)
    fig1 = visuals.plot_point_ops_grid(dummy, dummy, dummy, dummy)
    assert fig1 is not None

    fig2 = visuals.plot_filter_comparison([dummy, dummy], ["Orig", "Filt"])
    assert fig2 is not None
