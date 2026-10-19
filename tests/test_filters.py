"""Unit and smoke tests for point operators and spatial filtering module."""

import numpy as np
import pytest

from src.data.filters import (
    add_salt_pepper_noise,
    adjust_brightness,
    adjust_contrast,
    apply_bilateral_filter,
    apply_custom_filter,
    apply_gaussian_filter,
    apply_laplacian,
    apply_mean_filter,
    apply_median_filter,
    apply_prewitt,
    apply_sharpen_filter,
    apply_sobel,
    apply_threshold,
    get_gaussian_kernel,
    get_laplacian_kernel,
    get_mean_kernel,
    get_prewitt_kernels,
    get_sharpen_kernel,
    get_sobel_kernels,
    gradient_magnitude,
    kernel_sum,
    negative_transform,
)


@pytest.fixture
def flat_image() -> np.ndarray:
    """Fixture providing a flat 32x32 grayscale image with constant value 100."""
    return np.full((32, 32), 100, dtype=np.uint8)


@pytest.fixture
def edge_image() -> np.ndarray:
    """Fixture providing a 32x32 grayscale image with a vertical edge (100 | 50)."""
    img = np.full((32, 32), 100, dtype=np.uint8)
    img[:, 16:] = 50
    return img


@pytest.fixture
def dummy_bgr_image() -> np.ndarray:
    """Fixture providing a synthetic 40x40 3-channel BGR image."""
    img = np.zeros((40, 40, 3), dtype=np.uint8)
    img[:20, :20] = [255, 0, 0]
    img[20:, 20:] = [0, 0, 255]
    return img


# ==============================================================================
# Smoke Tests (@pytest.mark.smoke)
# ==============================================================================


@pytest.mark.smoke
def test_smoke_point_operations(dummy_bgr_image: np.ndarray) -> None:
    """Smoke test: all point operations execute and preserve shape/dtype."""
    for result in (
        adjust_brightness(dummy_bgr_image, 30),
        adjust_contrast(dummy_bgr_image, 1.5),
        negative_transform(dummy_bgr_image),
        apply_threshold(dummy_bgr_image, 127),
    ):
        assert result.shape == dummy_bgr_image.shape
        assert result.dtype == np.uint8


@pytest.mark.smoke
def test_smoke_linear_filters(dummy_bgr_image: np.ndarray) -> None:
    """Smoke test: all linear filters execute and preserve shape."""
    for result in (
        apply_mean_filter(dummy_bgr_image, 3),
        apply_gaussian_filter(dummy_bgr_image, 3, 1.0),
        apply_sharpen_filter(dummy_bgr_image),
    ):
        assert result.shape == dummy_bgr_image.shape


@pytest.mark.smoke
def test_smoke_edge_and_nonlinear(dummy_bgr_image: np.ndarray) -> None:
    """Smoke test: edge detectors and non-linear filters execute cleanly."""
    for result in (
        apply_sobel(dummy_bgr_image, "x"),
        apply_sobel(dummy_bgr_image, "y"),
        apply_prewitt(dummy_bgr_image, "x"),
        apply_laplacian(dummy_bgr_image),
        apply_median_filter(dummy_bgr_image, 3),
        apply_bilateral_filter(dummy_bgr_image),
    ):
        assert result.shape == dummy_bgr_image.shape


# ==============================================================================
# Normal Unit Tests
# ==============================================================================


def test_brightness_clipping(dummy_bgr_image: np.ndarray) -> None:
    """Verify brightness addition saturates at [0, 255] without overflow."""
    bright = adjust_brightness(dummy_bgr_image, beta=1000)
    assert bright.max() == 255
    dark = adjust_brightness(dummy_bgr_image, beta=-1000)
    assert dark.min() == 0


def test_contrast_validation(dummy_bgr_image: np.ndarray) -> None:
    """Verify non-positive contrast gain raises ValueError."""
    with pytest.raises(ValueError, match="must be positive"):
        adjust_contrast(dummy_bgr_image, alpha=0.0)
    with pytest.raises(ValueError, match="must be positive"):
        adjust_contrast(dummy_bgr_image, alpha=-1.0)


def test_negative_involution(dummy_bgr_image: np.ndarray) -> None:
    """Verify negative transform is its own inverse (applied twice = identity)."""
    twice = negative_transform(negative_transform(dummy_bgr_image))
    np.testing.assert_array_equal(twice, dummy_bgr_image)


def test_threshold_binary_output(flat_image: np.ndarray) -> None:
    """Verify thresholding yields only 0 or 255 values."""
    binary = apply_threshold(flat_image, threshold=127)
    assert set(np.unique(binary).tolist()) <= {0, 255}
    # Flat value 100 < 127 -> all background (0)
    assert binary.max() == 0


def test_threshold_validation(flat_image: np.ndarray) -> None:
    """Verify out-of-range threshold raises ValueError."""
    with pytest.raises(ValueError, match="must be in"):
        apply_threshold(flat_image, threshold=300)


def test_mean_kernel_sum_is_one() -> None:
    """Verify mean kernel coefficients sum to 1 (brightness-preserving)."""
    assert abs(kernel_sum(get_mean_kernel(3)) - 1.0) < 1e-6
    assert abs(kernel_sum(get_mean_kernel(5)) - 1.0) < 1e-6


def test_gaussian_kernel_sum_is_one() -> None:
    """Verify Gaussian kernel coefficients sum to 1 and are symmetric."""
    kernel = get_gaussian_kernel(5, sigma=1.5)
    assert abs(kernel_sum(kernel) - 1.0) < 1e-6
    np.testing.assert_allclose(kernel, kernel[::-1, ::-1], atol=1e-6)


def test_edge_kernels_sum_to_zero() -> None:
    """Verify all derivative (edge) kernels have sum = 0."""
    sobel_x, sobel_y = get_sobel_kernels()
    prewitt_x, prewitt_y = get_prewitt_kernels()
    for kernel in (sobel_x, sobel_y, prewitt_x, prewitt_y, get_laplacian_kernel()):
        assert abs(kernel_sum(kernel)) < 1e-6


def test_sharpen_kernel_sum_is_one() -> None:
    """Verify sharpen kernel sums to 1 (Identity - Laplacian)."""
    sharpen = get_sharpen_kernel()
    assert abs(kernel_sum(sharpen) - 1.0) < 1e-6
    assert kernel_sum(sharpen) == pytest.approx(
        kernel_sum(np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]))
        - kernel_sum(get_laplacian_kernel())
    )


def test_flat_region_derivative_is_zero(flat_image: np.ndarray) -> None:
    """Verify derivative filters output zero on a flat region (constant derivative = 0)."""
    laplacian = apply_laplacian(flat_image)
    # Interior of the flat image should be exactly zero
    assert laplacian[1:-1, 1:-1].max() == 0
    sobel = apply_sobel(flat_image, "x")
    assert sobel[1:-1, 1:-1].max() == 0


def test_flat_region_mean_filter_preserves_value(flat_image: np.ndarray) -> None:
    """Verify mean filter on a flat region keeps the same value (sum = 1)."""
    blurred = apply_mean_filter(flat_image, ksize=3)
    assert blurred[1:-1, 1:-1].max() == 100
    assert blurred[1:-1, 1:-1].min() == 100


def test_edge_detection_nonzero_on_edge(edge_image: np.ndarray) -> None:
    """Verify Sobel produces a non-zero response at a vertical boundary."""
    sobel_x = apply_sobel(edge_image, "x")
    # A vertical edge should activate the x-gradient near column 16
    assert sobel_x[:, 15:17].max() > 0


def test_gradient_magnitude_shape_mismatch() -> None:
    """Verify mismatched gradient shapes raise ValueError."""
    grad_x = np.zeros((10, 10), dtype=np.uint8)
    grad_y = np.zeros((10, 12), dtype=np.uint8)
    with pytest.raises(ValueError, match="must match"):
        gradient_magnitude(grad_x, grad_y)


def test_median_filter_removes_salt_pepper(dummy_bgr_image: np.ndarray) -> None:
    """Verify median filter reduces salt-and-pepper noise."""
    noisy = add_salt_pepper_noise(dummy_bgr_image, amount=0.05, seed=1)
    denoised = apply_median_filter(noisy, ksize=3)
    # Denoised image should be closer to the original than the noisy one
    err_noisy = np.abs(noisy.astype(int) - dummy_bgr_image.astype(int)).mean()
    err_denoised = np.abs(denoised.astype(int) - dummy_bgr_image.astype(int)).mean()
    assert err_denoised < err_noisy


def test_salt_pepper_reproducible(dummy_bgr_image: np.ndarray) -> None:
    """Verify the noise generator is deterministic for a fixed seed."""
    noisy_a = add_salt_pepper_noise(dummy_bgr_image, amount=0.05, seed=7)
    noisy_b = add_salt_pepper_noise(dummy_bgr_image, amount=0.05, seed=7)
    np.testing.assert_array_equal(noisy_a, noisy_b)


def test_bilateral_preserves_edges(dummy_bgr_image: np.ndarray) -> None:
    """Verify bilateral filter keeps a hard edge between the two color patches."""
    filtered = apply_bilateral_filter(
        dummy_bgr_image, d=9, sigma_color=75, sigma_space=75
    )
    # Center of the blue patch stays blue-ish, center of red patch stays red-ish
    assert filtered[10, 10, 0] > filtered[30, 30, 0]
    assert filtered[30, 30, 2] > filtered[10, 10, 2]


def test_invalid_image_input() -> None:
    """Verify non-uint8 or empty inputs are rejected."""
    with pytest.raises(ValueError, match="dtype uint8"):
        adjust_brightness(np.zeros((10, 10), dtype=np.float32), 10)
    with pytest.raises(ValueError, match="non-empty"):
        negative_transform(np.array([], dtype=np.uint8))


def test_custom_filter_kernel_validation(dummy_bgr_image: np.ndarray) -> None:
    """Verify a non-2D kernel is rejected by apply_custom_filter."""
    with pytest.raises(ValueError, match="must be 2D"):
        apply_custom_filter(dummy_bgr_image, np.ones((3, 3, 3), dtype=np.float32))


def test_median_ksize_validation(dummy_bgr_image: np.ndarray) -> None:
    """Verify invalid median kernel sizes raise ValueError."""
    with pytest.raises(ValueError, match="odd integer > 1"):
        apply_median_filter(dummy_bgr_image, ksize=2)
    with pytest.raises(ValueError, match="odd integer > 1"):
        apply_median_filter(dummy_bgr_image, ksize=1)
