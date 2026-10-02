"""Unit and smoke tests for image transformations and I/O module."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.data.transforms import (
    bgr_to_grayscale,
    bgr_to_hsv,
    bgr_to_lab,
    bgr_to_rgb,
    convert_image_format,
    crop_roi,
    crop_roi_pil,
    draw_circle,
    draw_line,
    draw_rectangle,
    draw_text,
    load_image_cv,
    load_image_pil,
    pil_to_grayscale,
    pil_to_hsv,
    resize_by_dimensions,
    resize_by_scale,
    resize_pil_by_dimensions,
    resize_pil_by_scale,
    rgb_to_bgr,
    save_image_cv,
)


@pytest.fixture
def dummy_bgr_image() -> np.ndarray:
    """Fixture providing a synthetic 100x120 3-channel BGR image."""
    img = np.zeros((100, 120, 3), dtype=np.uint8)
    img[:50, :60] = [255, 0, 0]  # Blue patch
    img[50:, 60:] = [0, 0, 255]  # Red patch
    return img


@pytest.fixture
def dummy_pil_image() -> Image.Image:
    """Fixture providing a synthetic 120x100 RGB PIL image."""
    return Image.new("RGB", (120, 100), color=(100, 150, 200))


# ==============================================================================
# Smoke Tests (@pytest.mark.smoke)
# ==============================================================================


@pytest.mark.smoke
def test_smoke_color_conversions(dummy_bgr_image: np.ndarray) -> None:
    """Smoke test: Verify all color space conversions execute without errors."""
    gray = bgr_to_grayscale(dummy_bgr_image)
    assert gray.shape == (100, 120)
    assert gray.dtype == np.uint8

    hsv = bgr_to_hsv(dummy_bgr_image)
    assert hsv.shape == (100, 120, 3)

    lab = bgr_to_lab(dummy_bgr_image)
    assert lab.shape == (100, 120, 3)

    rgb = bgr_to_rgb(dummy_bgr_image)
    assert rgb.shape == (100, 120, 3)


@pytest.mark.smoke
def test_smoke_crop_and_resize(dummy_bgr_image: np.ndarray) -> None:
    """Smoke test: Verify cropping and resizing basic operations."""
    roi = crop_roi(dummy_bgr_image, (10, 20, 50, 60))
    assert roi.shape == (40, 40, 3)

    scaled = resize_by_scale(dummy_bgr_image, fx=0.5, fy=0.5)
    assert scaled.shape == (50, 60, 3)

    fixed = resize_by_dimensions(dummy_bgr_image, target_width=80, target_height=60)
    assert fixed.shape == (60, 80, 3)


# ==============================================================================
# Normal Unit Tests
# ==============================================================================


def test_bgr_rgb_roundtrip(dummy_bgr_image: np.ndarray) -> None:
    """Verify BGR -> RGB -> BGR preserves identical values."""
    rgb = bgr_to_rgb(dummy_bgr_image)
    reconstructed_bgr = rgb_to_bgr(rgb)
    np.testing.assert_array_equal(dummy_bgr_image, reconstructed_bgr)


def test_invalid_color_space_input() -> None:
    """Verify ValueError is raised when 2D array is passed to 3-channel conversion."""
    invalid_2d = np.zeros((10, 10), dtype=np.uint8)
    with pytest.raises(ValueError, match="Expected 3-channel"):
        bgr_to_grayscale(invalid_2d)
    with pytest.raises(ValueError, match="Expected 3-channel"):
        bgr_to_hsv(invalid_2d)
    with pytest.raises(ValueError, match="Expected 3-channel"):
        bgr_to_lab(invalid_2d)


def test_crop_roi_validation(dummy_bgr_image: np.ndarray) -> None:
    """Verify out-of-bound and inverted crop boxes raise ValueError."""
    # x2 > width (width is 120)
    with pytest.raises(ValueError, match="Invalid crop box"):
        crop_roi(dummy_bgr_image, (0, 0, 150, 50))

    # inverted coords: x1 >= x2
    with pytest.raises(ValueError, match="Invalid crop box"):
        crop_roi(dummy_bgr_image, (50, 0, 20, 50))

    # negative coords
    with pytest.raises(ValueError, match="Invalid crop box"):
        crop_roi(dummy_bgr_image, (-5, 0, 20, 50))


def test_resize_validation(dummy_bgr_image: np.ndarray) -> None:
    """Verify non-positive resize parameters raise ValueError."""
    with pytest.raises(ValueError, match="must be positive"):
        resize_by_scale(dummy_bgr_image, fx=-1.0, fy=1.0)

    with pytest.raises(ValueError, match="must be positive"):
        resize_by_dimensions(dummy_bgr_image, target_width=0, target_height=50)


def test_pil_operations(dummy_pil_image: Image.Image) -> None:
    """Verify Pillow image color conversion, crop, and resize operations."""
    gray_pil = pil_to_grayscale(dummy_pil_image)
    assert gray_pil.mode == "L"
    assert gray_pil.size == (120, 100)

    hsv_pil = pil_to_hsv(dummy_pil_image)
    assert hsv_pil.mode == "HSV"

    cropped = crop_roi_pil(dummy_pil_image, (10, 10, 50, 40))
    assert cropped.size == (40, 30)

    resized_scale = resize_pil_by_scale(dummy_pil_image, 0.5, 0.5)
    assert resized_scale.size == (60, 50)

    resized_fixed = resize_pil_by_dimensions(dummy_pil_image, 200, 150)
    assert resized_fixed.size == (200, 150)


def test_drawing_functions(dummy_bgr_image: np.ndarray) -> None:
    """Verify geometric drawing and text annotation modify canvas without mutating original."""
    original_copy = dummy_bgr_image.copy()

    with_line = draw_line(dummy_bgr_image, (0, 0), (50, 50), color=(0, 255, 0))
    assert not np.array_equal(with_line, dummy_bgr_image)
    np.testing.assert_array_equal(dummy_bgr_image, original_copy)  # Immutable original

    with_rect = draw_rectangle(dummy_bgr_image, (10, 10), (40, 40), color=(0, 0, 255))
    assert not np.array_equal(with_rect, dummy_bgr_image)

    with_circle = draw_circle(dummy_bgr_image, (50, 50), radius=10, color=(255, 0, 0))
    assert not np.array_equal(with_circle, dummy_bgr_image)

    with_text = draw_text(dummy_bgr_image, "Test", (10, 30))
    assert not np.array_equal(with_text, dummy_bgr_image)


def test_file_io_and_conversions(tmp_path: Path, dummy_bgr_image: np.ndarray) -> None:
    """Verify saving, loading, and format conversions on disk."""
    save_path = tmp_path / "test_out.png"
    saved = save_image_cv(dummy_bgr_image, save_path)
    assert saved.is_file()

    loaded = load_image_cv(save_path)
    np.testing.assert_array_equal(loaded, dummy_bgr_image)

    # Format conversion: PNG -> BMP -> JPG
    bmp_path = tmp_path / "test_out.bmp"
    convert_image_format(save_path, bmp_path)
    assert bmp_path.is_file()

    jpg_path = tmp_path / "test_out.jpg"
    convert_image_format(bmp_path, jpg_path)
    assert jpg_path.is_file()


def test_file_io_nonexistent_path(tmp_path: Path) -> None:
    """Verify FileNotFoundError when loading a nonexistent file."""
    nonexistent = tmp_path / "ghost_image.jpg"
    with pytest.raises(FileNotFoundError):
        load_image_cv(nonexistent)
    with pytest.raises(FileNotFoundError):
        load_image_pil(nonexistent)
