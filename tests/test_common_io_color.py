import numpy as np
import pytest
from src.common.color import (
    bgr_to_grayscale,
    bgr_to_hsv,
    bgr_to_lab,
    bgr_to_rgb,
    rgb_to_bgr,
)
from src.common.io import get_image_properties, load_image_cv, save_image_cv


def test_common_color_conversions():
    img_bgr = np.zeros((20, 20, 3), dtype=np.uint8)
    img_bgr[:, :, 0] = 255  # Pure Blue in BGR
    img_rgb = bgr_to_rgb(img_bgr)
    assert img_rgb[0, 0, 2] == 255
    assert img_rgb[0, 0, 0] == 0
    assert np.array_equal(rgb_to_bgr(img_rgb), img_bgr)
    assert bgr_to_grayscale(img_bgr).shape == (20, 20)
    assert bgr_to_hsv(img_bgr).shape == (20, 20, 3)
    assert bgr_to_lab(img_bgr).shape == (20, 20, 3)


def test_common_io(tmp_path):
    img = np.zeros((30, 40, 3), dtype=np.uint8)
    save_path = tmp_path / "test.png"
    saved_path = save_image_cv(img, save_path)
    assert saved_path.is_file()
    loaded = load_image_cv(save_path)
    assert loaded.shape == (30, 40, 3)
    props = get_image_properties(loaded)
    assert props["height"] == 30 and props["width"] == 40
