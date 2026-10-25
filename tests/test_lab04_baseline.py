"""Check the documented baseline and calibration-only threshold workflow."""

import cv2
import numpy as np
import pytest
import pywt

from src.models.classical.wavelet_hash import (
    hamming_distance,
    wavelet_hash,
    wavelet_hash_color,
)
from src.pipelines.lab04_evaluation import (
    compare_hash_methods,
    score_pairs,
    select_threshold,
)


def test_documented_baseline_matches_intermediate_coefficients() -> None:
    """The grayscale comparison matches explicit LL resampling and median bits."""
    image = np.random.default_rng(42).integers(0, 256, (141, 141), dtype=np.uint8)
    gray = cv2.resize(
        image.astype(np.float32) / 255, (256, 256), interpolation=cv2.INTER_AREA
    )
    ll = pywt.wavedec2(gray, "db4", mode="symmetric", level=1)[0]
    ll8 = cv2.resize(ll, (8, 8), interpolation=cv2.INTER_AREA)
    expected = (ll8 > np.median(ll8)).astype(np.uint8).ravel()
    assert np.array_equal(wavelet_hash(image), expected)
    assert np.array_equal(wavelet_hash(image.astype(np.float32) / 255), expected)


def test_rgb_baseline_preserves_channel_order_and_192_bits() -> None:
    """RGB hash is the R|G|B concatenation without grayscale conversion."""
    rgb = np.random.default_rng(17).integers(0, 256, (141, 141, 3), dtype=np.uint8)
    expected = np.concatenate([wavelet_hash(rgb[:, :, i]) for i in range(3)])
    actual = wavelet_hash_color(rgb)
    assert actual.shape == (192,)
    assert actual.dtype == np.uint8
    assert np.array_equal(actual, expected)
    assert np.array_equal(wavelet_hash_color(rgb.astype(np.float32) / 255), expected)
    assert not np.array_equal(actual, wavelet_hash_color(rgb[:, :, ::-1]))


@pytest.mark.parametrize("shape", [(10, 10), (10, 10, 4), (0, 10, 3)])
def test_rgb_baseline_rejects_non_rgb_shapes(shape: tuple[int, ...]) -> None:
    """The RGB API does not silently discard an alpha channel or use grayscale."""
    with pytest.raises(ValueError):
        wavelet_hash_color(np.zeros(shape, dtype=np.uint8))


@pytest.mark.parametrize(
    "left,right", [([], []), ([0], [1, 0]), ([0, 256], [1, 0]), ([[0]], [[1]])]
)
def test_invalid_hashes_are_rejected(left: list, right: list) -> None:
    """Do not silently truncate, wrap values or compare nonbinary data."""
    with pytest.raises(ValueError):
        hamming_distance(left, right)


@pytest.mark.smoke
def test_calibration_threshold_is_frozen_for_evaluation() -> None:
    """Evaluation errors remain visible rather than refitting its threshold."""
    labels = np.array([1, 1, 0, 0])
    threshold = select_threshold(labels, np.array([0, 2, 10, 12]))
    assert threshold == 2
    scores = score_pairs(labels, np.array([0, 6, 2, 12]), threshold)
    assert (scores["tp"], scores["tn"], scores["fp"], scores["fn"]) == (1, 1, 1, 1)
    assert scores["accuracy"] == 0.5


def test_method_comparison_keeps_calibration_threshold() -> None:
    """The advanced comparison must leave an evaluation false negative visible."""
    images = {
        "a": np.array([0, 0], dtype=np.uint8),
        "a_edit": np.array([0, 1], dtype=np.uint8),
        "b": np.array([1, 1], dtype=np.uint8),
        "b_edit": np.array([0, 0], dtype=np.uint8),
    }
    pairs = {
        "calibration": [
            {"left_id": "a", "right_id": "a_edit"},
            {"left_id": "a", "right_id": "b"},
        ],
        "evaluation": [
            {"left_id": "b", "right_id": "b_edit"},
            {"left_id": "b", "right_id": "a"},
        ],
    }
    labels = {split: np.array([1, 0]) for split in pairs}
    result = compare_hash_methods(
        [("example", lambda image: image, images)], pairs, labels
    )[0]
    assert result["bits"] == 2
    assert result["evaluation"]["threshold"] == 0.5
    assert result["evaluation"]["fn"] == 1


def test_no_positive_predictions_and_missing_class() -> None:
    """Zero prediction denominator is explicit; ROC needs both labels."""
    result = score_pairs(np.array([0, 1]), np.array([2, 4]), -1)
    assert result["precision"] == result["f1"] == 0
    with pytest.raises(ValueError):
        score_pairs(np.array([1, 1]), np.array([0, 1]), 1)


def test_constant_image_has_zero_baseline_hash() -> None:
    """Ties at the median map to zero for an all-black image."""
    assert not wavelet_hash(np.zeros((64, 64), dtype=np.uint8)).any()
