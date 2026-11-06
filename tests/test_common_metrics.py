import numpy as np
import pytest

from src.common.metrics import (
    calculate_binary_confusion_matrix,
    calculate_classification_metrics,
    hamming_distance,
    normalized_hamming_distance,
)


def test_common_classification_metrics():
    preds = np.array([1, 0, 1, 1, 0])
    gt = np.array([1, 0, 0, 1, 0])
    cm = calculate_binary_confusion_matrix(preds, gt)
    assert cm == {"tp": 2, "fp": 1, "tn": 2, "fn": 0}
    metrics = calculate_classification_metrics(preds, gt)
    assert metrics["accuracy"] == pytest.approx(4 / 5)


def test_common_hamming_distance():
    h1 = np.array([1, 0, 1, 0])
    h2 = np.array([1, 1, 1, 0])
    assert hamming_distance(h1, h2) == 1
    assert normalized_hamming_distance(h1, h2) == 0.25
