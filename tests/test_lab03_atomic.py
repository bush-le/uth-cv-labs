"""Tests for Lab 03 Canny edge detection atomic modules."""

import numpy as np

from src.labs.lab03_canny import (
    canny_advanced,
    canny_core,
    canny_metrics,
    canny_wrappers,
    visuals,
)


def test_canny_wrappers():
    img = np.zeros((60, 60), dtype=np.uint8)
    img[10:50, 10:50] = 200
    cv_edges = canny_wrappers.run_opencv_canny(img, 50, 150)
    assert cv_edges.shape == (60, 60)
    assert cv_edges.dtype == np.uint8

    auto_edges, low_t, high_t = canny_advanced.run_auto_canny(img, sigma_factor=0.33)
    assert auto_edges.shape == (60, 60)


def test_canny_core_pipeline():
    img = np.zeros((60, 60), dtype=np.uint8)
    img[15:45, 15:45] = 255
    res = canny_core.custom_canny_pipeline(img, low_threshold=30, high_threshold=90)
    edges = res["edges"] if isinstance(res, dict) else res
    assert edges.shape == (60, 60)


def test_canny_metrics():
    gt = np.zeros((40, 40), dtype=bool)
    gt[10, :] = True
    pred = np.zeros((40, 40), dtype=bool)
    pred[10, :] = True

    f1_dict = canny_metrics.calculate_edge_f1_score(pred, gt)
    assert np.isclose(f1_dict["f1"], 1.0)

    fom = canny_metrics.calculate_pratt_fom(pred, gt)
    assert np.isclose(fom, 1.0)


def test_canny_visuals():
    img = np.zeros((30, 30), dtype=np.uint8)
    fig1 = visuals.plot_canny_comparison([img, img], ["Img1", "Img2"])
    assert fig1 is not None

    fig2 = visuals.plot_hysteresis_steps(img, img, img, img)
    assert fig2 is not None
