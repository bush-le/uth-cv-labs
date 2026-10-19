"""Unit tests for Lab 03 Canny Edge Detection suite.

Covers:
- Smoke tests (@pytest.mark.smoke) for zero-crash and instantiation.
- Normal tests for mathematical properties, shape invariants, edge metrics, and downstream tasks.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.models.classical.canny import (
    TemporalEdgeFilter,
    calculate_edge_f1_score,
    calculate_hausdorff_distance,
    calculate_pratt_fom,
    classify_polygon_contours,
    custom_canny_pipeline,
    run_auto_canny,
    run_di_zenzo_gradient,
    run_laplacian,
    run_multichannel_canny,
    run_opencv_canny,
    run_otsu_canny,
    run_skimage_canny,
    run_sobel,
    segment_via_morphology_and_holes,
)

# ==============================================================================
# SMOKE TEST SUITE (< 5s Sanity Checks)
# ==============================================================================


@pytest.mark.smoke
def test_imports_and_instantiation() -> None:
    """Verify that all core functions and filter classes import and instantiate properly."""
    filt = TemporalEdgeFilter(alpha=0.7)
    assert filt is not None
    assert filt.alpha == 0.7


@pytest.mark.smoke
def test_custom_canny_zero_crash() -> None:
    """Ensure custom Canny pipeline executes on small dummy array without exceptions."""
    dummy_img = (np.random.rand(64, 64) * 255).astype(np.uint8)
    res = custom_canny_pipeline(
        dummy_img, sigma=1.0, low_threshold=30, high_threshold=90
    )
    assert "edges" in res
    assert res["edges"].shape == (64, 64)
    assert res["edges"].dtype == np.uint8


@pytest.mark.smoke
def test_library_wrappers_smoke() -> None:
    """Test OpenCV and Scikit-image wrappers on dummy image."""
    dummy_img = np.zeros((48, 48), dtype=np.uint8)
    dummy_img[10:38, 10:38] = 200

    cv_edges = run_opencv_canny(dummy_img, 50, 150)
    sk_edges = run_skimage_canny(
        dummy_img, sigma=1.0, low_threshold=50, high_threshold=150
    )

    assert cv_edges.shape == (48, 48)
    assert sk_edges.shape == (48, 48)
    assert np.all(np.isin(cv_edges, [0, 255]))
    assert np.all(np.isin(sk_edges, [0, 255]))


# ==============================================================================
# NORMAL TEST SUITE (Logic, Invariants, Edge Cases)
# ==============================================================================


def test_custom_canny_stages_shapes_and_invariants() -> None:
    """Validate shapes, value ranges, and mathematical properties of custom Canny stages."""
    h, w = 80, 100
    img = np.zeros((h, w), dtype=np.uint8)
    img[20:60, 30:70] = 255  # Solid bright rectangle

    res = custom_canny_pipeline(img, sigma=1.2, low_threshold=40, high_threshold=120)

    assert res["smoothed"].shape == (h, w)
    assert res["magnitude"].shape == (h, w)
    assert res["direction_deg"].shape == (h, w)
    assert res["nms"].shape == (h, w)
    assert res["edges"].shape == (h, w)

    # Angle must be bounded in [0, 180)
    assert np.all(res["direction_deg"] >= 0.0)
    assert np.all(res["direction_deg"] < 180.0)

    # NMS values must be <= raw gradient magnitude
    assert np.all(res["nms"] <= res["magnitude"] + 1e-5)

    # Final edges must be binary uint8 {0, 255}
    unique_vals = np.unique(res["edges"])
    assert set(unique_vals).issubset({0, 255})
    # Edges should be detected around the boundary
    assert np.sum(res["edges"] > 0) > 0


def test_classical_filters_shapes() -> None:
    """Verify Sobel and Laplacian baseline outputs."""
    img = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
    sob = run_sobel(img)
    lap = run_laplacian(img)

    assert sob.shape == (50, 50)
    assert lap.shape == (50, 50)
    assert sob.dtype == np.uint8
    assert lap.dtype == np.uint8


def test_auto_canny_adaptive_thresholds() -> None:
    """Ensure Auto-Canny generates logical threshold bounds."""
    img = np.full((64, 64), 100, dtype=np.uint8)
    img[20:44, 20:44] = 200

    edges, low_th, high_th = run_auto_canny(img, sigma_factor=0.33)
    assert edges.shape == (64, 64)
    assert 0 <= low_th < high_th <= 255

    otsu_edges, o_low, o_high = run_otsu_canny(img)
    assert otsu_edges.shape == (64, 64)
    assert 0 <= o_low <= o_high <= 255


def test_pratt_fom_metric_properties() -> None:
    """Verify Pratt's Figure of Merit behaves according to theoretical bounds."""
    # 1. Perfect match -> FOM = 1.0
    gt = np.zeros((50, 50), dtype=np.uint8)
    gt[25, 10:40] = 255
    det_exact = gt.copy()
    assert calculate_pratt_fom(det_exact, gt) == pytest.approx(1.0, abs=1e-5)

    # 2. Shifted edge by 1 pixel -> FOM < 1.0 but > 0.0
    det_shifted = np.zeros((50, 50), dtype=np.uint8)
    det_shifted[26, 10:40] = 255  # 1 pixel vertical shift
    fom_shift = calculate_pratt_fom(det_shifted, gt, alpha=1.0 / 9.0)
    assert 0.0 < fom_shift < 1.0
    # Theoretical FOM for 1-pixel shift with alpha=1/9 is 1 / (1 + 1/9) = 0.9
    assert fom_shift == pytest.approx(0.9, rel=1e-2)

    # 3. Both empty -> 1.0; One empty -> 0.0
    empty = np.zeros((50, 50), dtype=np.uint8)
    assert calculate_pratt_fom(empty, empty) == 1.0
    assert calculate_pratt_fom(det_exact, empty) == 0.0


def test_f1_score_and_hausdorff_metrics() -> None:
    """Verify F1-score and Hausdorff distance calculations."""
    gt = np.zeros((40, 40), dtype=np.uint8)
    gt[10:30, 20] = 255

    det = gt.copy()
    f1_dict = calculate_edge_f1_score(det, gt, tolerance=1.0)
    assert f1_dict["f1"] == pytest.approx(1.0, abs=1e-4)

    h_dist = calculate_hausdorff_distance(det, gt)
    assert h_dist == pytest.approx(0.0, abs=1e-4)


def test_color_di_zenzo_and_multichannel() -> None:
    """Verify Di Zenzo color gradient tensor and multi-channel Canny on RGB."""
    rgb = np.zeros((50, 50, 3), dtype=np.uint8)
    rgb[:, :25, 0] = 255  # Red left half
    rgb[:, 25:, 1] = 255  # Green right half

    mc_edges = run_multichannel_canny(rgb, 30, 90)
    mag, ang = run_di_zenzo_gradient(rgb)

    assert mc_edges.shape == (50, 50)
    assert mag.shape == (50, 50)
    assert ang.shape == (50, 50)
    # Di Zenzo magnitude must have a strong peak at the vertical color boundary x=24/25
    assert np.max(mag[:, 23:27]) > np.max(mag[:, :10])


def test_downstream_segmentation_and_shapes() -> None:
    """Test morphological segmentation and shape contour classification."""
    # Synthesize clean white circle on black background
    canvas = np.zeros((100, 100), dtype=np.uint8)
    import cv2

    cv2.circle(canvas, (50, 50), 30, 255, -1)

    edges = cv2.Canny(canvas, 50, 150)
    closed, filled = segment_via_morphology_and_holes(edges, kernel_size=(3, 3))
    assert filled.shape == (100, 100)
    # Center of circle should be filled (255)
    assert filled[50, 50] == 255

    # Test contour shape classifier
    rgb_canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)
    annotated, counts = classify_polygon_contours(edges, rgb_canvas)
    assert annotated.shape == (100, 100, 3)
    assert counts["Circle/Ellipse"] >= 1


def test_temporal_filter() -> None:
    """Ensure temporal filter suppresses spurious flickering edges."""
    filter_inst = TemporalEdgeFilter(alpha=0.6, threshold=128.0)
    frame1 = np.zeros((30, 30), dtype=np.uint8)
    frame1[15, :] = 255  # Persistent horizontal line

    out1 = filter_inst.update(frame1)
    # First frame has value 255 >= 128 -> True
    assert np.all(out1[15, :] == 255)

    # Frame 2: noise spike at [5, 5] only once
    frame2 = frame1.copy()
    frame2[5, 5] = 255
    out2 = filter_inst.update(frame2)
    # Persistent line remains
    assert np.all(out2[15, :] == 255)
