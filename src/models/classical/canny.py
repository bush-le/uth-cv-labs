"""Canny Edge Detector algorithmic suite facade.

Consolidates and re-exports:
1. Fully vectorized 5-stage custom Canny pipeline (`canny_core`).
2. OpenCV and Scikit-image wrappers with baseline filters (`canny_wrappers`).
3. Auto-Canny, Otsu, and Di Zenzo color gradient tensor detectors (`canny_advanced`).
4. Quantitative evaluation metrics: Pratt's FOM, F1-Score, Hausdorff (`canny_metrics`).
5. Downstream applications: morphology segmentation, Hough lines/circles, contours, video filter (`canny_downstream`).
"""

from __future__ import annotations

from src.models.classical.canny_advanced import (
    run_auto_canny,
    run_di_zenzo_gradient,
    run_multichannel_canny,
    run_otsu_canny,
)
from src.models.classical.canny_core import (
    custom_canny_pipeline,
)
from src.models.classical.canny_downstream import (
    TemporalEdgeFilter,
    classify_polygon_contours,
    detect_hough_circles,
    detect_hough_lines,
    segment_via_morphology_and_holes,
)
from src.models.classical.canny_metrics import (
    calculate_edge_f1_score,
    calculate_hausdorff_distance,
    calculate_pratt_fom,
)
from src.models.classical.canny_wrappers import (
    run_laplacian,
    run_opencv_canny,
    run_skimage_canny,
    run_sobel,
)

__all__ = [
    # Core Pipeline
    "custom_canny_pipeline",
    # Library & Baseline Wrappers
    "run_opencv_canny",
    "run_skimage_canny",
    "run_sobel",
    "run_laplacian",
    # Advanced & Color Gradient
    "run_auto_canny",
    "run_otsu_canny",
    "run_multichannel_canny",
    "run_di_zenzo_gradient",
    # Evaluation Metrics
    "calculate_pratt_fom",
    "calculate_edge_f1_score",
    "calculate_hausdorff_distance",
    # Downstream Tasks & Temporal Filter
    "segment_via_morphology_and_holes",
    "detect_hough_lines",
    "detect_hough_circles",
    "classify_polygon_contours",
    "TemporalEdgeFilter",
]
