"""Canny Edge Detector facade module.

Backward-compatibility shim re-exporting from src.labs.lab03_canny.
"""

from __future__ import annotations

from src.labs.lab03_canny.canny_advanced import (
    run_auto_canny,
    run_di_zenzo_gradient,
    run_multichannel_canny,
    run_otsu_canny,
)
from src.labs.lab03_canny.canny_core import custom_canny_pipeline
from src.labs.lab03_canny.canny_downstream import (
    TemporalEdgeFilter,
    classify_polygon_contours,
    detect_hough_circles,
    detect_hough_lines,
    segment_via_morphology_and_holes,
)
from src.labs.lab03_canny.canny_metrics import (
    calculate_edge_f1_score,
    calculate_hausdorff_distance,
    calculate_pratt_fom,
)
from src.labs.lab03_canny.canny_wrappers import (
    run_laplacian,
    run_opencv_canny,
    run_skimage_canny,
    run_sobel,
)

__all__ = [
    # Core
    "custom_canny_pipeline",
    # Wrappers
    "run_opencv_canny",
    "run_skimage_canny",
    "run_sobel",
    "run_laplacian",
    # Advanced
    "run_auto_canny",
    "run_otsu_canny",
    "run_multichannel_canny",
    "run_di_zenzo_gradient",
    # Downstream Applications
    "detect_hough_circles",
    "detect_hough_lines",
    "classify_polygon_contours",
    "segment_via_morphology_and_holes",
    "TemporalEdgeFilter",
    # Evaluation Metrics
    "calculate_pratt_fom",
    "calculate_hausdorff_distance",
    "calculate_edge_f1_score",
]
