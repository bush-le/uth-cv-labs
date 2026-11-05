"""Lab 03 Canny Edge Detection Package: Core 5-Stage, Wrappers, Advanced, Downstream, Metrics, and Pipeline."""

from src.labs.lab03_canny import (
    canny_advanced,
    canny_core,
    canny_downstream,
    canny_metrics,
    canny_wrappers,
    pipeline,
    visuals,
)
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
from src.labs.lab03_canny.pipeline import Lab03Pipeline

__all__ = [
    "canny_advanced",
    "canny_core",
    "canny_downstream",
    "canny_metrics",
    "canny_wrappers",
    "visuals",
    "pipeline",
    "custom_canny_pipeline",
    "run_opencv_canny",
    "run_skimage_canny",
    "run_auto_canny",
    "run_otsu_canny",
    "run_sobel",
    "run_laplacian",
    "run_di_zenzo_gradient",
    "run_multichannel_canny",
    "detect_hough_circles",
    "detect_hough_lines",
    "classify_polygon_contours",
    "segment_via_morphology_and_holes",
    "TemporalEdgeFilter",
    "calculate_pratt_fom",
    "calculate_hausdorff_distance",
    "calculate_edge_f1_score",
    "Lab03Pipeline",
]
