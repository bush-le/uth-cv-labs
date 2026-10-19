"""Canny Edge Detector algorithmic suite, wrappers, and edge analysis tools.

Implements:
1. Fully vectorized 5-stage custom Canny pipeline.
2. OpenCV and Scikit-image wrappers with standardized interfaces.
3. Classic baseline comparison filters (Sobel, Laplacian).
4. Auto-Canny and adaptive thresholding methods.
5. Multi-channel bitwise OR and Di Zenzo color gradient tensor detectors.
6. Quantitative evaluation metrics: Pratt's Figure of Merit (FOM), Precision/Recall/F1, Hausdorff.
7. Downstream task modules: Morphological segmentation, Hough transforms, Contour polygons.
"""

from __future__ import annotations

import cv2
import numpy as np
import scipy.ndimage
from scipy.spatial.distance import directed_hausdorff
from skimage.feature import canny as skimage_canny

from src.common.logger import get_logger

logger = get_logger(__name__)


# ==============================================================================
# 1. Fully Vectorized Custom 5-Step Canny Pipeline
# ==============================================================================


def custom_canny_pipeline(
    image: np.ndarray,
    sigma: float = 1.0,
    low_threshold: float = 50.0,
    high_threshold: float = 150.0,
) -> dict[str, np.ndarray]:
    """Execute the 5-stage Canny Edge Detector pipeline using pure vectorized NumPy.

    Stages:
        1. Gaussian Smoothing (noise reduction)
        2. Gradient Magnitude & Direction (Sobel operators)
        3. Non-Maximum Suppression (1-pixel thinning along gradient direction)
        4. Double Thresholding (partitioning strong, weak, non-edges)
        5. Edge Tracking by Hysteresis (8-connected component propagation)

    Args:
        image: Grayscale input image of shape (H, W), dtype uint8 or float.
        sigma: Standard deviation for Gaussian smoothing.
        low_threshold: Lower gradient threshold.
        high_threshold: Upper gradient threshold.

    Returns:
        dict[str, np.ndarray]: Intermediate outputs and final edge map:
            - 'smoothed': (H, W) float32
            - 'grad_x': (H, W) float32
            - 'grad_y': (H, W) float32
            - 'magnitude': (H, W) float32
            - 'direction_deg': (H, W) float32 in [0, 180)
            - 'nms': (H, W) float32
            - 'strong_edges': (H, W) bool
            - 'weak_edges': (H, W) bool
            - 'edges': (H, W) uint8 with values 0 and 255
    """
    if image.ndim != 2:
        raise ValueError(f"Input image must be 2D grayscale, got shape {image.shape}")

    img_float = image.astype(np.float32)

    # --------------------------------------------------------------------------
    # Step 1: Gaussian Smoothing
    # --------------------------------------------------------------------------
    radius = int(np.ceil(3.0 * sigma))
    ksize = 2 * radius + 1
    smoothed = cv2.GaussianBlur(img_float, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)

    # --------------------------------------------------------------------------
    # Step 2: Sobel Gradients
    # --------------------------------------------------------------------------
    grad_x = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)

    magnitude = np.hypot(grad_x, grad_y)
    # Angle in radians -> degrees in [0, 180)
    direction_rad = np.arctan2(grad_y, grad_x)
    direction_deg = np.rad2deg(direction_rad) % 180.0

    # --------------------------------------------------------------------------
    # Step 3: Vectorized Non-Maximum Suppression (NMS)
    # --------------------------------------------------------------------------
    h, w = magnitude.shape
    pad_m = np.pad(magnitude, 1, mode="constant", constant_values=0.0)

    # Sector masks
    mask0 = ((direction_deg >= 0.0) & (direction_deg < 22.5)) | (direction_deg >= 157.5)
    mask1 = (direction_deg >= 22.5) & (direction_deg < 67.5)
    mask2 = (direction_deg >= 67.5) & (direction_deg < 112.5)
    mask3 = (direction_deg >= 112.5) & (direction_deg < 157.5)

    # Neighbor comparisons
    is_max0 = mask0 & (magnitude >= pad_m[1:-1, :-2]) & (magnitude >= pad_m[1:-1, 2:])
    is_max1 = mask1 & (magnitude >= pad_m[:-2, 2:]) & (magnitude >= pad_m[2:, :-2])
    is_max2 = mask2 & (magnitude >= pad_m[:-2, 1:-1]) & (magnitude >= pad_m[2:, 1:-1])
    is_max3 = mask3 & (magnitude >= pad_m[:-2, :-2]) & (magnitude >= pad_m[2:, 2:])

    keep_mask = is_max0 | is_max1 | is_max2 | is_max3
    nms = np.where(keep_mask, magnitude, 0.0).astype(np.float32)

    # --------------------------------------------------------------------------
    # Step 4: Double Thresholding
    # --------------------------------------------------------------------------
    strong_edges = nms >= high_threshold
    weak_edges = (nms >= low_threshold) & (nms < high_threshold)

    # --------------------------------------------------------------------------
    # Step 5: Edge Tracking by Hysteresis (Vectorized 8-Connectivity)
    # --------------------------------------------------------------------------
    candidate_map = strong_edges | weak_edges
    structuring_element = np.ones((3, 3), dtype=bool)
    labeled_map, num_features = scipy.ndimage.label(
        candidate_map, structure=structuring_element
    )

    strong_labels = np.unique(labeled_map[strong_edges])
    strong_labels = strong_labels[strong_labels > 0]

    final_edges_bool = np.isin(labeled_map, strong_labels)
    final_edges = (final_edges_bool.astype(np.uint8)) * 255

    logger.debug(
        f"Custom Canny completed: {num_features} connected components evaluated, "
        f"{len(strong_labels)} retained via strong hysteresis seeds."
    )

    return {
        "smoothed": smoothed,
        "grad_x": grad_x,
        "grad_y": grad_y,
        "magnitude": magnitude,
        "direction_deg": direction_deg,
        "nms": nms,
        "strong_edges": strong_edges,
        "weak_edges": weak_edges,
        "edges": final_edges,
    }


# ==============================================================================
# 2. Standard Library Wrappers
# ==============================================================================


def run_opencv_canny(
    image: np.ndarray,
    low_threshold: float = 50.0,
    high_threshold: float = 150.0,
    aperture_size: int = 3,
    l2_gradient: bool = False,
) -> np.ndarray:
    """Execute OpenCV Canny edge detector.

    Args:
        image: Grayscale uint8 image of shape (H, W).
        low_threshold: Lower hysteresis threshold.
        high_threshold: Upper hysteresis threshold.
        aperture_size: Sobel aperture size (3, 5, or 7).
        l2_gradient: If True, uses L2 norm sqrt(Gx^2 + Gy^2); else |Gx| + |Gy|.

    Returns:
        np.ndarray: Binary edge map of shape (H, W), dtype uint8 with {0, 255}.
    """
    if image.dtype != np.uint8:
        norm_img = np.clip(image, 0, 255).astype(np.uint8)
    else:
        norm_img = image

    edges = cv2.Canny(
        norm_img,
        threshold1=low_threshold,
        threshold2=high_threshold,
        apertureSize=aperture_size,
        L2gradient=l2_gradient,
    )
    return edges


def run_skimage_canny(
    image: np.ndarray,
    sigma: float = 1.0,
    low_threshold: float | None = 50.0,
    high_threshold: float | None = 150.0,
) -> np.ndarray:
    """Execute Scikit-Image Canny edge detector.

    Note: skimage.feature.canny accepts thresholds normalized to [0, 1] if image is float,
    or absolute scale matching the image dtype range. Here we standardize to [0, 255] uint8 output.

    Args:
        image: Grayscale image of shape (H, W).
        sigma: Standard deviation of Gaussian filter.
        low_threshold: Lower threshold.
        high_threshold: Upper threshold.

    Returns:
        np.ndarray: Binary edge map of shape (H, W), dtype uint8 with {0, 255}.
    """
    # Convert image to float in [0.0, 1.0] for stable skimage behavior
    if image.dtype == np.uint8:
        img_float = image.astype(np.float32) / 255.0
        low_th_norm = low_threshold / 255.0 if low_threshold is not None else None
        high_th_norm = high_threshold / 255.0 if high_threshold is not None else None
    else:
        max_val = max(1.0, float(np.max(image)))
        img_float = image.astype(np.float32) / max_val
        low_th_norm = low_threshold / max_val if low_threshold is not None else None
        high_th_norm = high_threshold / max_val if high_threshold is not None else None

    edges_bool = skimage_canny(
        img_float,
        sigma=sigma,
        low_threshold=low_th_norm,
        high_threshold=high_th_norm,
    )
    return (edges_bool.astype(np.uint8)) * 255


# ==============================================================================
# 3. Classic Baseline Derivatives (Sobel & Laplacian)
# ==============================================================================


def run_sobel(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Compute normalized Sobel gradient magnitude for baseline comparison.

    Args:
        image: Grayscale image of shape (H, W).
        ksize: Kernel aperture size.

    Returns:
        np.ndarray: Gradient magnitude normalized to uint8 [0, 255].
    """
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=ksize)
    mag = np.hypot(gx, gy)
    max_val = np.max(mag)
    if max_val > 1e-7:
        norm = (mag / max_val * 255.0).astype(np.uint8)
    else:
        norm = np.zeros_like(image, dtype=np.uint8)
    return norm


def run_laplacian(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Compute normalized Laplacian of Gaussian (LoG) second derivative.

    Args:
        image: Grayscale image of shape (H, W).
        ksize: Kernel aperture size.

    Returns:
        np.ndarray: Absolute second derivative normalized to uint8 [0, 255].
    """
    blurred = cv2.GaussianBlur(image, (5, 5), sigmaX=1.0)
    lap = cv2.Laplacian(blurred, cv2.CV_32F, ksize=ksize)
    abs_lap = np.abs(lap)
    max_val = np.max(abs_lap)
    if max_val > 1e-7:
        norm = (abs_lap / max_val * 255.0).astype(np.uint8)
    else:
        norm = np.zeros_like(image, dtype=np.uint8)
    return norm


# ==============================================================================
# 4. Auto-Canny / Adaptive Thresholding
# ==============================================================================


def run_auto_canny(
    image: np.ndarray,
    sigma_factor: float = 0.33,
) -> tuple[np.ndarray, int, int]:
    """Execute Auto-Canny with median-derived adaptive thresholds.

    Calculates thresholds based on the median intensity:
        low_threshold = max(0, (1.0 - sigma_factor) * median)
        high_threshold = min(255, (1.0 + sigma_factor) * median)

    Args:
        image: Grayscale uint8 image of shape (H, W).
        sigma_factor: Percentage deviation from median (standard default 0.33).

    Returns:
        tuple[np.ndarray, int, int]: (edge_map, low_threshold, high_threshold).
    """
    v = float(np.median(image))
    low_thresh = int(max(0, (1.0 - sigma_factor) * v))
    high_thresh = int(min(255, (1.0 + sigma_factor) * v))

    # Defensive guard to ensure high_thresh > low_thresh
    if high_thresh <= low_thresh:
        high_thresh = min(255, low_thresh + 30)

    edges = cv2.Canny(image, low_thresh, high_thresh)
    return edges, low_thresh, high_thresh


def run_otsu_canny(
    image: np.ndarray,
) -> tuple[np.ndarray, int, int]:
    """Compute Canny thresholds adaptively using Otsu's bimodal threshold.

    Args:
        image: Grayscale uint8 image of shape (H, W).

    Returns:
        tuple[np.ndarray, int, int]: (edge_map, low_threshold, high_threshold).
    """
    blurred = cv2.GaussianBlur(image, (5, 5), 1.0)
    otsu_val, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    high_thresh = int(otsu_val)
    low_thresh = int(0.5 * high_thresh)

    edges = cv2.Canny(blurred, low_thresh, high_thresh)
    return edges, low_thresh, high_thresh


# ==============================================================================
# 5. Color Image Canny (Multi-Channel vs Di Zenzo Tensor)
# ==============================================================================


def run_multichannel_canny(
    rgb_image: np.ndarray,
    low_threshold: float = 50.0,
    high_threshold: float = 150.0,
) -> np.ndarray:
    """Execute Canny on R, G, and B channels independently, merging via bitwise OR.

    Args:
        rgb_image: Color image of shape (H, W, 3), dtype uint8 RGB.
        low_threshold: Lower threshold.
        high_threshold: Upper threshold.

    Returns:
        np.ndarray: Merged binary edge map (H, W) uint8 {0, 255}.
    """
    edge_r = cv2.Canny(rgb_image[:, :, 0], low_threshold, high_threshold)
    edge_g = cv2.Canny(rgb_image[:, :, 1], low_threshold, high_threshold)
    edge_b = cv2.Canny(rgb_image[:, :, 2], low_threshold, high_threshold)

    merged = cv2.bitwise_or(edge_r, edge_g)
    merged = cv2.bitwise_or(merged, edge_b)
    return merged


def run_di_zenzo_gradient(
    rgb_image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute color gradient magnitude and direction using the Di Zenzo Tensor.

    Given I(x, y) = [R(x,y), G(x,y), B(x,y)]^T:
        g_xx = (dR/dx)^2 + (dG/dx)^2 + (dB/dx)^2
        g_yy = (dR/dy)^2 + (dG/dy)^2 + (dB/dy)^2
        g_xy = (dR/dx)(dR/dy) + (dG/dx)(dG/dy) + (dB/dx)(dB/dy)

    Max Eigenvalue:
        lambda_plus = 0.5 * (g_xx + g_yy + sqrt((g_xx - g_yy)^2 + 4 * g_xy^2))
        Magnitude = sqrt(lambda_plus)
        Angle = 0.5 * arctan2(2 * g_xy, g_xx - g_yy)

    Args:
        rgb_image: RGB image of shape (H, W, 3), dtype uint8 or float.

    Returns:
        tuple[np.ndarray, np.ndarray]: (magnitude_float32, direction_deg_float32 in [0, 180)).
    """
    img = rgb_image.astype(np.float32)

    # Compute color gradients per channel
    rx = cv2.Sobel(img[:, :, 0], cv2.CV_32F, 1, 0, ksize=3)
    ry = cv2.Sobel(img[:, :, 0], cv2.CV_32F, 0, 1, ksize=3)

    gx = cv2.Sobel(img[:, :, 1], cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img[:, :, 1], cv2.CV_32F, 0, 1, ksize=3)

    bx = cv2.Sobel(img[:, :, 2], cv2.CV_32F, 1, 0, ksize=3)
    by = cv2.Sobel(img[:, :, 2], cv2.CV_32F, 0, 1, ksize=3)

    # Tensor components
    g_xx = rx**2 + gx**2 + bx**2
    g_yy = ry**2 + gy**2 + by**2
    g_xy = rx * ry + gx * gy + bx * by

    # Eigenvalues of 2x2 symmetric tensor
    delta = np.sqrt(np.maximum(0.0, (g_xx - g_yy) ** 2 + 4.0 * (g_xy**2)))
    lambda_plus = 0.5 * (g_xx + g_yy + delta)

    magnitude = np.sqrt(np.maximum(0.0, lambda_plus))
    angle_rad = 0.5 * np.arctan2(2.0 * g_xy, g_xx - g_yy + 1e-7)
    angle_deg = np.rad2deg(angle_rad) % 180.0

    return magnitude, angle_deg


# ==============================================================================
# 6. Quantitative Evaluation Metrics
# ==============================================================================


def calculate_pratt_fom(
    detected_edges: np.ndarray,
    ground_truth_edges: np.ndarray,
    alpha: float = 1.0 / 9.0,
) -> float:
    """Compute Pratt's Figure of Merit (FOM) for edge detection accuracy.

    FOM = (1 / max(I_N, I_A)) * sum_{i=1}^{I_A} (1 / (1 + alpha * d_i^2))
    where:
        I_N: Number of ideal ground-truth edge points.
        I_A: Number of detected edge points.
        d_i: Euclidean distance from i-th detected point to nearest ground-truth point.
        alpha: Scaling constant (standard default 1/9).

    Args:
        detected_edges: Binary edge map of shape (H, W).
        ground_truth_edges: Binary ground truth edge map of shape (H, W).
        alpha: Penalty scale factor.

    Returns:
        float: Pratt's FOM score in [0.0, 1.0]. Perfect match equals 1.0.
    """
    gt_pts = np.argwhere(ground_truth_edges > 0)
    det_pts = np.argwhere(detected_edges > 0)

    i_n = len(gt_pts)
    i_a = len(det_pts)

    if i_n == 0 and i_a == 0:
        return 1.0
    if i_n == 0 or i_a == 0:
        return 0.0

    # Distance transform from ground truth edges
    gt_binary_inv = (ground_truth_edges == 0).astype(np.uint8)
    dist_map = cv2.distanceTransform(gt_binary_inv, cv2.DIST_L2, 5)

    distances = dist_map[detected_edges > 0]
    penalties = 1.0 / (1.0 + alpha * (distances**2))

    fom = float(np.sum(penalties) / max(i_n, i_a))
    return float(np.clip(fom, 0.0, 1.0))


def calculate_edge_f1_score(
    detected_edges: np.ndarray,
    ground_truth_edges: np.ndarray,
    tolerance: float = 2.0,
) -> dict[str, float]:
    """Calculate Precision, Recall, and F1-Score with distance tolerance.

    Args:
        detected_edges: Binary edge map (H, W).
        ground_truth_edges: Ground truth edge map (H, W).
        tolerance: Maximum pixel distance to count as a true positive.

    Returns:
        dict[str, float]: Precision, Recall, F1 scores in [0.0, 1.0].
    """
    det_mask = detected_edges > 0
    gt_mask = ground_truth_edges > 0

    det_count = int(np.sum(det_mask))
    gt_count = int(np.sum(gt_mask))

    if det_count == 0 and gt_count == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if det_count == 0 or gt_count == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # Distance transforms
    gt_dist = cv2.distanceTransform((~gt_mask).astype(np.uint8), cv2.DIST_L2, 5)
    det_dist = cv2.distanceTransform((~det_mask).astype(np.uint8), cv2.DIST_L2, 5)

    tp_det = np.sum(gt_dist[det_mask] <= tolerance)
    tp_gt = np.sum(det_dist[gt_mask] <= tolerance)

    precision = float(tp_det / (det_count + 1e-7))
    recall = float(tp_gt / (gt_count + 1e-7))
    f1 = float(2.0 * precision * recall / (precision + recall + 1e-7))

    return {
        "precision": min(1.0, precision),
        "recall": min(1.0, recall),
        "f1": min(1.0, f1),
    }


def calculate_hausdorff_distance(
    detected_edges: np.ndarray,
    ground_truth_edges: np.ndarray,
) -> float:
    """Calculate bidirectional symmetric Hausdorff distance between edge point sets.

    Args:
        detected_edges: Binary edge map (H, W).
        ground_truth_edges: Ground truth edge map (H, W).

    Returns:
        float: Hausdorff distance in pixels.
    """
    pts1 = np.argwhere(detected_edges > 0)
    pts2 = np.argwhere(ground_truth_edges > 0)

    if len(pts1) == 0 or len(pts2) == 0:
        return float("inf")

    d1 = directed_hausdorff(pts1, pts2)[0]
    d2 = directed_hausdorff(pts2, pts1)[0]
    return float(max(d1, d2))


# ==============================================================================
# 7. Downstream Applications: Segmentation & Shape Recognition
# ==============================================================================


def segment_via_morphology_and_holes(
    edges: np.ndarray,
    kernel_size: tuple[int, int] = (3, 3),
    closing_iterations: int = 2,
    dilation_iterations: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Segment image regions from Canny edges using morphology and hole filling.

    Pipeline:
        Edges -> Morphological Closing/Dilation (bridge gaps) -> binary_fill_holes.

    Args:
        edges: Binary edge map of shape (H, W) uint8 {0, 255}.
        kernel_size: Morphological structuring element shape.
        closing_iterations: Number of morphological closing iterations.
        dilation_iterations: Number of dilation iterations.

    Returns:
        tuple[np.ndarray, np.ndarray]: (closed_edges, filled_mask) uint8 {0, 255}.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    closed = cv2.morphologyEx(
        edges, cv2.MORPH_CLOSE, kernel, iterations=closing_iterations
    )
    if dilation_iterations > 0:
        closed = cv2.dilate(closed, kernel, iterations=dilation_iterations)

    filled = scipy.ndimage.binary_fill_holes(closed > 0)
    filled_uint8 = (filled.astype(np.uint8)) * 255

    return closed, filled_uint8


def detect_hough_lines(
    edges: np.ndarray,
    rgb_canvas: np.ndarray,
    rho: float = 1.0,
    theta_deg: float = 1.0,
    threshold: int = 40,
    min_line_length: int = 30,
    max_line_gap: int = 10,
) -> tuple[np.ndarray, int]:
    """Detect straight line segments using Probabilistic Hough Transform.

    Args:
        edges: Binary edge map (H, W).
        rgb_canvas: Canvas image to draw detected lines onto (H, W, 3) RGB.
        rho: Distance resolution in pixels.
        theta_deg: Angle resolution in degrees.
        threshold: Accumulator threshold parameter.
        min_line_length: Minimum line length.
        max_line_gap: Maximum allowed gap between points on the same line.

    Returns:
        tuple[np.ndarray, int]: (annotated_canvas, line_count).
    """
    canvas = rgb_canvas.copy()
    theta_rad = np.deg2rad(theta_deg)
    lines = cv2.HoughLinesP(
        edges,
        rho=rho,
        theta=theta_rad,
        threshold=threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )

    count = 0
    if lines is not None:
        lines_flat = np.asarray(lines).reshape(-1, 4)
        count = len(lines_flat)
        for line in lines_flat:
            x1, y1, x2, y2 = int(line[0]), int(line[1]), int(line[2]), int(line[3])
            cv2.line(canvas, (x1, y1), (x2, y2), (255, 0, 0), 2)

    return canvas, count


def detect_hough_circles(
    image_gray: np.ndarray,
    rgb_canvas: np.ndarray,
    dp: float = 1.2,
    min_dist: float = 40.0,
    param1: float = 100.0,
    param2: float = 25.0,
    min_radius: int = 15,
    max_radius: int = 80,
) -> tuple[np.ndarray, int]:
    """Detect circular features using Hough Circle Transform.

    Args:
        image_gray: Grayscale image (H, W).
        rgb_canvas: Canvas image (H, W, 3) RGB.
        dp: Inverse ratio of accumulator resolution to image resolution.
        min_dist: Minimum distance between detected circle centers.
        param1: Upper threshold for internal Canny detector.
        param2: Accumulator threshold for circle centers.
        min_radius: Minimum circle radius.
        max_radius: Maximum circle radius.

    Returns:
        tuple[np.ndarray, int]: (annotated_canvas, circle_count).
    """
    canvas = rgb_canvas.copy()
    blurred = cv2.GaussianBlur(image_gray, (9, 9), 2.0)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    count = 0
    if circles is not None:
        circles_flat = np.around(circles).reshape(-1, 3)
        count = len(circles_flat)
        for circle in circles_flat:
            cx, cy, r = int(circle[0]), int(circle[1]), int(circle[2])
            # Draw outer circle
            cv2.circle(canvas, (cx, cy), r, (0, 255, 0), 2)
            # Draw center point
            cv2.circle(canvas, (cx, cy), 3, (0, 0, 255), -1)

    return canvas, count


def classify_polygon_contours(
    edges: np.ndarray,
    rgb_canvas: np.ndarray,
    epsilon_factor: float = 0.03,
) -> tuple[np.ndarray, dict[str, int]]:
    """Detect and classify geometric polygons via contour polygon approximation.

    Args:
        edges: Binary edge map (H, W).
        rgb_canvas: Canvas image (H, W, 3) RGB.
        epsilon_factor: Approximation accuracy multiplier relative to perimeter.

    Returns:
        tuple[np.ndarray, dict[str, int]]: (annotated_canvas, shape_histogram).
    """
    canvas = rgb_canvas.copy()
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    counts: dict[str, int] = {
        "Triangle": 0,
        "Quadrilateral": 0,
        "Pentagon": 0,
        "Hexagon": 0,
        "Circle/Ellipse": 0,
    }

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:  # Ignore tiny noise artifacts
            continue

        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon_factor * perimeter, True)
        vertices = len(approx)

        if vertices == 3:
            shape_name = "Triangle"
            color = (255, 100, 0)
        elif vertices == 4:
            shape_name = "Quadrilateral"
            color = (0, 200, 255)
        elif vertices == 5:
            shape_name = "Pentagon"
            color = (255, 0, 200)
        elif vertices == 6:
            shape_name = "Hexagon"
            color = (150, 0, 255)
        else:
            shape_name = "Circle/Ellipse"
            color = (0, 255, 100)

        counts[shape_name] = counts.get(shape_name, 0) + 1

        # Draw contour and label
        cv2.drawContours(canvas, [approx], -1, color, 2)
        m = cv2.moments(cnt)
        if m["m00"] > 1e-5:
            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])
            cv2.putText(
                canvas,
                shape_name,
                (cx - 30, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    return canvas, counts


# ==============================================================================
# 8. Video Temporal Edge Stabilization
# ==============================================================================


class TemporalEdgeFilter:
    """Stabilizes edge detection across sequential video frames to mitigate temporal flickering."""

    def __init__(self, alpha: float = 0.7, threshold: float = 128.0) -> None:
        """Initialize filter.

        Args:
            alpha: Exponential moving average momentum factor in [0.0, 1.0].
            threshold: Binarization cutoff threshold for smoothed edge response.
        """
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.threshold = threshold
        self.running_edge_map: np.ndarray | None = None

    def update(self, current_edges: np.ndarray) -> np.ndarray:
        """Process current frame edges and return temporally smoothed binary edges.

        Args:
            current_edges: Raw Canny edges of shape (H, W), dtype uint8 {0, 255}.

        Returns:
            np.ndarray: Smoothed binary edge map of shape (H, W) uint8 {0, 255}.
        """
        edges_float = current_edges.astype(np.float32)

        if self.running_edge_map is None:
            self.running_edge_map = edges_float
        else:
            self.running_edge_map = (
                self.alpha * self.running_edge_map + (1.0 - self.alpha) * edges_float
            )

        stabilized = (self.running_edge_map >= self.threshold).astype(np.uint8) * 255
        return stabilized
