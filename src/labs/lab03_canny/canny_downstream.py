"""Downstream tasks utilizing Canny edges (segmentation, Hough shapes, contours, temporal filter)."""

from __future__ import annotations

import cv2
import numpy as np
import scipy.ndimage


def segment_via_morphology_and_holes(
    edges: np.ndarray,
    kernel_size: tuple[int, int] = (3, 3),
    closing_iterations: int = 2,
    dilation_iterations: int = 1,
    *,
    low_threshold: float | None = None,
    high_threshold: float | None = None,
    gaussian_ksize: tuple[int, int] | None = None,
    gaussian_sigma: float | None = None,
    morph_kernel_size: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray] | dict[str, np.ndarray]:
    """Segment image regions from Canny edges using morphology and hole filling.

    Can operate either directly on an edge map returning a (closed, mask) tuple,
    or on a grayscale image with Canny threshold parameters returning a dictionary
    containing 'edges', 'closed', 'mask', and 'segmented' result.

    Args:
        edges: Binary edge map of shape (H, W) or grayscale image if low_threshold is passed.
        kernel_size: Morphological structuring element shape.
        closing_iterations: Number of morphological closing iterations.
        dilation_iterations: Number of dilation iterations.
        low_threshold: Optional lower threshold for Canny edge detection.
        high_threshold: Optional higher threshold for Canny edge detection.
        gaussian_ksize: Optional Gaussian kernel size before Canny.
        gaussian_sigma: Optional Gaussian sigma before Canny.
        morph_kernel_size: Alias for kernel_size.

    Returns:
        tuple[np.ndarray, np.ndarray] | dict[str, np.ndarray]:
            Tuple (closed_edges, filled_mask) or dictionary with segmented components.
    """
    k_size = morph_kernel_size if morph_kernel_size is not None else kernel_size

    if low_threshold is not None and high_threshold is not None:
        img_gray = edges.copy()
        if gaussian_ksize is not None and gaussian_sigma is not None:
            blurred = cv2.GaussianBlur(img_gray, gaussian_ksize, gaussian_sigma)
        else:
            blurred = img_gray
        edge_map = cv2.Canny(blurred, int(low_threshold), int(high_threshold))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, k_size)
        closed = cv2.morphologyEx(
            edge_map, cv2.MORPH_CLOSE, kernel, iterations=closing_iterations
        )
        if dilation_iterations > 0:
            closed = cv2.dilate(closed, kernel, iterations=dilation_iterations)
        filled = scipy.ndimage.binary_fill_holes(closed > 0)
        filled_uint8 = (filled.astype(np.uint8)) * 255
        segmented = cv2.bitwise_and(img_gray, img_gray, mask=filled_uint8)
        return {
            "edges": edge_map,
            "closed": closed,
            "mask": filled_uint8,
            "segmented": segmented,
        }

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, k_size)
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
    rgb_canvas: np.ndarray | None = None,
    dp: float = 1.2,
    min_dist: float = 40.0,
    param1: float = 100.0,
    param2: float = 25.0,
    min_radius: int = 15,
    max_radius: int = 80,
    *,
    blur_ksize: tuple[int, int] = (9, 9),
    blur_sigma: float = 2.0,
) -> tuple[np.ndarray, int] | dict[str, object]:
    """Detect circular features using Hough Circle Transform.

    Args:
        image_gray: Grayscale image (H, W).
        rgb_canvas: Optional canvas image (H, W, 3) RGB/BGR. If None, produces a dictionary.
        dp: Inverse ratio of accumulator resolution to image resolution.
        min_dist: Minimum distance between detected circle centers.
        param1: Upper threshold for internal Canny detector.
        param2: Accumulator threshold for circle centers.
        min_radius: Minimum circle radius.
        max_radius: Maximum circle radius.
        blur_ksize: Gaussian blur kernel size prior to circle Hough transform.
        blur_sigma: Gaussian blur standard deviation.

    Returns:
        tuple[np.ndarray, int] | dict[str, object]:
            If rgb_canvas is provided: (annotated_canvas, circle_count).
            If rgb_canvas is None: dictionary with 'annotated_bgr', 'circles', and 'count'.
    """
    blurred = cv2.GaussianBlur(image_gray, blur_ksize, blur_sigma)
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

    if rgb_canvas is None:
        circle_result = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
        circle_count = 0
        if circles is not None:
            found_circles = np.rint(circles[0]).astype(int)
            found_circles = found_circles[
                np.lexsort((found_circles[:, 0], found_circles[:, 1]))
            ]
            circle_count = len(found_circles)
            for number, (x, y, r) in enumerate(found_circles, start=1):
                cv2.circle(circle_result, (x, y), r, (0, 0, 255), 2)
                cv2.circle(circle_result, (x, y), 2, (0, 255, 0), -1)
                cv2.putText(
                    circle_result,
                    str(number),
                    (x - 8, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
        return {
            "annotated_bgr": circle_result,
            "circles": circles,
            "count": circle_count,
        }

    canvas = rgb_canvas.copy()
    count = 0
    if circles is not None:
        circles_flat = np.around(circles).reshape(-1, 3)
        count = len(circles_flat)
        for circle in circles_flat:
            cx, cy, r = int(circle[0]), int(circle[1]), int(circle[2])
            cv2.circle(canvas, (cx, cy), r, (0, 255, 0), 2)
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
