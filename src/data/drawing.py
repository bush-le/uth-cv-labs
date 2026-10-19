"""Geometric annotations, drawing primitives, and text placement."""

from __future__ import annotations

import cv2
import numpy as np


def draw_line(
    image: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
) -> np.ndarray:
    """Draw a straight line segment on an image copy.

    Args:
        image: Source image array of shape (H, W, 3).
        pt1: First point coordinate (x1, y1).
        pt2: Second point coordinate (x2, y2).
        color: Line color in BGR format, default Red (0, 0, 255).
        thickness: Line thickness in pixels.

    Returns:
        np.ndarray: Modified image copy with line drawn.
    """
    output = image.copy()
    cv2.line(output, pt1, pt2, color, thickness=thickness, lineType=cv2.LINE_AA)
    return output


def draw_rectangle(
    image: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw a rectangle on an image copy.

    Args:
        image: Source image array of shape (H, W, 3).
        pt1: Top-left coordinate (x_min, y_min).
        pt2: Bottom-right coordinate (x_max, y_max).
        color: Rectangle color in BGR format, default Green (0, 255, 0).
        thickness: Border thickness in pixels (or cv2.FILLED for solid box).

    Returns:
        np.ndarray: Modified image copy.
    """
    output = image.copy()
    cv2.rectangle(output, pt1, pt2, color, thickness=thickness, lineType=cv2.LINE_AA)
    return output


def draw_circle(
    image: np.ndarray,
    center: tuple[int, int],
    radius: int,
    color: tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw a circle on an image copy.

    Args:
        image: Source image array of shape (H, W, 3).
        center: Circle center coordinate (center_x, center_y).
        radius: Circle radius in pixels (must be > 0).
        color: Circle color in BGR format, default Blue (255, 0, 0).
        thickness: Border thickness in pixels (or cv2.FILLED).

    Returns:
        np.ndarray: Modified image copy.

    Raises:
        ValueError: If radius <= 0.
    """
    if radius <= 0:
        raise ValueError(f"Circle radius must be positive, got {radius}")

    output = image.copy()
    cv2.circle(output, center, radius, color, thickness=thickness, lineType=cv2.LINE_AA)
    return output


def draw_text(
    image: np.ndarray,
    text: str,
    org: tuple[int, int],
    font_face: int = cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 0.8,
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
) -> np.ndarray:
    """Render a text string onto an image copy.

    Args:
        image: Source image array of shape (H, W, 3).
        text: String of text to render.
        org: Bottom-left coordinate of text string (x, y).
        font_face: OpenCV font style identifier.
        font_scale: Scale factor for font size.
        color: Text color in BGR format, default White (255, 255, 255).
        thickness: Font stroke thickness in pixels.

    Returns:
        np.ndarray: Modified image copy.
    """
    output = image.copy()
    cv2.putText(
        output,
        text,
        org,
        fontFace=font_face,
        fontScale=font_scale,
        color=color,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )
    return output
