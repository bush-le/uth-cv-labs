"""Geometric annotations and drawing atomic primitives for Lab 01."""

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
    """Draw a straight line segment on an image copy."""
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
    """Draw a rectangle on an image copy."""
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
    """Draw a circle on an image copy."""
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

    Gracefully handles when 4th argument is passed as color tuple instead of font_face int.
    """
    actual_font = font_face
    actual_color = color
    if isinstance(font_face, (tuple, list)):
        actual_color = tuple(font_face)
        actual_font = cv2.FONT_HERSHEY_SIMPLEX

    output = image.copy()
    cv2.putText(
        output,
        text,
        org,
        fontFace=actual_font,
        fontScale=font_scale,
        color=actual_color,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )
    return output
