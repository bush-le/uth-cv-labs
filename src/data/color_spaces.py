"""Color space transformation utilities for Computer Vision pipelines."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from src.common.logger import get_logger

logger = get_logger(__name__)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert an image from OpenCV BGR color space to RGB color space.

    Args:
        image: Input array of shape (H, W, 3) in BGR order.

    Returns:
        np.ndarray: Converted image array in RGB order.

    Raises:
        ValueError: If input is not a 3-channel array.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected 3-channel image for BGR->RGB conversion, got shape {image.shape}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    """Convert an image from RGB color space to OpenCV BGR color space.

    Args:
        image: Input array of shape (H, W, 3) in RGB order.

    Returns:
        np.ndarray: Converted image array in BGR order.

    Raises:
        ValueError: If input is not a 3-channel array.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected 3-channel image for RGB->BGR conversion, got shape {image.shape}")
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def bgr_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR color image to a single-channel grayscale image.

    Args:
        image: Input array of shape (H, W, 3) in BGR order.

    Returns:
        np.ndarray: Grayscale image array of shape (H, W) with dtype uint8.

    Raises:
        ValueError: If input is not a 3-channel array.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected 3-channel BGR image, got shape {image.shape}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def bgr_to_hsv(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to the HSV (Hue, Saturation, Value) color space.

    Args:
        image: Input array of shape (H, W, 3) in BGR order.

    Returns:
        np.ndarray: HSV image array of shape (H, W, 3).
            - H in range [0, 179]
            - S in range [0, 255]
            - V in range [0, 255]

    Raises:
        ValueError: If input is not a 3-channel array.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected 3-channel BGR image, got shape {image.shape}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def bgr_to_lab(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to the CIE L*a*b* perceptual color space.

    Args:
        image: Input array of shape (H, W, 3) in BGR order.

    Returns:
        np.ndarray: LAB image array of shape (H, W, 3).
            - L (Lightness) in range [0, 255]
            - a (green-red) in range [0, 255]
            - b (blue-yellow) in range [0, 255]

    Raises:
        ValueError: If input is not a 3-channel array.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected 3-channel BGR image, got shape {image.shape}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)


def pil_to_grayscale(image: Image.Image) -> Image.Image:
    """Convert a PIL Image to grayscale (mode 'L').

    Args:
        image: Input PIL Image.

    Returns:
        Image.Image: Grayscale PIL Image.
    """
    return image.convert("L")


def pil_to_hsv(image: Image.Image) -> Image.Image:
    """Convert a PIL Image to HSV color mode.

    Args:
        image: Input PIL Image.

    Returns:
        Image.Image: Converted HSV PIL Image.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image.convert("HSV")
