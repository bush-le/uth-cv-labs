"""Color space transformation atomic utilities for Lab 01."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from src.common.logger import get_logger

logger = get_logger(__name__)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert an image from OpenCV BGR color space to RGB color space."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected 3-channel image for BGR->RGB conversion, got shape {image.shape}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    """Convert an image from RGB color space to OpenCV BGR color space."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected 3-channel image for RGB->BGR conversion, got shape {image.shape}")
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def bgr_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR color image to a single-channel grayscale image."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected 3-channel BGR image, got shape {image.shape}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def bgr_to_hsv(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to the HSV color space."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected 3-channel BGR image, got shape {image.shape}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def bgr_to_lab(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to the CIE L*a*b* perceptual color space."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected 3-channel BGR image, got shape {image.shape}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)


def pil_to_grayscale(image: Image.Image) -> Image.Image:
    """Convert a PIL Image to grayscale (mode 'L')."""
    return image.convert("L")


def pil_to_hsv(image: Image.Image) -> Image.Image:
    """Convert a PIL Image to HSV color mode."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image.convert("HSV")
