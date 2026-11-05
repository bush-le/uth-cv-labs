"""Geometric and spatial image transformations for Lab 01."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from src.common.logger import get_logger

logger = get_logger(__name__)


def crop_roi(image: np.ndarray, *args) -> np.ndarray:
    """Crop a Region of Interest (ROI) from an image array.

    Coordinates format is Pascal VOC / XYXY: (x1, y1, x2, y2).
    Accepts either crop_roi(image, (x1, y1, x2, y2)) or crop_roi(image, x1, y1, x2, y2).
    """
    if len(args) == 1:
        box = args[0]
    elif len(args) == 4:
        box = args
    else:
        raise ValueError("crop_roi expects either a box tuple (x1, y1, x2, y2) or 4 coordinates x1, y1, x2, y2")

    x1, y1, x2, y2 = box
    height, width = image.shape[:2]

    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(
            f"Invalid crop box {box} for image of size width={width}, height={height}."
        )

    return image[y1:y2, x1:x2].copy()


def crop_roi_pil(image: Image.Image, *args) -> Image.Image:
    """Crop a Region of Interest (ROI) from a Pillow Image."""
    if len(args) == 1:
        box = args[0]
    elif len(args) == 4:
        box = args
    else:
        raise ValueError("crop_roi_pil expects either a box tuple (x1, y1, x2, y2) or 4 coordinates x1, y1, x2, y2")

    x1, y1, x2, y2 = box
    width, height = image.size

    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(
            f"Invalid crop box {box} for PIL image of size width={width}, height={height}."
        )

    return image.crop((x1, y1, x2, y2))


def resize_by_scale(
    image: np.ndarray,
    fx: float,
    fy: float | None = None,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Resize an image by scaling factors along width and height."""
    if fy is None:
        fy = fx
    if fx <= 0 or fy <= 0:
        raise ValueError(f"Scale factors fx and fy must be positive, got fx={fx}, fy={fy}")

    return cv2.resize(image, dsize=(0, 0), fx=fx, fy=fy, interpolation=interpolation)


def resize_by_dimensions(
    image: np.ndarray,
    target_width: int,
    target_height: int,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Resize an image to fixed target dimensions (width, height)."""
    if target_width <= 0 or target_height <= 0:
        raise ValueError(
            f"Target dimensions must be positive integers, got width={target_width}, height={target_height}"
        )

    return cv2.resize(image, (target_width, target_height), interpolation=interpolation)


def resize_pil_by_scale(
    image: Image.Image,
    fx: float,
    fy: float | None = None,
    resample: Image.Resampling = Image.Resampling.BILINEAR,
) -> Image.Image:
    """Resize a Pillow Image by scaling factors along width and height."""
    if fy is None:
        fy = fx
    if fx <= 0 or fy <= 0:
        raise ValueError(f"Scale factors fx and fy must be positive, got fx={fx}, fy={fy}")

    new_width = max(1, int(round(image.width * fx)))
    new_height = max(1, int(round(image.height * fy)))
    return image.resize((new_width, new_height), resample=resample)


def resize_pil_by_dimensions(
    image: Image.Image,
    target_width: int,
    target_height: int,
    resample: Image.Resampling = Image.Resampling.BILINEAR,
) -> Image.Image:
    """Resize a Pillow Image to fixed target dimensions (width, height)."""
    if target_width <= 0 or target_height <= 0:
        raise ValueError(
            f"Target dimensions must be positive integers, got width={target_width}, height={target_height}"
        )

    return image.resize((target_width, target_height), resample=resample)
