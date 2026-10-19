"""Geometric and spatial image transformations (cropping, resizing)."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from src.common.logger import get_logger

logger = get_logger(__name__)


def crop_roi(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Crop a Region of Interest (ROI) from an image array.

    Coordinates format is Pascal VOC / XYXY: (x1, y1, x2, y2).

    Args:
        image: Input image array of shape (H, W, C) or (H, W).
        box: Tuple of 4 integers (x1, y1, x2, y2) defining top-left and bottom-right coords.

    Returns:
        np.ndarray: Cropped image slice of shape (y2 - y1, x2 - x1, C) or (y2 - y1, x2 - x1).

    Raises:
        ValueError: If bounding box coordinates are invalid or out of image bounds.
    """
    x1, y1, x2, y2 = box
    height, width = image.shape[:2]

    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(
            f"Invalid crop box {box} for image of size width={width}, height={height}."
        )

    return image[y1:y2, x1:x2].copy()


def crop_roi_pil(image: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    """Crop a Region of Interest (ROI) from a Pillow Image.

    Coordinates format is (x1, y1, x2, y2).

    Args:
        image: PIL Image instance.
        box: (x1, y1, x2, y2) bounding box tuple.

    Returns:
        Image.Image: Cropped PIL Image.

    Raises:
        ValueError: If box coordinates are invalid or out of bounds.
    """
    x1, y1, x2, y2 = box
    width, height = image.size

    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(
            f"Invalid crop box {box} for PIL image of size width={width}, height={height}."
        )

    return image.crop(box)


def resize_by_scale(
    image: np.ndarray,
    fx: float,
    fy: float,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Resize an image by scaling factors along width and height.

    Args:
        image: Input image array of shape (H, W, C) or (H, W).
        fx: Scaling factor along horizontal axis (width). Must be > 0.
        fy: Scaling factor along vertical axis (height). Must be > 0.
        interpolation: OpenCV interpolation method flag (default cv2.INTER_LINEAR).

    Returns:
        np.ndarray: Resized image array.

    Raises:
        ValueError: If fx or fy are non-positive.
    """
    if fx <= 0 or fy <= 0:
        raise ValueError(f"Scale factors fx and fy must be positive, got fx={fx}, fy={fy}")

    return cv2.resize(image, dsize=(0, 0), fx=fx, fy=fy, interpolation=interpolation)


def resize_by_dimensions(
    image: np.ndarray,
    target_width: int,
    target_height: int,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Resize an image to fixed target dimensions (width, height).

    Args:
        image: Input image array of shape (H, W, C) or (H, W).
        target_width: Desired output width (> 0).
        target_height: Desired output height (> 0).
        interpolation: OpenCV interpolation method flag.

    Returns:
        np.ndarray: Resized image array.

    Raises:
        ValueError: If target_width or target_height are non-positive.
    """
    if target_width <= 0 or target_height <= 0:
        raise ValueError(
            f"Target dimensions must be positive integers, got width={target_width}, height={target_height}"
        )

    return cv2.resize(image, (target_width, target_height), interpolation=interpolation)


def resize_pil_by_scale(
    image: Image.Image,
    fx: float,
    fy: float,
    resample: Image.Resampling = Image.Resampling.BILINEAR,
) -> Image.Image:
    """Resize a Pillow Image by scaling factors along width and height.

    Args:
        image: PIL Image object.
        fx: Scale factor for width (> 0).
        fy: Scale factor for height (> 0).
        resample: PIL Resampling filter (default BILINEAR).

    Returns:
        Image.Image: Resized PIL image.

    Raises:
        ValueError: If fx or fy are non-positive.
    """
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
    """Resize a Pillow Image to fixed target dimensions (width, height).

    Args:
        image: PIL Image object.
        target_width: Desired output width (> 0).
        target_height: Desired output height (> 0).
        resample: PIL Resampling filter.

    Returns:
        Image.Image: Resized PIL image.

    Raises:
        ValueError: If target dimensions are non-positive.
    """
    if target_width <= 0 or target_height <= 0:
        raise ValueError(
            f"Target dimensions must be positive integers, got width={target_width}, height={target_height}"
        )

    return image.resize((target_width, target_height), resample=resample)
