"""Universal image input and output utilities using OpenCV and Pillow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from src.common.logger import get_logger

logger = get_logger(__name__)


def load_image_cv(image_path: str | Path, as_grayscale: bool = False) -> np.ndarray:
    """Load an image from disk using OpenCV."""
    path = Path(image_path)
    if not path.is_file():
        logger.error("Image file not found at: %s", path)
        raise FileNotFoundError(f"Image file not found: {path}")

    flags = cv2.IMREAD_GRAYSCALE if as_grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), flags)

    if image is None:
        logger.error("Failed to decode image from: %s", path)
        raise ValueError(f"Failed to decode image at path: {path}")

    logger.debug("Loaded image via OpenCV with shape %s from %s", image.shape, path)
    return image


def save_image_cv(image: np.ndarray, output_path: str | Path) -> Path:
    """Save an image array to disk using OpenCV."""
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Invalid image array provided for saving.")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(str(path), image)
    if not success:
        logger.error("Failed to write image to %s", path)
        raise ValueError(f"OpenCV failed to write image to {path}")

    logger.info("Saved image via OpenCV to: %s", path)
    return path


def load_image_pil(image_path: str | Path, to_rgb: bool = True) -> Image.Image:
    """Load an image from disk using Pillow."""
    path = Path(image_path)
    if not path.is_file():
        logger.error("Image file not found at: %s", path)
        raise FileNotFoundError(f"Image file not found: {path}")

    try:
        image = Image.open(path)
        image.load()
        if to_rgb and image.mode != "RGB":
            image = image.convert("RGB")
        logger.debug("Loaded image via PIL with size %s and mode %s", image.size, image.mode)
        return image
    except Exception as exc:
        logger.error("Pillow failed to open image %s: %s", path, exc)
        raise ValueError(f"Could not open image at {path}: {exc}") from exc


def save_image_pil(
    image: Image.Image,
    output_path: str | Path,
    image_format: str | None = None,
) -> Path:
    """Save a PIL Image object to disk."""
    if not isinstance(image, Image.Image):
        raise ValueError("Input must be a valid PIL Image instance.")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    target_format = (image_format or path.suffix.lstrip(".").upper()).replace("JPG", "JPEG")
    to_save = image
    if target_format in {"JPEG", "BMP"} and image.mode in {"RGBA", "P"}:
        to_save = image.convert("RGB")

    to_save.save(path, format=image_format)
    logger.info("Saved image via PIL to: %s (format=%s)", path, target_format)
    return path


def convert_image_format(
    input_path: str | Path,
    output_path: str | Path,
    use_pil: bool = False,
) -> Path:
    """Convert an image from one format to another."""
    if use_pil:
        pil_img = load_image_pil(input_path, to_rgb=False)
        return save_image_pil(pil_img, output_path)

    cv_img = load_image_cv(input_path)
    return save_image_cv(cv_img, output_path)


def get_image_properties(image: np.ndarray | Image.Image) -> dict[str, Any]:
    """Extract standard dimensions, channels, and metadata properties of an image."""
    if isinstance(image, np.ndarray):
        h = int(image.shape[0])
        w = int(image.shape[1])
        c = int(image.shape[2]) if image.ndim == 3 else 1
        return {
            "height": h,
            "width": w,
            "channels": c,
            "dtype": str(image.dtype),
            "size_bytes": int(image.nbytes),
            "type": "numpy",
        }
    if isinstance(image, Image.Image):
        w, h = image.size
        c = len(image.getbands())
        return {
            "height": h,
            "width": w,
            "channels": c,
            "mode": image.mode,
            "format": image.format,
            "type": "pil",
        }
    raise ValueError(f"Unsupported image type: {type(image)}")


__all__ = [
    "load_image_cv",
    "save_image_cv",
    "load_image_pil",
    "save_image_pil",
    "convert_image_format",
    "get_image_properties",
]
