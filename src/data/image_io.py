"""Image input and output utilities using OpenCV and Pillow."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.common.logger import get_logger

logger = get_logger(__name__)


def load_image_cv(image_path: str | Path, as_grayscale: bool = False) -> np.ndarray:
    """Load an image from disk using OpenCV.

    Args:
        image_path: Path to the image file.
        as_grayscale: If True, loads image in grayscale mode (H, W).
            Otherwise, loads in BGR mode (H, W, 3).

    Returns:
        np.ndarray: Loaded image array of shape (H, W, 3) with dtype uint8 for color,
            or shape (H, W) with dtype uint8 for grayscale.

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If OpenCV fails to decode the image.
    """
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
    """Save an image array to disk using OpenCV.

    Args:
        image: Image array of shape (H, W, 3) or (H, W), dtype uint8.
        output_path: Path where the image will be saved.

    Returns:
        Path: Resolved path to the saved image file.

    Raises:
        ValueError: If the input array is invalid or OpenCV fails to write.
    """
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
    """Load an image from disk using Pillow.

    Args:
        image_path: Path to the image file.
        to_rgb: If True, converts image to 'RGB' color mode.

    Returns:
        Image.Image: Loaded PIL Image object.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If Pillow fails to open the image.
    """
    path = Path(image_path)
    if not path.is_file():
        logger.error("Image file not found at: %s", path)
        raise FileNotFoundError(f"Image file not found: {path}")

    try:
        image = Image.open(path)
        image.load()  # Force loading image data into memory
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
    """Save a PIL Image object to disk.

    Args:
        image: PIL Image object.
        output_path: Path where the image will be saved.
        image_format: Optional explicit format override (e.g., 'PNG', 'JPEG', 'BMP').

    Returns:
        Path: Resolved path to the saved image file.

    Raises:
        ValueError: If the image object is invalid.
    """
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
    """Convert an image from one format to another (e.g., .png to .jpg or .bmp).

    Args:
        input_path: Path to the input image file.
        output_path: Path to the destination file with new extension.
        use_pil: If True, uses Pillow for conversion; otherwise uses OpenCV.

    Returns:
        Path: Resolved path to the converted image.
    """
    if use_pil:
        pil_img = load_image_pil(input_path, to_rgb=False)
        return save_image_pil(pil_img, output_path)

    cv_img = load_image_cv(input_path)
    return save_image_cv(cv_img, output_path)
