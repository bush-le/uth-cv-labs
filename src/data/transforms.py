"""Image processing and transformation module for Computer Vision tasks.

Provides standardized functions for image I/O, color space conversion,
spatial transformations (cropping, resizing), and geometric annotations
using OpenCV and Pillow.
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.common.logger import get_logger

logger = get_logger(__name__)


# ==============================================================================
# 1. Image Input / Output Functions
# ==============================================================================


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

    # Convert RGBA to RGB if saving to formats that don't support alpha channel (like JPEG or BMP)
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


# ==============================================================================
# 2. Color Space Conversion Functions
# ==============================================================================


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert an image from OpenCV BGR color space to RGB color space.

    Args:
        image: BGR image array of shape (H, W, 3), dtype uint8.

    Returns:
        np.ndarray: RGB image array of shape (H, W, 3), dtype uint8.

    Raises:
        ValueError: If input is not a 3-channel image.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected 3-channel BGR image, got shape {image.shape}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    """Convert an image from RGB color space to OpenCV BGR color space.

    Args:
        image: RGB image array of shape (H, W, 3), dtype uint8.

    Returns:
        np.ndarray: BGR image array of shape (H, W, 3), dtype uint8.

    Raises:
        ValueError: If input is not a 3-channel image.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected 3-channel RGB image, got shape {image.shape}")
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def bgr_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an image from OpenCV BGR to Grayscale.

    Formula: Y = 0.299*R + 0.587*G + 0.114*B (Rec. 601 standard).

    Args:
        image: BGR image array of shape (H, W, 3), dtype uint8.

    Returns:
        np.ndarray: Grayscale image array of shape (H, W), dtype uint8.

    Raises:
        ValueError: If input is not a 3-channel image.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected 3-channel BGR image, got shape {image.shape}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def bgr_to_hsv(image: np.ndarray) -> np.ndarray:
    """Convert an image from OpenCV BGR to HSV (Hue, Saturation, Value).

    OpenCV ranges for 8-bit images:
    - H: [0, 179]
    - S: [0, 255]
    - V: [0, 255]

    Args:
        image: BGR image array of shape (H, W, 3), dtype uint8.

    Returns:
        np.ndarray: HSV image array of shape (H, W, 3), dtype uint8.

    Raises:
        ValueError: If input is not a 3-channel image.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected 3-channel BGR image, got shape {image.shape}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def bgr_to_lab(image: np.ndarray) -> np.ndarray:
    """Convert an image from OpenCV BGR to CIE L*a*b* color space.

    OpenCV ranges for 8-bit images:
    - L*: [0, 255] (corresponds to L* * 255 / 100)
    - a*: [0, 255] (corresponds to a* + 128)
    - b*: [0, 255] (corresponds to b* + 128)

    Args:
        image: BGR image array of shape (H, W, 3), dtype uint8.

    Returns:
        np.ndarray: LAB image array of shape (H, W, 3), dtype uint8.

    Raises:
        ValueError: If input is not a 3-channel image.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected 3-channel BGR image, got shape {image.shape}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)


def pil_to_grayscale(image: Image.Image) -> Image.Image:
    """Convert a Pillow Image to Grayscale (mode 'L').

    Args:
        image: PIL Image object.

    Returns:
        Image.Image: Converted image in 'L' (luminance) mode.
    """
    return image.convert("L")


def pil_to_hsv(image: Image.Image) -> Image.Image:
    """Convert a Pillow Image to HSV color space (mode 'HSV').

    Args:
        image: PIL Image object.

    Returns:
        Image.Image: Converted image in 'HSV' mode.
    """
    return image.convert("HSV")


# ==============================================================================
# 3. Cropping and Resizing Functions
# ==============================================================================


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
        Image.Image: Cropped PIL image.

    Raises:
        ValueError: If box coordinates are outside image bounds or degenerate.
    """
    x1, y1, x2, y2 = box
    width, height = image.size

    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(
            f"Invalid crop box {box} for PIL image with size {image.size}."
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
    """Resize an image to exact target dimensions (width, height).

    Args:
        image: Input image array of shape (H, W, C) or (H, W).
        target_width: Desired output width in pixels. Must be > 0.
        target_height: Desired output height in pixels. Must be > 0.
        interpolation: OpenCV interpolation method flag (default cv2.INTER_LINEAR).

    Returns:
        np.ndarray: Resized image array of shape (target_height, target_width, C) or (target_height, target_width).

    Raises:
        ValueError: If target_width or target_height are non-positive.
    """
    if target_width <= 0 or target_height <= 0:
        raise ValueError(
            f"Target dimensions must be positive, got width={target_width}, height={target_height}"
        )

    return cv2.resize(image, dsize=(target_width, target_height), interpolation=interpolation)


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
    """Resize a Pillow Image to exact dimensions (target_width, target_height).

    Args:
        image: PIL Image object.
        target_width: Target width in pixels (> 0).
        target_height: Target height in pixels (> 0).
        resample: PIL Resampling filter (default BILINEAR).

    Returns:
        Image.Image: Resized PIL image.
    """
    if target_width <= 0 or target_height <= 0:
        raise ValueError(
            f"Target dimensions must be positive, got width={target_width}, height={target_height}"
        )

    return image.resize((target_width, target_height), resample=resample)


# ==============================================================================
# 4. Drawing and Annotations Functions
# ==============================================================================


def draw_line(
    image: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw a line segment on an image copy.

    Args:
        image: Input image array of shape (H, W, 3).
        pt1: Starting point (x, y).
        pt2: Ending point (x, y).
        color: Line color in BGR format (default green: (0, 255, 0)).
        thickness: Line thickness in pixels (default 2).

    Returns:
        np.ndarray: Annotated image copy of shape (H, W, 3).
    """
    canvas = image.copy()
    cv2.line(canvas, pt1=pt1, pt2=pt2, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    return canvas


def draw_rectangle(
    image: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
) -> np.ndarray:
    """Draw a rectangle (bounding box) on an image copy.

    Args:
        image: Input image array of shape (H, W, 3).
        pt1: Top-left coordinate (x1, y1).
        pt2: Bottom-right coordinate (x2, y2).
        color: Rectangle color in BGR format (default red: (0, 0, 255)).
        thickness: Line thickness in pixels (default 2; negative value fills rectangle).

    Returns:
        np.ndarray: Annotated image copy of shape (H, W, 3).
    """
    canvas = image.copy()
    cv2.rectangle(canvas, pt1=pt1, pt2=pt2, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    return canvas


def draw_circle(
    image: np.ndarray,
    center: tuple[int, int],
    radius: int,
    color: tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw a circle on an image copy.

    Args:
        image: Input image array of shape (H, W, 3).
        center: Center point coordinate (x, y).
        radius: Radius in pixels (> 0).
        color: Circle color in BGR format (default blue: (255, 0, 0)).
        thickness: Line thickness in pixels (default 2; negative value fills circle).

    Returns:
        np.ndarray: Annotated image copy of shape (H, W, 3).

    Raises:
        ValueError: If radius is non-positive.
    """
    if radius <= 0:
        raise ValueError(f"Radius must be positive, got {radius}")

    canvas = image.copy()
    cv2.circle(canvas, center=center, radius=radius, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    return canvas


def draw_text(
    image: np.ndarray,
    text: str,
    org: tuple[int, int],
    font_face: int = cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 1.0,
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
) -> np.ndarray:
    """Render a text string onto an image copy.

    Args:
        image: Input image array of shape (H, W, 3).
        text: String of text to render.
        org: Bottom-left coordinate (x, y) where text starts.
        font_face: OpenCV font type (e.g., cv2.FONT_HERSHEY_SIMPLEX).
        font_scale: Font scale factor.
        color: Text color in BGR format (default white: (255, 255, 255)).
        thickness: Text line thickness in pixels.

    Returns:
        np.ndarray: Annotated image copy of shape (H, W, 3).
    """
    canvas = image.copy()
    cv2.putText(
        canvas,
        text=text,
        org=org,
        fontFace=font_face,
        fontScale=font_scale,
        color=color,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )
    return canvas
