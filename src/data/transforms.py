"""Image processing and transformation facade for Computer Vision tasks.

Backward-compatibility shim re-exporting from src.labs.lab01_basics.
"""

from __future__ import annotations

from src.labs.lab01_basics.color import (
    bgr_to_grayscale,
    bgr_to_hsv,
    bgr_to_lab,
    bgr_to_rgb,
    pil_to_grayscale,
    pil_to_hsv,
    rgb_to_bgr,
)
from src.labs.lab01_basics.drawing import (
    draw_circle,
    draw_line,
    draw_rectangle,
    draw_text,
)
from src.labs.lab01_basics.geometry import (
    crop_roi,
    crop_roi_pil,
    resize_by_dimensions,
    resize_by_scale,
    resize_pil_by_dimensions,
    resize_pil_by_scale,
)
from src.labs.lab01_basics.io import (
    convert_image_format,
    load_image_cv,
    load_image_pil,
    save_image_cv,
    save_image_pil,
)

__all__ = [
    # Image I/O
    "load_image_cv",
    "save_image_cv",
    "load_image_pil",
    "save_image_pil",
    "convert_image_format",
    # Color Spaces
    "bgr_to_rgb",
    "rgb_to_bgr",
    "bgr_to_grayscale",
    "bgr_to_hsv",
    "bgr_to_lab",
    "pil_to_grayscale",
    "pil_to_hsv",
    # Geometric Transformations
    "crop_roi",
    "crop_roi_pil",
    "resize_by_scale",
    "resize_by_dimensions",
    "resize_pil_by_scale",
    "resize_pil_by_dimensions",
    # Drawing Primitives
    "draw_line",
    "draw_rectangle",
    "draw_circle",
    "draw_text",
]
