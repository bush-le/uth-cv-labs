"""Image processing and transformation facade for Computer Vision tasks.

Consolidates and re-exports modular image I/O, color space conversions,
geometric transformations, and annotation primitives.
"""

from __future__ import annotations

from src.data.color_spaces import (
    bgr_to_grayscale,
    bgr_to_hsv,
    bgr_to_lab,
    bgr_to_rgb,
    pil_to_grayscale,
    pil_to_hsv,
    rgb_to_bgr,
)
from src.data.drawing import (
    draw_circle,
    draw_line,
    draw_rectangle,
    draw_text,
)
from src.data.geometric import (
    crop_roi,
    crop_roi_pil,
    resize_by_dimensions,
    resize_by_scale,
    resize_pil_by_dimensions,
    resize_pil_by_scale,
)
from src.data.image_io import (
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
    # Drawing & Annotations
    "draw_line",
    "draw_rectangle",
    "draw_circle",
    "draw_text",
]
