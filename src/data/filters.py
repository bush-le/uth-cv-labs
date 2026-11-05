"""Point operators and spatial filtering facade for Computer Vision.

Backward-compatibility shim re-exporting from src.labs.lab02_filters.
"""

from __future__ import annotations

from src.labs.lab02_filters.edge_filters import (
    apply_laplacian,
    apply_prewitt,
    apply_sobel,
    get_prewitt_kernels,
    get_sobel_kernels,
    gradient_magnitude,
)
from src.labs.lab02_filters.filter_utils import (
    _validate_image,
    _validate_kernel_size,
    find_project_root,
)
from src.labs.lab02_filters.linear_filters import (
    apply_custom_filter,
    apply_gaussian_filter,
    apply_mean_filter,
    apply_sharpen_filter,
    get_gaussian_kernel,
    get_laplacian_kernel,
    get_mean_kernel,
    get_sharpen_kernel,
    kernel_sum,
)
from src.labs.lab02_filters.nonlinear_filters import (
    add_salt_pepper_noise,
    apply_bilateral_filter,
    apply_median_filter,
)
from src.labs.lab02_filters.point_ops import (
    adjust_brightness,
    adjust_contrast,
    apply_threshold,
    negative_transform,
)

__all__ = [
    # Point Operations
    "adjust_brightness",
    "adjust_contrast",
    "negative_transform",
    "apply_threshold",
    # Linear Filters
    "get_mean_kernel",
    "get_gaussian_kernel",
    "get_sharpen_kernel",
    "get_laplacian_kernel",
    "apply_custom_filter",
    "apply_mean_filter",
    "apply_gaussian_filter",
    "apply_sharpen_filter",
    "kernel_sum",
    # Edge Detection Filters
    "get_sobel_kernels",
    "get_prewitt_kernels",
    "apply_sobel",
    "apply_prewitt",
    "apply_laplacian",
    "gradient_magnitude",
    # Non-linear Filters
    "apply_median_filter",
    "apply_bilateral_filter",
    "add_salt_pepper_noise",
    # Validation Utilities
    "_validate_image",
    "_validate_kernel_size",
    "find_project_root",
]
