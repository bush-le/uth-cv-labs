"""Point operators and spatial filtering facade for Computer Vision.

Consolidates and re-exports modular point operations, linear convolution filters,
edge detection derivatives, non-linear denoising, and kernel inspection tools.
"""

from __future__ import annotations

from src.data.edge_filters import (
    apply_laplacian,
    apply_prewitt,
    apply_sobel,
    get_prewitt_kernels,
    get_sobel_kernels,
    gradient_magnitude,
)
from src.data.filter_utils import (
    _validate_image,
    _validate_kernel_size,
    find_project_root,
)
from src.data.linear_filters import (
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
from src.data.nonlinear_filters import (
    add_salt_pepper_noise,
    apply_bilateral_filter,
    apply_median_filter,
)
from src.data.point_ops import (
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
    # Linear Filters & Kernels
    "get_mean_kernel",
    "get_gaussian_kernel",
    "get_sharpen_kernel",
    "get_laplacian_kernel",
    "apply_custom_filter",
    "apply_mean_filter",
    "apply_gaussian_filter",
    "apply_sharpen_filter",
    "kernel_sum",
    # Edge Detection Operators
    "get_sobel_kernels",
    "get_prewitt_kernels",
    "apply_sobel",
    "apply_prewitt",
    "apply_laplacian",
    "gradient_magnitude",
    # Non-linear Filters & Noise
    "apply_median_filter",
    "apply_bilateral_filter",
    "add_salt_pepper_noise",
    # Helpers
    "find_project_root",
    "_validate_image",
    "_validate_kernel_size",
]
