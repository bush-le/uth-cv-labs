"""Visual presentation utilities for Lab 01 notebook."""

from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np


def plot_side_by_side(
    img1: np.ndarray,
    img2: np.ndarray,
    title1: str = "Image 1",
    title2: str = "Image 2",
    figsize: tuple[int, int] = (14, 6),
) -> plt.Figure:
    """Plot two images side by side for visual comparison."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].imshow(
        img1 if img1.ndim == 3 else img1, cmap=None if img1.ndim == 3 else "gray"
    )
    axes[0].set_title(title1, fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(
        img2 if img2.ndim == 3 else img2, cmap=None if img2.ndim == 3 else "gray"
    )
    axes[1].set_title(title2, fontsize=12)
    axes[1].axis("off")

    plt.tight_layout()
    return fig


def plot_color_spaces(
    img_bgr: np.ndarray,
    img_gray: np.ndarray,
    img_hsv: np.ndarray,
    img_lab: np.ndarray,
    figsize: tuple[int, int] = (12, 10),
) -> plt.Figure:
    """Plot 4 color spaces (RGB, Grayscale, HSV, Lab) in a 2x2 grid."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("1. RGB (Chuyển từ BGR gốc)", fontsize=11)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(img_gray, cmap="gray")
    axes[0, 1].set_title("2. Grayscale (Đơn kênh mức xám)", fontsize=11)
    axes[0, 1].axis("off")

    axes[1, 0].imshow(img_hsv)
    axes[1, 0].set_title("3. HSV (Hiển thị mảng HSV trên colormap RGB)", fontsize=11)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(img_lab)
    axes[1, 1].set_title("4. CIE L*a*b* (Mảng LAB trên colormap RGB)", fontsize=11)
    axes[1, 1].axis("off")

    plt.tight_layout()
    return fig


def plot_hsv_channels(
    h: np.ndarray,
    s: np.ndarray,
    v: np.ndarray,
    figsize: tuple[int, int] = (15, 4),
) -> plt.Figure:
    """Plot separated H, S, V channels side by side."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    axes[0].imshow(h, cmap="hsv")
    axes[0].set_title("Kênh H (Hue - Sắc độ)", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(s, cmap="gray")
    axes[1].set_title("Kênh S (Saturation - Độ bão hòa)", fontsize=11)
    axes[1].axis("off")

    axes[2].imshow(v, cmap="gray")
    axes[2].set_title("Kênh V (Value - Độ sáng)", fontsize=11)
    axes[2].axis("off")

    plt.tight_layout()
    return fig


def plot_roi_and_resize(
    original: np.ndarray,
    roi: np.ndarray,
    resized_scale: np.ndarray,
    resized_dims: np.ndarray,
    figsize: tuple[int, int] = (14, 10),
) -> plt.Figure:
    """Plot original, ROI, scale-resized, and fixed-dim resized images."""
    orig_rgb = (
        cv2.cvtColor(original, cv2.COLOR_BGR2RGB) if original.ndim == 3 else original
    )
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB) if roi.ndim == 3 else roi
    scale_rgb = (
        cv2.cvtColor(resized_scale, cv2.COLOR_BGR2RGB)
        if resized_scale.ndim == 3
        else resized_scale
    )
    dims_rgb = (
        cv2.cvtColor(resized_dims, cv2.COLOR_BGR2RGB)
        if resized_dims.ndim == 3
        else resized_dims
    )

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes[0, 0].imshow(orig_rgb)
    axes[0, 0].set_title(
        f"Ảnh gốc ({original.shape[1]}x{original.shape[0]})", fontsize=11
    )
    axes[0, 0].axis("off")

    axes[0, 1].imshow(roi_rgb)
    axes[0, 1].set_title(
        f"Vùng ROI trích xuất ({roi.shape[1]}x{roi.shape[0]})", fontsize=11
    )
    axes[0, 1].axis("off")

    axes[1, 0].imshow(scale_rgb)
    axes[1, 0].set_title(
        f"Thu nhỏ tỷ lệ ({resized_scale.shape[1]}x{resized_scale.shape[0]})",
        fontsize=11,
    )
    axes[1, 0].axis("off")

    axes[1, 1].imshow(dims_rgb)
    axes[1, 1].set_title(
        f"Kích thước cố định ({resized_dims.shape[1]}x{resized_dims.shape[0]})",
        fontsize=11,
    )
    axes[1, 1].axis("off")

    plt.tight_layout()
    return fig


def plot_annotated_image(
    image: np.ndarray,
    title: str = "Ảnh sau khi vẽ và chú thích",
    figsize: tuple[int, int] = (10, 8),
    is_bgr: bool = True,
) -> plt.Figure:
    """Plot an annotated or single image converted to RGB."""
    img_rgb = (
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if (is_bgr and image.ndim == 3 and image.shape[2] == 3)
        else image
    )
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img_rgb, cmap="gray" if img_rgb.ndim == 2 else None)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    return fig


plot_single_image = plot_annotated_image
