"""Visual presentation utilities for Lab 02 notebook."""

from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np


def plot_point_ops_grid(
    original: np.ndarray,
    bright: np.ndarray,
    contrast: np.ndarray,
    negative: np.ndarray,
    threshold: np.ndarray | None = None,
    figsize: tuple[int, int] = (14, 10),
) -> plt.Figure:
    """Plot comparison grid for point operations."""
    def to_rgb(im: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB) if (im.ndim == 3 and im.shape[2] == 3) else im

    n_plots = 5 if threshold is not None else 4
    cols = 2
    rows = (n_plots + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    axes_flat[0].imshow(to_rgb(original), cmap=None if original.ndim == 3 else "gray")
    axes_flat[0].set_title("Ảnh gốc", fontsize=11)
    axes_flat[0].axis("off")

    axes_flat[1].imshow(to_rgb(bright), cmap=None if bright.ndim == 3 else "gray")
    axes_flat[1].set_title("Tăng sáng (Beta = +50)", fontsize=11)
    axes_flat[1].axis("off")

    axes_flat[2].imshow(to_rgb(contrast), cmap=None if contrast.ndim == 3 else "gray")
    axes_flat[2].set_title("Tăng tương phản (Alpha = 1.5)", fontsize=11)
    axes_flat[2].axis("off")

    axes_flat[3].imshow(to_rgb(negative), cmap=None if negative.ndim == 3 else "gray")
    axes_flat[3].set_title("Ảnh âm bản (Negative)", fontsize=11)
    axes_flat[3].axis("off")

    if threshold is not None:
        axes_flat[4].imshow(threshold, cmap="gray")
        axes_flat[4].set_title("Ngưỡng hóa (Threshold = 127)", fontsize=11)
        axes_flat[4].axis("off")
        if len(axes_flat) > 5:
            axes_flat[5].axis("off")

    plt.tight_layout()
    return fig


def plot_filter_comparison(
    images: list[np.ndarray],
    titles: list[str],
    ncols: int = 3,
    figsize: tuple[int, int] | None = None,
) -> plt.Figure:
    """Plot an arbitrary list of filtered images in an organized grid."""
    n = len(images)
    ncols = min(n, ncols)
    nrows = (n + ncols - 1) // ncols
    if figsize is None:
        figsize = (5 * ncols, 4 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat = np.array(axes).reshape(-1)

    for i in range(len(axes_flat)):
        if i < n:
            im = images[i]
            cmap = "gray" if im.ndim == 2 else None
            disp = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) if (im.ndim == 3 and im.shape[2] == 3) else im
            axes_flat[i].imshow(disp, cmap=cmap)
            axes_flat[i].set_title(titles[i], fontsize=11)
            axes_flat[i].axis("off")
        else:
            axes_flat[i].axis("off")

    plt.tight_layout()
    return fig


def plot_sobel_gradient_grid(
    original_gray: np.ndarray,
    grad_x: np.ndarray,
    grad_y: np.ndarray,
    magnitude: np.ndarray,
    figsize: tuple[int, int] = (12, 10),
) -> plt.Figure:
    """Plot Sobel gradient x, y, and magnitude in a 2x2 grid."""
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes[0, 0].imshow(original_gray, cmap="gray")
    axes[0, 0].set_title("Ảnh gốc mức xám", fontsize=11)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(grad_x, cmap="gray")
    axes[0, 1].set_title("Đạo hàm Sobel theo trục X (Biên đứng)", fontsize=11)
    axes[0, 1].axis("off")

    axes[1, 0].imshow(grad_y, cmap="gray")
    axes[1, 0].set_title("Đạo hàm Sobel theo trục Y (Biên ngang)", fontsize=11)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(magnitude, cmap="gray")
    axes[1, 1].set_title("Độ lớn Gradient Magnitude (Căn bậc hai Gx^2 + Gy^2)", fontsize=11)
    axes[1, 1].axis("off")

    plt.tight_layout()
    return fig
