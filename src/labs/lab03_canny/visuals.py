"""Visual presentation utilities for Lab 03 Canny Edge Detection."""

from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np


def plot_canny_comparison(
    images: list[np.ndarray],
    titles: list[str],
    ncols: int = 3,
    figsize: tuple[int, int] | None = None,
) -> plt.Figure:
    """Plot edge detection comparison across multiple algorithms."""
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


def plot_hysteresis_steps(
    smoothed: np.ndarray,
    magnitude: np.ndarray,
    nms: np.ndarray,
    final_edges: np.ndarray,
    figsize: tuple[int, int] = (14, 10),
) -> plt.Figure:
    """Plot the 4 key stages of Canny: Gaussian blur, Gradient magnitude, NMS thinning, and Hysteresis."""
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    axes[0, 0].imshow(smoothed, cmap="gray")
    axes[0, 0].set_title("1. Gaussian Smoothing", fontsize=11)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(magnitude, cmap="gray")
    axes[0, 1].set_title("2. Gradient Magnitude (Sobel)", fontsize=11)
    axes[0, 1].axis("off")

    axes[1, 0].imshow(nms, cmap="gray")
    axes[1, 0].set_title("3. Non-Maximum Suppression (Thinning)", fontsize=11)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(final_edges, cmap="gray")
    axes[1, 1].set_title("4. Double Thresholding & Hysteresis Tracking", fontsize=11)
    axes[1, 1].axis("off")

    plt.tight_layout()
    return fig
