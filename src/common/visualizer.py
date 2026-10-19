"""Visualization utilities for Computer Vision experiments.

Provides functions to plot side-by-side comparisons, multi-image grids, edge overlays,
and analytical metric charts with standard styling.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.common.logger import get_logger

logger = get_logger(__name__)


def save_image(
    image: np.ndarray,
    output_path: str | Path,
    is_bgr: bool = False,
) -> Path:
    """Save an image array to disk safely using OpenCV.

    Args:
        image: Image array of shape (H, W) or (H, W, C).
        output_path: Target file path.
        is_bgr: True if color image is in BGR format, False if RGB or Grayscale.

    Returns:
        Path: Path object pointing to the written image.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if image.ndim == 3 and not is_bgr:
        write_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        write_img = image

    cv2.imwrite(str(path), write_img)
    logger.debug(f"Saved image to {path}")
    return path


def plot_side_by_side(
    images: Sequence[np.ndarray],
    titles: Sequence[str],
    save_path: str | Path,
    main_title: str = "",
    figsize: tuple[float, float] = (12.0, 6.0),
    cmaps: Sequence[str] | str = "gray",
) -> Path:
    """Plot images side-by-side with individual titles and save to disk.

    Args:
        images: Sequence of 2D or 3D image arrays of shape (H, W) or (H, W, C).
        titles: Sequence of title strings for each subplot.
        save_path: Destination path for the generated figure.
        main_title: Super title for the figure.
        figsize: Matplotlib figure dimensions (width, height).
        cmaps: Colormap name or sequence of colormap names.

    Returns:
        Path: Path to saved figure.
    """
    n_images = len(images)
    fig, axes = plt.subplots(1, n_images, figsize=figsize)
    if n_images == 1:
        axes = [axes]

    if isinstance(cmaps, str):
        cmaps = [cmaps] * n_images

    for ax, img, title, cmap in zip(axes, images, titles, cmaps, strict=False):
        if img.ndim == 2:
            ax.imshow(img, cmap=cmap)
        elif img.ndim == 3:
            # Assume RGB for matplotlib
            ax.imshow(img)
        ax.set_title(title, fontsize=12, fontweight="semibold")
        ax.axis("off")

    if main_title:
        fig.suptitle(main_title, fontsize=14, fontweight="bold", y=0.98)

    plt.tight_layout()
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved side-by-side plot to {out}")
    return out


def plot_grid(
    images: Sequence[np.ndarray],
    titles: Sequence[str],
    rows: int,
    cols: int,
    save_path: str | Path,
    main_title: str = "",
    figsize: tuple[float, float] = (16.0, 10.0),
    cmaps: Sequence[str] | str = "gray",
) -> Path:
    """Plot an arbitrary 2D grid of images.

    Args:
        images: Sequence of image arrays (H, W) or (H, W, C).
        titles: Sequence of titles for each cell.
        rows: Number of grid rows.
        cols: Number of grid columns.
        save_path: Destination file path.
        main_title: Overall figure super title.
        figsize: Matplotlib figure size tuple.
        cmaps: Colormap string or list of colormap strings.

    Returns:
        Path: Path to saved figure.
    """
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes_flat = np.array(axes).flatten()

    if isinstance(cmaps, str):
        cmaps_list = [cmaps] * len(images)
    else:
        cmaps_list = list(cmaps)

    for i in range(rows * cols):
        ax = axes_flat[i]
        if i < len(images):
            img = images[i]
            title = titles[i]
            cmap = cmaps_list[i] if i < len(cmaps_list) else "gray"
            if img.ndim == 2:
                ax.imshow(img, cmap=cmap)
            else:
                ax.imshow(img)
            ax.set_title(title, fontsize=10, fontweight="semibold")
        ax.axis("off")

    if main_title:
        fig.suptitle(main_title, fontsize=15, fontweight="bold", y=0.99)

    plt.tight_layout()
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved grid plot to {out}")
    return out


def create_edge_overlay(
    image: np.ndarray,
    edges: np.ndarray,
    color: tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.6,
) -> np.ndarray:
    """Overlay detected edges onto the original image.

    Args:
        image: Original RGB image (H, W, 3) or Grayscale image (H, W).
        edges: Binary edge map of shape (H, W) with boolean or uint8 [0, 255].
        color: RGB tuple for edge color, default bright red (255, 0, 0).
        alpha: Blending weight for edge overlay in [0.0, 1.0].

    Returns:
        np.ndarray: Blended RGB image of shape (H, W, 3) and dtype uint8.
    """
    if image.ndim == 2:
        base = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        base = image.copy()

    overlay = base.copy()
    edge_mask = edges > 0
    overlay[edge_mask] = color

    blended = cv2.addWeighted(overlay, alpha, base, 1.0 - alpha, 0)
    return blended
