"""Multi-image grid plotting and edge overlay visualization utilities."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.common.logger import get_logger
from src.data.transforms import bgr_to_rgb

logger = get_logger(__name__)


def plot_color_spaces_grid(
    original_bgr: np.ndarray,
    gray: np.ndarray,
    hsv: np.ndarray,
    lab: np.ndarray,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot a 2x2 grid comparing Original RGB, Grayscale, HSV, and LAB representations.

    Args:
        original_bgr: Original image in BGR format, shape (H, W, 3).
        gray: Grayscale image, shape (H, W).
        hsv: HSV color space image, shape (H, W, 3).
        lab: LAB color space image, shape (H, W, 3).
        save_path: Optional path to save the figure.

    Returns:
        plt.Figure: Created Matplotlib figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Original (RGB)
    axes[0, 0].imshow(bgr_to_rgb(original_bgr))
    axes[0, 0].set_title("1. Original Image (RGB)", fontsize=13, fontweight="bold")
    axes[0, 0].axis("off")

    # 2. Grayscale
    axes[0, 1].imshow(gray, cmap="gray")
    axes[0, 1].set_title("2. Grayscale (Luminance Y)", fontsize=13, fontweight="bold")
    axes[0, 1].axis("off")

    # 3. HSV
    axes[1, 0].imshow(hsv)
    axes[1, 0].set_title(
        "3. HSV (Hue, Saturation, Value)", fontsize=13, fontweight="bold"
    )
    axes[1, 0].axis("off")

    # 4. LAB
    axes[1, 1].imshow(lab)
    axes[1, 1].set_title("4. CIE L*a*b* Color Space", fontsize=13, fontweight="bold")
    axes[1, 1].axis("off")

    plt.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        logger.info("Saved color spaces grid to: %s", path)

    return fig


def plot_filter_grid(
    images: Sequence[np.ndarray],
    titles: Sequence[str],
    ncols: int = 3,
    is_bgr: bool = True,
    save_path: str | Path | None = None,
    figsize_per_cell: tuple[float, float] = (4.5, 4.0),
) -> plt.Figure:
    """Plot a grid of images for filter comparison.

    Args:
        images: Sequence of image arrays of shape (H, W, 3) or (H, W).
        titles: Sequence of titles corresponding to each image.
        ncols: Number of columns in the grid (default 3).
        is_bgr: If True, converts 3-channel images from BGR to RGB.
        save_path: Optional path to save the generated figure.
        figsize_per_cell: (width, height) in inches allocated per grid cell.

    Returns:
        plt.Figure: Created Matplotlib figure.

    Raises:
        ValueError: If lengths mismatch or ncols is non-positive.
    """
    if len(images) != len(titles):
        raise ValueError(
            f"Number of images ({len(images)}) must match titles ({len(titles)})"
        )
    if ncols <= 0:
        raise ValueError(f"ncols must be positive, got {ncols}")

    num_images = len(images)
    nrows = int(np.ceil(num_images / ncols))
    fig_w = figsize_per_cell[0] * ncols
    fig_h = figsize_per_cell[1] * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

    for idx, (img, title) in enumerate(zip(images, titles, strict=True)):
        ax = axes[idx // ncols][idx % ncols]
        if img.ndim == 2:
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        elif img.ndim == 3 and is_bgr:
            ax.imshow(bgr_to_rgb(img))
        else:
            ax.imshow(img)
        ax.set_title(title, fontsize=11, fontweight="semibold")
        ax.axis("off")

    # Hide unused subplots in the last row
    for idx in range(num_images, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    plt.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        logger.info("Saved filter grid figure to: %s", path)

    return fig


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
