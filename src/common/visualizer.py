"""Visualization utilities for Computer Vision experiments.

Provides side-by-side comparison, multi-channel grids, and publication-ready
figure generation with safe headless Matplotlib backend.
"""

from collections.abc import Sequence
from pathlib import Path

import matplotlib
import numpy as np

# Ensure headless compatibility
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.common.logger import get_logger
from src.data.transforms import bgr_to_rgb

logger = get_logger(__name__)


def plot_image(
    image: np.ndarray,
    title: str = "",
    is_bgr: bool = True,
    cmap: str | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot a single image with Matplotlib.

    Args:
        image: Image array of shape (H, W, 3) or (H, W).
        title: Title of the plot.
        is_bgr: If True and image has 3 channels, converts BGR to RGB for correct display.
        cmap: Colormap to use (defaults to 'gray' if grayscale image).
        save_path: Optional path to save the generated figure.

    Returns:
        plt.Figure: Created Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    if image.ndim == 2:
        colormap = cmap or "gray"
        ax.imshow(image, cmap=colormap)
    elif image.ndim == 3 and is_bgr:
        ax.imshow(bgr_to_rgb(image))
    else:
        ax.imshow(image)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        logger.info("Saved figure to: %s", path)

    return fig


def plot_side_by_side(
    images: Sequence[np.ndarray],
    titles: Sequence[str],
    is_bgr: bool = True,
    save_path: str | Path | None = None,
    figsize: tuple[int, int] | None = None,
) -> plt.Figure:
    """Plot multiple images horizontally in a single row for comparison.

    Args:
        images: Sequence of image arrays.
        titles: Sequence of titles corresponding to each image.
        is_bgr: If True, converts 3-channel images from BGR to RGB.
        save_path: Optional path to save the generated figure.
        figsize: Optional figure size tuple (width, height).

    Returns:
        plt.Figure: Created Matplotlib figure.

    Raises:
        ValueError: If length of images and titles do not match.
    """
    if len(images) != len(titles):
        raise ValueError(
            f"Number of images ({len(images)}) must match titles ({len(titles)})"
        )

    num_images = len(images)
    size = figsize or (5 * num_images, 5)
    fig, axes = plt.subplots(1, num_images, figsize=size)

    if num_images == 1:
        axes = [axes]

    for ax, img, title in zip(axes, images, titles, strict=True):
        if img.ndim == 2:
            ax.imshow(img, cmap="gray")
        elif img.ndim == 3 and is_bgr:
            ax.imshow(bgr_to_rgb(img))
        else:
            ax.imshow(img)

        ax.set_title(title, fontsize=12, fontweight="semibold")
        ax.axis("off")

    plt.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        logger.info("Saved side-by-side figure to: %s", path)

    return fig


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
