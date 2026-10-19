"""Single-image and side-by-side comparison plotting utilities."""

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
    save_path: str | Path | None = None,
    is_bgr: bool = True,
    main_title: str = "",
    figsize: tuple[float, float] | None = None,
    cmaps: Sequence[str] | str = "gray",
) -> Path | plt.Figure:
    """Plot multiple images horizontally in a single row for comparison.

    Args:
        images: Sequence of image arrays of shape (H, W) or (H, W, C).
        titles: Sequence of titles corresponding to each image.
        save_path: Optional destination path for the generated figure.
        is_bgr: If True and image is 3-channel, converts BGR to RGB.
        main_title: Optional figure super title.
        figsize: Optional Matplotlib figure dimensions (width, height).
        cmaps: Colormap name or sequence of colormap names.

    Returns:
        Path | plt.Figure: Saved file Path if save_path is provided, else Figure.

    Raises:
        ValueError: If length of images and titles do not match.
    """
    if len(images) != len(titles):
        raise ValueError(
            f"Number of images ({len(images)}) must match titles ({len(titles)})"
        )

    num_images = len(images)
    size = figsize or (5.0 * num_images, 5.0)
    fig, axes = plt.subplots(1, num_images, figsize=size)

    if num_images == 1:
        axes = [axes]

    if isinstance(cmaps, str):
        cmaps_list = [cmaps] * num_images
    else:
        cmaps_list = list(cmaps)

    for ax, img, title, cmap in zip(axes, images, titles, cmaps_list, strict=False):
        if img.ndim == 2:
            ax.imshow(img, cmap=cmap)
        elif img.ndim == 3 and is_bgr:
            ax.imshow(bgr_to_rgb(img))
        else:
            ax.imshow(img)

        ax.set_title(title, fontsize=12, fontweight="semibold")
        ax.axis("off")

    if main_title:
        fig.suptitle(main_title, fontsize=14, fontweight="bold", y=0.98)

    plt.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved side-by-side figure to: %s", path)
        return path

    return fig
