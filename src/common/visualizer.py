"""Visualization utilities for Computer Vision experiments.

Provides side-by-side comparison, multi-channel grids, edge overlays,
analytical metric charts, and publication-ready figure generation
with safe headless Matplotlib backend.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import matplotlib
import numpy as np

# Ensure headless compatibility
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


def plot_histogram(
    image: np.ndarray,
    title: str = "Intensity Histogram",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot the intensity histogram of a grayscale or color image.

    Args:
        image: Image array of shape (H, W, 3) or (H, W), dtype uint8.
        title: Title of the plot.
        save_path: Optional path to save the generated figure.

    Returns:
        plt.Figure: Created Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    if image.ndim == 3:
        colors = ("b", "g", "r")  # BGR order matches OpenCV channel order
        for i, color in enumerate(colors):
            ax.hist(
                image[:, :, i].ravel(),
                bins=256,
                range=(0, 256),
                color=color,
                alpha=0.5,
                label=color.upper(),
            )
        ax.legend(loc="upper right")
    else:
        ax.hist(image.ravel(), bins=256, range=(0, 256), color="gray")

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Frequency")
    plt.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        logger.info("Saved histogram to: %s", path)

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


def plot_distance_heatmap(
    matrix: np.ndarray,
    labels: Sequence[str],
    save_path: str | Path,
    title: str = "Pairwise Hamming Distance Matrix",
    figsize: tuple[float, float] = (10.0, 8.5),
    cmap: str = "viridis",
) -> Path:
    """Plot a labeled heatmap for pairwise Hamming distance matrices.

    Args:
        matrix: Square matrix of shape (M, M) containing distance values.
        labels: Sequence of image labels of length M.
        save_path: Target file path.
        title: Plot title.
        figsize: Dimensions of the figure.
        cmap: Matplotlib colormap.

    Returns:
        Path: Path to saved plot.
    """
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, cmap=cmap, interpolation="nearest")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Hamming Distance", fontsize=11, fontweight="semibold")

    m = len(labels)
    ax.set_xticks(np.arange(m))
    ax.set_yticks(np.arange(m))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    # Annotate numeric values if matrix is reasonably sized
    if m <= 15:
        max_val = np.max(matrix)
        for i in range(m):
            for j in range(m):
                val = matrix[i, j]
                color = "white" if val < 0.5 * max_val else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.0f}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=8,
                )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved heatmap to {out}")
    return out


def plot_roc_and_pr_curves(
    fpr: Sequence[float],
    tpr: Sequence[float],
    auc_score: float,
    precisions: Sequence[float],
    recalls: Sequence[float],
    save_path: str | Path,
    optimal_fpr: float | None = None,
    optimal_tpr: float | None = None,
    figsize: tuple[float, float] = (14.0, 5.5),
) -> Path:
    """Plot dual Receiver Operating Characteristic (ROC) and Precision-Recall (PR) curves.

    Args:
        fpr: False positive rate values.
        tpr: True positive rate values.
        auc_score: Area under ROC curve.
        precisions: Precision values across thresholds.
        recalls: Recall values across thresholds.
        save_path: Destination file path.
        optimal_fpr: Optional FPR coordinate of optimal threshold.
        optimal_tpr: Optional TPR coordinate of optimal threshold.
        figsize: Matplotlib figure size tuple.

    Returns:
        Path: Path to saved figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # ROC Curve
    ax1.plot(
        fpr, tpr, color="#06b6d4", lw=2.5, label=f"ROC Curve (AUC = {auc_score:.4f})"
    )
    ax1.plot(
        [0, 1], [0, 1], color="#94a3b8", lw=1.5, linestyle="--", label="Random Chance"
    )
    if optimal_fpr is not None and optimal_tpr is not None:
        ax1.scatter(
            [optimal_fpr],
            [optimal_tpr],
            color="#ef4444",
            s=80,
            zorder=5,
            label=f"Optimal Point (TPR={optimal_tpr:.2f}, FPR={optimal_fpr:.2f})",
        )
    ax1.set_xlabel(
        "False Positive Rate (1 - Specificity)", fontsize=11, fontweight="semibold"
    )
    ax1.set_ylabel(
        "True Positive Rate (Recall / Sensitivity)", fontsize=11, fontweight="semibold"
    )
    ax1.set_title(
        "Đường cong ROC (Receiver Operating Characteristic)",
        fontsize=12,
        fontweight="bold",
    )
    ax1.legend(loc="lower right", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Precision-Recall Curve
    ax2.plot(recalls, precisions, color="#8b5cf6", lw=2.5, label="PR Curve")
    ax2.set_xlabel("Recall (Độ nhạy)", fontsize=11, fontweight="semibold")
    ax2.set_ylabel("Precision (Độ xác thực)", fontsize=11, fontweight="semibold")
    ax2.set_title("Đường cong Precision - Recall", fontsize=12, fontweight="bold")
    ax2.legend(loc="lower left", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved ROC/PR curves to {out}")
    return out
