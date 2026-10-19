"""Visualization utilities for Computer Vision experiments.

Provides functions to plot side-by-side comparisons, multi-image grids, distance heatmaps,
and ROC/Precision-Recall curves with standard styling.
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
            ax.imshow(img)
        ax.set_title(title, fontsize=11, fontweight="semibold")
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
