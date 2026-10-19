"""Statistical, analytical, and metric evaluation plotting utilities."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.common.logger import get_logger

logger = get_logger(__name__)


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
