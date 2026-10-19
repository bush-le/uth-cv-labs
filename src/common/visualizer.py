"""Visualization utilities facade for Computer Vision experiments.

Consolidates and re-exports modular image plots, side-by-side grids,
edge overlays, distance heatmaps, and ROC/PR metric curves.
"""

from __future__ import annotations

from src.common.visualizers.grid_plots import (
    create_edge_overlay,
    plot_color_spaces_grid,
    plot_filter_grid,
    plot_grid,
)
from src.common.visualizers.image_plots import (
    plot_image,
    plot_side_by_side,
    save_image,
)
from src.common.visualizers.metric_plots import (
    plot_distance_heatmap,
    plot_histogram,
    plot_roc_and_pr_curves,
)

__all__ = [
    # Image & Comparison Plots
    "save_image",
    "plot_image",
    "plot_side_by_side",
    # Grid Visualizations & Overlays
    "plot_grid",
    "plot_filter_grid",
    "plot_color_spaces_grid",
    "create_edge_overlay",
    # Metric & Analytical Charts
    "plot_histogram",
    "plot_distance_heatmap",
    "plot_roc_and_pr_curves",
]
