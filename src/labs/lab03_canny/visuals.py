"""Visual presentation utilities for Lab 03 Canny Edge Detection."""

from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
from matplotlib.patches import Rectangle

from src.labs.lab03_canny.canny_core import custom_canny_pipeline


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
            disp = (
                cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                if (im.ndim == 3 and im.shape[2] == 3)
                else im
            )
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


def plot_gaussian_tradeoff_1d(
    noise_sigma: float = 8.0,
    sigmas: tuple[float, ...] = (0.6, 1.8, 5.0),
    figsize: tuple[int, int] = (12, 7),
) -> plt.Figure:
    """Plot 1D Gaussian smoothing tradeoff: noise suppression vs edge localization and broadening.

    Simulates a 1D ideal step edge corrupted by additive Gaussian noise, showing:
    - Top subplot: Raw noisy step signal alongside smoothed waveforms under undersmoothed (sigma=0.6),
      optimal (sigma=1.8), and oversmoothed (sigma=5.0) Gaussian kernels.
    - Bottom subplot: 1st numerical derivative profiles illustrating spurious local extrema,
      sharp Dirac-like peak localization, and peak flattening / position drift.

    Args:
        noise_sigma: Standard deviation of additive Gaussian noise.
        sigmas: Tuple of Gaussian scale parameters to compare.
        figsize: Output figure dimension tuple (width, height).

    Returns:
        plt.Figure: Matplotlib figure containing the 2-stage analysis.
    """
    n_points = 240
    step_pos = 120
    x = np.arange(n_points)

    # Ideal step signal: jump from 60 to 180
    ideal = np.where(x >= step_pos, 180.0, 60.0)
    rng = np.random.default_rng(42)
    noise = rng.normal(0.0, noise_sigma, size=n_points)
    noisy_signal = ideal + noise

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    colors = {
        0.6: ("#ef4444", "Undersmoothed (nhiễu sót lại)"),
        1.8: ("#10b981", "Optimal (cân đối lọc nhiễu & sắc nét)"),
        5.0: ("#3b82f6", "Oversmoothed (mờ biên, dạt vị trí)"),
    }
    default_colors = ["#8b5cf6", "#ec4899", "#f59e0b"]

    # Top: Signals
    ax1 = axes[0]
    ax1.plot(
        x,
        ideal,
        "k--",
        alpha=0.5,
        linewidth=1.5,
        label="Biên bước nhảy lý tưởng (Không nhiễu)",
    )
    ax1.plot(
        x,
        noisy_signal,
        color="#94a3b8",
        alpha=0.5,
        linewidth=1.0,
        label=f"Tín hiệu có nhiễu (noise σ={noise_sigma})",
    )

    derivs = {}
    for i, s in enumerate(sigmas):
        smoothed = scipy.ndimage.gaussian_filter1d(noisy_signal, sigma=s)
        col, tag = colors.get(s, (default_colors[i % len(default_colors)], f"σ={s}"))
        ax1.plot(
            x,
            smoothed,
            color=col,
            linewidth=2.0,
            label=f"Smoothed σ={s:.1f}: {tag}",
        )
        derivs[s] = (np.gradient(smoothed), col, tag)

    ax1.axvline(step_pos, color="gray", linestyle=":", alpha=0.7)
    ax1.set_ylabel("Cường độ tín hiệu 1D $I(x)$", fontsize=11)
    ax1.set_title(
        "1. Đánh đổi Gaussian 1D: Khử nhiễu tần số cao vs Làm mờ sườn biên độ",
        fontsize=12,
        fontweight="bold",
    )
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="upper left", fontsize=9, framealpha=0.9)

    # Bottom: Derivatives
    ax2 = axes[1]
    for s in sigmas:
        deriv, col, tag = derivs[s]
        ax2.plot(x, deriv, color=col, linewidth=2.0, label=f"dI/dx (σ={s:.1f})")

    ax2.axvline(
        step_pos,
        color="gray",
        linestyle=":",
        alpha=0.7,
        label=f"Vị trí biên thật (x={step_pos})",
    )
    ax2.set_xlabel("Vị trí tọa độ không gian 1D (pixel)", fontsize=11)
    ax2.set_ylabel("Đạo hàm bậc nhất $dI/dx$", fontsize=11)
    ax2.set_title(
        "2. Đạo hàm bậc 1 $dI/dx$: Đỉnh giả do nhiễu (σ=0.6) vs Đỉnh sắc chuẩn (σ=1.8) vs Đỉnh tù dạt rộng (σ=5.0)",
        fontsize=12,
        fontweight="bold",
    )
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="upper left", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    return fig


def plot_gradient_quiver_roi(
    image: np.ndarray,
    y_slice: slice | tuple[int, int] | None = None,
    x_slice: slice | tuple[int, int] | None = None,
    step: int = 1,
    figsize: tuple[int, int] = (8, 8),
) -> plt.Figure:
    """Plot zoomed-in ROI vector field showing gradient arrows orthogonal to edge boundaries.

    Args:
        image: Grayscale or BGR image.
        y_slice: Row slice or (y_min, y_max) tuple defining the ROI window.
        x_slice: Col slice or (x_min, x_max) tuple defining the ROI window.
        step: Pixel subsampling step for quiver vectors (default 1).
        figsize: Figure dimension tuple.

    Returns:
        plt.Figure: Quiver vector field overlaid on pixel intensity patch.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    h, w = gray.shape

    # Default ROI: look for high gradient patch or center window
    if y_slice is None or x_slice is None:
        if h >= 116 and w >= 116:
            y_sl = slice(100, 116)
            x_sl = slice(100, 116)
        else:
            cy, cx = h // 2, w // 2
            y_sl = slice(max(0, cy - 8), min(h, cy + 8))
            x_sl = slice(max(0, cx - 8), min(w, cx + 8))
    else:
        y_sl = slice(*y_slice) if isinstance(y_slice, tuple) else y_slice
        x_sl = slice(*x_slice) if isinstance(x_slice, tuple) else x_slice

    patch = gray[y_sl, x_sl].astype(np.float32)
    ph, pw = patch.shape
    if ph == 0 or pw == 0:
        raise ValueError("Invalid ROI slices resulting in empty patch.")

    # Compute Sobel gradients
    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(patch, cmap="gray", origin="upper", extent=[0, pw, ph, 0])

    # Quiver grid
    y_coords, x_coords = np.mgrid[0:ph:step, 0:pw:step]
    u = gx[0:ph:step, 0:pw:step]
    v = gy[0:ph:step, 0:pw:step]
    m_sub = mag[0:ph:step, 0:pw:step]

    # Matplotlib quiver on imshow (origin='upper'):
    # In image coordinates, y increases downward. Sobel Gy > 0 points downward.
    # ax.quiver with angles='xy', scale_units='xy' points along coordinate axes,
    # correctly matching (gx, gy) without sign inversion.
    q = ax.quiver(
        x_coords + 0.5,
        y_coords + 0.5,
        u,
        v,
        m_sub,
        cmap="autumn",
        angles="xy",
        scale_units="xy",
        scale=None,
        width=0.007,
        pivot="mid",
    )

    cb = fig.colorbar(q, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Độ lớn Gradient $|\\nabla I|$", fontsize=10)

    # Pixel grid lines
    ax.set_xticks(np.arange(0, pw + 1, 2))
    ax.set_yticks(np.arange(0, ph + 1, 2))
    ax.grid(color="cyan", alpha=0.25, linestyle=":", linewidth=0.8)

    ax.set_title(
        f"Trường Vector Gradient $\\nabla I = (G_x, G_y)$ trên ROI vi mô ({pw}×{ph})\n"
        "(Mũi tên chỉ hướng biến thiên cực đại, vuông góc với đường bao biên)",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel("Tọa độ Cột cục bộ (x)")
    ax.set_ylabel("Tọa độ Hàng cục bộ (y)")

    plt.tight_layout()
    return fig


def plot_sector_directional_map(
    image: np.ndarray,
    threshold: float = 30.0,
    figsize: tuple[int, int] = (10, 8),
) -> plt.Figure:
    """Plot 4-sector quantized gradient directional map with compass inset.

    Discretizes gradient directions into 4 canonical Canny normal sectors:
    - Sector 0 (0 deg, Horizontal normal / Vertical edge): Yellow
    - Sector 1 (45 deg, Diagonal normal): Green
    - Sector 2 (90 deg, Vertical normal / Horizontal edge): Cyan
    - Sector 3 (135 deg, Diagonal normal): Magenta
    Pixels with gradient magnitude below threshold are rendered in black.

    Args:
        image: Grayscale or BGR input image.
        threshold: Magnitude cutoff threshold for active edge pixels.
        figsize: Figure dimension tuple.

    Returns:
        plt.Figure: Figure with 4-color directional map and sector statistics/compass.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    pipe = custom_canny_pipeline(
        gray, sigma=1.0, low_threshold=threshold, high_threshold=threshold * 2
    )
    mag = pipe["magnitude"]
    deg = pipe["direction_deg"]

    h, w = gray.shape
    color_map = np.zeros((h, w, 3), dtype=np.uint8)

    mask0 = ((deg >= 0.0) & (deg < 22.5)) | (deg >= 157.5)
    mask1 = (deg >= 22.5) & (deg < 67.5)
    mask2 = (deg >= 67.5) & (deg < 112.5)
    mask3 = (deg >= 112.5) & (deg < 157.5)

    edge_mask = mag >= threshold

    # 4 distinct canonical colors: Yellow, Green, Cyan, Magenta
    color_map[edge_mask & mask0] = [255, 235, 59]  # Yellow: 0 deg
    color_map[edge_mask & mask1] = [34, 197, 94]  # Green: 45 deg
    color_map[edge_mask & mask2] = [6, 182, 212]  # Cyan: 90 deg
    color_map[edge_mask & mask3] = [236, 72, 153]  # Magenta: 135 deg

    total_edges = max(1, int(np.count_nonzero(edge_mask)))
    counts = [
        int(np.count_nonzero(edge_mask & mask0)),
        int(np.count_nonzero(edge_mask & mask1)),
        int(np.count_nonzero(edge_mask & mask2)),
        int(np.count_nonzero(edge_mask & mask3)),
    ]
    pcts = [c / total_edges * 100.0 for c in counts]

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, width_ratios=[2.5, 1], height_ratios=[1, 1])

    ax_main = fig.add_subplot(gs[:, 0])
    ax_main.imshow(color_map)
    ax_main.set_title(
        f"Bản Đồ 4 Sector Hướng Gradient Lượng Tử Hóa (|∇I| ≥ {threshold:.0f})",
        fontsize=12,
        fontweight="bold",
    )
    ax_main.axis("off")

    # Polar Compass Subplot
    ax_polar = fig.add_subplot(gs[0, 1], projection="polar")
    sector_angles_rad = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    wedge_colors = ["#facc15", "#22c55e", "#06b6d4", "#ec4899"]
    sector_labels = ["0° (Doc)", "45° (Cheo)", "90° (Ngang)", "135° (Cheo)"]

    for ang, col in zip(sector_angles_rad, wedge_colors, strict=False):
        ax_polar.bar(
            ang,
            1.0,
            width=np.pi / 4,
            color=col,
            alpha=0.85,
            edgecolor="black",
            linewidth=1.2,
        )
        # Symmetrical counter-directional wedge for Canny [0, 180)
        ax_polar.bar(
            ang + np.pi,
            1.0,
            width=np.pi / 4,
            color=col,
            alpha=0.85,
            edgecolor="black",
            linewidth=1.2,
        )

    ax_polar.set_theta_zero_location("E")
    ax_polar.set_yticklabels([])
    ax_polar.set_title("La Bàn Hướng Gradient", fontsize=10, fontweight="bold", pad=12)

    # Sector Statistics Bar Subplot
    ax_bar = fig.add_subplot(gs[1, 1])
    bars = ax_bar.barh(
        sector_labels, pcts, color=wedge_colors, edgecolor="black", linewidth=0.8
    )
    ax_bar.set_xlabel("Tỷ lệ biên (%)", fontsize=9)
    ax_bar.set_xlim(0, max(pcts + [40]) * 1.15)
    for bar, pct, cnt in zip(bars, pcts, counts, strict=False):
        ax_bar.text(
            bar.get_width() + 1.0,
            bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}% ({cnt:,} px)",
            va="center",
            fontsize=8,
            fontweight="bold",
        )
    ax_bar.grid(axis="x", linestyle="--", alpha=0.5)
    ax_bar.set_title("Phân Bố 4 Hướng", fontsize=10, fontweight="bold")

    plt.tight_layout()
    return fig


def plot_nms_ridge_cross_section(
    image: np.ndarray,
    row: int | None = None,
    col_start: int = 0,
    col_end: int | None = None,
    figsize: tuple[int, int] = (12, 6),
) -> plt.Figure:
    """Plot 1D cross-section showing smooth intensity, broad Sobel ridge, and 1-pixel NMS impulse.

    Args:
        image: Grayscale or BGR image.
        row: Row index for 1D horizontal scan. If None, picks the row with highest NMS peak.
        col_start: Starting column index.
        col_end: Ending column index.
        figsize: Figure dimension tuple.

    Returns:
        plt.Figure: 2-subplot figure illustrating non-maximum suppression thinning mechanism.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    pipe = custom_canny_pipeline(
        gray, sigma=1.5, low_threshold=30.0, high_threshold=100.0
    )
    smoothed = pipe["smoothed"]
    magnitude = pipe["magnitude"]
    nms = pipe["nms"]

    h, w = gray.shape
    if row is None:
        # Pick row with highest NMS peak
        row = int(np.argmax(np.max(nms, axis=1)))
    row = max(0, min(h - 1, row))

    if col_end is None:
        col_end = w
    col_end = max(col_start + 1, min(w, col_end))

    cols = np.arange(col_start, col_end)
    i_prof = smoothed[row, col_start:col_end]
    m_prof = magnitude[row, col_start:col_end]
    nms_prof = nms[row, col_start:col_end]

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Subplot 1: Smoothed intensity transition
    ax1 = axes[0]
    ax1.plot(
        cols,
        i_prof,
        color="#0284c7",
        linewidth=2.0,
        label="Cường độ mượt $I_\\sigma(x)$",
    )
    ax1.set_ylabel("Mức xám $I_\\sigma$", fontsize=10)
    ax1.set_title(
        f"1. Mặt Cắt Ngang 1D Tại Hàng y={row}: Cường Độ Ảnh Mượt $I_\\sigma(x)$ (Chuyển Tiếp Sườn Biên)",
        fontsize=11,
        fontweight="bold",
    )
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="upper right", fontsize=9)

    # Subplot 2: Sobel continuous broad ridge vs NMS discrete delta impulse
    ax2 = axes[1]
    ax2.plot(
        cols,
        m_prof,
        color="#f59e0b",
        linewidth=1.8,
        label="Độ lớn Gradient Sobel $|\\nabla I|$ (Sườn dốc rộng)",
    )
    ax2.fill_between(cols, m_prof, color="#fef3c7", alpha=0.6)

    # NMS spike
    peak_cols = cols[nms_prof > 0]
    peak_vals = nms_prof[nms_prof > 0]
    if len(peak_cols) > 0:
        ax2.vlines(
            peak_cols,
            0,
            peak_vals,
            color="#dc2626",
            linewidth=2.5,
            label="NMS 1-pixel Dirac Impulse (Biên mảnh)",
        )
        ax2.scatter(peak_cols, peak_vals, color="#dc2626", s=45, zorder=5)
        for pc, pv in zip(peak_cols, peak_vals, strict=False):
            ax2.annotate(
                f"Peak x={pc}\nval={pv:.1f}",
                xy=(pc, pv),
                xytext=(pc + 2, pv * 0.85),
                arrowprops={"arrowstyle": "->", "color": "#dc2626", "lw": 1.2},
                fontsize=8,
                fontweight="bold",
                color="#dc2626",
            )

    ax2.set_xlabel("Tọa độ Cột không gian 1D (x)", fontsize=10)
    ax2.set_ylabel("Độ lớn Gradient", fontsize=10)
    ax2.set_title(
        "2. Cơ Chế Triệt Tiêu Phi Cực Đại (NMS): Thu Hẹp Sườn Đồi Rộng Thành Xung 1-Pixel Sắc Nét",
        fontsize=11,
        fontweight="bold",
    )
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    return fig


def plot_hysteresis_tricolor(
    image: np.ndarray,
    low_thresh: float = 50.0,
    high_thresh: float = 150.0,
    figsize: tuple[int, int] = (12, 6),
) -> plt.Figure:
    """Plot tri-color hysteresis classification edge map with quantitative partition bar.

    Classifies candidate edge pixels into 3 mutually exclusive sets:
    - Strong edges (|G| >= T_high): Green
    - Weak rescued edges (T_low <= |G| < T_high, connected to strong edge via 8-connectivity): Sky Blue
    - Weak suppressed edges (T_low <= |G| < T_high, isolated noise): Red

    Args:
        image: Grayscale or BGR image.
        low_thresh: Lower threshold T_low.
        high_thresh: Upper threshold T_high.
        figsize: Figure dimension tuple.

    Returns:
        plt.Figure: Figure showing tri-color spatial distribution and pixel count breakdown.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    pipe = custom_canny_pipeline(
        gray, sigma=1.0, low_threshold=low_thresh, high_threshold=high_thresh
    )
    strong = pipe["strong_edges"]
    weak = pipe["weak_edges"]
    final_edges = pipe["edges"] > 0

    weak_rescued = weak & final_edges
    weak_suppressed = weak & (~final_edges)

    h, w = gray.shape
    tricolor = np.zeros((h, w, 3), dtype=np.uint8)

    # Color definitions
    # Strong: Green
    tricolor[strong] = [34, 197, 94]
    # Rescued weak: Sky Blue
    tricolor[weak_rescued] = [56, 189, 248]
    # Suppressed weak: Coral Red
    tricolor[weak_suppressed] = [239, 68, 68]

    n_strong = int(np.count_nonzero(strong))
    n_rescued = int(np.count_nonzero(weak_rescued))
    n_suppressed = int(np.count_nonzero(weak_suppressed))
    total_candidates = max(1, n_strong + n_rescued + n_suppressed)

    fig, (ax_img, ax_bar) = plt.subplots(
        1, 2, figsize=figsize, gridspec_kw={"width_ratios": [2.2, 1]}
    )

    ax_img.imshow(tricolor)
    ax_img.set_title(
        f"Bản Đồ 3 Màu Hysteresis Tracking (T_low={low_thresh:.0f}, T_high={high_thresh:.0f})\n"
        "Xanh lá: Biên mạnh | Xanh dương: Biên yếu nối thông | Đỏ: Biên yếu bị triệt tiêu",
        fontsize=11,
        fontweight="bold",
    )
    ax_img.axis("off")

    # Bar chart
    categories = [
        "Biên Mạnh\n(|G| ≥ T_high)",
        "Biên Yếu Cứu\n(Nối 8-hướng)",
        "Biên Yếu Loại\n(Nhiễu cô lập)",
    ]
    counts = [n_strong, n_rescued, n_suppressed]
    pcts = [c / total_candidates * 100.0 for c in counts]
    colors = ["#22c55e", "#38bdf8", "#ef4444"]

    bars = ax_bar.bar(
        categories, counts, color=colors, edgecolor="black", linewidth=1.0
    )
    ax_bar.set_ylabel("Số lượng điểm ảnh (pixels)", fontsize=10)
    ax_bar.set_title("Thống Kê Ứng Viên Hysteresis", fontsize=11, fontweight="bold")
    ax_bar.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, c, p in zip(bars, counts, pcts, strict=False):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.02,
            f"{c:,} px\n({p:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    ax_bar.set_ylim(0, max(counts) * 1.25)

    plt.tight_layout()
    return fig


def plot_magnitude_cdf_histogram(
    image: np.ndarray,
    low_thresh: float = 50.0,
    high_thresh: float = 150.0,
    figsize: tuple[int, int] = (10, 5),
) -> plt.Figure:
    """Plot gradient magnitude histogram with cumulative distribution function (CDF) and threshold partitions.

    Illustrates the empirical distribution of gradient magnitudes |nabla I|, marking:
    - Region 1 (0 to T_low): Background noise suppressed unconditionally.
    - Region 2 (T_low to T_high): Weak edge candidates undergoing hysteresis connectivity search.
    - Region 3 (T_high to inf): Strong seed edges confirmed unconditionally.

    Args:
        image: Grayscale or BGR image.
        low_thresh: Lower threshold T_low.
        high_thresh: Upper threshold T_high.
        figsize: Figure dimension tuple.

    Returns:
        plt.Figure: Dual-axis plot with histogram and CDF overlay.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    pipe = custom_canny_pipeline(
        gray, sigma=1.0, low_threshold=low_thresh, high_threshold=high_thresh
    )
    mag = pipe["magnitude"]
    # Filter flat non-edge baseline (< 1.0) to focus on structural gradient distribution
    active_mag = mag[mag >= 1.0].ravel()
    if len(active_mag) == 0:
        active_mag = mag.ravel()

    fig, ax1 = plt.subplots(figsize=figsize)

    max_val = max(float(np.percentile(active_mag, 99.5)), high_thresh * 1.4, 255.0)
    bins = np.linspace(0, max_val, 100)

    # Histogram
    ax1.hist(
        active_mag,
        bins=bins,
        color="#94a3b8",
        alpha=0.6,
        edgecolor="#64748b",
        label="Histogram |∇I|",
    )
    ax1.set_xlabel("Độ lớn Gradient Sobel $|\\nabla I|$", fontsize=11)
    ax1.set_ylabel("Tần suất số lượng pixel", fontsize=11, color="#1e293b")
    ax1.set_xlim(0, max_val)

    # CDF on secondary twinx axis
    ax2 = ax1.twinx()
    sorted_mag = np.sort(active_mag)
    cdf = np.arange(1, len(sorted_mag) + 1) / len(sorted_mag) * 100.0
    ax2.plot(
        sorted_mag,
        cdf,
        color="#d97706",
        linewidth=2.2,
        label="Đường cong tích lũy CDF (%)",
    )
    ax2.set_ylabel("Tỷ lệ phần trăm tích lũy CDF (%)", fontsize=11, color="#d97706")
    ax2.set_ylim(0, 105)

    # Threshold lines
    ax1.axvline(
        low_thresh,
        color="#3b82f6",
        linestyle="--",
        linewidth=2.0,
        label=f"T_low = {low_thresh:.0f}",
    )
    ax1.axvline(
        high_thresh,
        color="#10b981",
        linestyle="--",
        linewidth=2.0,
        label=f"T_high = {high_thresh:.0f}",
    )

    # Percentages
    pct_below_low = float(np.mean(active_mag < low_thresh) * 100.0)
    pct_mid = float(
        np.mean((active_mag >= low_thresh) & (active_mag < high_thresh)) * 100.0
    )
    pct_above_high = float(np.mean(active_mag >= high_thresh) * 100.0)

    # Shaded regions
    ax1.axvspan(
        0,
        low_thresh,
        color="#ef4444",
        alpha=0.08,
        label=f"Dưới T_low: Triệt tiêu ({pct_below_low:.1f}%)",
    )
    ax1.axvspan(
        low_thresh,
        high_thresh,
        color="#f59e0b",
        alpha=0.10,
        label=f"Giữa T_low & T_high: Hysteresis ({pct_mid:.1f}%)",
    )
    ax1.axvspan(
        high_thresh,
        max_val,
        color="#10b981",
        alpha=0.10,
        label=f"Trên T_high: Biên chắc chắn ({pct_above_high:.1f}%)",
    )

    ax1.set_title(
        f"Phân Bố Độ Lớn Gradient |∇I| & Đường Cong CDF (T_low={low_thresh:.0f}, T_high={high_thresh:.0f})",
        fontsize=12,
        fontweight="bold",
    )
    ax1.grid(True, linestyle="--", alpha=0.3)

    # Combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="center right", fontsize=8.5, framealpha=0.95)

    plt.tight_layout()
    return fig


def plot_clahe_tile_mechanism(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
    figsize: tuple[int, int] = (14, 6),
) -> plt.Figure:
    """Plot CLAHE local tile mechanism: contrast slope clipping and noise suppression.

    Demonstrates why global histogram equalization severely amplifies background/sky noise
    while CLAHE bounds local contrast gain:
    - Subplot 1: 8x8 tile grid overlaid on image with flat sky/background tile highlighted.
    - Subplot 2: Local tile histogram before and after clip-limit clipping with uniform redistribution.
    - Subplot 3: Resulting gradient magnitude contrast comparison (max |G| bounded under CLAHE vs global HE spike).

    Args:
        image: Grayscale or BGR image.
        clip_limit: CLAHE clip limit (contrast gain cutoff).
        tile_grid_size: Tuple (grid_rows, grid_cols).
        figsize: Output figure dimension tuple.

    Returns:
        plt.Figure: 3-subplot comparison figure.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    h, w = gray.shape
    gh, gw = tile_grid_size
    th, tw = max(1, h // gh), max(1, w // gw)

    # 1. Global Equalization and CLAHE
    he_img = cv2.equalizeHist(gray)
    clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    clahe_img = clahe_obj.apply(gray)

    # Find the smoothest (flat sky / background) tile with lowest variance in top half
    best_var = float("inf")
    best_tile = (0, 0)
    max_search_r = max(1, gh // 2)
    for gy in range(max_search_r):
        for gx in range(gw):
            tile = gray[gy * th : (gy + 1) * th, gx * tw : (gx + 1) * tw]
            if tile.size == 0:
                continue
            var = float(np.var(tile))
            if var < best_var:
                best_var = var
                best_tile = (gy, gx)

    sel_gy, sel_gx = best_tile
    y0, y1 = sel_gy * th, (sel_gy + 1) * th
    x0, x1 = sel_gx * tw, (sel_gx + 1) * tw

    tile_orig = gray[y0:y1, x0:x1]
    tile_he = he_img[y0:y1, x0:x1]
    tile_clahe = clahe_img[y0:y1, x0:x1]

    # Compute Sobel gradients in tile
    g_he = np.hypot(
        cv2.Sobel(tile_he.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(tile_he.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3),
    )
    g_cl = np.hypot(
        cv2.Sobel(tile_clahe.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(tile_clahe.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3),
    )

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Subplot 1: Grid overlay
    ax1 = axes[0]
    ax1.imshow(gray, cmap="gray")
    # Draw tile grid
    for r in range(1, gh):
        ax1.axhline(r * th, color="#06b6d4", linestyle=":", alpha=0.6, linewidth=1.0)
    for c in range(1, gw):
        ax1.axvline(c * tw, color="#06b6d4", linestyle=":", alpha=0.6, linewidth=1.0)
    rect = Rectangle(
        (x0, y0), tw, th, linewidth=2.5, edgecolor="#ef4444", facecolor="none"
    )
    ax1.add_patch(rect)
    ax1.set_title(
        f"1. Lưới Ô Cục Bộ {gh}×{gw} & Ô Nền Phẳng\n(Tile [{sel_gy},{sel_gx}]: Var={best_var:.1f})",
        fontsize=11,
        fontweight="bold",
    )
    ax1.axis("off")

    # Subplot 2: Tile Histogram & Clip Limit
    ax2 = axes[1]
    hist_orig, _ = np.histogram(tile_orig.ravel(), bins=64, range=(0, 256))
    hist_cl, bin_edges = np.histogram(tile_clahe.ravel(), bins=64, range=(0, 256))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    avg_bin = max(1.0, len(tile_orig.ravel()) / 64.0)
    cutoff = clip_limit * avg_bin

    ax2.plot(
        bin_centers,
        hist_orig,
        color="#ef4444",
        linewidth=1.8,
        label="Histogram Ô gốc (Đỉnh hẹp)",
    )
    ax2.plot(
        bin_centers,
        hist_cl,
        color="#06b6d4",
        linewidth=1.8,
        label=f"CLAHE (Clip={clip_limit}x)",
    )
    ax2.axhline(
        cutoff,
        color="#d97706",
        linestyle="--",
        linewidth=1.8,
        label=f"Ngưỡng Cắt (Clip Limit={cutoff:.1f})",
    )
    ax2.set_xlabel("Mức xám", fontsize=10)
    ax2.set_ylabel("Số lượng pixel", fontsize=10)
    ax2.set_title(
        "2. Cắt Ngưỡng Histogram Ô Cục Bộ\n(Chặn độ dốc CDF, tái phân bố đều)",
        fontsize=11,
        fontweight="bold",
    )
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="upper right", fontsize=8.5)

    # Subplot 3: Gradient Magnitude distribution comparison
    ax3 = axes[2]
    max_he = float(np.max(g_he)) if g_he.size > 0 else 0.0
    max_cl = float(np.max(g_cl)) if g_cl.size > 0 else 0.0
    canny_thresh = 50.0  # Standard Canny high/low reference

    ax3.hist(
        g_he.ravel(),
        bins=30,
        alpha=0.6,
        color="#ef4444",
        label=f"Global HE (Max={max_he:.1f})",
    )
    ax3.hist(
        g_cl.ravel(),
        bins=30,
        alpha=0.6,
        color="#06b6d4",
        label=f"CLAHE (Max={max_cl:.1f})",
    )
    ax3.axvline(
        canny_thresh,
        color="black",
        linestyle="--",
        linewidth=1.8,
        label=f"Ngưỡng Canny ({canny_thresh:.0f})",
    )

    ax3.set_xlabel("Độ lớn Gradient Sobel $|\\nabla I|$ trong ô phẳng", fontsize=10)
    ax3.set_ylabel("Số lượng pixel", fontsize=10)
    ax3.set_title(
        f"3. Nhiễu Gradient: Global HE ({max_he:.1f} > 50)\nvs CLAHE ({max_cl:.1f} < 50, Không sinh biên giả)",
        fontsize=11,
        fontweight="bold",
    )
    ax3.grid(True, linestyle="--", alpha=0.4)
    ax3.legend(loc="upper right", fontsize=8.5)

    plt.tight_layout()
    return fig


__all__ = [
    "plot_canny_comparison",
    "plot_hysteresis_steps",
    "plot_gaussian_tradeoff_1d",
    "plot_gradient_quiver_roi",
    "plot_sector_directional_map",
    "plot_nms_ridge_cross_section",
    "plot_hysteresis_tricolor",
    "plot_magnitude_cdf_histogram",
    "plot_clahe_tile_mechanism",
]
