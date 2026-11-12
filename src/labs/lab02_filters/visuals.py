"""Visual presentation utilities for Lab 02 notebook."""

from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - enables 3D projection


def plot_point_ops_grid(
    original: np.ndarray,
    bright: np.ndarray,
    contrast: np.ndarray,
    negative: np.ndarray,
    threshold: np.ndarray | None = None,
    figsize: tuple[int, int] = (14, 10),
) -> plt.Figure:
    """Plot comparison grid for point operations."""

    def to_rgb(im: np.ndarray) -> np.ndarray:
        return (
            cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            if (im.ndim == 3 and im.shape[2] == 3)
            else im
        )

    n_plots = 5 if threshold is not None else 4
    cols = 2
    rows = (n_plots + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    axes_flat[0].imshow(to_rgb(original), cmap=None if original.ndim == 3 else "gray")
    axes_flat[0].set_title("Ảnh gốc", fontsize=11)
    axes_flat[0].axis("off")

    axes_flat[1].imshow(to_rgb(bright), cmap=None if bright.ndim == 3 else "gray")
    axes_flat[1].set_title("Tăng sáng (Beta = +50)", fontsize=11)
    axes_flat[1].axis("off")

    axes_flat[2].imshow(to_rgb(contrast), cmap=None if contrast.ndim == 3 else "gray")
    axes_flat[2].set_title("Tăng tương phản (Alpha = 1.5)", fontsize=11)
    axes_flat[2].axis("off")

    axes_flat[3].imshow(to_rgb(negative), cmap=None if negative.ndim == 3 else "gray")
    axes_flat[3].set_title("Ảnh âm bản (Negative)", fontsize=11)
    axes_flat[3].axis("off")

    if threshold is not None:
        axes_flat[4].imshow(threshold, cmap="gray")
        axes_flat[4].set_title("Ngưỡng hóa (Threshold = 127)", fontsize=11)
        axes_flat[4].axis("off")
        if len(axes_flat) > 5:
            axes_flat[5].axis("off")

    plt.tight_layout()
    return fig


def plot_filter_comparison(
    images: list[np.ndarray],
    titles: list[str],
    ncols: int = 3,
    figsize: tuple[int, int] | None = None,
) -> plt.Figure:
    """Plot an arbitrary list of filtered images in an organized grid."""
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


def plot_sobel_gradient_grid(
    original_gray: np.ndarray,
    grad_x: np.ndarray,
    grad_y: np.ndarray,
    magnitude: np.ndarray,
    figsize: tuple[int, int] = (12, 10),
) -> plt.Figure:
    """Plot Sobel gradient x, y, and magnitude in a 2x2 grid."""
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes[0, 0].imshow(original_gray, cmap="gray")
    axes[0, 0].set_title("Ảnh gốc mức xám", fontsize=11)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(grad_x, cmap="gray")
    axes[0, 1].set_title("Đạo hàm Sobel theo trục X (Biên đứng)", fontsize=11)
    axes[0, 1].axis("off")

    axes[1, 0].imshow(grad_y, cmap="gray")
    axes[1, 0].set_title("Đạo hàm Sobel theo trục Y (Biên ngang)", fontsize=11)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(magnitude, cmap="gray")
    axes[1, 1].set_title(
        "Độ lớn Gradient Magnitude (Căn bậc hai Gx^2 + Gy^2)", fontsize=11
    )
    axes[1, 1].axis("off")

    plt.tight_layout()
    return fig


# ============================================================================
# Enriched Visual Analytics & Transformation Curves (8 Charts)
# ============================================================================


def plot_point_op_transfer_curves(
    figsize: tuple[int, int] = (14, 10),
) -> plt.Figure:
    """Plot 4-subplot transfer curves for point operations.

    Subplots:
    1. Brightness transfer curves (f +/- 50) highlighting saturation shelves.
    2. Contrast transfer curves (0.5f, 1.8f) highlighting slope and saturation.
    3. Negative transformation curve (255 - f).
    4. Thresholding step transfer curves (T = 80, 127, 180).
    """
    x = np.arange(256, dtype=np.float32)

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # 1. Brightness Transfer
    ax = axes[0, 0]
    ax.plot(x, x, "k--", label="Gốc f(x, y) (Identity)", alpha=0.7)
    y_bright_plus = np.clip(x + 50, 0, 255)
    y_bright_minus = np.clip(x - 50, 0, 255)
    ax.plot(x, y_bright_plus, "r-", linewidth=2, label="Tăng sáng (Beta = +50)")
    ax.plot(x, y_bright_minus, "b-", linewidth=2, label="Giảm sáng (Beta = -50)")
    ax.axhline(255, color="gray", linestyle=":", alpha=0.5)
    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
    # Highlight saturation shelves
    ax.axvspan(205, 255, color="red", alpha=0.15, label="Vùng bão hòa (x >= 205)")
    ax.axvspan(0, 50, color="blue", alpha=0.15, label="Vùng cắt tối (x <= 50)")
    ax.set_title("1. Đường cong Độ sáng: g(x) = clip(x + Beta, 0, 255)", fontsize=11)
    ax.set_xlabel("Mức xám vào f(x, y)")
    ax.set_ylabel("Mức xám ra g(x, y)")
    ax.set_xlim(0, 255)
    ax.set_ylim(-5, 260)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9, loc="lower right")

    # 2. Contrast Transfer
    ax = axes[0, 1]
    ax.plot(x, x, "k--", label="Gốc (Alpha = 1.0)", alpha=0.7)
    y_contrast_low = np.clip(0.5 * x, 0, 255)
    y_contrast_high = np.clip(1.8 * x, 0, 255)
    ax.plot(x, y_contrast_low, "g-", linewidth=2, label="Giảm tương phản (Alpha = 0.5)")
    ax.plot(
        x, y_contrast_high, "m-", linewidth=2, label="Tăng tương phản (Alpha = 1.8)"
    )
    ax.axhline(255, color="gray", linestyle=":", alpha=0.5)
    sat_idx = int(np.ceil(255.0 / 1.8))
    ax.axvline(sat_idx, color="magenta", linestyle=":", alpha=0.7)
    ax.text(
        sat_idx + 3,
        220,
        f"Bão hòa tại x >= {sat_idx}",
        color="magenta",
        fontsize=9,
        weight="bold",
    )
    ax.set_title(
        "2. Đường cong Tương phản: g(x) = clip(Alpha * x, 0, 255)", fontsize=11
    )
    ax.set_xlabel("Mức xám vào f(x, y)")
    ax.set_ylabel("Mức xám ra g(x, y)")
    ax.set_xlim(0, 255)
    ax.set_ylim(-5, 260)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9, loc="lower right")

    # 3. Negative Transformation
    ax = axes[1, 0]
    ax.plot(x, x, "k--", label="Gốc f(x, y)", alpha=0.7)
    y_neg = 255 - x
    ax.plot(x, y_neg, "c-", linewidth=2.5, label="Âm bản: g(x) = 255 - x")
    ax.plot(127.5, 127.5, "ro", markersize=6, label="Điểm bất động (127.5, 127.5)")
    ax.set_title("3. Đường cong Âm bản: g(x) = 255 - x", fontsize=11)
    ax.set_xlabel("Mức xám vào f(x, y)")
    ax.set_ylabel("Mức xám ra g(x, y)")
    ax.set_xlim(0, 255)
    ax.set_ylim(-5, 260)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9, loc="center left")

    # 4. Thresholding Step Transfer
    ax = axes[1, 1]
    ax.plot(x, x, "k--", label="Gốc f(x, y)", alpha=0.5)
    colors = ["#2ca02c", "#d62728", "#9467bd"]
    for t_val, color in zip([80, 127, 180], colors, strict=False):
        y_thresh = np.where(x >= t_val, 255.0, 0.0)
        ax.step(
            x,
            y_thresh,
            where="post",
            color=color,
            linewidth=2,
            label=f"Cắt ngưỡng T = {t_val}",
        )
        ax.axvline(t_val, color=color, linestyle=":", alpha=0.5)
    ax.set_title(
        "4. Hàm bước Cắt ngưỡng: g(x) = 255 nếu x >= T, ngược lại 0", fontsize=11
    )
    ax.set_xlabel("Mức xám vào f(x, y)")
    ax.set_ylabel("Mức xám ra g(x, y)")
    ax.set_xlim(0, 255)
    ax.set_ylim(-5, 260)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9, loc="center right")

    plt.tight_layout()
    return fig


def plot_threshold_cdf_analysis(
    image_gray: np.ndarray,
    thresholds: list[int] | tuple[int, ...] = (80, 127, 180),
    figsize: tuple[int, int] = (10, 6),
) -> plt.Figure:
    """Plot dual-axis histogram PDF and CDF curve displaying exact % black and % white."""
    if image_gray.ndim == 3:
        img = cv2.cvtColor(image_gray, cv2.COLOR_BGR2GRAY)
    else:
        img = image_gray

    pixels = img.ravel()
    counts, _ = np.histogram(pixels, bins=256, range=(0, 256))
    pdf = counts / float(counts.sum())
    cdf = np.cumsum(pdf) * 100.0

    fig, ax1 = plt.subplots(figsize=figsize)

    # Left axis: Histogram count / PDF
    ax1.bar(
        np.arange(256),
        counts,
        width=1.0,
        color="#1f77b4",
        alpha=0.35,
        label="Tần suất mức xám (Histogram / PDF)",
    )
    ax1.set_xlabel("Mức xám pixel [0 .. 255]", fontsize=11)
    ax1.set_ylabel("Số lượng pixel (Count)", color="#1f77b4", fontsize=11)
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_xlim(0, 255)

    # Right axis: Cumulative distribution curve
    ax2 = ax1.twinx()
    ax2.plot(
        np.arange(256),
        cdf,
        color="#d62728",
        linewidth=2.5,
        label="Hàm tích lũy (CDF %)",
    )
    ax2.set_ylabel("Tỉ lệ tích lũy (%)", color="#d62728", fontsize=11)
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax2.set_ylim(0, 105)

    # Threshold markers
    line_colors = ["#2ca02c", "#ff7f0e", "#9467bd"]
    for i, t in enumerate(thresholds):
        pct_black = cdf[int(np.clip(t - 1, 0, 255))]
        pct_white = 100.0 - pct_black
        c = line_colors[i % len(line_colors)]
        ax2.axvline(t, color=c, linestyle="--", linewidth=1.8)
        ax2.plot(t, pct_black, "o", color=c, markersize=6)
        y_text = min(92, max(8, pct_black + (8 if i % 2 == 0 else -10)))
        ax2.annotate(
            f"T={t}\n{pct_black:.1f}% Đen / {pct_white:.1f}% Trắng",
            xy=(t, pct_black),
            xytext=(t + 5, y_text),
            color=c,
            fontsize=9,
            weight="bold",
            arrowprops={"arrowstyle": "->", "color": c, "lw": 1.2},
        )

    plt.title(
        "Phân tích Dual-Axis: Phân bố Mức xám (PDF) & Tích lũy (CDF) với Điểm cắt ngưỡng",
        fontsize=12,
        pad=12,
    )
    fig.tight_layout()
    return fig


def plot_edge_step_response_profile(
    img_orig: np.ndarray,
    img_mean: np.ndarray,
    img_gauss: np.ndarray,
    img_sharp5: np.ndarray,
    img_sharp21: np.ndarray,
    row: int | None = None,
    col_range: tuple[int, int] | None = None,
    figsize: tuple[int, int] = (12, 6),
) -> plt.Figure:
    """Plot 1D scanline cross-section showing blur slope flattening and sharpening overshoot/undershoot."""

    def to_gray(im: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if im.ndim == 3 else im

    orig = to_gray(img_orig)
    mean = to_gray(img_mean)
    gauss = to_gray(img_gauss)
    s5 = to_gray(img_sharp5)
    s21 = to_gray(img_sharp21)

    if row is None:
        row = orig.shape[0] // 2
    if col_range is None:
        c_start, c_end = 0, orig.shape[1]
    else:
        c_start, c_end = col_range

    cols = np.arange(c_start, c_end)
    prof_orig = orig[row, c_start:c_end]
    prof_mean = mean[row, c_start:c_end]
    prof_gauss = gauss[row, c_start:c_end]
    prof_s5 = s5[row, c_start:c_end]
    prof_s21 = s21[row, c_start:c_end]

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        cols,
        prof_orig,
        "k-",
        linewidth=2.2,
        label="Ảnh gốc (Step Edge)",
        alpha=0.9,
    )
    ax.plot(
        cols,
        prof_mean,
        "b--",
        linewidth=1.8,
        label="Lọc Trung bình Mean (Làm phẳng độ dốc)",
    )
    ax.plot(
        cols,
        prof_gauss,
        "g-.",
        linewidth=1.8,
        label="Lọc Gauss Gaussian (Độ dốc chuông mượt)",
    )
    ax.plot(
        cols,
        prof_s5,
        color="orange",
        linestyle="-",
        linewidth=2.0,
        label="Làm nét Sharpen k=5 (Overshoot nhẹ)",
    )
    ax.plot(
        cols,
        prof_s21,
        color="crimson",
        linestyle="-",
        linewidth=2.2,
        label="Làm nét Sharpen k=21 (Hiệu ứng dải Mach rõ nét)",
    )

    # Highlight Mach band overshoot & undershoot if present
    diff_s21 = prof_s21.astype(float) - prof_orig.astype(float)
    max_overshoot_idx = int(np.argmax(diff_s21))
    min_undershoot_idx = int(np.argmin(diff_s21))

    if diff_s21[max_overshoot_idx] > 5:
        c_peak = cols[max_overshoot_idx]
        v_peak = int(prof_s21[max_overshoot_idx])
        ax.plot(c_peak, v_peak, "ro", markersize=6)
        ax.annotate(
            f"Overshoot (+{diff_s21[max_overshoot_idx]:.0f})",
            xy=(c_peak, v_peak),
            xytext=(c_peak + 2, min(250, v_peak + 15)),
            arrowprops={"arrowstyle": "->", "color": "crimson"},
            color="crimson",
            weight="bold",
            fontsize=9,
        )

    if diff_s21[min_undershoot_idx] < -5:
        c_dip = cols[min_undershoot_idx]
        v_dip = int(prof_s21[min_undershoot_idx])
        ax.plot(c_dip, v_dip, "bo", markersize=6)
        ax.annotate(
            f"Undershoot ({diff_s21[min_undershoot_idx]:.0f})",
            xy=(c_dip, v_dip),
            xytext=(c_dip + 2, max(5, v_dip - 20)),
            arrowprops={"arrowstyle": "->", "color": "blue"},
            color="blue",
            weight="bold",
            fontsize=9,
        )

    ax.set_title(
        f"Mặt cắt Quét ngang 1D (Scanline row={row}): Đáp ứng Bước nhảy Biên & Hiện tượng Dải Mach",
        fontsize=12,
        pad=10,
    )
    ax.set_xlabel("Tọa độ Cột ảnh X (Column Index)", fontsize=11)
    ax.set_ylabel("Mức xám Điểm ảnh (Intensity [0..255])", fontsize=11)
    ax.set_ylim(-5, 265)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=9, loc="best")

    plt.tight_layout()
    return fig


def plot_fft_filter_spectra(
    images: list[np.ndarray],
    titles: list[str],
    include_radial_profile: bool = False,
    figsize: tuple[int, int] | None = None,
) -> plt.Figure:
    """Plot 2D FFT magnitude spectra for filter comparison (optionally with 1D radial power curve)."""

    def to_gray(im: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if im.ndim == 3 else im

    n_imgs = len(images)
    gray_imgs = [to_gray(im).astype(np.float32) for im in images]

    # Compute 2D FFT magnitude spectra
    mags_2d = []
    for g_img in gray_imgs:
        f_shift = np.fft.fftshift(np.fft.fft2(g_img))
        mag = np.log1p(np.abs(f_shift))
        mags_2d.append(mag)

    if not include_radial_profile:
        # Clean 1-row layout: only 2D FFT magnitude spectra
        fig_size = figsize or (4.2 * n_imgs, 3.8)
        fig, axes = plt.subplots(1, n_imgs, figsize=fig_size)
        axes = np.atleast_1d(axes).ravel()

        for idx, (ax, mag_2d, title) in enumerate(zip(axes, mags_2d, titles, strict=False)):
            im = ax.imshow(mag_2d, cmap="inferno")
            ax.set_title(f"Phổ 2D FFT: {title}", fontsize=11, fontweight="bold")
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        return fig

    # Full layout with 1D Radial Profile (if explicitly requested)
    fig_size = figsize or (15, 8)
    radial_profiles = []
    max_radius = min(min(g.shape) // 2 for g in gray_imgs)

    for g_img, mag_2d in zip(gray_imgs, mags_2d, strict=False):
        f_shift = np.fft.fftshift(np.fft.fft2(g_img))
        h, w = g_img.shape
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        r_grid = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)

        power_spectrum = np.abs(f_shift) ** 2
        r_flat = r_grid.ravel()
        p_flat = power_spectrum.ravel()
        valid = r_flat < max_radius
        r_valid = r_flat[valid]
        p_valid = p_flat[valid]

        counts = np.bincount(r_valid, minlength=max_radius)
        sums = np.bincount(r_valid, weights=p_valid, minlength=max_radius)
        r_profile = np.zeros(max_radius, dtype=np.float32)
        nonzero = counts > 0
        r_profile[nonzero] = sums[nonzero] / counts[nonzero]
        radial_profiles.append(r_profile)

    fig = plt.figure(figsize=fig_size)
    gs = fig.add_gridspec(2, n_imgs)

    for idx, (mag_2d, title) in enumerate(zip(mags_2d, titles, strict=False)):
        ax = fig.add_subplot(gs[0, idx])
        im = ax.imshow(mag_2d, cmap="inferno")
        ax.set_title(f"Phổ 2D FFT: {title}", fontsize=10)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax_rad = fig.add_subplot(gs[1, :])
    r_axis = np.arange(max_radius)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for idx, (r_prof, title) in enumerate(zip(radial_profiles, titles, strict=False)):
        log_prof = np.log1p(r_prof[:max_radius])
        ax_rad.plot(
            r_axis,
            log_prof,
            label=title,
            color=colors[idx % len(colors)],
            linewidth=2.0,
        )

    ax_rad.set_title(
        "Mặt cắt Năng lượng Hướng tâm 1D: Radial Average Power Spectrum P(r)",
        fontsize=11,
    )
    ax_rad.set_xlabel("Tần số không gian bán kính r (Spatial Frequency Radius)")
    ax_rad.set_ylabel("Mật độ công suất log(1 + P(r))")
    ax_rad.grid(True, linestyle="--", alpha=0.6)
    ax_rad.legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    return fig


def plot_gradient_vector_field(
    img_gray: np.ndarray,
    roi: tuple[int, int, int, int] | None = None,
    step: int = 2,
    figsize: tuple[int, int] = (14, 6),
) -> plt.Figure:
    """Plot gradient quiver vector field overlaid on edge pixels + color-coded direction map."""
    if img_gray.ndim == 3:
        gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_gray

    h, w = gray.shape
    if roi is None:
        cy, cx = h // 2, w // 2
        roi = (max(0, cy - 16), min(h, cy + 16), max(0, cx - 16), min(w, cx + 16))

    ymin, ymax, xmin, xmax = roi
    patch = gray[ymin:ymax, xmin:xmax].astype(np.float32)
    ph, pw = patch.shape

    # Compute Sobel derivatives
    gx = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    angle = np.arctan2(gy, gx)  # in [-pi, pi]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Subplot 1: Quiver arrows on zoomed intensity patch
    ax1 = axes[0]
    ax1.imshow(patch, cmap="gray", origin="upper", extent=[0, pw, ph, 0])
    y_coords, x_coords = np.mgrid[0:ph:step, 0:pw:step]
    u = gx[0:ph:step, 0:pw:step]
    v = gy[0:ph:step, 0:pw:step]
    m_sub = mag[0:ph:step, 0:pw:step]

    # Matplotlib quiver with angles="xy", scale_units="xy" evaluates vectors in data coordinates
    q = ax1.quiver(
        x_coords + 0.5,
        y_coords + 0.5,
        u,
        v,
        m_sub,
        cmap="autumn",
        scale=None,
        scale_units="xy",
        angles="xy",
        width=0.006,
    )
    fig.colorbar(q, ax=ax1, label="Độ lớn Gradient |∇I|", fraction=0.046, pad=0.04)
    ax1.set_title("1. Trường Vector Gradient ∇I = (Gx, Gy) trên ROI vi mô", fontsize=11)
    ax1.set_xlabel("Tọa độ Cột (x)")
    ax1.set_ylabel("Tọa độ Hàng (y)")

    # Subplot 2: Gradient direction map with twilight / hsv colormap
    ax2 = axes[1]
    # Convert angle to degrees [-180, 180]
    angle_deg = np.rad2deg(angle)
    im2 = ax2.imshow(angle_deg, cmap="twilight", origin="upper", vmin=-180, vmax=180)
    cb2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cb2.set_label("Góc hướng Gradient (Độ [-180°, +180°])")
    ax2.set_title("2. Bản đồ Góc hướng Gradient θ = arctan2(Gy, Gx)", fontsize=11)
    ax2.set_xlabel("Tọa độ Cột (x)")
    ax2.set_ylabel("Tọa độ Hàng (y)")

    plt.tight_layout()
    return fig


def plot_gradient_3d_surface(
    mag_patch: np.ndarray,
    figsize: tuple[int, int] = (10, 7),
) -> plt.Figure:
    """Plot 3D topographic surface mesh showing edge ridge lines and flat valleys."""
    if mag_patch.ndim == 3:
        patch = cv2.cvtColor(mag_patch, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        patch = mag_patch.astype(np.float32)

    h, w = patch.shape
    x = np.arange(w)
    y = np.arange(h)
    x_mesh, y_mesh = np.meshgrid(x, y)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        x_mesh,
        y_mesh,
        patch,
        cmap="viridis",
        rstride=1,
        cstride=1,
        linewidth=0.2,
        edgecolor="black",
        alpha=0.9,
        antialiased=True,
    )

    ax.set_title(
        "Mô hình Địa hình 3D (3D Topographic Surface) của Gradient Magnitude",
        fontsize=12,
        pad=15,
    )
    ax.set_xlabel("Tọa độ X", labelpad=8)
    ax.set_ylabel("Tọa độ Y", labelpad=8)
    ax.set_zlabel("Độ lớn Gradient |∇I|", labelpad=8)
    ax.view_init(elev=38, azim=-55)

    fig.colorbar(surf, ax=ax, shrink=0.6, aspect=12, pad=0.1, label="|∇I|")
    plt.tight_layout()
    return fig


def plot_kernel_sum_dc_characteristics(
    figsize: tuple[int, int] = (10, 6),
) -> plt.Figure:
    """Plot DC transmission response line g(c) = clip(c * sum(K), 0, 255) for sum(K) in {0.0, 0.5, 1.0, 2.0}."""
    c = np.linspace(0, 255, 256)

    fig, ax = plt.subplots(figsize=figsize)

    kernel_sums = [1.0, 0.0, 0.5, 2.0]
    styles = [
        ("k-", 2.5, "ΣK = 1.0 (Bảo toàn DC - Identity Line, năng lượng giữ nguyên)"),
        (
            "r-",
            2.2,
            "ΣK = 0.0 (Triệt tiêu DC - Bộ dò biên / Sobel / Laplacian, nền triệt về 0)",
        ),
        ("g--", 2.0, "ΣK = 0.5 (Suy hao DC - Độ dốc 0.5, ảnh tối đi 50%)"),
        (
            "m-.",
            2.2,
            "ΣK = 2.0 (Khuếch đại DC - Độ dốc 2.0, bão hòa 255 tại c >= 127.5)",
        ),
    ]

    for k_sum, (style, lw, label) in zip(kernel_sums, styles, strict=False):
        g = np.clip(c * k_sum, 0, 255)
        ax.plot(c, g, style, linewidth=lw, label=label)

    # Highlight saturation shelf for sum = 2.0
    ax.axhline(255, color="magenta", linestyle=":", alpha=0.5)
    ax.axvline(127.5, color="magenta", linestyle=":", alpha=0.5)
    ax.axvspan(
        127.5,
        255,
        color="magenta",
        alpha=0.15,
        label="Vùng bão hòa trắng bệt (ΣK = 2.0)",
    )

    ax.set_title(
        "Quy luật Tổng Hệ số Kernel: Đặc tính Đáp ứng Truyền dẫn DC g(c) = clip(c * ΣK, 0, 255)",
        fontsize=12,
        pad=10,
    )
    ax.set_xlabel("Độ sáng Đầu vào c (Constant Background Intensity)", fontsize=11)
    ax.set_ylabel("Độ sáng Đầu ra g(c)", fontsize=11)
    ax.set_xlim(0, 255)
    ax.set_ylim(-5, 265)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=9, loc="center left")

    plt.tight_layout()
    return fig


def plot_bilateral_decomposition_profile(
    figsize: tuple[int, int] = (14, 6),
) -> plt.Figure:
    """Plot bilateral weights decomposition across step edge + 1D salt-and-pepper impulse denoising."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Subplot 1: Bilateral Decomposition across Step Edge
    ax1 = axes[0]
    # Spatial distance x from center pixel (x=0)
    x = np.linspace(-10, 10, 201)
    sigma_s = 3.0
    sigma_r = 30.0

    # Ideal step edge at x = 1.5: intensity is 50 for x < 1.5, and 200 for x >= 1.5
    center_intensity = 50.0
    edge_intensity = np.where(x < 1.5, 50.0, 200.0)
    delta_i = np.abs(edge_intensity - center_intensity)

    # Weights
    w_s = np.exp(-(x**2) / (2.0 * sigma_s**2))
    w_r = np.exp(-(delta_i**2) / (2.0 * sigma_r**2))
    w_bilateral = w_s * w_r

    # Normalize for display comparison
    ax1.plot(
        x,
        w_s,
        "b--",
        linewidth=2.0,
        label=f"Trọng số Không gian Ws (Gauss σ_s={sigma_s})",
    )
    ax1.plot(
        x,
        w_r,
        "r-.",
        linewidth=2.0,
        label=f"Trọng số Miền giá trị Wr (Gauss σ_r={sigma_r})",
    )
    ax1.plot(
        x,
        w_bilateral,
        "g-",
        linewidth=2.5,
        label="Trọng số Song phương W = Ws * Wr (Cắt cụt tại biên)",
    )
    ax1.axvline(
        1.5, color="gray", linestyle=":", alpha=0.7, label="Vị trí Biên nhảy (Edge)"
    )

    ax1.set_title(
        "1. Phân rã Trọng số Bilateral qua Biên: W_bilateral = Ws * Wr",
        fontsize=11,
    )
    ax1.set_xlabel("Khoảng cách Điểm ảnh đến Tâm (Pixel Offset)")
    ax1.set_ylabel("Độ lớn Trọng số Chuẩn hóa")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=9, loc="upper left")

    # Subplot 2: 1D Salt-and-Pepper Impulse Denoising (Median vs Mean)
    ax2 = axes[1]
    n_pts = 60
    t = np.arange(n_pts)
    # Step edge signal
    clean_signal = np.where(t < 30, 80.0, 180.0)
    noisy_signal = clean_signal.copy()

    # Add sparse impulse spikes (salt and pepper)
    spikes_salt = [10, 22, 45]
    spikes_pepper = [18, 38, 52]
    noisy_signal[spikes_salt] = 255.0
    noisy_signal[spikes_pepper] = 0.0

    # Apply 1D Mean filter and 1D Median filter (window=5)
    mean_filtered = scipy.ndimage.uniform_filter1d(noisy_signal, size=5)
    median_filtered = scipy.ndimage.median_filter(noisy_signal, size=5)

    ax2.plot(
        t,
        noisy_signal,
        "k.",
        markersize=7,
        alpha=0.6,
        label="Tín hiệu Nhiễu Muối tiêu (Salt & Pepper)",
    )
    ax2.step(
        t,
        clean_signal,
        where="mid",
        color="gray",
        linestyle="--",
        linewidth=1.8,
        label="Tín hiệu Biên lý tưởng (Ground Truth)",
    )
    ax2.plot(
        t,
        mean_filtered,
        "b-",
        linewidth=1.8,
        label="Lọc Trung bình Mean k=5 (Nhiễu bẹp thành vệt, mờ biên)",
    )
    ax2.plot(
        t,
        median_filtered,
        "crimson",
        linewidth=2.2,
        label="Lọc Trung vị Median k=5 (Triệt tiêu hoàn toàn nhiễu, giữ nét biên)",
    )

    ax2.set_title(
        "2. Khử Nhiễu Muối tiêu 1D: Median (Khử triệt để) vs Mean (Nhoè thành vệt)",
        fontsize=11,
    )
    ax2.set_xlabel("Tọa độ Mẫu 1D (Sample Index)")
    ax2.set_ylabel("Biên độ Tín hiệu [0 .. 255]")
    ax2.set_ylim(-10, 270)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(fontsize=9, loc="upper left")

    plt.tight_layout()
    return fig
