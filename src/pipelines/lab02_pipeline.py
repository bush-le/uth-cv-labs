"""Pipeline execution script for Computer Vision Lab 2.

Demonstrates point operators and linear filtering for image processing:
1. Point operations: brightness, contrast, negative, thresholding.
2. Linear filters: mean, Gaussian, sharpening.
3. Edge detection: Sobel, Prewitt, gradient magnitude.
4. Non-linear filters: median, bilateral.
5. Comprehensive comparison grid across all filters.
"""

from pathlib import Path

import cv2
import numpy as np

from src.common.logger import get_logger
from src.common.visualizer import plot_filter_grid, plot_histogram, plot_side_by_side
from src.data.filters import (
    add_salt_pepper_noise,
    adjust_brightness,
    adjust_contrast,
    apply_bilateral_filter,
    apply_gaussian_filter,
    apply_mean_filter,
    apply_median_filter,
    apply_prewitt,
    apply_sharpen_filter,
    apply_sobel,
    apply_threshold,
    find_project_root,
    get_gaussian_kernel,
    get_laplacian_kernel,
    get_mean_kernel,
    get_prewitt_kernels,
    get_sharpen_kernel,
    get_sobel_kernels,
    gradient_magnitude,
    kernel_sum,
    negative_transform,
)
from src.data.transforms import bgr_to_grayscale, load_image_cv, save_image_cv

logger = get_logger("lab02_pipeline")

# Reproducibility: fixed random seed for the noise demo
DEFAULT_SEED = 42
# Fixed threshold used in Yêu cầu 4
DEFAULT_THRESHOLD = 127


def _log_kernel(name: str, kernel: np.ndarray) -> None:
    """Log a kernel matrix, its coefficient sum, and its behavior class.

    Args:
        name: Human-readable kernel name.
        kernel: 2D convolution kernel.
    """
    total = kernel_sum(kernel)
    if abs(total) < 1e-9:
        behavior = "EDGE (sum = 0 -> removes flat regions)"
    elif abs(total - 1.0) < 1e-9:
        behavior = "BRIGHTNESS-PRESERVING (sum = 1)"
    else:
        behavior = f"GAIN {total:.3f} (not 0 or 1)"
    logger.info("  - %s: sum = %.3f -> %s", name, total, behavior)
    for row in kernel:
        logger.info("      [%s]", "  ".join(f"{v:+.2f}" for v in row))


def run_lab02(
    input_image_path: str | Path = "data/samples/image-2.jpg",
    output_dir: str | Path = "experiments/predictions/lab02",
) -> dict[str, Path]:
    """Execute the complete end-to-end pipeline for Lab 2.

    Args:
        input_image_path: Path to the input test image.
        output_dir: Directory where resulting images and figures are saved.

    Returns:
        dict[str, Path]: Mapping of artifact names to their saved paths.

    Raises:
        FileNotFoundError: If the input image cannot be found.
    """
    project_root = find_project_root()
    input_path = Path(input_image_path)
    if not input_path.is_absolute():
        input_path = project_root / input_path
    base_out = Path(output_dir)
    if not base_out.is_absolute():
        base_out = project_root / base_out

    artifacts: dict[str, Path] = {}

    logger.info("==================================================")
    logger.info("Starting Execution: UTH Computer Vision - Lab 2")
    logger.info("Input Image: %s", input_path.resolve())
    logger.info("Output Directory: %s", base_out.resolve())
    logger.info("==================================================")

    # --------------------------------------------------------------------------
    # Setup: load image and derive a grayscale version for filter demos
    # --------------------------------------------------------------------------
    logger.info("[Setup] Môi trường và Thư viện:")
    logger.info("  - OpenCV Version: %s", cv2.__version__)
    logger.info("  - NumPy Version: %s", np.__version__)

    img_bgr = load_image_cv(input_path)
    img_gray = bgr_to_grayscale(img_bgr)
    logger.info("  - Loaded image shape: %s (BGR)", img_bgr.shape)
    logger.info("  - Grayscale shape: %s", img_gray.shape)

    # --------------------------------------------------------------------------
    # Yêu cầu 1: Thay đổi độ sáng (Brightness)
    # --------------------------------------------------------------------------
    logger.info("[Yêu cầu 1] Thay đổi độ sáng (brightness += beta)...")
    point_dir = base_out / "point_ops"
    img_bright = adjust_brightness(img_bgr, beta=50)
    img_dark = adjust_brightness(img_bgr, beta=-50)
    artifacts["bright"] = save_image_cv(img_bright, point_dir / "brightness_plus50.jpg")
    artifacts["dark"] = save_image_cv(img_dark, point_dir / "brightness_minus50.jpg")
    plot_side_by_side(
        images=[img_bgr, img_dark, img_bright],
        titles=["Original", "Dark (beta=-50)", "Bright (beta=+50)"],
        save_path=point_dir / "brightness_comparison.png",
    )

    # --------------------------------------------------------------------------
    # Yêu cầu 2: Thay đổi độ tương phản (Contrast)
    # --------------------------------------------------------------------------
    logger.info("[Yêu cầu 2] Thay đổi độ tương phản (pixel *= alpha)...")
    img_contrast_low = adjust_contrast(img_bgr, alpha=0.5)
    img_contrast_high = adjust_contrast(img_bgr, alpha=1.8)
    artifacts["contrast_low"] = save_image_cv(
        img_contrast_low, point_dir / "contrast_0.5x.jpg"
    )
    artifacts["contrast_high"] = save_image_cv(
        img_contrast_high, point_dir / "contrast_1.8x.jpg"
    )
    plot_side_by_side(
        images=[img_bgr, img_contrast_low, img_contrast_high],
        titles=["Original", "Contrast x0.5", "Contrast x1.8"],
        save_path=point_dir / "contrast_comparison.png",
    )

    # --------------------------------------------------------------------------
    # Yêu cầu 3: Biến đổi âm bản (Negative)
    # --------------------------------------------------------------------------
    logger.info("[Yêu cầu 3] Biến đổi âm bản (g = 255 - f)...")
    img_negative = negative_transform(img_bgr)
    artifacts["negative"] = save_image_cv(img_negative, point_dir / "negative.jpg")
    plot_side_by_side(
        images=[img_bgr, img_negative],
        titles=["Original", "Negative (255 - f)"],
        save_path=point_dir / "negative_comparison.png",
    )

    # --------------------------------------------------------------------------
    # Yêu cầu 4: Cắt ngưỡng (Thresholding)
    # --------------------------------------------------------------------------
    logger.info("[Yêu cầu 4] Cắt ngưỡng (threshold = %d)...", DEFAULT_THRESHOLD)
    img_binary = apply_threshold(img_gray, threshold=DEFAULT_THRESHOLD)
    artifacts["binary"] = save_image_cv(img_binary, point_dir / "threshold_binary.png")
    plot_side_by_side(
        images=[img_gray, img_binary],
        titles=["Grayscale", f"Binary (T={DEFAULT_THRESHOLD})"],
        save_path=point_dir / "threshold_comparison.png",
    )
    logger.info("  - Point operation artifacts saved to: %s", point_dir)

    # --------------------------------------------------------------------------
    # Yêu cầu 5 & 6: Lọc trung bình & Lọc Gaussian
    # --------------------------------------------------------------------------
    logger.info("[Yêu cầu 5] Lọc trung bình (mean/box filter)...")
    linear_dir = base_out / "linear_filters"
    mean_k7 = get_mean_kernel(7)
    mean_k21 = get_mean_kernel(21)
    _log_kernel("Mean 7x7", mean_k7)
    _log_kernel("Mean 21x21", mean_k21)

    img_mean7 = apply_mean_filter(img_bgr, ksize=7)
    img_mean21 = apply_mean_filter(img_bgr, ksize=21)
    artifacts["mean7"] = save_image_cv(img_mean7, linear_dir / "mean_7x7.jpg")
    artifacts["mean21"] = save_image_cv(img_mean21, linear_dir / "mean_21x21.jpg")

    logger.info("[Yêu cầu 6] Lọc Gaussian (weighted blur)...")
    # ksize ~= 6*sigma + 1 keeps the Gaussian from being truncated.
    gauss_k19 = get_gaussian_kernel(19, sigma=3.0)
    _log_kernel("Gaussian 19x19 (sigma=3.0)", gauss_k19)
    img_gauss_s30 = apply_gaussian_filter(img_bgr, ksize=19, sigma=3.0)
    img_gauss_s70 = apply_gaussian_filter(img_bgr, ksize=43, sigma=7.0)
    artifacts["gauss_s30"] = save_image_cv(
        img_gauss_s30, linear_dir / "gaussian_19x19_s3.jpg"
    )
    artifacts["gauss_s70"] = save_image_cv(
        img_gauss_s70, linear_dir / "gaussian_43x43_s7.jpg"
    )

    plot_filter_grid(
        images=[img_bgr, img_mean7, img_mean21, img_gauss_s30, img_gauss_s70],
        titles=[
            "Original",
            "Mean 7x7",
            "Mean 21x21",
            "Gaussian s=3.0 (k19)",
            "Gaussian s=7.0 (k43)",
        ],
        ncols=5,
        save_path=linear_dir / "blur_comparison_grid.png",
    )

    # --------------------------------------------------------------------------
    # Yêu cầu 7: Làm sắc nét (Sharpening = Identity - Laplacian)
    # --------------------------------------------------------------------------
    logger.info("[Yêu cầu 7] Làm sắc nét (sharpen kernel = Identity - Laplacian)...")
    sharpen_k = get_sharpen_kernel()
    _log_kernel("Sharpen (Identity - Laplacian)", sharpen_k)

    img_sharp = apply_sharpen_filter(img_bgr)
    artifacts["sharpen"] = save_image_cv(img_sharp, linear_dir / "sharpen.jpg")
    plot_side_by_side(
        images=[img_bgr, img_sharp],
        titles=["Original", "Sharpened (Identity - Laplacian)"],
        save_path=linear_dir / "sharpen_comparison.png",
    )
    logger.info("  - Linear filter artifacts saved to: %s", linear_dir)

    # --------------------------------------------------------------------------
    # Yêu cầu 8: Phát hiện cạnh bằng Sobel & Prewitt
    # --------------------------------------------------------------------------
    logger.info("[Yêu cầu 8] Phát hiện cạnh (Sobel & Prewitt)...")
    edge_dir = base_out / "edge_detection"
    sobel_x_k, sobel_y_k = get_sobel_kernels()
    prewitt_x_k, prewitt_y_k = get_prewitt_kernels()
    _log_kernel("Sobel X", sobel_x_k)
    _log_kernel("Sobel Y", sobel_y_k)
    _log_kernel("Prewitt X", prewitt_x_k)
    _log_kernel("Prewitt Y", prewitt_y_k)

    sobel_x = apply_sobel(img_gray, direction="x")
    sobel_y = apply_sobel(img_gray, direction="y")
    prewitt_x = apply_prewitt(img_gray, direction="x")
    prewitt_y = apply_prewitt(img_gray, direction="y")
    sobel_mag = gradient_magnitude(sobel_x, sobel_y)

    artifacts["sobel_x"] = save_image_cv(sobel_x, edge_dir / "sobel_x.png")
    artifacts["sobel_y"] = save_image_cv(sobel_y, edge_dir / "sobel_y.png")
    artifacts["prewitt_x"] = save_image_cv(prewitt_x, edge_dir / "prewitt_x.png")
    artifacts["prewitt_y"] = save_image_cv(prewitt_y, edge_dir / "prewitt_y.png")
    artifacts["sobel_mag"] = save_image_cv(
        sobel_mag, edge_dir / "sobel_gradient_magnitude.png"
    )

    plot_filter_grid(
        images=[sobel_x, sobel_y, sobel_mag, prewitt_x, prewitt_y],
        titles=[
            "Sobel X",
            "Sobel Y",
            "Sobel |G| = sqrt(Gx^2+Gy^2)",
            "Prewitt X",
            "Prewitt Y",
        ],
        ncols=3,
        save_path=edge_dir / "edge_detection_grid.png",
    )

    logger.info("  - Edge detection artifacts saved to: %s", edge_dir)

    # --------------------------------------------------------------------------
    # Yêu cầu 9: Tự thiết kế kernel & kiểm chứng quy luật tổng hệ số
    # --------------------------------------------------------------------------
    logger.info("[Yêu cầu 9] Kiểm chứng quy luật tổng hệ số kernel...")
    kernel_dir = base_out / "kernel_rules"

    # Flat patch to demonstrate the "flat region -> output = c * sum(kernel)" rule
    flat_patch = np.full((64, 64), 100, dtype=np.uint8)
    kernel_cases = {
        "sum=0.5": np.full((3, 3), 0.5 / 9, dtype=np.float32),
        "sum=1.0": get_mean_kernel(3),
        "sum=2.0": np.full((3, 3), 2.0 / 9, dtype=np.float32),
        "sum=0.0": get_laplacian_kernel(),
    }
    rule_figs: list[np.ndarray] = [flat_patch]
    rule_titles: list[str] = ["Flat patch (c=100)"]
    for label, kern in kernel_cases.items():
        filtered = cv2.filter2D(flat_patch, -1, kern)
        unique_val = int(np.unique(filtered)[0])
        logger.info(
            "  - %s on flat patch -> output = %d (expected c * sum = %.1f)",
            label,
            unique_val,
            100 * kernel_sum(kern),
        )
        rule_figs.append(filtered)
        rule_titles.append(f"{label} -> {unique_val}")
    plot_filter_grid(
        images=rule_figs,
        titles=rule_titles,
        ncols=5,
        is_bgr=False,
        save_path=kernel_dir / "kernel_sum_rule.png",
    )

    # Apply the same self-designed kernels to the real image so their visual
    # effect is visible (darken / preserve / brighten / keep only edges).
    effect_figs: list[np.ndarray] = [img_bgr] + [
        cv2.filter2D(img_bgr, -1, kern) for kern in kernel_cases.values()
    ]
    effect_titles = [
        "Original (color)",
        "sum=0.5 (darker)",
        "sum=1.0 (unchanged)",
        "sum=2.0 (brighter)",
        "sum=0.0 (edges only)",
    ]
    plot_filter_grid(
        images=effect_figs,
        titles=effect_titles,
        ncols=5,
        is_bgr=True,
        save_path=kernel_dir / "kernel_effects_on_image.png",
    )
    logger.info("  - Kernel rule artifacts saved to: %s", kernel_dir)

    # --------------------------------------------------------------------------
    # Yêu cầu 10: Lọc phi tuyến tính (Median & Bilateral)
    # --------------------------------------------------------------------------
    logger.info("[Yêu cầu 10] Lọc phi tuyến tính (Median & Bilateral)...")
    nonlinear_dir = base_out / "nonlinear_filters"

    # Add salt-and-pepper noise (reproducible) to showcase median strength
    img_noisy = add_salt_pepper_noise(img_bgr, amount=0.03, seed=DEFAULT_SEED)
    img_median = apply_median_filter(img_noisy, ksize=3)
    img_bilateral = apply_bilateral_filter(img_noisy, d=9, sigma_color=75, sigma_space=75)

    artifacts["noisy"] = save_image_cv(img_noisy, nonlinear_dir / "salt_pepper_noisy.jpg")
    artifacts["median"] = save_image_cv(img_median, nonlinear_dir / "median_denoised.jpg")
    artifacts["bilateral"] = save_image_cv(
        img_bilateral, nonlinear_dir / "bilateral_filtered.jpg"
    )

    plot_filter_grid(
        images=[img_bgr, img_noisy, img_median, img_bilateral],
        titles=[
            "Original",
            "Salt & Pepper Noise (3%)",
            "Median 3x3 (denoise)",
            "Bilateral (edge-preserving)",
        ],
        ncols=4,
        save_path=nonlinear_dir / "nonlinear_comparison_grid.png",
    )
    logger.info("  - Non-linear filter artifacts saved to: %s", nonlinear_dir)

    # --------------------------------------------------------------------------
    # Yêu cầu 11: So sánh tổng hợp tất cả các bộ lọc
    # --------------------------------------------------------------------------
    logger.info("[Yêu cầu 11] So sánh tổng hợp tất cả các bộ lọc...")
    compare_dir = base_out / "comparison"

    all_filters = [
        img_bgr,
        img_mean7,
        img_mean21,
        img_gauss_s30,
        img_gauss_s70,
        img_sharp,
        sobel_mag,
        img_median,
        img_bilateral,
        img_negative,
    ]
    all_titles = [
        "Original",
        "Mean 7x7 (blur)",
        "Mean 21x21 (blur)",
        "Gaussian s=3.0 (blur)",
        "Gaussian s=7.0 (blur)",
        "Sharpen",
        "Sobel |G| (edge)",
        "Median (denoise)",
        "Bilateral (denoise)",
        "Negative",
    ]
    plot_filter_grid(
        images=all_filters,
        titles=all_titles,
        ncols=3,
        save_path=compare_dir / "all_filters_grid.png",
    )

    # Histograms: original vs point operations
    plot_histogram(
        img_gray,
        title="Original Grayscale Histogram",
        save_path=compare_dir / "histogram_original.png",
    )
    plot_histogram(
        img_binary,
        title=f"Histogram after Threshold (T={DEFAULT_THRESHOLD})",
        save_path=compare_dir / "histogram_binary.png",
    )
    logger.info("  - Comparison artifacts saved to: %s", compare_dir)

    logger.info("==================================================")
    logger.info("Lab 2 Execution Pipeline Completed Successfully!")
    logger.info("Total generated artifacts: %d", len(artifacts))
    logger.info("==================================================")

    return artifacts


if __name__ == "__main__":
    run_lab02()
