"""End-to-end execution pipeline for Lab 03: Canny Edge Detector.

Executes all 4 core lab exercises and 4 extended research topics, producing
visual artifacts in experiments/predictions/lab03/ and metrics in experiments/logs/lab03/.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage.data as sd

from src.common.logger import get_logger
from src.common.visualizer import plot_grid
from src.models.classical.canny import (
    TemporalEdgeFilter,
    calculate_edge_f1_score,
    calculate_hausdorff_distance,
    calculate_pratt_fom,
    classify_polygon_contours,
    custom_canny_pipeline,
    detect_hough_circles,
    detect_hough_lines,
    run_auto_canny,
    run_di_zenzo_gradient,
    run_laplacian,
    run_multichannel_canny,
    run_opencv_canny,
    run_otsu_canny,
    run_skimage_canny,
    run_sobel,
    segment_via_morphology_and_holes,
)

logger = get_logger(__name__)


class Lab03Pipeline:
    """Coordinator pipeline for all Lab 03 experimental workflows."""

    def __init__(
        self,
        samples_dir: str | Path = "data/samples/lab03",
        predictions_dir: str | Path = "experiments/predictions/lab03",
        logs_dir: str | Path = "experiments/logs/lab03",
    ) -> None:
        """Initialize pipeline paths and directories."""
        self.samples_dir = Path(samples_dir)
        self.predictions_dir = Path(predictions_dir)
        self.logs_dir = Path(logs_dir)

        self.samples_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_report: dict[str, Any] = {}

    # ==========================================================================
    # 0. Sample Data Synthesis & Curation
    # ==========================================================================

    def prepare_dataset_samples(self) -> dict[str, np.ndarray]:
        """Load and synthesize representative benchmark images."""
        logger.info("Preparing standardized benchmark datasets for Lab 03...")
        np.random.seed(42)

        # 1. Classic standard images
        coins = sd.coins()  # uint8 (303, 384)
        camera = sd.camera()  # uint8 (512, 512)

        # 2. Noisy images
        # Gaussian noise
        noise_gauss = np.random.normal(0, 25, camera.shape)
        noisy_gaussian = np.clip(
            camera.astype(np.float32) + noise_gauss, 0, 255
        ).astype(np.uint8)

        # Salt and pepper noise
        noisy_sp = camera.copy()
        sp_mask = np.random.rand(*camera.shape)
        noisy_sp[sp_mask < 0.03] = 0
        noisy_sp[sp_mask > 0.97] = 255

        # 3. Low contrast image (narrow histogram range [90, 140])
        low_contrast = np.clip(
            90 + (camera.astype(np.float32) / 255.0) * 50.0, 0, 255
        ).astype(np.uint8)

        # 4. Dense texture image (brick wall)
        brick = sd.brick()

        # 5. Clean geometric shapes and Ground Truth edges
        canvas_h, canvas_w = 400, 400
        shapes_canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        gt_canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

        # Circle
        cv2.circle(shapes_canvas, (120, 120), 50, 255, -1)
        cv2.circle(gt_canvas, (120, 120), 50, 255, 1)

        # Rectangle
        cv2.rectangle(shapes_canvas, (220, 70), (340, 170), 255, -1)
        cv2.rectangle(gt_canvas, (220, 70), (340, 170), 255, 1)

        # Triangle
        tri_pts = np.array([[120, 240], [60, 350], [180, 350]], np.int32)
        cv2.fillPoly(shapes_canvas, [tri_pts], 255)
        cv2.polylines(gt_canvas, [tri_pts], True, 255, 1)

        # Pentagon
        pent_pts = np.array(
            [[280, 230], [350, 270], [320, 350], [240, 350], [210, 270]],
            np.int32,
        )
        cv2.fillPoly(shapes_canvas, [pent_pts], 255)
        cv2.polylines(gt_canvas, [pent_pts], True, 255, 1)

        # 6. Color Chromatic Test Image (Pure color boundary with near-identical luminance)
        # R = [200, 50, 50], G = [50, 200, 50]
        color_chroma = np.zeros((300, 300, 3), dtype=np.uint8)
        color_chroma[:, :150] = [204, 51, 51]  # RGB Red-ish
        color_chroma[:, 150:] = [51, 153, 51]  # RGB Green-ish
        # Draw a bright blue circle spanning both halves
        cv2.circle(color_chroma, (150, 150), 60, (51, 102, 255), -1)

        # Save sample files
        samples = {
            "coins": coins,
            "camera": camera,
            "noisy_gaussian": noisy_gaussian,
            "noisy_sp": noisy_sp,
            "low_contrast": low_contrast,
            "brick": brick,
            "shapes": shapes_canvas,
            "shapes_gt": gt_canvas,
            "color_chroma": color_chroma,
        }

        for name, img in samples.items():
            out_file = self.samples_dir / f"{name}.png"
            if img.ndim == 3:
                cv2.imwrite(str(out_file), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            else:
                cv2.imwrite(str(out_file), img)

        logger.info(
            f"Generated and saved {len(samples)} benchmark samples to {self.samples_dir}"
        )
        return samples

    # ==========================================================================
    # 1. Bai 1: Library Implementation & Mathematical Pipeline
    # ==========================================================================

    def run_exercise_1(self, samples: dict[str, np.ndarray]) -> None:
        """Execute Exercise 1: 5-step custom Canny vs cv2.Canny vs skimage.feature.canny."""
        logger.info(
            "Executing Exercise 1: Pipeline Breakdown and Library Comparisons..."
        )
        img = samples["coins"]

        # 1.1 Custom 5-Step Mathematical Pipeline Visuals
        pipeline_res = custom_canny_pipeline(
            img, sigma=1.2, low_threshold=50, high_threshold=150
        )

        # Plot 5 stages
        stages_images = [
            img,
            pipeline_res["smoothed"],
            pipeline_res["magnitude"],
            pipeline_res["nms"],
            (pipeline_res["strong_edges"].astype(np.uint8) * 255),
            pipeline_res["edges"],
        ]
        stages_titles = [
            "(a) Goc (Raw Grayscale)",
            "(b) Buoc 1: Gaussian Smoothing",
            "(c) Buoc 2: Sobel Magnitude",
            "(d) Buoc 3: Non-Max Suppression",
            "(e) Buoc 4: Double Threshold",
            "(f) Buoc 5: Hysteresis Tracking",
        ]
        plot_grid(
            stages_images,
            stages_titles,
            rows=2,
            cols=3,
            save_path=self.predictions_dir / "ex1_canny_5_stages.png",
            main_title="Quy trinh 5 buoc Toan hoc Canny Edge Detector (Custom Vectorized Pipeline)",
        )

        # 1.2 Comparison: OpenCV vs Scikit-image vs Custom vs Sobel vs Laplacian
        t0 = time.perf_counter()
        cv_edges = run_opencv_canny(img, 50, 150)
        t_cv = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        sk_edges = run_skimage_canny(
            img, sigma=1.2, low_threshold=50, high_threshold=150
        )
        t_sk = (time.perf_counter() - t0) * 1000.0

        sob_edges = run_sobel(img)
        lap_edges = run_laplacian(img)

        comp_images = [
            img,
            pipeline_res["edges"],
            cv_edges,
            sk_edges,
            sob_edges,
            lap_edges,
        ]
        comp_titles = [
            "Anh goc (Coins)",
            "Custom Canny (5-stage)",
            f"OpenCV cv2.Canny ({t_cv:.2f} ms)",
            f"Scikit-Image canny ({t_sk:.2f} ms)",
            "Sobel Magnitude (Thick edges)",
            "Laplacian of Gaussian (LoG)",
        ]
        plot_grid(
            comp_images,
            comp_titles,
            rows=2,
            cols=3,
            save_path=self.predictions_dir / "ex1_library_comparison.png",
            main_title="So sanh Doi dau: Custom vs OpenCV vs Scikit-Image vs Sobel vs Laplacian",
        )

        self.metrics_report["ex1"] = {
            "opencv_latency_ms": t_cv,
            "skimage_latency_ms": t_sk,
            "edge_pixel_count_cv": int(np.sum(cv_edges > 0)),
            "edge_pixel_count_sk": int(np.sum(sk_edges > 0)),
            "edge_pixel_count_custom": int(np.sum(pipeline_res["edges"] > 0)),
        }
        logger.info(
            f"Exercise 1 completed. Latency: OpenCV={t_cv:.2f}ms, Scikit-Image={t_sk:.2f}ms"
        )

    # ==========================================================================
    # 2. Bai 2: Parameter Grid Search (Sigma, Low/High Thresholds)
    # ==========================================================================

    def run_exercise_2(self, samples: dict[str, np.ndarray]) -> None:
        """Execute Exercise 2: Systematic parameter variations and impact analysis."""
        logger.info(
            "Executing Exercise 2: Hyperparameter Sweeps (Sigma & Thresholds)..."
        )
        img = samples["camera"]

        # 2.1 Sigma sweep with fixed thresholds (50, 150)
        sigmas = [0.5, 1.0, 2.0, 3.5]
        sigma_imgs = []
        sigma_titles = []
        for s in sigmas:
            # OpenCV doesn't have direct sigma argument in cv2.Canny (it applies 3x3 Sobel)
            # So standard practice is pre-blurring with Gaussian sigma s
            k = int(np.ceil(3.0 * s)) * 2 + 1
            blurred = cv2.GaussianBlur(img, (k, k), sigmaX=s, sigmaY=s)
            edges = cv2.Canny(blurred, 50, 150)
            edge_count = int(np.sum(edges > 0))
            sigma_imgs.append(edges)
            sigma_titles.append(f"sigma = {s} (Edges: {edge_count} px)")

        plot_grid(
            sigma_imgs,
            sigma_titles,
            rows=1,
            cols=4,
            save_path=self.predictions_dir / "ex2_sigma_sweep.png",
            main_title="Khao sat Bien thien Tham so Gaussian sigma tren anh Camera (50, 150)",
            figsize=(16, 4.5),
        )

        # 2.2 Threshold Pairs Sweep with fixed sigma=1.0
        blurred_1 = cv2.GaussianBlur(img, (5, 5), 1.0)
        thresh_pairs = [
            ("Sensitive (30, 90) [1:3]", 30, 90),
            ("Standard (50, 150) [1:3]", 50, 150),
            ("Strict (100, 200) [1:2]", 100, 200),
            ("Wide Ratio (20, 180) [1:9]", 20, 180),
        ]
        thresh_imgs = []
        thresh_titles = []
        for label, low, high in thresh_pairs:
            edges = cv2.Canny(blurred_1, low, high)
            edge_count = int(np.sum(edges > 0))
            thresh_imgs.append(edges)
            thresh_titles.append(f"{label}\nEdges: {edge_count} px")

        plot_grid(
            thresh_imgs,
            thresh_titles,
            rows=1,
            cols=4,
            save_path=self.predictions_dir / "ex2_threshold_sweep.png",
            main_title="Khao sat Cap Nguong (T_low, T_high) voi Gaussian sigma=1.0",
            figsize=(16, 4.5),
        )

        # 2.3 Comprehensive 4x4 Grid Matrix (Sigma x Thresholds)
        matrix_imgs = []
        matrix_titles = []
        for s in sigmas:
            k = int(np.ceil(3.0 * s)) * 2 + 1
            b = cv2.GaussianBlur(img, (k, k), sigmaX=s, sigmaY=s)
            for _, low, high in thresh_pairs:
                ed = cv2.Canny(b, low, high)
                matrix_imgs.append(ed)
                matrix_titles.append(f"s={s} | ({low}, {high})")

        plot_grid(
            matrix_imgs,
            matrix_titles,
            rows=4,
            cols=4,
            save_path=self.predictions_dir / "ex2_full_grid_matrix.png",
            main_title="Ma tran Luoi 16 Tham so: 4 Mua Sigma x 4 Cap Nguong Hysteresis",
            figsize=(16, 16),
        )

        logger.info("Exercise 2 completed. Parameter grid visualizations saved.")

    # ==========================================================================
    # 3. Bai 3: Adverse Image Modalities & Preprocessing Impact
    # ==========================================================================

    def run_exercise_3(self, samples: dict[str, np.ndarray]) -> None:
        """Execute Exercise 3: Robustness across Noise, Low Contrast, Dense Texture."""
        logger.info(
            "Executing Exercise 3: Adverse Modalities & Preprocessing Impact..."
        )

        # 3.1 Modality 1: High Noise (Gaussian & Salt-and-Pepper)
        noisy_gauss = samples["noisy_gaussian"]
        noisy_sp = samples["noisy_sp"]

        # Direct vs Preprocessed for Gaussian Noise
        direct_gauss = cv2.Canny(noisy_gauss, 50, 150)
        pre_bilateral = cv2.bilateralFilter(
            noisy_gauss, d=9, sigmaColor=75, sigmaSpace=75
        )
        edges_bilateral = cv2.Canny(pre_bilateral, 50, 150)

        # Direct vs Preprocessed for Salt & Pepper Noise
        direct_sp = cv2.Canny(noisy_sp, 50, 150)
        pre_median = cv2.medianBlur(noisy_sp, 5)
        edges_median = cv2.Canny(pre_median, 50, 150)

        noise_imgs = [
            noisy_gauss,
            direct_gauss,
            edges_bilateral,
            noisy_sp,
            direct_sp,
            edges_median,
        ]
        noise_titles = [
            "(a) Nhieu Gaussian (sigma=25)",
            "(b) Canny truc tiep (Bi vo vun)",
            "(c) Tien xu ly Bilateral + Canny",
            "(d) Nhieu Muoi Tieu (S&P)",
            "(e) Canny truc tiep (Phao hoa)",
            "(f) Tien xu ly Median Blur + Canny",
        ]
        plot_grid(
            noise_imgs,
            noise_titles,
            rows=2,
            cols=3,
            save_path=self.predictions_dir / "ex3_noise_robustness.png",
            main_title="Khao sat Anh Nhieu: Hieu qua Dot pha cua Loc Song phuong va Loc Trung vi",
        )

        # 3.2 Modality 2: Low Contrast (Dark / Foggy)
        low_cont = samples["low_contrast"]
        direct_low = cv2.Canny(low_cont, 50, 150)

        # Preprocessing: Histogram Equalization vs CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        pre_clahe = clahe.apply(low_cont)
        edges_clahe = cv2.Canny(pre_clahe, 50, 150)

        # 3.3 Modality 3: Dense Texture (Brick)
        brick = samples["brick"]
        direct_brick = cv2.Canny(brick, 50, 150)
        # Preprocessing: Bilateral to smooth interior texture while keeping mortar edges
        pre_brick = cv2.bilateralFilter(brick, d=9, sigmaColor=100, sigmaSpace=100)
        edges_brick = cv2.Canny(pre_brick, 70, 180)

        adv_imgs = [
            low_cont,
            direct_low,
            edges_clahe,
            brick,
            direct_brick,
            edges_brick,
        ]
        adv_titles = [
            "(a) Tuong phan thap (Hep dai)",
            "(b) Canny truc tiep (Mat bien hoan toan)",
            "(c) CLAHE Tang cuong + Canny",
            "(d) Nhieu chi tiet (Gach/Van)",
            "(e) Canny truc tiep (Rung bien rac)",
            "(f) Bilateral Filter + Canny chon loc",
        ]
        plot_grid(
            adv_imgs,
            adv_titles,
            rows=2,
            cols=3,
            save_path=self.predictions_dir / "ex3_contrast_and_texture.png",
            main_title="Khao sat Anh Tuong phan Thap va Anh Nhieu Chi tiet Van Be mat",
        )

        logger.info("Exercise 3 completed. Adverse modality figures generated.")

    # ==========================================================================
    # 4. Bai 4: Downstream Task Integration (Segmentation & Shape Recognition)
    # ==========================================================================

    def run_exercise_4(self, samples: dict[str, np.ndarray]) -> None:
        """Execute Exercise 4: Morphology Segmentation and Shape Recognition."""
        logger.info("Executing Exercise 4: Downstream Tasks (Segmentation & Shapes)...")

        # 4.1 Segmentation via Canny + Morphology Closing + Hole Filling
        coins = samples["coins"]
        blurred_coins = cv2.GaussianBlur(coins, (5, 5), 1.2)
        canny_coins = cv2.Canny(blurred_coins, 60, 140)

        closed_edges, filled_mask = segment_via_morphology_and_holes(
            canny_coins, kernel_size=(5, 5), closing_iterations=3, dilation_iterations=1
        )

        # Color overlay segmented coins
        coins_rgb = cv2.cvtColor(coins, cv2.COLOR_GRAY2RGB)
        seg_overlay = coins_rgb.copy()
        seg_overlay[filled_mask > 0] = [0, 220, 100]
        blended_seg = cv2.addWeighted(seg_overlay, 0.5, coins_rgb, 0.5, 0)

        seg_imgs = [coins, canny_coins, closed_edges, filled_mask, blended_seg]
        seg_titles = [
            "(a) Anh goc Coins",
            "(b) Bien Canny ban dau",
            "(c) Morphology Closing (Noi bien)",
            "(d) binary_fill_holes (To mat na)",
            "(e) Ket qua Phan doan Vung dong",
        ]
        plot_grid(
            seg_imgs,
            seg_titles,
            rows=1,
            cols=5,
            save_path=self.predictions_dir / "ex4_segmentation_pipeline.png",
            main_title="Pipeline Phan doan Vung: Canny -> Morphological Closing -> Binary Fill Holes",
            figsize=(18, 4),
        )

        # 4.2 Shape Recognition: Hough Lines, Hough Circles, Contour Polygon Approx
        shapes = samples["shapes"]
        shapes_rgb = cv2.cvtColor(shapes, cv2.COLOR_GRAY2RGB)
        shapes_edges = cv2.Canny(shapes, 50, 150)

        # 4.2.1 Hough Lines P
        lines_canvas, line_count = detect_hough_lines(shapes_edges, shapes_rgb)

        # 4.2.2 Hough Circles
        circles_canvas, circle_count = detect_hough_circles(shapes, shapes_rgb)

        # 4.2.3 Contour Polygon Approximation
        poly_canvas, poly_counts = classify_polygon_contours(shapes_edges, shapes_rgb)

        shape_imgs = [
            shapes,
            shapes_edges,
            lines_canvas,
            circles_canvas,
            poly_canvas,
        ]
        shape_titles = [
            "(a) Cac khoi hinh hoc goc",
            "(b) Bien Canny 1-pixel",
            f"(c) Hough Lines P ({line_count} doan)",
            f"(d) Hough Circles ({circle_count} hinh tron)",
            f"(e) Polygon Approx ({sum(poly_counts.values())} da giac)",
        ]
        plot_grid(
            shape_imgs,
            shape_titles,
            rows=1,
            cols=5,
            save_path=self.predictions_dir / "ex4_shape_recognition.png",
            main_title="Nhan dang Hinh hoc: Bien doi Hough & Xap xi Da giac cv2.approxPolyDP",
            figsize=(18, 4),
        )

        self.metrics_report["ex4"] = {
            "hough_lines_detected": line_count,
            "hough_circles_detected": circle_count,
            "polygon_classification": poly_counts,
        }
        logger.info(f"Exercise 4 completed. Shapes recognized: {poly_counts}")

    # ==========================================================================
    # 5. Extended Research (Phần III)
    # ==========================================================================

    def run_extended_questions(self, samples: dict[str, np.ndarray]) -> None:
        """Execute in-depth experiments for the 4 extended theoretical & practical questions."""
        logger.info(
            "Executing Extended Research: Metrics, Auto-Canny, Color Tensor, Video..."
        )

        # ----------------------------------------------------------------------
        # Q1: Quantitative Evaluation with Pratt's FOM, F1, Hausdorff vs Noise
        # ----------------------------------------------------------------------
        gt = samples["shapes_gt"]
        clean_shapes = samples["shapes"]

        noise_levels = [0, 10, 20, 30, 40, 50]
        fom_scores: list[float] = []
        f1_scores: list[float] = []
        precision_scores: list[float] = []
        recall_scores: list[float] = []
        hausdorff_scores: list[float] = []

        np.random.seed(42)
        for nl in noise_levels:
            if nl == 0:
                noisy_img = clean_shapes
            else:
                noise = np.random.normal(0, nl, clean_shapes.shape)
                noisy_img = np.clip(
                    clean_shapes.astype(np.float32) + noise, 0, 255
                ).astype(np.uint8)

            blurred = cv2.GaussianBlur(noisy_img, (5, 5), 1.0)
            det_edges = cv2.Canny(blurred, 50, 150)

            fom = calculate_pratt_fom(det_edges, gt)
            f1_dict = calculate_edge_f1_score(det_edges, gt, tolerance=2.0)
            h_dist = calculate_hausdorff_distance(det_edges, gt)

            fom_scores.append(fom)
            f1_scores.append(f1_dict["f1"])
            precision_scores.append(f1_dict["precision"])
            recall_scores.append(f1_dict["recall"])
            hausdorff_scores.append(min(h_dist, 50.0))  # Cap for plotting

        # Plot Quantitative Curves
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(
            noise_levels,
            fom_scores,
            "o-",
            color="#1f77b4",
            linewidth=2,
            label="Pratt's FOM",
        )
        ax1.plot(
            noise_levels,
            f1_scores,
            "s--",
            color="#2ca02c",
            linewidth=2,
            label="F1-Score",
        )
        ax1.plot(
            noise_levels, precision_scores, "^:", color="#ff7f0e", label="Precision"
        )
        ax1.plot(noise_levels, recall_scores, "v:", color="#9467bd", label="Recall")
        ax1.set_xlabel(
            "Muc Do Nhieu Gaussian (Standard Deviation sigma)",
            fontsize=11,
            fontweight="semibold",
        )
        ax1.set_ylabel("Chi so Danh gia [0.0, 1.0]", fontsize=11, fontweight="semibold")
        ax1.set_ylim(-0.05, 1.05)
        ax1.grid(True, linestyle="--", alpha=0.6)

        ax2 = ax1.twinx()
        ax2.plot(
            noise_levels,
            hausdorff_scores,
            "d-.",
            color="#d62728",
            linewidth=2,
            label="Hausdorff Distance (px)",
        )
        ax2.set_ylabel(
            "Khoang cach Hausdorff (pixels)",
            fontsize=11,
            fontweight="semibold",
            color="#d62728",
        )
        ax2.tick_params(axis="y", labelcolor="#d62728")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        plt.title(
            "Danh gia Dinh luong Do chinh xac Canny theo Muc Nhieu (Q1)",
            fontsize=12,
            fontweight="bold",
        )
        plt.tight_layout()
        q1_chart_path = self.predictions_dir / "ext_q1_quantitative_metrics.png"
        plt.savefig(q1_chart_path, dpi=200)
        plt.close(fig)

        self.metrics_report["q1_evaluation"] = {
            "noise_levels": noise_levels,
            "pratt_fom": fom_scores,
            "f1": f1_scores,
            "precision": precision_scores,
            "recall": recall_scores,
            "hausdorff_dist": hausdorff_scores,
        }

        # ----------------------------------------------------------------------
        # Q2: Auto-Canny & Adaptive Thresholding
        # ----------------------------------------------------------------------
        coins = samples["coins"]
        fixed_edges = cv2.Canny(coins, 50, 150)
        auto_edges, a_low, a_high = run_auto_canny(coins, sigma_factor=0.33)
        otsu_edges, o_low, o_high = run_otsu_canny(coins)

        auto_imgs = [coins, fixed_edges, auto_edges, otsu_edges]
        auto_titles = [
            "Anh goc Coins",
            "Fixed Canny (50, 150)",
            f"Auto-Canny Median ({a_low}, {a_high})",
            f"Otsu-Canny ({o_low}, {o_high})",
        ]
        plot_grid(
            auto_imgs,
            auto_titles,
            rows=1,
            cols=4,
            save_path=self.predictions_dir / "ext_q2_auto_canny.png",
            main_title="Cai tien Hieu suat: Tu dong hoa Chon Nguong Thich nghi (Auto-Canny & Otsu)",
            figsize=(16, 4.5),
        )

        # ----------------------------------------------------------------------
        # Q3: Color Canny (Multi-Channel bitwise OR vs Di Zenzo Tensor)
        # ----------------------------------------------------------------------
        chroma = samples["color_chroma"]
        chroma_gray = cv2.cvtColor(chroma, cv2.COLOR_RGB2GRAY)
        gray_canny = cv2.Canny(chroma_gray, 40, 100)

        mc_canny = run_multichannel_canny(chroma, 40, 100)
        dizenzo_mag, dizenzo_ang = run_di_zenzo_gradient(chroma)
        dizenzo_norm = np.clip(
            dizenzo_mag / (np.max(dizenzo_mag) + 1e-7) * 255.0, 0, 255
        ).astype(np.uint8)
        dizenzo_edges = cv2.Canny(dizenzo_norm, 40, 100)

        color_imgs = [
            chroma,
            chroma_gray,
            gray_canny,
            mc_canny,
            dizenzo_norm,
            dizenzo_edges,
        ]
        color_titles = [
            "(a) Anh Mau Sac dong do sang",
            "(b) Chuyen Grayscale (Bien mau bi triet tieu)",
            "(c) Grayscale Canny (Mat bien giua 2 vung mau)",
            "(d) Multi-channel OR Canny (Bat duoc bien)",
            "(e) Di Zenzo Gradient Tensor Magnitude",
            "(f) Di Zenzo Canny (Toi uu gradient da kenh)",
        ]
        plot_grid(
            color_imgs,
            color_titles,
            rows=2,
            cols=3,
            save_path=self.predictions_dir / "ext_q3_color_canny.png",
            main_title="Canny tren Anh Mau: Phan tich Han che Grayscale vs Di Zenzo Tensor",
        )

        # ----------------------------------------------------------------------
        # Q4: Video Edge Detection & Temporal Flickering Mitigation
        # ----------------------------------------------------------------------
        # Synthesize a 10-frame mini video sequence with moving object + random noise
        n_frames = 10
        video_frames = []
        raw_edges_seq = []
        stabilized_edges_seq = []

        temp_filter = TemporalEdgeFilter(alpha=0.65, threshold=100.0)

        for f_idx in range(n_frames):
            frame = np.full((120, 160), 40, dtype=np.uint8)
            # Moving circle
            center_x = 40 + f_idx * 8
            cv2.circle(frame, (center_x, 60), 25, 200, -1)
            # Add dynamic temporal noise spike
            noise = np.random.normal(0, 15, frame.shape)
            noisy_f = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

            raw_ed = cv2.Canny(noisy_f, 50, 140)
            stab_ed = temp_filter.update(raw_ed)

            video_frames.append(noisy_f)
            raw_edges_seq.append(raw_ed)
            stabilized_edges_seq.append(stab_ed)

        # Compare Frame 4 and Frame 5 to inspect flicker
        f4_diff = np.abs(
            raw_edges_seq[4].astype(np.int16) - raw_edges_seq[3].astype(np.int16)
        ).astype(np.uint8)
        f4_stab_diff = np.abs(
            stabilized_edges_seq[4].astype(np.int16)
            - stabilized_edges_seq[3].astype(np.int16)
        ).astype(np.uint8)

        vid_imgs = [
            video_frames[4],
            raw_edges_seq[4],
            stabilized_edges_seq[4],
            f4_diff,
            f4_stab_diff,
        ]
        vid_titles = [
            "Video Frame t=4",
            "Raw Canny (Flickering Noise)",
            "Stabilized Canny (Temporal Filter)",
            f"Raw Temporal Diff ({np.sum(f4_diff > 0)} px diff)",
            f"Filtered Temp Diff ({np.sum(f4_stab_diff > 0)} px diff)",
        ]
        plot_grid(
            vid_imgs,
            vid_titles,
            rows=1,
            cols=5,
            save_path=self.predictions_dir / "ext_q4_video_stabilization.png",
            main_title="Xu ly Video Thoi gian Thuc: Khac phuc Hien tuong Nhap nhay (Temporal Flickering)",
            figsize=(18, 4),
        )

        logger.info("Extended research experiments completed.")

    # ==========================================================================
    # 6. Master Run & Log Export
    # ==========================================================================

    def run_all(self) -> dict[str, Any]:
        """Execute the entire Lab 03 experimental suite."""
        logger.info("Starting Lab 03 Master Pipeline execution...")
        start_time = time.time()

        samples = self.prepare_dataset_samples()
        self.run_exercise_1(samples)
        self.run_exercise_2(samples)
        self.run_exercise_3(samples)
        self.run_exercise_4(samples)
        self.run_extended_questions(samples)

        total_elapsed = time.time() - start_time
        self.metrics_report["total_runtime_seconds"] = total_elapsed
        logger.info(
            f"Lab 03 Pipeline execution finished successfully in {total_elapsed:.2f} seconds."
        )

        # Save metrics summary to JSON
        metrics_file = self.logs_dir / "metrics_summary.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(self.metrics_report, f, indent=2)
        logger.info(f"Exported metrics report to {metrics_file}")

        return self.metrics_report


if __name__ == "__main__":
    pipeline = Lab03Pipeline()
    pipeline.run_all()
