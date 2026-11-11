"""Pipeline execution coordinator for Computer Vision Lab 1."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import PIL

from src.common.logger import get_logger
from src.labs.lab01_basics.color import bgr_to_grayscale, bgr_to_hsv, bgr_to_lab
from src.labs.lab01_basics.drawing import (
    draw_circle,
    draw_line,
    draw_rectangle,
    draw_text,
)
from src.labs.lab01_basics.geometry import (
    crop_roi,
    resize_by_dimensions,
    resize_by_scale,
)
from src.labs.lab01_basics.io import (
    convert_image_format,
    load_image_cv,
    load_image_pil,
    save_image_cv,
)
from src.labs.lab01_basics.visuals import plot_color_spaces

logger = get_logger(__name__)


def run_lab01(
    input_image_path: str | Path = "data/samples/sample_traffic.jpg",
    output_dir: str | Path = "experiments/predictions/lab01",
) -> dict[str, Path]:
    """Execute complete end-to-end pipeline for Lab 1.

    Args:
        input_image_path: Path to the input test image.
        output_dir: Directory where resulting images and figures will be saved.

    Returns:
        dict[str, Path]: Dictionary mapping artifact names to their saved paths.
    """
    input_path = Path(input_image_path)
    base_out = Path(output_dir)
    artifacts: dict[str, Path] = {}

    logger.info("==================================================")
    logger.info("Starting Execution: UTH Computer Vision - Lab 1")
    logger.info("Input Image: %s", input_path.resolve())
    logger.info("Output Directory: %s", base_out.resolve())
    logger.info("==================================================")

    # 1. Environment verification
    logger.info("[Yêu cầu 1] Môi trường và Thư viện:")
    logger.info("  - OpenCV Version: %s", cv2.__version__)
    logger.info("  - Pillow Version: %s", PIL.__version__)
    logger.info("  - NumPy Version: %s", np.__version__)

    # 2. Image IO & format conversion
    logger.info("[Yêu cầu 2] Đọc và Lưu ảnh đa định dạng...")
    img_cv = load_image_cv(input_path)
    img_pil = load_image_pil(input_path)

    height, width, channels = img_cv.shape
    logger.info(
        "  - OpenCV image shape: (H=%d, W=%d, C=%d), dtype=%s",
        height,
        width,
        channels,
        img_cv.dtype,
    )
    logger.info("  - Pillow image size: %s, mode=%s", img_pil.size, img_pil.mode)

    conv_dir = base_out / "converted"
    artifacts["png"] = convert_image_format(input_path, conv_dir / "sample.png")
    artifacts["bmp"] = convert_image_format(input_path, conv_dir / "sample.bmp")
    artifacts["jpg"] = convert_image_format(input_path, conv_dir / "sample_copy.jpg")

    # 3. Color spaces
    logger.info("[Yêu cầu 3] Chuyển đổi không gian màu (Grayscale, HSV, LAB)...")
    colors_dir = base_out / "color_spaces"

    img_gray = bgr_to_grayscale(img_cv)
    img_hsv = bgr_to_hsv(img_cv)
    img_lab = bgr_to_lab(img_cv)

    artifacts["grayscale"] = save_image_cv(img_gray, colors_dir / "grayscale.jpg")
    artifacts["hsv"] = save_image_cv(img_hsv, colors_dir / "hsv.jpg")
    artifacts["lab"] = save_image_cv(img_lab, colors_dir / "lab.jpg")

    fig_grid = plot_color_spaces(img_cv, img_gray, img_hsv, img_lab)
    fig_grid_path = colors_dir / "color_spaces_grid.png"
    fig_grid_path.parent.mkdir(parents=True, exist_ok=True)
    fig_grid.savefig(fig_grid_path, bbox_inches="tight", dpi=150)
    import matplotlib.pyplot as plt

    plt.close(fig_grid)
    artifacts["color_spaces_grid"] = fig_grid_path

    # 4. Spatial transforms
    logger.info("[Yêu cầu 4] Biến đổi hình học: Cắt ROI và Thay đổi kích thước...")
    trans_dir = base_out / "transformed"

    roi_box = (
        int(width * 0.25),
        int(height * 0.25),
        int(width * 0.75),
        int(height * 0.75),
    )
    roi_img = crop_roi(img_cv, roi_box)
    artifacts["roi"] = save_image_cv(roi_img, trans_dir / "roi_crop.jpg")

    scaled_img = resize_by_scale(img_cv, fx=0.5, fy=0.5)
    artifacts["scaled"] = save_image_cv(scaled_img, trans_dir / "scaled_50_percent.jpg")

    fixed_img = resize_by_dimensions(img_cv, target_width=300, target_height=300)
    artifacts["fixed_size"] = save_image_cv(
        fixed_img, trans_dir / "resized_300x300.jpg"
    )

    # 5. Drawing and annotations
    logger.info("[Yêu cầu 5] Vẽ hình học cơ bản và chú thích văn bản...")
    draw_dir = base_out / "annotations"
    annotated = img_cv.copy()

    annotated = draw_rectangle(
        annotated, (50, 50), (250, 200), (0, 255, 0), thickness=3
    )
    annotated = draw_circle(
        annotated, (400, 150), radius=60, color=(255, 0, 0), thickness=-1
    )
    annotated = draw_line(
        annotated, (0, 0), (width - 1, height - 1), (0, 0, 255), thickness=2
    )
    annotated = draw_text(
        annotated,
        "Lab 1: OpenCV Basics",
        (50, height - 50),
        font_scale=1.2,
        color=(0, 255, 255),
        thickness=2,
    )
    artifacts["annotated"] = save_image_cv(annotated, draw_dir / "annotated_sample.jpg")

    logger.info("Lab 1 pipeline execution completed successfully!")
    return artifacts


if __name__ == "__main__":
    run_lab01()
