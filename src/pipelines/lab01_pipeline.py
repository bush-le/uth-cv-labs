"""Pipeline execution script for Computer Vision Lab 1.

Demonstrates fundamental image processing operations using OpenCV and Pillow:
1. Environment setup and library verification.
2. Image input, output, and format conversion (.jpg, .png, .bmp).
3. Color space conversion (Grayscale, HSV, CIE L*a*b*).
4. Spatial transformations: ROI cropping, scaling, fixed resizing.
5. Geometric drawings and text annotations.
"""

from pathlib import Path

import cv2
import numpy as np
import PIL

from src.common.logger import get_logger
from src.common.visualizer import plot_color_spaces_grid, plot_side_by_side
from src.data.transforms import (
    bgr_to_grayscale,
    bgr_to_hsv,
    bgr_to_lab,
    convert_image_format,
    crop_roi,
    draw_circle,
    draw_line,
    draw_rectangle,
    draw_text,
    load_image_cv,
    load_image_pil,
    resize_by_dimensions,
    resize_by_scale,
    save_image_cv,
)

logger = get_logger("lab01_pipeline")


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

    # --------------------------------------------------------------------------
    # Yêu cầu 1: Kiểm tra môi trường
    # --------------------------------------------------------------------------
    logger.info("[Yêu cầu 1] Môi trường và Thư viện:")
    logger.info("  - OpenCV Version: %s", cv2.__version__)
    logger.info("  - Pillow Version: %s", PIL.__version__)
    logger.info("  - NumPy Version: %s", np.__version__)

    # --------------------------------------------------------------------------
    # Yêu cầu 2: Đọc, Hiển thị và Lưu ảnh đa định dạng
    # --------------------------------------------------------------------------
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
    logger.info("  - Converted formats saved to: %s", conv_dir)

    # --------------------------------------------------------------------------
    # Yêu cầu 3: Chuyển đổi không gian màu (Grayscale, HSV, LAB)
    # --------------------------------------------------------------------------
    logger.info("[Yêu cầu 3] Chuyển đổi không gian màu (Grayscale, HSV, LAB)...")
    colors_dir = base_out / "color_spaces"

    img_gray = bgr_to_grayscale(img_cv)
    img_hsv = bgr_to_hsv(img_cv)
    img_lab = bgr_to_lab(img_cv)

    artifacts["gray"] = save_image_cv(img_gray, colors_dir / "grayscale.png")
    artifacts["hsv"] = save_image_cv(img_hsv, colors_dir / "hsv.png")
    artifacts["lab"] = save_image_cv(img_lab, colors_dir / "lab.png")

    # Lưu biểu đồ lưới so sánh 2x2
    grid_fig_path = colors_dir / "color_spaces_grid.png"
    plot_color_spaces_grid(
        original_bgr=img_cv,
        gray=img_gray,
        hsv=img_hsv,
        lab=img_lab,
        save_path=grid_fig_path,
    )
    artifacts["color_grid_fig"] = grid_fig_path
    logger.info("  - Color space artifacts saved to: %s", colors_dir)

    # --------------------------------------------------------------------------
    # Yêu cầu 4: Cắt xén và Thay đổi kích thước (Cropping & Resizing)
    # --------------------------------------------------------------------------
    logger.info("[Yêu cầu 4] Cắt xén và Thay đổi kích thước (ROI Crop & Resize)...")
    spatial_dir = base_out / "spatial"

    # Trích xuất ROI trung tâm (ví dụ: đối tượng xe / phương tiện giao thông)
    # Tọa độ ROI động theo kích thước ảnh: [x1, y1, x2, y2]
    x1 = int(width * 0.25)
    y1 = int(height * 0.35)
    x2 = int(width * 0.75)
    y2 = int(height * 0.85)
    roi_box = (x1, y1, x2, y2)

    img_roi = crop_roi(img_cv, roi_box)
    artifacts["roi"] = save_image_cv(img_roi, spatial_dir / "roi_cropped.jpg")
    logger.info(
        "  - Cropped ROI with box %s -> output shape: %s", roi_box, img_roi.shape
    )

    # Resize theo tỷ lệ (Scale factor: 0.5x và 1.5x)
    img_scale_half = resize_by_scale(img_cv, fx=0.5, fy=0.5)
    img_scale_150 = resize_by_scale(img_cv, fx=1.5, fy=1.5)
    artifacts["scale_0.5x"] = save_image_cv(
        img_scale_half, spatial_dir / "resize_scale_0.5x.jpg"
    )
    artifacts["scale_1.5x"] = save_image_cv(
        img_scale_150, spatial_dir / "resize_scale_1.5x.jpg"
    )

    # Resize về kích thước cố định (width=400, height=300)
    fixed_w, fixed_h = 400, 300
    img_fixed = resize_by_dimensions(
        img_cv, target_width=fixed_w, target_height=fixed_h
    )
    artifacts["resize_fixed"] = save_image_cv(
        img_fixed, spatial_dir / f"resize_fixed_{fixed_w}x{fixed_h}.jpg"
    )

    # Lưu biểu đồ so sánh các phép biến đổi không gian
    spatial_fig_path = spatial_dir / "spatial_transformations_comparison.png"
    plot_side_by_side(
        images=[img_cv, img_roi, img_fixed],
        titles=[
            "Original (800x533)",
            f"ROI Crop {roi_box}",
            f"Fixed Resize ({fixed_w}x{fixed_h})",
        ],
        save_path=spatial_fig_path,
    )
    artifacts["spatial_fig"] = spatial_fig_path
    logger.info("  - Spatial transformation artifacts saved to: %s", spatial_dir)

    # --------------------------------------------------------------------------
    # Yêu cầu 5: Vẽ hình cơ bản và chèn chữ lên ảnh (Drawing & Annotations)
    # --------------------------------------------------------------------------
    logger.info("[Yêu cầu 5] Vẽ hình cơ bản và chèn chữ lên ảnh...")
    annot_dir = base_out / "annotations"

    # Bước 5.1: Vẽ Line (Đường kẻ phân cách / đường dẫn hướng)
    img_annotated = draw_line(
        img_cv,
        pt1=(int(width * 0.1), int(height * 0.9)),
        pt2=(int(width * 0.9), int(height * 0.9)),
        color=(0, 255, 0),  # Màu xanh lá (Green in BGR)
        thickness=3,
    )

    # Bước 5.2: Vẽ Rectangle (Bounding box quanh vùng ROI)
    img_annotated = draw_rectangle(
        img_annotated,
        pt1=(x1, y1),
        pt2=(x2, y2),
        color=(0, 0, 255),  # Màu đỏ (Red in BGR)
        thickness=3,
    )

    # Bước 5.3: Vẽ Circle (Điểm mốc / tâm quan tâm)
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    img_annotated = draw_circle(
        img_annotated,
        center=(center_x, center_y),
        radius=25,
        color=(255, 165, 0),  # Màu cam / xanh dương BGR (255, 165, 0)
        thickness=3,
    )

    # Bước 5.4: Chèn Text (Nhãn chú thích và thông tin)
    img_annotated = draw_text(
        img_annotated,
        text="UTH CV Lab 1: Object ROI",
        org=(x1, max(30, y1 - 15)),
        font_scale=0.8,
        color=(0, 255, 255),  # Màu vàng (Yellow in BGR)
        thickness=2,
    )

    img_annotated = draw_text(
        img_annotated,
        text="Target Centroid",
        org=(center_x + 35, center_y + 8),
        font_scale=0.6,
        color=(255, 255, 255),  # Màu trắng
        thickness=2,
    )

    artifacts["annotated_img"] = save_image_cv(
        img_annotated, annot_dir / "annotated_sample.jpg"
    )

    # Lưu biểu đồ so sánh Original vs Annotated
    annot_fig_path = annot_dir / "annotations_comparison.png"
    plot_side_by_side(
        images=[img_cv, img_annotated],
        titles=["Original Raw Image", "Annotated Image (Line, Rect, Circle, Text)"],
        save_path=annot_fig_path,
    )
    artifacts["annot_fig"] = annot_fig_path
    logger.info("  - Annotation artifacts saved to: %s", annot_dir)

    logger.info("==================================================")
    logger.info("Lab 1 Execution Pipeline Completed Successfully!")
    logger.info("Total generated artifacts: %d", len(artifacts))
    logger.info("==================================================")

    return artifacts


if __name__ == "__main__":
    run_lab01()
