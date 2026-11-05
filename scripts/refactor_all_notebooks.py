"""Refactor Jupyter notebooks to thin presentation layers calling atomic modules."""

from __future__ import annotations

import json
from pathlib import Path


def set_cell_source(cell: dict, code: str) -> None:
    lines = code.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] = lines[-1] + "\n"
    cell["source"] = lines


def refactor_lab01() -> None:
    nb_path = Path("notebooks/labs/lab_01.ipynb")
    nb = json.loads(nb_path.read_text(encoding="utf-8"))

    updates = {
        3: """import sys
from pathlib import Path

CURRENT_DIR = Path.cwd().resolve()
PROJECT_ROOT = CURRENT_DIR
for parent in [CURRENT_DIR, *CURRENT_DIR.parents]:
    if (parent / "configs").exists() and (parent / "src").exists():
        PROJECT_ROOT = parent
        break

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL

from src.labs.lab01_basics import color, drawing, geometry, io, visuals

%matplotlib inline
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["image.interpolation"] = "nearest"

print(f"Python Version  : {sys.version.split()[0]}")
print(f"OpenCV Version  : {cv2.__version__}")
print(f"Pillow Version  : {PIL.__version__}")
print(f"NumPy Version   : {np.__version__}")
print("\\n[+] Đã tải thành công các atomic modules từ src.labs.lab01_basics: (color, drawing, geometry, io, visuals)")
""",
        5: """image_path = PROJECT_ROOT / "data/samples/sample_traffic.jpg"
if not image_path.exists():
    image_path = Path("data/samples/sample_traffic.jpg")

img_bgr = io.load_image_cv(image_path)
print("=== OpenCV Image Metadata ===")
print(f"Type       : {type(img_bgr)}")
print(f"Shape      : {img_bgr.shape} (Chiều cao H, Chiều rộng W, Số kênh C)")
print(f"Dtype      : {img_bgr.dtype}")
print(f"Min / Max  : [{img_bgr.min()}, {img_bgr.max()}]")

img_pil = io.load_image_pil(image_path)
print("\\n=== Pillow Image Metadata ===")
print(f"Size       : {img_pil.size} (Chiều rộng W, Chiều cao H)")
print(f"Format     : {img_pil.format}")
print(f"Mode       : {img_pil.mode}")
""",
        7: """img_rgb = color.bgr_to_rgb(img_bgr)
fig = visuals.plot_bgr_vs_rgb(img_bgr, img_rgb)
plt.show()
""",
        9: """out_dir = PROJECT_ROOT / "experiments/predictions/lab01/converted"
out_dir.mkdir(parents=True, exist_ok=True)

png_path = io.save_image_cv(img_bgr, out_dir / "output.png")
bmp_path = io.save_image_cv(img_bgr, out_dir / "output.bmp")
jpg_path = io.save_image_cv(img_bgr, out_dir / "output_q95.jpg")

print("Dung lượng lưu trữ các định dạng:")
print(f"  - Original JPEG: {image_path.stat().st_size / 1024:.2f} KB")
print(f"  - Saved PNG    : {png_path.stat().st_size / 1024:.2f} KB")
print(f"  - Saved BMP    : {bmp_path.stat().st_size / 1024:.2f} KB")
print(f"  - Saved JPG Q95: {jpg_path.stat().st_size / 1024:.2f} KB")
""",
        11: """img_gray = color.bgr_to_grayscale(img_bgr)
img_hsv  = color.bgr_to_hsv(img_bgr)
img_lab  = color.bgr_to_lab(img_bgr)

print(f"Shape Grayscale : {img_gray.shape} (2 chiều: H x W)")
print(f"Shape HSV       : {img_hsv.shape} (3 chiều: H x W x 3)")
print(f"Shape LAB       : {img_lab.shape} (3 chiều: H x W x 3)")
""",
        12: """fig = visuals.plot_color_spaces(img_rgb, img_gray, img_hsv, img_lab)
plt.show()
""",
        14: """h, s, v = color.split_channels(img_hsv)
fig = visuals.plot_hsv_channels(h, s, v)
plt.show()
""",
        16: """h_orig, w_orig = img_bgr.shape[:2]

x1, y1 = int(w_orig * 0.25), int(h_orig * 0.35)
x2, y2 = int(w_orig * 0.75), int(h_orig * 0.85)

img_roi = geometry.crop_roi(img_bgr, x1, y1, x2, y2)
print(f"Kích thước ảnh gốc: {img_bgr.shape}")
print(f"Tọa độ ROI        : [x1={x1}, y1={y1}, x2={x2}, y2={y2}]")
print(f"Kích thước ROI    : {img_roi.shape}")

fig = visuals.plot_single_image(img_roi, f"Vùng Quan Tâm Đã Cắt (ROI: {img_roi.shape[1]}x{img_roi.shape[0]})", is_bgr=True)
plt.show()
""",
        18: """img_half = geometry.resize_by_scale(img_bgr, 0.5)
img_upscaled = geometry.resize_by_scale(img_bgr, 1.5)
target_w, target_h = 400, 300
img_fixed = geometry.resize_by_dimensions(img_bgr, target_w, target_h)

print(f"Kích thước ban đầu     : {img_bgr.shape[1]} x {img_bgr.shape[0]}")
print(f"Kích thước thu nhỏ 0.5x: {img_half.shape[1]} x {img_half.shape[0]}")
print(f"Kích thước phóng to 1.5x: {img_upscaled.shape[1]} x {img_upscaled.shape[0]}")
print(f"Kích thước cố định      : {img_fixed.shape[1]} x {img_fixed.shape[0]}")
""",
        19: """fig = visuals.plot_roi_and_resize(img_bgr, img_roi, img_half, img_fixed)
plt.show()
""",
        21: """canvas = img_bgr.copy()

canvas = drawing.draw_line(
    canvas,
    pt1=(int(w_orig * 0.05), int(h_orig * 0.92)),
    pt2=(int(w_orig * 0.95), int(h_orig * 0.92)),
    color=(0, 255, 0),
    thickness=3,
)

canvas = drawing.draw_rectangle(
    canvas,
    pt1=(x1, y1),
    pt2=(x2, y2),
    color=(0, 0, 255),
    thickness=3,
)

cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
canvas = drawing.draw_circle(canvas, center=(cx, cy), radius=20, color=(0, 165, 255), thickness=3)
canvas = drawing.draw_circle(canvas, center=(cx, cy), radius=4, color=(0, 0, 255), thickness=-1)

canvas = drawing.draw_text(
    canvas,
    text="Vehicle ROI: 98.5%",
    org=(x1 + 5, y1 - 5),
    color=(255, 255, 255),
    scale=0.75,
    thickness=2,
    bg_color=(0, 0, 255),
)
canvas = drawing.draw_text(
    canvas,
    text="UTH AI & Computer Vision Lab 1",
    org=(20, 40),
    color=(0, 255, 255),
    scale=0.8,
    thickness=2,
)
""",
        22: """fig = visuals.plot_single_image(canvas, "Kết Quả Hoàn Chỉnh: Vẽ Hình Học & Chú Thích (Line, Rectangle, Circle, Text)", is_bgr=True)
plt.show()
""",
    }

    for idx, code in updates.items():
        set_cell_source(nb["cells"][idx], code)

    nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Refactored Lab 01 notebook successfully.")


def refactor_lab02() -> None:
    nb_path = Path("notebooks/labs/lab_02.ipynb")
    nb = json.loads(nb_path.read_text(encoding="utf-8"))

    updates = {
        3: """import sys
from pathlib import Path

CURRENT_DIR = Path.cwd().resolve()
PROJECT_ROOT = CURRENT_DIR
for parent in [CURRENT_DIR, *CURRENT_DIR.parents]:
    if (parent / "configs").exists() and (parent / "src").exists():
        PROJECT_ROOT = parent
        break

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.labs.lab02_filters import (
    edge_filters,
    filter_utils,
    linear_filters,
    nonlinear_filters,
    point_ops,
    visuals,
)

%matplotlib inline
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 80
plt.rcParams["image.interpolation"] = "nearest"

print(f"Python {sys.version.split()[0]} | OpenCV {cv2.__version__} | NumPy {np.__version__}")
print("[+] Đã nạp atomic modules từ src.labs.lab02_filters: point_ops, linear_filters, nonlinear_filters, edge_filters, visuals")


def show_grid(images, titles, ncols=None, normalize_display=False):
    ncols = ncols or len(images)
    nrows = (len(images) + ncols - 1) // ncols
    axes = np.atleast_1d(plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.2 * nrows))[1]).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, img, title in zip(axes, images, titles):
        disp = np.abs(img.astype(np.float32)) if normalize_display else img
        if normalize_display:
            disp = disp / (np.percentile(disp, 99) or 1.0) * 255.0
        disp = np.clip(disp, 0, 255).astype(np.uint8)
        if disp.ndim == 3:
            disp = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        ax.imshow(disp, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontweight="bold")
    plt.tight_layout()
    plt.show()
""",
        4: """image_path = PROJECT_ROOT / "data/samples/image-2.jpg"
if not image_path.exists():
    image_path = Path("data/samples/image-2.jpg")

img_bgr = cv2.imread(str(image_path))
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

print(f"Ảnh       : {image_path}")
print(f"Shape BGR : {img_bgr.shape} (H, W, C)")
print(f"Shape Gray: {img_gray.shape}")
print(f"Dtype     : {img_bgr.dtype}")

show_grid([img_bgr, img_gray], ["Ảnh gốc (BGR)", "Grayscale"])
""",
        6: """beta = 50
img_bright = point_ops.adjust_brightness(img_bgr, beta)
img_dark = point_ops.adjust_brightness(img_bgr, -beta)

print(f"β = +{beta}: min/max = [{img_bright.min()}, {img_bright.max()}]")
print(f"β = -{beta}: min/max = [{img_dark.min()}, {img_dark.max()}]")

show_grid(
    [img_bgr, img_dark, img_bright],
    ["Gốc", f"Tối hơn (β=-{beta})", f"Sáng hơn (β=+{beta})"],
)
""",
        8: """alpha_low, alpha_high = 0.5, 1.8
img_contrast_low = point_ops.adjust_contrast(img_bgr, alpha_low)
img_contrast_high = point_ops.adjust_contrast(img_bgr, alpha_high)

show_grid(
    [img_bgr, img_contrast_low, img_contrast_high],
    ["Gốc", f"α = {alpha_low} (tối hơn)", f"α = {alpha_high} (tương phản mạnh)"],
)
""",
        11: """img_negative = point_ops.negative_transform(img_bgr)

show_grid([img_bgr, img_negative], ["Gốc", "Âm bản (255 - f)"])
""",
        13: """thresholds = [80, 127, 180]
binaries = [point_ops.apply_threshold(img_gray, t) for t in thresholds]

show_grid(
    [img_gray] + binaries,
    ["Grayscale"] + [f"T = {t}" for t in thresholds],
    ncols=4,
)
""",
        15: """mean_k7 = linear_filters.get_mean_kernel(7)
mean_k21 = linear_filters.get_mean_kernel(21)

print(f"Mean 7x7  : sum = {linear_filters.kernel_sum(mean_k7):.2f}")
print(f"Mean 21x21: sum = {linear_filters.kernel_sum(mean_k21):.2f}")

img_mean7 = linear_filters.apply_mean_filter(img_bgr, 7)
img_mean21 = linear_filters.apply_mean_filter(img_bgr, 21)
img_blur_cv2 = cv2.blur(img_bgr, (7, 7))

show_grid(
    [img_bgr, img_mean7, img_mean21, img_blur_cv2],
    ["Gốc", "Mean 7x7", "Mean 21x21", "cv2.blur() 7x7"],
    ncols=4,
)
""",
        17: """gauss_k19 = linear_filters.get_gaussian_kernel(19, sigma=3.0)
print(f"Gaussian 19x19 (sigma=3.0): sum = {linear_filters.kernel_sum(gauss_k19):.4f}")

img_gauss_manual = linear_filters.apply_gaussian_filter(img_bgr, 19, 3.0)
img_gauss_s30 = cv2.GaussianBlur(img_bgr, (19, 19), sigmaX=3.0)
img_gauss_s70 = cv2.GaussianBlur(img_bgr, (43, 43), sigmaX=7.0)

show_grid(
    [img_bgr, img_gauss_manual, img_gauss_s30, img_gauss_s70],
    ["Gốc", "Gaussian thủ công σ=3.0 (k19)",
     "GaussianBlur σ=3.0 (k19)", "GaussianBlur σ=7.0 (k43)"],
    ncols=4,
)
""",
        19: """sharpen_k = linear_filters.get_sharpen_kernel()
print(f"Sharpen kernel: sum = {linear_filters.kernel_sum(sharpen_k):.0f}  (tổng = 1 -> giữ nguyên độ sáng)")

img_sharpen = linear_filters.apply_filter2d(img_bgr, sharpen_k)

show_grid(
    [img_bgr, img_sharpen],
    ["Gốc", "Sharpen kernel (tổng=1)"],
    ncols=2,
)
""",
        21: """sobel_x, sobel_y = edge_filters.get_sobel_kernels()
prewitt_x, prewitt_y = edge_filters.get_prewitt_kernels()

print("Sobel X :")
print(sobel_x)
print(f"  sum = {linear_filters.kernel_sum(sobel_x):.0f}")
print("Prewitt X :")
print(prewitt_x)
print(f"  sum = {linear_filters.kernel_sum(prewitt_x):.0f}  (chỉ khác Sobel ở hàng giữa: 1 thay vì 2)")

gx_sobel, gy_sobel, sobel_mag = edge_filters.apply_sobel(img_gray)
gx_prewitt, gy_prewitt, prewitt_mag = edge_filters.apply_prewitt(img_gray)

corr = np.corrcoef(sobel_mag.ravel(), prewitt_mag.ravel())[0, 1]
print()
print(f"Tương quan Sobel vs Prewitt     : {corr:.4f}  (~1 -> gần như trùng cấu trúc)")
print(f"Biên độ trung bình Sobel        : {sobel_mag.mean():.1f}  (lớn hơn Prewitt vì nhấn hàng giữa)")
print(f"Biên độ trung bình Prewitt      : {prewitt_mag.mean():.1f}")

show_grid(
    [np.abs(gx_sobel), np.abs(gy_sobel), sobel_mag,
     np.abs(gx_prewitt), np.abs(gy_prewitt), prewitt_mag],
    ["Sobel Gx", "Sobel Gy", "Sobel |G|",
     "Prewitt Gx", "Prewitt Gy", "Prewitt |G|"],
    ncols=3,
    normalize_display=True,
)
""",
        23: """flat = np.full((64, 64), 100, dtype=np.uint8)

kernel_cases = {
    "Tổng = 0.5": np.full((3, 3), 0.5 / 9, dtype=np.float32),
    "Tổng = 1.0": np.full((3, 3), 1.0 / 9, dtype=np.float32),
    "Tổng = 2.0": np.full((3, 3), 2.0 / 9, dtype=np.float32),
    "Tổng = 0.0": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32),
}

figs, titles = [flat], ["Vùng phẳng (c=100)"]
for label, kern in kernel_cases.items():
    out = linear_filters.apply_filter2d(flat, kern)
    val = int(np.unique(out)[0])
    print(f"{label}: sum={linear_filters.kernel_sum(kern):+.2f} -> output = {val:>3} (kỳ vọng {100 * linear_filters.kernel_sum(kern):.0f})")
    figs.append(out)
    titles.append(f"{label} -> {val}")

show_grid(figs, titles)

img_effects = [img_bgr] + [linear_filters.apply_filter2d(img_bgr, k) for k in kernel_cases.values()]
img_effect_titles = [
    "Ảnh gốc (màu)",
    "Σ=0.5 (tối hơn)",
    "Σ=1.0 (giữ nguyên)",
    "Σ=2.0 (sáng hơn)",
    "Σ=0.0 (chỉ còn viền)",
]
show_grid(img_effects, img_effect_titles, ncols=5)
""",
        25: """img_noisy = nonlinear_filters.add_salt_pepper_noise(img_bgr, amount=0.03, seed=42)
img_median = nonlinear_filters.apply_median_filter(img_noisy, 3)
img_bilateral = nonlinear_filters.apply_bilateral_filter(img_noisy, d=9, sigma_color=75.0, sigma_space=75.0)

show_grid(
    [img_bgr, img_noisy, img_median, img_bilateral],
    ["Gốc", "Nhiễu muối tiêu (3%)", "Median 3x3", "Bilateral"],
)
""",
    }

    for idx, code in updates.items():
        set_cell_source(nb["cells"][idx], code)

    nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Refactored Lab 02 notebook successfully.")


def refactor_lab03() -> None:
    nb_path = Path("notebooks/labs/lab_03.ipynb")
    nb = json.loads(nb_path.read_text(encoding="utf-8"))

    updates = {
        3: """import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from skimage import data, feature, util

root = next(
    folder
    for folder in (Path.cwd(), *Path.cwd().parents)
    if (folder / "pyproject.toml").is_file()
)
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.labs.lab03_canny import (
    canny_advanced,
    canny_core,
    canny_downstream,
    canny_metrics,
    canny_wrappers,
    visuals,
)

config = yaml.safe_load(
    (root / "configs/experiment/lab03_canny.yaml").read_text(encoding="utf-8")
)
params = config["default_params"]
settings = config["notebook"]
sigma = params["sigma"]
low, high = params["low_threshold"], params["high_threshold"]
aperture = params["aperture_size"]
l2_gradient = settings["l2_gradient"]
radius = int(np.ceil(settings["gaussian_radius_sigmas"] * sigma))
ksize = 2 * radius + 1
plt.rcParams.update({"figure.dpi": 80, "image.interpolation": "nearest"})
print("[+] Đã nạp thành công atomic modules từ src.labs.lab03_canny: canny_core, canny_wrappers, canny_advanced, canny_downstream, canny_metrics, visuals")
""",
        7: """edges_cv = canny_wrappers.run_opencv_canny(img, low, high, aperture_size=aperture, l2_gradient=l2_gradient)
edges_sk = canny_wrappers.run_skimage_canny(
    img,
    sigma=sigma,
    low_threshold=low / 255.0,
    high_threshold=high / 255.0,
)
show_grid([edges_cv, edges_sk], ["OpenCV Canny", "Scikit-image Canny"], ncols=2)
""",
        25: """from scipy.ndimage import binary_fill_holes

region_image = data.coins()
segment_params = config["segmentation"]
morph_res = canny_downstream.segment_via_morphology_and_holes(
    region_image,
    low_threshold=low,
    high_threshold=high,
    gaussian_ksize=(ksize, ksize),
    gaussian_sigma=sigma,
    morph_kernel_size=tuple(segment_params["morph_kernel_size"]),
    closing_iterations=segment_params["closing_iterations"],
)

region_edges = morph_res["edges"]
region_closed = morph_res["closed"]
region_mask = morph_res["mask"]
region_result = morph_res["segmented"]

show_grid(
    [region_image, region_edges, region_closed, region_mask, region_result],
    ["Ảnh đồng xu", "Canny", "Sau closing", "Tô đầy vùng kín", "Vùng ảnh được giữ"],
    ncols=3,
)
""",
        27: """circle_params = config["shape_detection"]["hough_circles"]
hough_res = canny_downstream.detect_hough_circles(
    region_image,
    blur_ksize=tuple(circle_params["blur_ksize"]),
    blur_sigma=circle_params["blur_sigma"],
    dp=circle_params["dp"],
    min_dist=circle_params["min_dist"],
    param1=circle_params["param1"],
    param2=circle_params["param2"],
    min_radius=circle_params["min_radius"],
    max_radius=circle_params["max_radius"],
)

circle_result = hough_res["annotated_bgr"]
circles = hough_res["circles"]
circle_count = hough_res["count"]

print(f"Số hình tròn phát hiện được: {circle_count}")
show_grid(
    [region_image, cv2.cvtColor(circle_result, cv2.COLOR_BGR2RGB)],
    ["Ảnh gốc", f"Hough: {circle_count} hình tròn"],
    ncols=2,
)
""",
    }

    for idx, code in updates.items():
        set_cell_source(nb["cells"][idx], code)

    nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Refactored Lab 03 notebook successfully.")


def refactor_lab04() -> None:
    nb_path = Path("notebooks/labs/lab_04.ipynb")
    nb = json.loads(nb_path.read_text(encoding="utf-8"))

    updates = {
        4: """from src.labs.lab04_wavelet.wavelet_pairs import load_pair_dataset
from src.labs import lab04_wavelet as wh
from src.labs.lab04_wavelet import wavelet_evaluation as evaluation
from src.labs.lab04_wavelet import visuals as visual

config = yaml.safe_load(
    (root / "configs/experiment/lab04_wavelet.yaml").read_text(encoding="utf-8")
)
params, settings = config["default_params"], config["notebook"]
dataset_dir = root / settings["dataset_dir"]
plt.rcParams.update({"figure.dpi": 90, "image.interpolation": "nearest"})
print("[+] Đã nạp thành công atomic modules từ src.labs.lab04_wavelet: wh, wavelet_pairs, wavelet_evaluation, visuals")
""",
    }

    for idx, code in updates.items():
        set_cell_source(nb["cells"][idx], code)

    nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Refactored Lab 04 notebook successfully.")


def refactor_lab05() -> None:
    nb_path = Path("notebooks/labs/lab_05.ipynb")
    nb = json.loads(nb_path.read_text(encoding="utf-8"))

    updates = {
        3: """import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import auc, roc_curve

CURRENT_DIR = Path.cwd().resolve()
PROJECT_ROOT = CURRENT_DIR
for parent in [CURRENT_DIR, *CURRENT_DIR.parents]:
    if (parent / "configs").exists() and (parent / "src").exists():
        PROJECT_ROOT = parent
        break

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.labs.lab05_facenet.face_database import FaceDatabase, l2_normalize_numpy
from src.labs.lab05_facenet.face_preprocessing import preprocess_face
from src.labs.lab05_facenet.facenet_extractor import FaceNetExtractor
from src.labs.lab05_facenet.mtcnn_detector import MTCNNDetector
from src.labs.lab05_facenet.similarity_matcher import SimilarityMatcher
from src.labs.lab05_facenet.face_registration import FaceRegistrationPipeline
from src.labs.lab05_facenet.webcam_recognition import RealtimeRecognitionPipeline
from src.labs.lab05_facenet.visuals import FaceVisualizer, RecognitionResult
from src.labs.lab05_facenet.pipeline import load_yaml_config
from src.common.device import resolve_device
from src.common.logger import setup_logger

%matplotlib inline
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

DATASET_CFG_PATH = PROJECT_ROOT / "configs/dataset/lab05_faces.yaml"
EXP_CFG_PATH = PROJECT_ROOT / "configs/experiment/lab05_facenet.yaml"

ds_cfg = load_yaml_config(DATASET_CFG_PATH)["dataset"]
exp_full_cfg = load_yaml_config(EXP_CFG_PATH)
exp_cfg = exp_full_cfg["experiment"]
mtcnn_cfg = exp_full_cfg["mtcnn"]
facenet_cfg = exp_full_cfg["facenet"]
matcher_cfg = exp_full_cfg["matcher"]

DEVICE = resolve_device(exp_cfg["device"])
OUTPUT_DIR = PROJECT_ROOT / exp_cfg["output_dir"]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"[+] Thư mục gốc dự án : {PROJECT_ROOT}")
print(f"[+] Thiết bị tính toán: {DEVICE} (PyTorch {torch.__version__}, OpenCV {cv2.__version__})")
print(f"[+] Kích thước khuôn mặt : {ds_cfg['image_size']}x{ds_cfg['image_size']} px")
print(f"[+] Ngưỡng Similarity : {matcher_cfg['similarity_threshold']} (Biên: {matcher_cfg['boundary_policy']})")
print("[+] Đã nạp thành công atomic modules từ src.labs.lab05_facenet: mtcnn, facenet, similarity, database, visuals")
""",
    }

    for idx, code in updates.items():
        set_cell_source(nb["cells"][idx], code)

    nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Refactored Lab 05 notebook successfully.")


def main() -> None:
    refactor_lab01()
    refactor_lab02()
    refactor_lab03()
    refactor_lab04()
    refactor_lab05()
    print("All 5 notebooks refactored into thin visual presentation layers successfully.")


if __name__ == "__main__":
    main()
