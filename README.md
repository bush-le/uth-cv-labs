# UTH Computer Vision Workspace (`uth-cv-labs`)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Tests: pytest](https://img.shields.io/badge/tests-passing-brightgreen.svg)](https://docs.pytest.org/)

---

## 📌 Giới thiệu Tổng quan (Overview)

Repository này chứa toàn bộ mã nguồn, dữ liệu thực nghiệm, pipeline phân tầng Clean Architecture và hệ thống báo cáo kỹ thuật tương tác đa phương tiện (Markdown & HTML) cho chuỗi bài thực hành **Thị giác Máy tính (Computer Vision)** tại **Trường Đại học Giao thông Vận tải TP.HCM (UTH)**.

### 📋 Tiến Độ Thực Hiện Các Bài Thực Hành (Labs Roadmap)

| Lab | Chủ đề Kỹ thuật | Trạng thái | Notebook | Pipeline Thực thi | Báo cáo Kỹ thuật | Unit Tests |
| :---: | :--- | :---: | :---: | :--- | :---: | :---: |
| **Lab 01** | **Xử lý ảnh cơ bản & Không gian màu** (OpenCV, Pillow, BGR/RGB/HSV/LAB, ROI, Annotations) | ✅ **100%** | [`lab_01.ipynb`](notebooks/labs/lab_01.ipynb) | [`lab01_pipeline.py`](src/pipelines/lab01_pipeline.py) | [Markdown](docs/labs/lab01/LAB_01_REPORT.md) · [HTML](docs/labs/lab01/LAB_01_REPORT.html) | 10 passed |
| **Lab 02** | **Toán tử điểm ảnh & Lọc không gian** (Brightness/Contrast/Negative, Gaussian, Sobel/Prewitt, Bilateral/Median) | ✅ **100%** | [`lab_02.ipynb`](notebooks/labs/lab_02.ipynb) | [`lab02_pipeline.py`](src/pipelines/lab02_pipeline.py) | [Markdown](docs/labs/lab02/LAB_02_REPORT.md) · [HTML](docs/labs/lab02/LAB_02_REPORT.html) | 17 passed |
| **Lab 03** | **Canny Edge Detector Chuyên sâu** (5-stage Canny, Noise Robustness, Pratt FOM, Hough Shapes, Segmentation) | ✅ **100%** | [`lab_03.ipynb`](notebooks/labs/lab_03.ipynb) | [`lab03_pipeline.py`](src/pipelines/lab03_pipeline.py) | [Markdown](docs/labs/lab03/LAB_03_REPORT.md) · [HTML](docs/labs/lab03/LAB_03_REPORT.html) | 16 passed |
| **Lab 04** | **So sánh tương đồng ảnh dùng Wavelet Hashing** (2D DWT Haar/db4, Quantization, ROC/PR Curves, Top-k Retrieval, MIH) | ✅ **100%** | [`lab_04.ipynb`](notebooks/labs/lab_04.ipynb) | [`lab04_pipeline.py`](src/pipelines/lab04_pipeline.py) | [Markdown](docs/labs/lab04/LAB_04_REPORT.md) · [HTML](docs/labs/lab04/LAB_04_REPORT.html) | 33 passed |
| **Lab 05** | **Nhận diện khuôn mặt thời gian thực** (MTCNN 3-Stage Cascaded CNN, FaceNet 512-D, Cosine Similarity &gt; 0.70, Webcam OSD + Batch Fallback) | ✅ **100%** | [`lab_05.ipynb`](notebooks/labs/lab_05.ipynb) | [`lab05_pipeline.py`](src/pipelines/lab05_pipeline.py) | [Markdown](docs/labs/lab05/LAB_05_DESCRIPTION.md) · [HTML](docs/labs/lab05/LAB_05_REPORT.html) | 17 passed |

- **Lab 03 Report Tool**: Chạy `python scripts/lab03/render_report.py` để tạo lại báo cáo HTML theo sơ đồ thực nghiệm Canny.
- **Lab 04 Report Tool**: Chạy `python scripts/lab04/run_notebook.py` để thực thi notebook và `python scripts/lab04/render_report.py` để tạo báo cáo HTML theo sơ đồ luồng Wavelet Hash.
- **Lab 05 Report Tool**: Chạy `python scripts/lab05/render_report.py` để tạo báo cáo HTML theo sơ đồ MTCNN & FaceNet.
- **Master Presentation Deck**: Chạy `python scripts/render_visual_reports.py --lab all` để kết xuất toàn bộ slide presentation 100vh cho Labs 01-05 và báo cáo tổng kết.

---

## 🗂️ Cấu trúc Thư mục Tổng thể (Architecture Layout)

```text
.
├── agents/                            # Hệ thống quy trình kỹ thuật & Second Brain cho AI Agent
│   ├── README.md                      # Cổng nhận thức trung tâm & bản đồ điều phối
│   ├── WORKFLOW_OVERVIEW.md           # Tổng quan luồng thực thi khép kín
│   ├── AI_FIRST_LIFECYCLE.md          # 9 giai đoạn phát triển chuẩn
│   ├── PIPELINE_SEPARATION.md         # Phân tách 5 pipelines Clean Architecture
│   ├── AI_REFERENCE.md                # Bất biến CV/DL, chuẩn PEP 484 & docstrings
│   └── CODEBASE_AUDIT.md              # Kiểm toán mã nguồn & an toàn CV
├── configs/                           # Quản lý cấu hình bằng Hydra / YAML
│   ├── dataset/
│   │   └── lab04_pairs.yaml           # Cấu hình bộ cặp ảnh đánh giá Lab 04
│   └── experiment/
│       ├── lab03_canny.yaml           # Cấu hình siêu tham số Lab 03 Canny
│       └── lab04_wavelet.yaml         # Cấu hình siêu tham số Lab 04 Wavelet Hashing
├── data/                              # Dữ liệu phân tầng
│   └── samples/                       # Ảnh mẫu kiểm chuẩn (Self-contained & Git-tracked)
│       ├── image-2.jpg                # Ảnh mẫu cho Lab 01 & Lab 02
│       ├── sample_traffic.jpg         # Ảnh giao thông cho Lab 01
│       ├── lab03/                     # 6 ảnh mẫu kiểm chuẩn đặc thù cho Lab 03
│       ├── lab04/                     # Dữ liệu ảnh Chihuahua/Muffin & Mèo/Kem
│       └── lab05/                     # Dữ liệu ảnh khuôn mặt mẫu & faces_db
├── docs/                              # Tài liệu & Báo cáo lab
│   ├── assets/                        # Sơ đồ kiến trúc & hình ảnh trích xuất (lab03, lab04, lab05)
│   └── labs/                          # Báo cáo kỹ thuật chi tiết của từng bài Lab
│       ├── lab01/                     # Markdown & HTML Report Lab 01
│       ├── lab02/                     # Markdown & HTML Report Lab 02
│       ├── lab03/                     # Markdown & HTML Report Lab 03
│       ├── lab04/                     # Markdown & HTML Report Lab 04
│       └── lab05/                     # Markdown & HTML Report Lab 05
├── notebooks/                         # Jupyter Notebooks tương tác trực tiếp
│   └── labs/
│       ├── lab_01.ipynb
│       ├── lab_02.ipynb
│       ├── lab_03.ipynb
│       ├── lab_04.ipynb
│       └── lab_05.ipynb
├── experiments/                       # Nơi lưu trữ artifacts (Ignored by Git)
│   ├── logs/                          # Metrics summary JSON
│   └── predictions/                   # Biểu đồ và ảnh đầu ra của từng lab
│       ├── lab01/
│       ├── lab02/
│       ├── lab03/
│       ├── lab04/
│       └── lab05/
├── requirements/                      # Quản lý dependencies theo môi trường
│   ├── base.txt                       # OpenCV, NumPy, Matplotlib, Pillow, Scikit-Image, SciPy, PyWavelets
│   └── dev.txt                        # Pytest, Ruff, Pre-commit
├── scripts/                           # Shell scripts tự động hóa
│   ├── run_lab.sh                     # Script chạy lab tự động (1, 2, 3, 4, 5, all)
│   ├── lab04/                         # Scripts phụ trợ cho Lab 04
│   └── lab05/                         # Scripts chạy webcam và kết xuất báo cáo Lab 05
├── src/                               # Package mã nguồn chính (Lab-Centric Atomic Architecture)
│   ├── common/                        # Tiện ích dùng chung (device, logger, visualizers, metrics)
│   ├── labs/                          # Các gói thực hành tự trị & độc lập theo từng bài lab
│   │   ├── lab01_basics/              # Lab 1: io, color, geometry, drawing, visuals, pipeline
│   │   ├── lab02_filters/             # Lab 2: point_ops, linear_filters, nonlinear_filters, edge_filters, visuals, pipeline
│   │   ├── lab03_canny/               # Lab 3: canny_core, canny_wrappers, canny_advanced, canny_downstream, canny_metrics, visuals, pipeline
│   │   ├── lab04_wavelet/             # Lab 4: wavelet_core, wavelet_distance, wavelet_distortions, wavelet_pairs, wavelet_evaluation, visuals, pipeline
│   │   └── lab05_facenet/             # Lab 5: mtcnn_detector, facenet_extractor, similarity_matcher, face_database, face_preprocessing, visuals, pipeline
│   ├── data/                          # Shims tương thích ngược (Transforms, Filters, Face Database)
│   ├── models/                        # Shims tương thích ngược (Canny, Wavelet Hash, MTCNN, FaceNet)
│   └── pipelines/                     # Shims CLI runners tương thích ngược
├── tests/                             # Bộ kiểm thử unit tests (119 test cases)
└── README.md
```

---

## 🏛️ Kiến Trúc Module Hóa Tự Trị (Lab-Centric Atomic Architecture)

Toàn bộ codebase của dự án đã được tái cấu trúc triệt để theo kiến trúc **Lab-Centric Atomic Modules**:

1. **Phân tách Atomic Module rõ ràng**: Mọi logic tính toán, giải thuật xử lý ảnh, biến đổi ma trận và trích xuất đặc trưng được tách biệt thành các file Python độc lập trong `src/labs/lab0X_.../`:
   - `lab01_basics`: `io.py`, `color.py`, `geometry.py`, `drawing.py`, `visuals.py`, `pipeline.py`.
   - `lab02_filters`: `filter_utils.py`, `point_ops.py`, `linear_filters.py`, `nonlinear_filters.py`, `edge_filters.py`, `visuals.py`, `pipeline.py`.
   - `lab03_canny`: `canny_core.py`, `canny_wrappers.py`, `canny_advanced.py`, `canny_downstream.py`, `canny_metrics.py`, `visuals.py`, `pipeline.py`.
   - `lab04_wavelet`: `wavelet_core.py`, `wavelet_distance.py`, `wavelet_distortions.py`, `wavelet_pairs.py`, `wavelet_evaluation.py`, `visuals.py`, `pipeline.py`.
   - `lab05_facenet`: `mtcnn_detector.py`, `facenet_extractor.py`, `similarity_matcher.py`, `face_database.py`, `face_preprocessing.py`, `face_registration.py`, `webcam_recognition.py`, `visuals.py`, `pipeline.py`.
2. **Jupyter Notebooks là Tầng Trình Diễn Mỏng (Thin Presentation Layer)**: Toàn bộ code trong `notebooks/labs/lab_01.ipynb` đến `lab_05.ipynb` đã được tinh gọn, không chứa code thuật toán tự viết dài dòng mà chỉ import module từ `src.labs` và gọi hàm hiển thị từ `visuals.py`. Các notebook ban đầu được lưu trữ an toàn tại `notebooks/archive/`.
3. **Tương thích ngược 100% (Backward Compatibility Shims)**: Các module tại `src/data/`, `src/models/`, `src/pipelines/` và `src/common/lab05_utils.py` được duy trì dưới dạng shim re-export, đảm bảo 119 unit test và các công cụ xuất báo cáo HTML (`render_report.py`, `run_notebook.py`) hoạt động hoàn hảo không bị gián đoạn.
4. **Chuẩn Mực Mã Nguồn**: 100% NumPy vectorization (không lặp pixel bằng for-loop Python), PEP 484 Type Hints đầy đủ, Docstrings chuẩn Google-style, logging có cấu trúc qua `src/common/logger.py`.

---

## 🚀 Hướng dẫn Cài đặt & Sử dụng (Quickstart)

### 1. Khởi tạo Môi trường

```bash
# 1. Tạo và kích hoạt virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Cài đặt toàn bộ dependencies
pip install --upgrade pip
pip install -r requirements/base.txt
pip install -r requirements/dev.txt
```

### 2. Chạy Các Bài Thực Hành

```bash
# Chạy từng bài lab đơn lẻ:
bash scripts/run_lab.sh 1
bash scripts/run_lab.sh 2
bash scripts/run_lab.sh 3
bash scripts/run_lab.sh 4
bash scripts/run_lab.sh 5

# Chạy toàn bộ tất cả bài lab (Lab 1 đến Lab 5):
bash scripts/run_lab.sh all
```

### 3. Kiểm thử & Kiểm toán Mã nguồn

```bash
# Chạy toàn bộ unit tests
pytest tests/

# Kiểm tra chất lượng mã nguồn với linter
ruff check src tests scripts
```
