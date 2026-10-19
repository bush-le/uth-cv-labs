# UTH Computer Vision Workspace (`uth-cv-labs`)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Tests: pytest](https://img.shields.io/badge/tests-51%20passed-brightgreen.svg)](https://docs.pytest.org/)

---

## 📌 Giới thiệu Tổng quan (Overview)

Repository này chứa toàn bộ mã nguồn, dữ liệu thực nghiệm, pipeline phân tầng Clean Architecture và hệ thống báo cáo kỹ thuật tương tác đa phương tiện (Markdown & HTML) cho chuỗi bài thực hành **Thị giác Máy tính (Computer Vision)** tại **Trường Đại học Giao thông Vận tải TP.HCM (UTH)**.

### 📋 Tiến Độ Thực Hiện Các Bài Thực Hành (Labs Roadmap)

| Lab | Chủ đề Kỹ thuật | Trạng thái | Pipeline Thực thi | Báo cáo Kỹ thuật | Unit Tests |
| :---: | :--- | :---: | :--- | :---: | :---: |
| **Lab 01** | **Xử lý ảnh cơ bản & Không gian màu** (OpenCV, Pillow, BGR/RGB/HSV/LAB, ROI, Annotations) | ✅ **100%** | [`lab01_pipeline.py`](src/pipelines/lab01_pipeline.py) | [Markdown](docs/labs/lab01/LAB_01_REPORT.md) · [HTML](docs/labs/lab01/LAB_01_REPORT.html) | 10 passed |
| **Lab 02** | **Toán tử điểm ảnh & Lọc không gian** (Brightness/Contrast/Negative, Gaussian, Sobel/Prewitt, Bilateral/Median) | ✅ **100%** | [`lab02_pipeline.py`](src/pipelines/lab02_pipeline.py) | [Markdown](docs/labs/lab02/LAB_02_REPORT.md) · [HTML](docs/labs/lab02/LAB_02_REPORT.html) | 17 passed |
| **Lab 03** | **Canny Edge Detector Chuyên sâu** (5-stage Canny, Noise Robustness, Pratt FOM, Hough Shapes, Di Zenzo Tensor) | ✅ **100%** | [`lab03_pipeline.py`](src/pipelines/lab03_pipeline.py) | [Markdown](docs/labs/lab03/LAB_03_REPORT.md) · [HTML](docs/labs/lab03/LAB_03_REPORT.html) | 16 passed |
| **Lab 04** | **So sánh tương đồng ảnh dùng Wavelet Hashing** (2D DWT Haar/db4, Quantization, ROC/PR Curves, Top-k Retrieval, MIH) | ✅ **100%** | [`lab04_pipeline.py`](src/pipelines/lab04_pipeline.py) | [Markdown](docs/labs/lab04/LAB_04_REPORT.md) · [HTML](docs/labs/lab04/LAB_04_REPORT.html) | 8 passed |

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
│   ├── experiment/
│   │   ├── lab03_canny.yaml           # Cấu hình siêu tham số Lab 03 Canny
│   │   └── lab04_wavelet.yaml         # Cấu hình siêu tham số Lab 04 Wavelet Hashing
│   └── ...
├── data/                              # Dữ liệu phân tầng
│   └── samples/                       # Ảnh mẫu kiểm chuẩn (Self-contained & Git-tracked)
│       ├── image-2.jpg                # Ảnh mẫu cho Lab 01 & Lab 02
│       ├── sample_traffic.jpg         # Ảnh giao thông cho Lab 01
│       ├── lab03/                     # 9 ảnh mẫu kiểm chuẩn đặc thù cho Lab 03
│       └── lab04/                     # 6 ảnh mẫu danh mục chuẩn cho Lab 04
├── docs/                              # Tài liệu & Báo cáo lab
│   └── labs/                          # Báo cáo kỹ thuật chi tiết của từng bài Lab
│       ├── lab01/                     # Markdown & HTML Report Lab 01
│       ├── lab02/                     # Markdown & HTML Report Lab 02
│       ├── lab03/                     # Markdown & HTML Report Lab 03
│       └── lab04/                     # Markdown & HTML Report Lab 04
├── experiments/                       # Nơi lưu trữ artifacts (Ignored by Git)
│   ├── logs/                          # Metrics summary JSON
│   └── predictions/                   # Biểu đồ và ảnh đầu ra của từng lab
│       ├── lab01/
│       ├── lab02/
│       ├── lab03/
│       └── lab04/
├── requirements/                      # Quản lý dependencies theo môi trường
│   ├── base.txt                       # OpenCV, NumPy, Matplotlib, Pillow, Scikit-Image, SciPy, PyWavelets
│   └── dev.txt                        # Pytest, Ruff, Pre-commit
├── scripts/                           # Shell scripts tự động hóa
│   └── run_lab.sh                     # Script chạy lab tự động (1, 2, 3, 4, all)
├── src/                               # Package mã nguồn chính (Clean Architecture)
│   ├── common/                        # Tiện ích dùng chung
│   │   ├── logger.py                  # Logger chuẩn hóa, không dùng print()
│   │   └── visualizer.py              # Suite trực quan hóa đa phương tiện
│   ├── data/
│   │   ├── transforms.py              # Xử lý ảnh không gian, kênh màu, annotations
│   │   └── filters.py                 # Bộ lọc tuyến tính và phi tuyến tính
│   ├── models/classical/
│   │   ├── canny.py                   # Canny 5-stage, Di Zenzo tensor, Pratt FOM
│   │   └── wavelet_hash.py            # DWT 2D, Wavelet Hashing, Hamming distance, ROC
│   └── pipelines/                     # Master pipelines điều phối
│       ├── lab01_pipeline.py
│       ├── lab02_pipeline.py
│       ├── lab03_pipeline.py
│       └── lab04_pipeline.py
├── tests/                             # Bộ kiểm thử 51 unit tests
│   ├── test_transforms.py
│   ├── test_filters.py
│   ├── test_canny.py
│   └── test_wavelet_hash.py
└── README.md
```

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

# Chạy toàn bộ tất cả bài lab (Lab 1 đến Lab 4):
bash scripts/run_lab.sh all
```

### 3. Kiểm thử & Kiểm toán Mã nguồn

```bash
# Chạy toàn bộ 51 unit tests
pytest tests/

# Kiểm tra chất lượng mã nguồn với linter
ruff check src tests
```
