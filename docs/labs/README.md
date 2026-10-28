# UTH COMPUTER VISION - LABORATORIES DIRECTORY
## Danh Mục Bài Thực Hành Thị Giác Máy Tính

Thư mục `docs/labs/` quản lý toàn bộ tài liệu hướng dẫn, đề bài và báo cáo kỹ thuật cho các bài thực hành Thị giác Máy tính (Computer Vision) tại **Trường Đại học Giao thông Vận tải TP.HCM (UTH)**.

---

## 🗂️ Quy Chuẩn Cấu Trúc Khép Kín (Zero-Conflict Multi-Lab Architecture)

Để đảm bảo các bài thực hành từ **Lab 1 đến Lab 4** hoàn toàn độc lập, không gây xung đột (conflict) mã nguồn, tài liệu hoặc dữ liệu đầu ra, toàn bộ hệ thống tuân thủ nghiêm ngặt quy ước phân tầng (`labXX`):

```text
docs/labs/
├── README.md               # 📍 Danh mục tổng quan toàn bộ các bài lab (tệp này)
├── TEMPLATE.md             # 📋 Mẫu báo cáo chuẩn cho các bài thực hành
│
├── lab01/                  # 🟢 LAB 1: XỬ LÝ ẢNH CƠ BẢN VỚI OPENCV & PILLOW
│   ├── LAB_01_DESCRIPTION.md # Đặc tả yêu cầu đề bài
│   ├── LAB_01_REPORT.md    # Báo cáo kỹ thuật chi tiết
│   └── LAB_01_REPORT.html  # Báo cáo trực quan HTML tương tác
│
├── lab02/                  # 🟢 LAB 2: TOÁN TỬ ĐIỂM ẢNH & LỌC TUYẾN TÍNH / PHI TUYẾN
│   ├── LAB_02_DESCRIPTION.md
│   ├── LAB_02_REPORT.md
│   └── LAB_02_REPORT.html
│
├── lab03/                  # 🟢 LAB 3: CANNY EDGE DETECTOR & PHÂN ĐOẠN HÌNH THÁI HỌC
│   ├── LAB_03_DESCRIPTION.md
│   ├── LAB_03_REPORT.md
│   └── LAB_03_REPORT.html
│
└── lab04/                  # 🟢 LAB 4: SO SÁNH TƯƠNG ĐỒNG ẢNH VỚI WAVELET HASHING
    ├── LAB_04_DESCRIPTION.md
    ├── LAB_04_REPORT.md
    └── LAB_04_REPORT.html
```

---

## 📋 Danh Sách Các Bài Thực Hành (Labs Index)

| Lab | Tên Chủ Đề Thực Hành | Trạng Thái | Thư Mục Tài Liệu | Pipeline Điều Phối | Test Suite |
| :---: | :--- | :---: | :--- | :--- | :--- |
| **Lab 1** | **Xử lý ảnh cơ bản với OpenCV & Pillow** | ✅ **Hoàn thành** (10 Tests) | [`lab01/`](lab01/LAB_01_REPORT.md) · [`HTML`](lab01/LAB_01_REPORT.html) | [`lab01_pipeline.py`](../../src/pipelines/lab01_pipeline.py) | [`test_transforms.py`](../../tests/test_transforms.py) |
| **Lab 2** | **Toán tử điểm ảnh & Lọc không gian (Tuyến tính / Phi tuyến)** | ✅ **Hoàn thành** (17 Tests) | [`lab02/`](lab02/LAB_02_REPORT.md) · [`HTML`](lab02/LAB_02_REPORT.html) | [`lab02_pipeline.py`](../../src/pipelines/lab02_pipeline.py) | [`test_filters.py`](../../tests/test_filters.py) |
| **Lab 3** | **Canny Edge Detector & Phân đoạn hình thái học** | ✅ **Hoàn thành** (16 Tests) | [`lab03/`](lab03/LAB_03_REPORT.md) · [`HTML`](lab03/LAB_03_REPORT.html) | [`lab03_pipeline.py`](../../src/pipelines/lab03_pipeline.py) | [`test_canny.py`](../../tests/test_canny.py) |
| **Lab 4** | **So sánh tương đồng ảnh dùng Wavelet Hashing (DWT 2D)** | ✅ **Hoàn thành** (8 Tests) | [`lab04/`](lab04/LAB_04_DESCRIPTION.md) · [`HTML`](lab04/LAB_04_REPORT.html) | [`lab04_pipeline.py`](../../src/pipelines/lab04_pipeline.py) | [`test_wavelet_hash.py`](../../tests/test_wavelet_hash.py) |
| **Labs 02–04** | **Báo Cáo Tổng Hợp Tương Tác Trực Quan (Gộp 3 Lab)** | ✅ **Hoàn thành** (58 Tests) | [`LABS_02_03_04_REPORT.html`](LABS_02_03_04_REPORT.html) | [`render_unified_report.py`](../../scripts/render_unified_report.py) | `58/58 passed` |

---

## 🚀 Chạy Nhanh Toàn Bộ Labs (Quickstart)

```bash
# Chạy từng bài lab cụ thể
bash scripts/run_lab.sh 1
bash scripts/run_lab.sh 2
bash scripts/run_lab.sh 3
bash scripts/run_lab.sh 4

# Hoặc chạy toàn bộ tất cả bài lab cùng lúc
bash scripts/run_lab.sh all

# Chạy toàn bộ 51 unit tests
pytest tests/
```
