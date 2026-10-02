# UTH COMPUTER VISION - LABORATORIES DIRECTORY
## Danh Mục Bài Thực Hành Thị Giác Máy Tính

Thư mục `docs/labs/` quản lý toàn bộ tài liệu hướng dẫn, đề bài và báo cáo kỹ thuật cho các bài thực hành Thị giác Máy tính (Computer Vision).

---

## 🗂️ Quy Chuẩn Cấu Trúc Khép Kín (Zero-Conflict Multi-Lab Architecture)

Để đảm bảo các bài thực hành từ **Lab 1, Lab 2, Lab 3...** hoàn toàn độc lập, không gây xung đột (conflict) mã nguồn, tài liệu hoặc dữ liệu đầu ra, toàn bộ hệ thống tuân thủ nghiêm ngặt quy ước đặt tên 2 chữ số (`labXX`):

```text
docs/labs/
├── README.md               # 📍 Danh mục tổng quan toàn bộ các bài lab (tệp này)
├── TEMPLATE.md             # 📋 Mẫu báo cáo chuẩn cho các bài thực hành mới
│
├── lab01/                  # 🟢 LAB 1: XỬ LÝ ẢNH CƠ BẢN VỚI OPENCV & PILLOW
│   ├── LAB_01_DESCRIPTION.md # Đặc tả yêu cầu đề bài
│   ├── LAB_01_REPORT.md    # Báo cáo kỹ thuật chi tiết
│   └── LAB_01_REPORT.html  # Báo cáo trực quan HTML tương tác
│
├── lab02/                  # 🟡 LAB 2: TIỀN XỬ LÝ & LỌC ẢNH KHÔNG GIAN (Sắp tới)
│   ├── LAB_02_DESCRIPTION.md
│   ├── LAB_02_REPORT.md
│   └── LAB_02_REPORT.html
│
└── lab03/                  # 🔵 LAB 3: PHÁT HIỆN BIÊN & BIẾN ĐỔI HÌNH THÁI (Sắp tới)
    └── ...
```

---

## 📋 Danh Sách Các Bài Thực Hành (Labs Index)

| Lab | Tên Chủ Đề Thực Hành | Trạng Thái | Thư Mục Tài Liệu | Pipeline Điều Phối | Notebook Thực Hành |
| :---: | :--- | :---: | :--- | :--- | :--- |
| **Lab 1** | **Xử lý ảnh cơ bản với OpenCV & Pillow** | ✅ **Hoàn thành** (10/10 Tests) | [`lab01/`](lab01/LAB_01_REPORT.md) | [`lab01_pipeline.py`](../../src/pipelines/lab01_pipeline.py) | [`lab_01.ipynb`](../../notebooks/labs/lab_01.ipynb) |
| **Lab 2** | **Lọc ảnh không gian & Nâng cao chất lượng ảnh** | ⏳ *Sẵn sàng khởi tạo* | `lab02/` | `lab02_pipeline.py` | `lab_02.ipynb` |
| **Lab 3** | **Biến đổi hình thái học & Tách biên (Edge Detection)** | ⏳ *Sẵn sàng khởi tạo* | `lab03/` | `lab03_pipeline.py` | `lab_03.ipynb` |

---

## 🛠️ Quy Trình Khởi Tạo Lab Mới Không Gây Xung Đột

Khi bắt đầu một bài lab mới (ví dụ **Lab 2**):
1. **Tạo thư mục tài liệu riêng**: Tạo `docs/labs/lab02/`.
2. **Khởi tạo tài liệu từ mẫu**: Sao chép `docs/labs/TEMPLATE.md` vào `docs/labs/lab02/LAB_02_REPORT.md`.
3. **Tạo Pipeline riêng**: Tạo `src/pipelines/lab02_pipeline.py`.
4. **Tạo Notebook riêng**: Sao chép `notebooks/labs/template.ipynb` thành `notebooks/labs/lab_02.ipynb`.
5. **Thư mục Artifacts riêng**: Cấu hình pipeline lưu kết quả vào `experiments/predictions/lab02/`.
6. **Thêm case vào Runner**: Cập nhật case `2|"02"|"lab2"|"lab02"` trong [`scripts/run_lab.sh`](../../scripts/run_lab.sh).
