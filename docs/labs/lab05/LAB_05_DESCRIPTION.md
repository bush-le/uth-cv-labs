# BÀI THỰC HÀNH LAB 05: NHẬN DIỆN KHUÔN MẶT THỜI GIAN THỰC VỚI FACENET & MTCNN TRÊN WEBCAM

> **Học phần:** Thị giác Máy tính (Computer Vision)  
> **Lab:** 05 · **Chương:** 3 (Part 2)  
> **Chủ đề:** Real-Time Face Recognition using MTCNN & FaceNet Deep Embeddings on Webcam Stream  
> **Giảng viên hướng dẫn:** TS. Nguyễn Thị Khánh Tiên

---

## 1. Giới thiệu chung
Bài thực hành số 05 tập trung vào việc xây dựng một hệ thống nhận diện khuôn mặt thời gian thực (Real-time Face Recognition) kết hợp hai mạng học sâu chuyên biệt theo nguyên lý Clean Architecture:
- **MTCNN (Multi-task Cascaded Convolutional Networks):** Phát hiện vị trí khuôn mặt (Face Detection), căn chỉnh (Alignment) và trích xuất vùng mặt trên từng khung hình video.
- **FaceNet (`InceptionResnetV1`):** Ánh xạ ảnh khuôn mặt sang không gian vector đặc trưng 512 chiều (Face Embedding) chuẩn hóa $L_2$ sao cho các khuôn mặt của cùng một người có khoảng cách tối thiểu và độ tương đồng Cosine cao nhất.

---

## 2. Yêu cầu kỹ thuật cốt lõi
1. **Thu thập luồng video qua Webcam:** Sử dụng thư viện OpenCV (`cv2.VideoCapture`) để đọc luồng khung hình thời gian thực với cơ chế quản lý tài nguyên an toàn (`WebcamManager` context manager).
2. **Phát hiện khuôn mặt bằng MTCNN:** Xác định tọa độ Bounding Box của một hoặc nhiều khuôn mặt trên mỗi khung hình với ngưỡng tin cậy xác định.
3. **Trích xuất đặc trưng & Đối sánh với FaceNet:**
   - Chuẩn hóa ảnh khuôn mặt về kích thước $160 \times 160$ và phân phối pixel $[-1, 1]$.
   - Trích xuất vector đặc trưng 512 chiều chuẩn hóa $L_2$ qua mạng `InceptionResnetV1` (pretrained VGGFace2).
   - Tính toán độ tương đồng **Cosine Similarity**:
     $$\text{Cosine Similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$
4. **Quy tắc phân ngưỡng quyết định (Threshold Rule):**
   - Nếu $\text{similarity} > 0.700000$: Hiển thị trạng thái `"Matched"` kèm tên danh tính đã đăng ký.
   - Nếu $\text{similarity} \le 0.700000$: Hiển thị trạng thái `"Unknown"`.
5. **Trực quan hóa thời gian thực (OSD Overlay):**
   - Bounding Box màu xanh lá cho danh tính `"Matched"`.
   - Bounding Box màu đỏ cảnh báo cho đối tượng `"Unknown"`.
   - Bảng thông số HUD trên khung hình: FPS, số khuôn mặt, số danh tính trong DB, ngưỡng quyết định.

---

## 3. Cấu trúc tệp tin bàn giao

```text
configs/
├── dataset/lab05_faces.yaml           # Cấu hình đường dẫn thư mục ảnh đăng ký & định dạng
└── experiment/lab05_facenet.yaml      # Siêu tham số MTCNN, FaceNet, ngưỡng 0.7, camera device

src/
├── common/lab05_utils.py              # Tiện ích cấu hình, FPSMeter, FaceVisualizer
├── data/
│   ├── face_preprocessing.py          # Tiền xử lý, crop, chuẩn hóa 160x160 whitening
│   └── face_database.py               # Quản trị cơ sở dữ liệu vector đặc trưng .npz
├── models/
│   ├── deep/
│   │   ├── mtcnn_detector.py          # Wrapper phát hiện khuôn mặt MTCNN
│   │   └── facenet_extractor.py       # Wrapper trích xuất embedding InceptionResnetV1
│   └── similarity_matcher.py          # Tính toán Cosine Similarity và phân ngưỡng 0.7
└── pipelines/
    ├── lab05_pipeline.py              # Master pipeline điều phối tương thích đa chế độ
    ├── lab05_face_registration.py     # Pipeline quét thư mục / chụp webcam đăng ký danh tính
    └── lab05_webcam_recognition.py    # Pipeline vòng lặp nhận diện thời gian thực qua webcam

scripts/lab05/
├── download_lab05_weights.py          # Tải và xác thực trọng số pretrained
├── register_faces.py                  # CLI đăng ký danh tính khuôn mặt
├── run_lab05_webcam.py                # Chạy ứng dụng webcam nhận diện trực tiếp
└── generate_lab05_artifacts.py        # Tạo ảnh minh chứng nghiệm thu tự động

tests/
├── test_lab05_smoke.py                # Kiểm thử nhanh khởi tạo và nạp cấu hình
└── test_lab05_unit.py                 # Kiểm thử đơn vị các hàm tiền xử lý, embedding, ngưỡng 0.7
```

---

## 4. Tiêu chí Nghiệm thu (Acceptance Criteria)

| ID | Tiêu chí | Điều kiện nghiệm thu | Trạng thái |
| :--- | :--- | :--- | :---: |
| **AC01** | Architecture Compliance | Phân tầng theo chuẩn Clean Architecture (`configs/`, `src/`, `scripts/`, `tests/`) | **PASS** |
| **AC02** | Webcam Acquisition | Quản lý vòng đời `cv2.VideoCapture` an toàn bằng context manager | **PASS** |
| **AC03** | MTCNN Detection | Phát hiện chính xác BBox khuôn mặt và kẹp biên ảnh tránh out-of-bounds | **PASS** |
| **AC04** | FaceNet Embedding | Trích xuất vector 512 chiều chuẩn hóa $L_2$ từ ảnh $160 \times 160$ | **PASS** |
| **AC05** | Face Database | Lưu trữ và nạp cơ sở dữ liệu danh tính định dạng `.npz` | **PASS** |
| **AC06** | Cosine Similarity | Tính chính xác tích vô hướng các vector đã chuẩn hóa độ dài đơn vị | **PASS** |
| **AC07** | Strict Threshold 0.7 | Áp dụng nghiêm ngặt $\text{sim} > 0.7$ là `Matched`, $\text{sim} \le 0.7$ là `Unknown` | **PASS** |
| **AC08** | Real-Time OSD Overlay | Vẽ bounding box xanh lá / đỏ, nhãn danh tính, FPS và similarity score | **PASS** |
| **AC09** | Adaptive Pipeline | Hỗ trợ tự động fallback khi không có camera để phục vụ kiểm thử tự động | **PASS** |
| **AC10** | Testing & QA | Đầy đủ smoke tests và unit tests đạt chuẩn, 100% pass | **PASS** |
| **AC11** | Documentation | Đầy đủ đặc tả, báo cáo Markdown và báo cáo HTML tương tác theo flow | **PASS** |

---

## 5. Hướng dẫn Thực thi (Quickstart)

### 5.1. Chạy Pipeline Tự động
```bash
# Chạy pipeline Lab 05 qua script điều phối tổng thể
bash scripts/run_lab.sh 5

# Hoặc thực thi trực tiếp qua Python module
python -m src.pipelines.lab05_pipeline --mode auto
```

### 5.2. Đăng ký Khuôn mặt Mới
```bash
# Đăng ký qua quét thư mục data/raw/lab05/faces/
python scripts/lab05/register_faces.py --mode folder

# Đăng ký trực tiếp qua webcam
python scripts/lab05/register_faces.py --mode capture --identity "Ten_Cua_Ban" --num-samples 5
```

### 5.3. Khởi chạy Ứng dụng Webcam Thời gian Thực
```bash
python scripts/lab05/run_lab05_webcam.py
# Phím tắt trong cửa sổ webcam:
#   [Q] : Thoát ứng dụng an toàn
#   [S] : Lưu ảnh chụp thực nghiệm vào experiments/predictions/lab05/
```

### 5.4. Kiểm thử Đơn vị
```bash
pytest tests/test_lab05_smoke.py tests/test_lab05_unit.py -v
```
