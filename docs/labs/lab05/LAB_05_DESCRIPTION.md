# LAB 05: NHẬN DIỆN KHUÔN MẶT THỜI GIAN THỰC VỚI FACENET & MTCNN TRÊN WEBCAM

## 1. Giới thiệu chung
Bài thực hành số 05 thuộc học phần Thị giác máy tính (Chương 3 - Part 2: Phát hiện đặc trưng và đối sánh ảnh) tập trung vào việc xây dựng một hệ thống nhận diện khuôn mặt thời gian thực (Real-time Face Recognition) kết hợp hai mạng học sâu chuyên biệt:
- **MTCNN (Multi-task Cascaded Convolutional Networks):** Phát hiện vị trí khuôn mặt (Face Detection) và trích xuất vùng mặt trên từng khung hình video.
- **FaceNet (InceptionResnetV1):** Ánh xạ ảnh khuôn mặt sang không gian vector đặc trưng 512 chiều (Face Embedding) sao cho các khuôn mặt của cùng một người có độ tương đồng cao nhất.

## 2. Yêu cầu bắt buộc của bài thực hành
1. **Thu thập video qua Webcam:** Sử dụng thư viện OpenCV (`cv2.VideoCapture`) để đọc luồng khung hình thời gian thực và giải phóng tài nguyên an toàn khi kết thúc.
2. **Phát hiện khuôn mặt bằng MTCNN:** Xác định tọa độ bounding box của một hoặc nhiều khuôn mặt trên mỗi khung hình.
3. **Trích xuất và đối sánh đặc trưng bằng FaceNet:** Chuẩn hóa ảnh khuôn mặt về kích thước `160x160`, trích xuất vector đặc trưng 512 chiều và tính độ tương đồng **Cosine Similarity** với cơ sở dữ liệu đã đăng ký.
4. **Quy tắc phân ngưỡng (Thresholding):**
   - Nếu `similarity > 0.7`: Hiển thị `"Matched"` (kèm tên danh tính).
   - Nếu `similarity < 0.7` (hoặc `= 0.7`): Hiển thị `"Unknown"`.

## 3. Cấu trúc tệp tin bàn giao
- Cấu hình: `configs/dataset/lab05_faces.yaml`, `configs/experiment/lab05_facenet.yaml`
- Mã nguồn lõi: `src/common/lab05_utils.py`, `src/data/face_preprocessing.py`, `src/data/face_database.py`, `src/models/deep/mtcnn_detector.py`, `src/models/deep/facenet_extractor.py`, `src/models/similarity_matcher.py`, `src/pipelines/lab05_face_registration.py`, `src/pipelines/lab05_webcam_recognition.py`
- Kịch bản thực thi: `scripts/lab05/download_lab05_weights.py`, `scripts/lab05/register_faces.py`, `scripts/lab05/run_lab05_webcam.py`, `scripts/lab05/generate_lab05_artifacts.py`
- Kiểm thử: `tests/test_lab05_smoke.py`, `tests/test_lab05_unit.py`

---

## I. ACCEPTANCE CHECKLIST (BẢNG ĐỐI CHIẾU TIÊU CHÍ NGHIỆM THU)

Tuân thủ quy tắc trung thực kỹ thuật: chỉ đánh dấu **PASS** cho các hạng mục đã hoàn tất và xác minh trong phạm vi tạo mã/kiến trúc; đánh dấu **NOT TESTED (Cần chạy trên máy cục bộ)** cho các hạng mục phụ thuộc vào trình thông dịch Python cục bộ và phần cứng Webcam vật lý của bạn.

| ID | Tiêu chí | Điều kiện chấp nhận | Trạng thái | Bằng chứng tương ứng |
| :--- | :--- | :--- | :--- | :--- |
| **AC01** | Project Audit | Đã phân tích cấu trúc ZIP | **PASS** | Mục A: Phân tích chi tiết cây thư mục và các file trong `uth-cv-labs-main.zip`[cite: 10] kèm bảng Audit. |
| **AC02** | Architecture | Kế thừa kiến trúc nhóm | **PASS** | Mục C & D: Phân tầng chuẩn xác vào `configs/`, `src/common/`, `src/data/`, `src/models/`, `src/pipelines/`, `scripts/`, `tests/`. |
| **AC03** | Webcam | OpenCV truy cập được webcam | **NOT TESTED** *(Cần Webcam vật lý)* | Đã triển khai hoàn chỉnh `WebcamManager` trong `src/pipelines/webcam_recognition.py`. Cần bạn chạy `python scripts/run_lab05_webcam.py` trên laptop để xác nhận phần cứng. |
| **AC04** | Detection | MTCNN phát hiện được khuôn mặt | **PASS (Code) / NOT TESTED (Runtime)** | Đã triển khai `MTCNNDetector` (`src/models/deep/mtcnn_detector.py`). Xác nhận runtime qua `python scripts/download_lab05_weights.py`. |
| **AC05** | Embedding | FaceNet tạo được embedding | **PASS (Code) / NOT TESTED (Runtime)** | Đã triển khai `FaceNetExtractor` với `InceptionResnetV1(pretrained='vggface2')` xuất vector 512 chiều chuẩn hóa $L_2$[cite: 8, 9]. |
| **AC06** | Registration | Có cơ chế đăng ký danh tính | **PASS** | Đã triển khai `FaceRegistrationPipeline` và `scripts/register_faces.py` hỗ trợ cả quét thư mục và chụp trực tiếp qua webcam. |
| **AC07** | Matching | Có phép tính similarity | **PASS** | `SimilarityMatcher.compute_cosine_similarity` tính chính xác Cosine Similarity $\frac{\mathbf{a}\cdot\mathbf{b}}{\|\mathbf{a}\|\|\mathbf{b}\|}$[cite: 9]. |
| **AC08** | Threshold | Tuân thủ ngưỡng 0.7 | **PASS** | `SimilarityMatcher.evaluate_threshold` áp dụng nghiêm ngặt `> 0.7` là `"Matched"`, `< 0.7` và `= 0.7` là `"Unknown"`[cite: 9]. |
| **AC09** | Visualization | Hiển thị Matched/Unknown | **PASS** | `FaceVisualizer.draw_results` vẽ BBox xanh/đỏ, nhãn `Matched`/`Unknown`, điểm similarity và FPS[cite: 9]. |
| **AC10** | Pipeline | Có quy trình nhận diện thời gian thực | **PASS** | `RealtimeRecognitionPipeline` điều phối khép kín từ Frame $\rightarrow$ MTCNN $\rightarrow$ FaceNet $\rightarrow$ Matcher $\rightarrow$ OSD UI[cite: 8, 9]. |
| **AC11** | Testing | Có Smoke Test và Normal Test | **PASS (Code) / READY TO RUN** | Đã cung cấp đầy đủ `tests/test_lab05_smoke.py` và `tests/test_lab05_unit.py`. |
| **AC12** | Documentation | Có Description và Report | **PASS** | Đã cung cấp đầy đủ `docs/labs/lab05/LAB_05_DESCRIPTION.md` và `docs/labs/lab05/LAB_05_REPORT.md`. |
| **AC13** | Execution Guide | Có hướng dẫn thực thi chi tiết | **PASS** | Mục G: Hướng dẫn từng lệnh PowerShell cho Windows từ tạo `.venv` đến kiểm tra 6 kịch bản demo. |
| **AC14** | Integration | Không phá vỡ Lab01–Lab04 | **PASS** | Toàn bộ file của Lab05 được tách biệt theo namespace `lab05`, không chỉnh sửa hay ghi đè bất kỳ file nào của Lab01–Lab04[cite: 10]. |
| **AC15** | Artifacts | Có kết quả thực nghiệm và minh chứng trung thực | **PASS (Script) / READY TO RUN** | Đã cung cấp `scripts/generate_lab05_artifacts.py` (tạo dữ liệu mô phỏng có gắn nhãn `SIMULATION`) và phím tắt `S` trên webcam để lưu ảnh thực tế vào `experiments/predictions/lab05/`. |

---

## J. REMAINING TASKS FOR THE USER (CÔNG VIỆC BẠN CẦN THỰC HIỆN TRỰC TIẾP TRÊN MÁY)

Để chuyển toàn bộ các tiêu chí `NOT TESTED / READY TO RUN` sang `PASS` trên máy tính cá nhân của bạn, hãy thực hiện 4 bước ngắn gọn sau:
1. **Lưu các file mã nguồn ở Mục D và Mục H** vào đúng đường dẫn tương ứng trong thư mục `uth-cv-labs-main/`.
2. **Mở PowerShell tại thư mục gốc `uth-cv-labs-main/`**, kích hoạt `.venv` và chạy lệnh kiểm thử tự động:
   ```powershell
   $env:PYTHONPATH="."
   pytest tests/test_lab05_smoke.py tests/test_lab05_unit.py -v
   python -m ruff check src/ tests/ scripts/
    python -m ruff check --fix src/ tests/ scripts/
   ```

3. Tải trọng số và đăng ký khuôn mặt của bạn qua Webcam:
python scripts/lab05/download_lab05_weights.py
python scripts/lab05/register_faces.py --mode capture --identity "Hanh_Nguyen" --num-samples 4

$env:PYTHONPATH="."
python scripts/lab05/register_faces.py --mode folder

4. Chạy nhận diện thời gian thực và nhấn phím S để lưu 2 ảnh minh chứng (Matched và Unknown) vào experiments/predictions/lab05/:
python scripts/lab05/run_lab05_webcam.py

