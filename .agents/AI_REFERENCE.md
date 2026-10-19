# AI REFERENCE - BỘ QUY CHUẨN THAM CHIẾU BẮT BUỘC CHO AI AGENT

AI Agent phải tuyệt đối tuân thủ toàn bộ quy chuẩn trong tài liệu này. Bất kỳ sai lệch nào sẽ bị từ chối ở bước Codebase Audit.

---

## 1. Quy chuẩn Đặt tên (Naming Conventions)

| Đối tượng | Quy chuẩn | Ví dụ Chuẩn (Good) | Ví dụ Sai (Bad) |
| :--- | :--- | :--- | :--- |
| **Tên Thư mục** | `snake_case` hoặc kebab-case ngắn | `models/`, `data_loader/`, `pre_trained/` | `Models/`, `dataLoader/`, `MyData/` |
| **Tên File Python** | `snake_case.py` | `datamodule.py`, `metrics.py`, `train.py` | `DataModule.py`, `my_Train.py` |
| **Tên File Cấu hình** | `snake_case.yaml` | `default.yaml`, `template.yaml` | `Default.YAML`, `my-config.yml` |
| **Tên Class** | `PascalCase` | `StandardCVDataset`, `CVDataModule`, `CustomCNN` | `standard_cv_dataset`, `cvDataModule` |
| **Tên Function / Method** | `snake_case` (bắt đầu bằng động từ) | `calculate_iou()`, `apply_filter()`, `build_model()` | `CalculateIoU()`, `iou()`, `DoProcess()` |
| **Tên Biến thông thường** | `snake_case` | `image_tensor`, `batch_size`, `learning_rate` | `ImageTensor`, `bs`, `lrVal` |
| **Tên Hằng số** | `UPPER_SNAKE_CASE` | `DEFAULT_SEED`, `IMAGE_SIZE`, `NUM_CLASSES` | `default_seed`, `ImageSize` |
| **Tên Biến Riêng tư (Private)**| `_leading_snake_case` | `_init_weights()`, `_cached_features` | `__init_weights()`, `privateFeatures` |

---

## 2. Quy chuẩn Cấu trúc Thư mục (Folder Structure Invariants)

AI không được tự ý tạo thêm thư mục gốc mới ngoài danh mục đã được phê duyệt:

```text
uth-cv-labs/
├── configs/          # CHỈ chứa file .yaml cấu hình (Hydra)
├── data/             # CHỈ chứa dữ liệu thô, trung gian và processed (ignored by Git)
├── docs/             # CHỈ chứa tài liệu hướng dẫn và báo cáo lab markdown
├── experiments/      # CHỈ chứa output sinh ra tự động: checkpoints, logs, predictions
├── notebooks/        # CHỈ chứa notebook tương tác .ipynb
├── requirements/     # CHỈ chứa danh sách dependencies theo môi trường .txt
├── scripts/          # CHỈ chứa shell scripts tự động hóa .sh
├── src/              # CHỈ chứa mã nguồn Python chính thức của package
└── tests/            # CHỈ chứa unit tests và fixtures phục vụ pytest
```

---

## 3. Danh sách Điều cấm Kỵ (Negative Constraints for AI)

1. **Cấm tự ý tạo file chứa nội dung giả định**: Không tạo các file có tên cụ thể (như `lab01_report.md`, `resnet50.py`) khi chưa được chỉ định trong task.
2. **Cấm hardcode đường dẫn tuyệt đối**: Không dùng `/home/user/...` hay `C:\...`; luôn dùng đường dẫn tương đối hoặc `pathlib.Path`.
3. **Cấm sử dụng `print` trong module `src/`**: Luôn sử dụng logger từ `src.common.logger`.
4. **Cấm commit file nhị phân lớn**: Không lưu trọng số model nặng, file zip hoặc video vào Git.

---

## 4. Quy chuẩn Conventional Commits

Mọi commit do AI đề xuất hoặc kỹ sư thực hiện phải tuân thủ nghiêm ngặt cú pháp:

```text
<type>(<scope>): <subject>

[optional body: mô tả nguyên nhân và giải pháp kỹ thuật]

[optional footer: liên kết issue hoặc breaking changes]
```

### Danh mục Types cho Computer Vision:
- `feat`: Thêm tính năng mới (thuật toán lọc, kiến trúc model, metric mới, datamodule).
- `fix`: Sửa lỗi (shape mismatch, zero-division, NaN loss, off-by-one index).
- `perf`: Tối ưu hóa hiệu năng (vectorization, DataLoader prefetching, GPU memory).
- `refactor`: Tái cấu trúc mã nguồn mà không làm thay đổi tính năng hay kết quả.
- `test`: Thêm hoặc cập nhật unit test, smoke test, benchmark fixtures.
- `docs`: Cập nhật tài liệu kỹ thuật, docstring, báo cáo thí nghiệm.
- `chore`: Thay đổi cấu hình build, pre-commit, package dependencies.

### Danh mục Scopes chuẩn:
`data`, `models`, `pipelines`, `metrics`, `visualizer`, `configs`, `tests`, `scripts`

### Quy tắc commit:
1. Viết subject ở thể mệnh lệnh hiện tại (`implement`, `add`, `fix`, không dùng `implemented`, `fixing`).
2. Subject không viết hoa chữ đầu, không kết thúc bằng dấu chấm, tối đa 72 ký tự.
3. Commit nguyên tử (Atomic Commit): Một commit chỉ giải quyết trọn vẹn một tác vụ độc lập.

---

## 5. Bất biến Kỹ thuật Ngành Thị giác Máy tính (CV/DL Domain Invariants)

Mọi dòng code xử lý hình ảnh và mô hình học sâu phải tuân thủ nghiêm ngặt 5 tiêu chuẩn kỹ thuật:

### 5.1. Quy chuẩn Tensor & Kích thước Chiều (Shape Conventions)
- **OpenCV / NumPy**: Luôn ở định dạng `(H, W, C)` hoặc `(H, W)` (grayscale), kiểu dữ liệu `uint8` [0, 255] hoặc `float32` [0.0, 1.0].
- **PyTorch**: Luôn ở định dạng `(B, C, H, W)` (batch) hoặc `(C, H, W)` (đơn mẫu), kiểu dữ liệu `float32`.
- **Docstrings**: Bắt buộc ghi rõ shape tensor đầu vào và đầu ra trong mọi hàm xử lý ảnh.

### 5.2. Nhận diện Không gian Màu (Color Space Awareness)
- **OpenCV**: Đọc và ghi mặc định ở hệ màu `BGR`.
- **PIL / Matplotlib / Torchvision**: Sử dụng mặc định hệ màu `RGB`.
- **Quy tắc**: Phải chuyển đổi tường minh qua `cv2.cvtColor(image, cv2.COLOR_BGR2RGB)` khi trao đổi dữ liệu giữa OpenCV và Torchvision / Matplotlib. Cấm để lệch màu ẩn.

### 5.3. Định dạng Tọa độ Hộp bao (Bounding Box Formats)
Không bao giờ ngầm định định dạng hộp bao. Phải ghi rõ 1 trong các chuẩn sau:
- `XYXY` / Pascal VOC: `[x1, y1, x2, y2]` (pixel tuyệt đối).
- `XYWH` / COCO: `[x_min, y_min, width, height]` (pixel tuyệt đối).
- `CXCYWH` / YOLO: `[cx, cy, w, h]` (chuẩn hóa về khoảng [0.0, 1.0]).

### 5.4. Chuẩn hóa Tính Tái lập Thí nghiệm (Reproducibility Mandate)
Mọi thí nghiệm phải sử dụng hàm seed tập trung để khóa ngẫu nhiên trên toàn bộ môi trường:
- `random.seed(seed)`
- `np.random.seed(seed)`
- `torch.manual_seed(seed)`
- `torch.cuda.manual_seed_all(seed)`
- `torch.backends.cudnn.deterministic = True`

### 5.5. Quản lý Tài nguyên GPU & Tối ưu Băng thông (GPU Hygiene)
- Không hardcode `.cuda()` hoặc `'cuda:0'`, luôn dùng `device = torch.device(...)` nhận từ file cấu hình.
- DataLoader trên GPU phải bật `pin_memory=True` và cấu hình `num_workers > 0` phù hợp với số CPU cores.
- Sử dụng `torch.cuda.amp.autocast()` khi huấn luyện để tối ưu tốc độ và dung lượng VRAM.

