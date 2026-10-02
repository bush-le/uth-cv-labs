# Bài Thực Hành Chương 1: Xử Lý Ảnh Cơ Bản với OpenCV và Pillow

## 1. Mục tiêu
Làm quen với các thao tác cơ bản trong xử lý ảnh số sử dụng thư viện **OpenCV** và **Pillow** (PIL) trên ngôn ngữ Python.

---

## 2. Môi trường & Thư viện yêu cầu
- **Ngôn ngữ:** Python 3.x
- **Cài đặt thư viện:**
  ```bash
  pip install opencv-python pillow
  ```

---

## 3. Nội dung thực hành chi tiết

### Yêu cầu 1: Cài đặt môi trường
- Cài đặt thành công hai thư viện `opencv-python` và `Pillow`.

### Yêu cầu 2: Đọc và hiển thị ảnh
- **Đọc ảnh:** Tải một hình ảnh bất kỳ từ bộ nhớ máy tính vào chương trình.
- **Hiển thị:** Mở cửa sổ hiển thị hình ảnh vừa tải lên màn hình.
- **Lưu ảnh:** Lưu lại hình ảnh (hoặc ảnh đã xử lý) dưới một định dạng khác (ví dụ: chuyển đổi qua lại giữa `.png`, `.jpg`, `.bmp`, v.v.).

### Yêu cầu 3: Chuyển đổi không gian màu (Color Space Conversion)
- Chuyển đổi ảnh gốc (hệ màu RGB/BGR) sang ảnh mức xám (**Grayscale**).
- Chuyển đổi ảnh sang các không gian màu khác:
  - Không gian màu **HSV** (Hue, Saturation, Value).
  - Không gian màu **LAB** ($L^*a^*b^*$).

### Yêu cầu 4: Cắt xén và thay đổi kích thước ảnh (Cropping & Resizing)
- **Cắt xén (Crop):** Trích xuất một vùng quan tâm (ROI - Region of Interest) bất kỳ từ ảnh gốc.
- **Thay đổi kích thước (Resize):**
  - Đổi kích thước theo tỷ lệ phần trăm (Scale factor / Aspect ratio).
  - Đổi kích thước về một kích thước cố định (width $\times$ height chỉ định trước).

### Yêu cầu 5: Vẽ hình cơ bản và chèn chữ lên ảnh (Drawing & Annotations)
- Vẽ các hình học cơ bản lên ảnh:
  - Đường thẳng (Line).
  - Hình chữ nhật (Rectangle / Bounding Box).
  - Hình tròn (Circle).
- Thêm văn bản (Text / Annotation) lên hình ảnh với phông chữ, kích thước và màu sắc tùy chỉnh.

---

## 4. Kết quả cần bàn giao
- Mã nguồn thực hiện các yêu cầu trên (file `.py` hoặc Jupyter Notebook `.ipynb`).
- Thư mục chứa ảnh gốc (input) và ảnh kết quả sau từng thao tác (output).