# BÀI THỰC HÀNH CHƯƠNG 2: TOÁN TỬ ĐIỂM ẢNH & LỌC TUYẾN TÍNH

## 1. Mục tiêu

1. Hiểu và cài đặt được các **toán tử điểm ảnh** (point operations): thay đổi độ sáng, độ tương phản, biến đổi âm bản, cắt ngưỡng.
2. Hiểu bản chất **phép tích chập (convolution)** trong ảnh số và cách kernel quyết định kết quả đầu ra.
3. Cài đặt và so sánh được các **bộ lọc tuyến tính**: lọc trung bình, lọc Gaussian, làm nét.
4. Hiểu và áp dụng được các **bộ phát hiện biên** (Sobel, Prewitt) thông qua đạo hàm bậc 1.
5. Nắm được **quy luật tổng hệ số kernel** và mối liên hệ giữa đạo hàm — kernel — hành vi bộ lọc.

---

## 2. Môi trường & Thư viện yêu cầu

- **Ngôn ngữ:** Python 3.x
- **Thư viện chính:**
  ```bash
  pip install opencv-python numpy matplotlib
  ```
  - `opencv-python`: đọc/ghi ảnh, `cv2.filter2D`, `cv2.GaussianBlur`, `cv2.medianBlur`, `cv2.bilateralFilter`.
  - `numpy`: thao tác mảng và vector hoá.
  - `matplotlib`: hiển thị và so sánh ảnh trực quan.
- **Dữ liệu đầu vào:** Một ảnh mẫu đặt trong `data/samples/` (ví dụ: `data/samples/lab02_input.jpg`) để đảm bảo **tính tái lập kết quả**. Ghi rõ tên ảnh sử dụng trong báo cáo.

> 📌 **Lưu ý kỹ thuật:** Toàn bộ bộ lọc trong bài này hoạt động độc lập trên **từng kênh màu** (B, G, R). Không bắt buộc chuyển sang ảnh xám; tuy nhiên khi phân tích kết quả gradient, nên chạy thêm trên ảnh xám để quan sát rõ đặc tính của từng bộ lọc.

---

## 3. Nội dung thực hành chi tiết

### I. Toán tử điểm ảnh (Point Operations)

> **Đặc điểm chung:** mỗi pixel đầu ra chỉ phụ thuộc vào **chính pixel đó** (không phụ thuộc lân cận). Có thể thực hiện bằng phép toán mảng thuần, **không dùng vòng lặp duyệt từng pixel**.

#### Yêu cầu 1: Thay đổi độ sáng
- Tăng hoặc giảm độ sáng toàn ảnh bằng cách **cộng/trừ một hằng số** `β` vào mỗi điểm ảnh: `g(x, y) = f(x, y) + β`.
- Lưu ý hiện tượng **bão hoà (clipping)**: giá trị vượt [0, 255] phải được cắt biên (dùng `np.clip`).

#### Yêu cầu 2: Thay đổi độ tương phản
- Tăng hoặc giảm độ tương phản bằng cách **nhân mỗi điểm ảnh với một hằng số** `α`: `g(x, y) = α · f(x, y)`.
- Nhận xét: `α > 1` làm ảnh tương phản mạnh hơn, `0 < α < 1` làm ảnh "bạc màu" hơn.

#### Yêu cầu 3: Biến đổi âm bản (Negative)
- Đảo ngược giá trị điểm ảnh: `g(x, y) = 255 − f(x, y)` (với ảnh `uint8`).
- Ứng dụng: quan sát rõ chi tiết ở vùng tối.

#### Yêu cầu 4: Cắt ngưỡng (Thresholding)
- Tạo ảnh nhị phân bằng cách so sánh mỗi điểm ảnh với một ngưỡng `T`:
  `g(x, y) = 255 nếu f(x, y) > T, ngược lại g(x, y) = 0`.
- Thử nhiều giá trị `T` khác nhau và nhận xét.

---

### II. Lọc tuyến tính (Linear Filtering)

> **Đặc điểm chung:** mỗi pixel đầu ra là **tổ hợp tuyến tính có trọng số** của các pixel lân cận (phép tích chập với kernel). Sử dụng `cv2.filter2D(image, -1, kernel)`.
>
> **Nguyên tắc cần nắm — tổng hệ số kernel:**
> Trên một vùng ảnh phẳng (mọi pixel = `c`), output = `c × (tổng hệ số kernel)`. Do đó:
> - **Tổng = 1** → giữ nguyên độ sáng trung bình (nhóm giữ ảnh / làm mờ / làm nét).
> - **Tổng = 0** → xoá vùng phẳng, chỉ giữ lại vùng có biên (nhóm đạo hàm / phát hiện biên).

#### Yêu cầu 5: Lọc trung bình (Mean Filter)
- Làm mờ ảnh bằng cách lấy **trung bình cộng** các điểm ảnh trong vùng lân cận (kernel toàn số dương, tổng = 1).
- Thử kernel 3×3 và 5×5, nhận xét mức độ mờ và mất chi tiết.
- Đối chiếu với `cv2.blur()`.

#### Yêu cầu 6: Lọc Gaussian (Gaussian Filter)
- Làm mờ bằng kernel Gaussian (trọng số giảm dần theo khoảng cách tới tâm, tổng = 1).
- So sánh với lọc trung bình: Gaussian **giữ cạnh tốt hơn** và mờ tự nhiên hơn.
- Đối chiếu với `cv2.GaussianBlur()`, thử thay đổi `sigma`.

#### Yêu cầu 7: Làm sắc nét (Sharpening)
- **Bản chất:** làm nét = **Ảnh gốc + thành phần biên**, tức `Identity − Laplacian`.
- Kernel làm nét kinh điển (tâm dương LỚN, xung quanh ÂM):
  ```
  [ 0  -1   0]
  [-1   5  -1]      →  tổng = 1
  [ 0  -1   0]
  ```
- **Phân biệt rõ (tránh nhầm lẫn):**
  - **Laplacian** (`[0 1 0; 1 -4 1; 0 1 0]`, tổng = 0) là **bộ phát hiện biên**, **KHÔNG** phải bộ làm nét.
  - Muốn làm nét từ Laplacian: lấy `Ảnh gốc − Laplacian`.
  - Vì sao kernel làm nét có tổng = 1? Vì `Identity (tổng 1) − Laplacian (tổng 0) = tổng 1`.
- Giải thích tại sao làm nét cần **hệ số âm**: hệ số âm ở lân cận khuếch đại chênh lệch giữa tâm và vùng xung quanh.

---

### III. Bài tập nâng cao

#### Yêu cầu 8: Phát hiện cạnh bằng Sobel & Prewitt
- **Sobel:** dùng hai kernel (ngang & dọc) để tính đạo hàm bậc 1 theo hai phương. Kernel ngang:
  ```
  [-1  0  1]
  [-2  0  2]      →  tổng = 0
  [-1  0  1]
  ```
- **Prewitt:** tương tự nhưng hệ số đơn giản hơn (không nhấn mạnh hàng giữa):
  ```
  [-1  0  1]
  [-1  0  1]      →  tổng = 0
  [-1  0  1]
  ```
- **So sánh:** trình bày sự khác nhau giữa hai kernel (trọng số hàng giữa) và ảnh hưởng tới độ nhạy nhiễu.
- **Gợi ý mở rộng:** tính độ lớn gradient tổng hợp `G = √(Gx² + Gy²)` và hiển thị ảnh biên.

#### Yêu cầu 9: Tự thiết kế kernel & hiểu ý nghĩa tổng hệ số
- Tự tạo các kernel tuỳ chỉnh và **giải thích bằng tổng hệ số**:
  1. Vì sao **tổng = 0** cho ra **ảnh chỉ còn biên** (đạo hàm)?
  2. Vì sao **tổng = 1** giữ nguyên độ sáng? Và vì sao **tổng = 1 vẫn có thể là làm mờ** (lọc trung bình, Gaussian) chứ chưa chắc là làm nét?
  3. Điều kiện nào để một kernel **thực sự làm nét**? (Gợi ý: tổng = 1 **và** có hệ số âm.)
- **Bài tập kiểm chứng:** thử các kernel có tổng lần lượt **0.5, 1, 2, 0** trên cùng một ảnh phẳng và quan sát ảnh tối đi / giữ nguyên / sáng lên / chỉ còn viền.

#### Yêu cầu 10: Lọc phi tuyến tính (Non-linear Filters)
- **Lọc trung vị (Median):** thay mỗi pixel bằng **trung vị** vùng lân cận. Hiệu quả với nhiễu muối tiêu. Đối chiếu `cv2.medianBlur()`.
- **Lọc Bilateral:** kết hợp thông tin **không gian** và **cường độ** → làm mờ vùng đồng nhất nhưng giữ cạnh. Đối chiếu `cv2.bilateralFilter()`.
- **Nhận xét:** vì sao các lọc này **không thể** biểu diễn bằng một kernel tích chập (phi tuyến tính)?

#### Yêu cầu 11: So sánh tổng hợp các bộ lọc
- Áp tất cả bộ lọc lên **cùng một ảnh** và lập bảng so sánh:
  - Mức độ làm mờ / làm nét.
  - Khả năng giữ cạnh.
  - Hiệu quả khử nhiễu.
- Trực quan hoá kết quả thành lưới ảnh (grid) có chú thích.

---

## 4. Kết quả cần bàn giao

1. **Mã nguồn** thực hiện đầy đủ các yêu cầu trên (`.py` hoặc `.ipynb`).
2. **Ảnh kết quả** của từng thao tác, lưu trong `experiments/predictions/lab02/`.
3. **Ảnh so sánh tổng hợp** (lưới grid) cho Yêu cầu 11.
4. **Báo cáo kỹ thuật** `docs/labs/lab02/LAB_02_REPORT.md` + báo cáo trực quan `LAB_02_REPORT.html` (theo quy chuẩn repo), trong đó giải thích rõ:
   - Bản chất phép tích chập và cách kernel quyết định kết quả.
   - Ý nghĩa tổng hệ số kernel (= 1 và = 0) kèm minh chứng bằng ảnh thực tế.
   - Sự khác biệt Sobel vs Prewitt.
   - Vì sao Laplacian là **phát hiện biên**, còn làm nét là `Identity − Laplacian`.