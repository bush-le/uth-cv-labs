# BÀI THỰC HÀNH CHƯƠNG 2 (PHẦN 2): CANNY EDGE DETECTOR

> **Học phần:** Thị giác Máy tính (Computer Vision)  
> **Lab:** 03 · **Chương:** 2 (Phần 2)  

---

## 1. Mục tiêu

1. Nắm vững và giải thích được chi tiết từng bước toán học và thuật toán trong pipeline **Canny Edge Detector**.
2. Hiểu rõ ý nghĩa, vai trò và làm chủ cách tinh chỉnh các tham số ($\sigma, T_{\text{low}}, T_{\text{high}}$).
3. Sử dụng thành thạo hai thư viện **OpenCV** (`cv2.Canny`) và **Scikit-image** (`skimage.feature.canny`).
4. Phân tích ưu / nhược điểm của Canny so với các bộ lọc đạo hàm kinh điển (Sobel, Laplacian) và hiểu rõ các ứng dụng thực tế.
5. Ứng dụng Canny kết hợp với các kỹ thuật phân đoạn (Segmentation) và nhận dạng hình dạng (Shape Detection) trên đa dạng loại ảnh.

---

## 2. Môi trường & Thư viện yêu cầu

- **Ngôn ngữ:** Python 3.x
- **Cài đặt thư viện:**
  ```bash
  pip install -r requirements/base.txt
pip install jupyter nbclient nbformat ipykernel
  ```

---

## 3. Nội dung thực hành chi tiết (Bám sát Slide Bài giảng)

### I> Lý thuyết:

#### 1. Tìm hiểu về các bước của thuật toán Canny:
- **a. Giải thích chi tiết từng bước:**
  - **Giảm nhiễu (Làm mịn Gaussian):** Khử nhiễu tần số cao bằng bộ lọc Gaussian 2D. Giải thích vai trò của tham số $\sigma$ và kích thước kernel, sự đánh đổi giữa mức độ khử nhiễu và độ mờ biên.
  - **Tính toán gradient:** Áp dụng đạo hàm bậc nhất (kernel Sobel theo trục $X$ và $Y$) để xác định:
    - Độ lớn Gradient (Gradient Magnitude): $G = \sqrt{G_x^2 + G_y^2}$ (hoặc xấp xỉ $|G_x| + |G_y|$).
    - Hướng Gradient (Gradient Direction / Angle): $\theta = \arctan(G_y / G_x)$.
  - **Non-maximum suppression (Triệt tiêu cực đại cục bộ):** Cơ chế làm mỏng đường biên về độ rộng đúng 1 pixel bằng cách rời rạc hóa hướng $\theta$ về 4 hướng ($0^\circ, 45^\circ, 90^\circ, 135^\circ$) và chỉ giữ lại điểm có độ lớn gradient lớn nhất so với 2 điểm láng giềng dọc theo hướng gradient.
  - **Ngưỡng kép (Double Thresholding):** Phân loại các điểm biên thành 3 miền giá trị dựa trên ngưỡng thấp ($T_{\text{low}}$) và ngưỡng cao ($T_{\text{high}}$):
    - *Strong edge* ($G \ge T_{\text{high}}$): Biên mạnh / chắc chắn.
    - *Weak edge* ($T_{\text{low}} \le G < T_{\text{high}}$): Biên yếu / tiềm năng.
    - *Non-edge* ($G < T_{\text{low}}$): Không phải biên (bị loại bỏ).
  - **Theo dõi cạnh (Edge Tracking by Hysteresis):** Duyệt liên thông 8-láng giềng. Chỉ giữ lại các điểm *weak edge* nếu chúng kết nối liên tục với ít nhất một điểm *strong edge*; loại bỏ các điểm weak edge ngắt quãng do nhiễu.
- **b. So sánh với các thuật toán khác như Sobel, Laplacian:**
  - So sánh chi tiết về: số bước xử lý, khả năng khử nhiễu, độ chính xác định vị tâm biên (localization), độ mỏng biên (1-pixel vs đa pixel), hiện tượng phản hồi kép và độ phức tạp tính toán.

#### 2. Các tham số của thuật toán và ảnh hưởng của chúng:
- **a. Sigma trong Gaussian filter:**
  - Ảnh hưởng của $\sigma$ đến kích thước kernel và mức độ làm mịn ảnh trước khi tính gradient.
  - Tác động khi $\sigma$ nhỏ (nhạy nhiễu, giữ chi tiết) vs $\sigma$ lớn (khử nhiễu mạnh, dễ làm mờ/mất biên mảnh).
- **b. Ngưỡng thấp và ngưỡng cao:**
  - Khoảng cách và tỷ lệ khuyến nghị giữa $T_{\text{high}}$ và $T_{\text{low}}$ (thường từ $2:1$ hoặc $3:1$).
  - Hiện tượng khi cặp ngưỡng quá cao (bỏ sót biên thực, biên đứt nét) vs khi quá thấp (nhiều biên giả do nhiễu và vân bề mặt).

#### 3. Ưu điểm và nhược điểm của Canny edge detector. Các ứng dụng thực tế của Canny edge detector:
- **a. So sánh với các thuật toán khác về độ chính xác, tốc độ, khả năng xử lý nhiễu:**
  - Độ chính xác định vị tâm biên (tối ưu theo 3 tiêu chuẩn Canny).
  - Tốc độ xử lý (chậm hơn Sobel/Prewitt do qua 5 giai đoạn).
  - Khả năng xử lý/kháng nhiễu (vượt trội nhờ tiền lọc Gaussian kết hợp ngưỡng kép).
- **b. Trong lĩnh vực nào Canny được sử dụng phổ biến nhất?**
  - Xe tự hành & giao thông thông minh (ADAS, phát hiện làn đường, biển báo).
  - Thị giác máy tính công nghiệp (kiểm tra ngoại quan, phát hiện lỗi sản phẩm, đo kích thước).
  - Xử lý ảnh y tế (phân tách xương, khối u, viền cơ quan trên X-quang/MRI/CT).
  - Nhận dạng ký tự quang học (OCR) và nhận dạng biển số xe.
- **c. Ví dụ cụ thể về các ứng dụng:**
  - Trình bày pipeline thực tế cụ thể (ví dụ: Hệ thống phát hiện làn đường Lane Detection kết hợp Canny + ROI Polygon + Hough Transform).

---

### II> Bài tập thực hành:

#### 1. Thực hiện thuật toán Canny bằng các thư viện:
- **OpenCV:** Sử dụng hàm `cv2.Canny` để phát hiện cạnh.
- **Scikit-image:** Sử dụng hàm `skimage.feature.canny` để phát hiện cạnh.
- **Yêu cầu:** Hiển thị song song kết quả của 2 thư viện; nhận xét về cú pháp truyền tham số và sự khác biệt trên ảnh đầu ra.

#### 2. Thay đổi các tham số và quan sát kết quả:
- **Thay đổi sigma, ngưỡng thấp, ngưỡng cao để xem ảnh hưởng của chúng đến kết quả:**
  - Khảo sát biến thiên $\sigma$ qua các mức khác nhau (ví dụ: 0.5, 1.0, 2.0, 3.5).
  - Khảo sát các dải cặp ngưỡng $(T_{\text{low}}, T_{\text{high}})$: nhạy cao, chuẩn, khắt khe.
- **So sánh kết quả với các giá trị mặc định:**
  - Vẽ lưới ảnh (Grid plot) để quan sát trực quan sự thay đổi so với cấu hình tham chiếu của bài (sigma=1, ngưỡng 50/150); đây không phải ngưỡng mặc định của OpenCV.

#### 3. Áp dụng Canny cho các loại ảnh khác nhau:
- **Thử nghiệm trên 3 loại ảnh:**
  - Ảnh có nhiều nhiễu (nhiễu hạt Gaussian, nhiễu muối tiêu).
  - Ảnh có độ tương phản thấp (thiếu sáng, mờ sương, dải động hẹp).
  - Ảnh có nhiều chi tiết (tán cây, thảm cỏ, vải dệt, kiến trúc phức tạp).
- **Đánh giá kết quả và rút ra kết luận:**
  - So sánh kết quả chạy Canny trực tiếp và chạy Canny sau khi áp dụng tiền xử lý phù hợp (lọc trung vị, cân bằng histogram,..).
  - Rút ra kết luận về tầm quan trọng của khâu tiền xử lý.

#### 4. Kết hợp Canny với các kỹ thuật khác:

- **YC4.1 — Kết hợp phân đoạn để tìm các vùng trong ảnh:**
  - Dùng ảnh đồng xu `skimage.data.coins()`: Gaussian giảm nhiễu → Canny → closing nối khe hở nhỏ → `scipy.ndimage.binary_fill_holes` tô đầy vùng kín → mask vùng và ảnh giữ lại bằng mask.
  - Hiển thị ảnh gốc, biên Canny, biên sau closing, mask và vùng ảnh được giữ. Nhận xét vùng bỏ sót do biên hở hoặc vùng sai do nối nhầm biên; không bắt buộc tô màu đối tượng.
- **YC4.2 — Kết hợp nhận dạng hình dạng để tìm đối tượng:**
  - Dùng cùng ảnh đồng xu, chọn **Hough Circle Transform** (`cv2.HoughCircles`) để tìm các cấu trúc gần tròn; vẽ đường tròn, đánh số và in số lượng.
  - `HoughCircles` nhận ảnh xám đã làm mịn, xử lý gradient/Canny bên trong; không truyền ảnh biên Canny như input grayscale. Phân biệt `param1` (ngưỡng cao Canny nội bộ) với `param2` (ngưỡng quyết định tâm tròn).
  - Giải thích khoảng bán kính, khoảng cách tâm và ngưỡng; đối chiếu kết quả với ảnh gốc. Đây là nhận dạng hình học tròn, không xác nhận ngữ nghĩa đối tượng là tiền xu.
  - Hough đường thẳng và contour/xấp xỉ đa giác là các hướng mở rộng, không bắt buộc triển khai trong ví dụ đồng xu này.

---

### III> Các câu hỏi mở rộng:

1. **Làm thế nào để đánh giá chất lượng của các cạnh được phát hiện bởi Canny?**
   - Đánh giá định tính (độ mảnh 1-pixel, tính liên tục, tỷ lệ biên giả).
   - Đánh giá định lượng khi có nhãn Ground Truth (Precision, Recall, F1/BF-Score, chỉ số Pratt's FOM, khoảng cách Hausdorff).
2. **Có những phương pháp nào khác để cải thiện hiệu suất của Canny?**
   - Tự động chọn ngưỡng (Auto-Canny / Adaptive Thresholding theo median hoặc Otsu).
   - Thay thế Gaussian Blur bằng Bilateral Filter để khử nhiễu nhưng bảo toàn biên sắc nét.
   - Tối ưu tăng tốc phần cứng (GPU CUDA, bộ lọc phân tách separable filter).
3. **Canny có thể được sử dụng để phát hiện các cạnh trong ảnh màu không? Nếu có, thì như thế nào?**
   - Phân tích hạn chế mất mát biên khi chuyển ảnh xám (Grayscale).
   - Phương pháp Multi-channel: chạy Canny trên từng kênh màu độc lập và hợp nhất bằng toán tử bitwise OR (`cv2.bitwise_or`).
   - Phương pháp Color Gradient Tensor (thuật toán Di Zenzo tính ma trận Jacobian không gian đa kênh).
4. **Làm thế nào để áp dụng Canny cho các video?**
   - Pipeline xử lý từng khung hình (frame-by-frame) với `cv2.VideoCapture`.
   - Xử lý hiện tượng giật/nhấp nháy biên giữa các khung hình liên tiếp (Temporal Flickering).
   - Tối ưu hóa FPS đảm bảo thời gian thực (Real-time).

---

## 4. Yêu cầu sản phẩm đầu ra (Deliverables)

Theo đúng yêu cầu nghiệm thu, sản phẩm bài lab bao gồm **02 file báo cáo hoàn chỉnh**:

| # | Sản phẩm đầu ra | Tên file quy ước | Mô tả chi tiết yêu cầu |
|:---:|:---|:---|:---|
| 1 | **Báo cáo Kỹ thuật Toàn diện** | [`LAB_03_REPORT.md`](LAB_03_REPORT.md) | File Markdown báo cáo **chi tiết, toàn bộ và hoàn chỉnh nhất có thể**: trình bày cặn kẽ lý thuyết toán học 5 bước Canny, phân tích định tính và định lượng cho cả 4 bài thực hành, trả lời sâu sắc 4 câu hỏi mở rộng kèm công thức, bảng đối chiếu và phân tích khoa học. |
| 2 | **Báo cáo Trực quan Đa phương tiện** | [`LAB_03_REPORT.html`](LAB_03_REPORT.html) | File HTML **trực quan nhất, show được chi tiết toàn bộ bài làm để báo cáo và thuyết trình**: giao diện hiện đại, responsive, trực quan hóa từng bước trong pipeline, hiển thị so sánh ảnh kết quả (trước/sau, grid tuning tham số, phân đoạn, nhận dạng hình dạng), bảng so sánh và code snippets có tô màu cú pháp. |
