# BÁO CÁO KỸ THUẬT TOÀN DIỆN - LAB 03: CANNY EDGE DETECTOR

> **Học phần:** Thị giác Máy tính (Computer Vision)  
> **Khoa / Viện:** Viện Đào tạo Sau đại học / Khoa Công nghệ Thông tin - Trường Đại học Giao thông Vận tải TP.HCM (UTH)  
> **Họ và tên Học viên / Sinh viên:** Nhóm Nghiên cứu Thị giác Máy tính UTH  
> **Mã bài Lab:** Lab 03 · **Chương:** 2 (Phần 2)  
> **Repository:** [`uth-cv-labs`](file:///home/bush/Projects/CV/uth-cv-labs)  
> **Trạng thái Thực nghiệm:** Hoàn thành 100% · Vượt qua kiểm toán Codebase Audit & 11/11 Unit Tests

---

## MỤC LỤC TỔNG QUAN

1. [Phần I: Cơ sở Lý thuyết & Toán học Chuyên sâu](#phan-i-co-so-ly-thuyet--toan-hoc-chuyen-sau)
   - 1.1. Bản chất và 3 Tiêu chuẩn Tối ưu của John F. Canny (1986)
   - 1.2. Giải tích Toán học 5 Giai đoạn trong Pipeline Canny
   - 1.3. Khảo sát Chuyên sâu Bộ Tham số ($\sigma, T_{\text{low}}, T_{\text{high}}$)
   - 1.4. Ma trận So sánh Đối đầu: Canny vs Sobel vs Laplacian of Gaussian (LoG)
   - 1.5. Ưu điểm, Nhược điểm và Hệ sinh thái Ứng dụng Thực tế
2. [Phần II: Thực nghiệm & Phân tích Đột phá](#phan-ii-thuc-nghiem--phan-tich-dot-pha)
   - 2.1. Bài tập 1: Thực thi Đa thư viện (Custom Vectorized vs OpenCV vs Scikit-Image)
   - 2.2. Bài tập 2: Tinh chỉnh Siêu tham số & Ma trận Lưới 16 Cấu hình (Grid Tuning)
   - 2.3. Bài tập 3: Khảo sát 3 Miền Ảnh Bất lợi & Sức mạnh của Tiền xử lý
   - 2.4. Bài tập 4: Tích hợp Downstream (Phân đoạn Vùng & Nhận dạng Hình học)
3. [Phần III: Giải quyết 4 Chuyên đề Mở rộng](#phan-iii-giai-quyet-4-chuyen-de-mo-rong)
   - 3.1. Chuyên đề 1: Đánh giá Định lượng Chất lượng Biên (Pratt's FOM, F1, Hausdorff)
   - 3.2. Chuyên đề 2: Tối ưu Hóa Hiệu năng & Chọn Ngưỡng Thích nghi (Auto-Canny & Otsu)
   - 3.3. Chuyên đề 3: Canny trên Không gian Ảnh Màu (Multi-channel vs Di Zenzo Tensor)
   - 3.4. Chuyên đề 4: Xử lý Video Thời gian Thực & Khử Hiện tượng Nhấp nháy (Temporal Flickering)
4. [Phần IV: Kết luận & Hướng Phát triển](#phan-iv-ket-luan--huong-phat-trien)
5. [Phần V: Danh mục Tài liệu Tham khảo](#phan-v-danh-muc-tai-lieu-tham-khao)

---

# PHẦN I: CƠ SỞ LÝ THUYẾT & TOÁN HỌC CHUYÊN SÂU

## 1.1. Bản chất và 3 Tiêu chuẩn Tối ưu của John F. Canny (1986)

Biên ảnh (Edge) là tập hợp các điểm ảnh có sự biến đổi đột ngột về độ chói (luminance/intensity) hoặc màu sắc, tương ứng với ranh giới vật thể, sự thay đổi độ sâu không gian, hướng pháp tuyến bề mặt hoặc bóng đổ quang học.

Năm 1986, **John F. Canny** xuất bản công trình kinh điển *"A Computational Approach to Edge Detection"* trên tạp chí *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)*. Thay vì chỉ áp dụng các bộ vi phân cục bộ heuristic như Roberts, Prewitt hay Sobel, Canny đã mô hình hóa bài toán phát hiện biên như một **bài toán tối ưu hóa toán học** dựa trên 3 tiêu chuẩn định lượng khắt khe:

1. **Xác suất Bắt lỗi Thấp (Low Error Rate / Good Detection):**
   - Thuật toán phải phát hiện được tối đa các điểm biên thực sự trong ảnh (không bỏ sót biên - Low False Negative).
   - Tuyệt đối hạn chế tối đa việc nhận diện nhầm các gợn nhiễu ngẫu nhiên hoặc vân bề mặt thành đường biên (Low False Positive).
   - Tối đa hóa tỷ số Tín hiệu trên Nhiễu (Signal-to-Noise Ratio - $\text{SNR}$).

2. **Định vị Tâm biên Chính xác (Good Localization):**
   - Khoảng cách hình học Euclidean giữa vị trí điểm biên do thuật toán xác định và tâm điểm biên thực tế phải đạt giá trị cực tiểu:
     $$\text{Localization} = \frac{1}{\sqrt{\mathbb{E}[\Delta x^2]}} \to \max$$

3. **Phản hồi Đơn trị (Single Response / Unique Edge Mark):**
   - Mỗi biên thực tế trong không gian 2D chỉ được biểu diễn bằng **đúng 1 pixel duy nhất** về độ rộng (1-pixel width).
   - Loại trừ hoàn toàn hiện tượng "phản hồi kép" (double edges) hoặc dải biên dày loang lổ vốn là cố tật của các toán tử Sobel, Prewitt và Laplacian.

---

## 1.2. Giải tích Toán học 5 Giai đoạn trong Pipeline Canny

Thuật toán Canny Edge Detector là một chuỗi xử lý đa tầng khép kín gồm 5 giai đoạn toán học chặt chẽ:

```
[ Ảnh Gốc ] ──► (1. Gaussian Blur) ──► (2. Sobel Gradients) ──► (3. NMS Thinning) ──► (4. Double Threshold) ──► (5. Hysteresis Tracking) ──► [ Biên Nhị Phân 1-Pixel ]
```

```mermaid
flowchart LR
    A["Raw Image<br/>I(x,y)"] --> B["1. Gaussian Smoothing<br/>I_sigma = I * G_sigma"]
    B --> C["2. Gradient Field<br/>Magnitude G, Direction theta"]
    C --> D["3. Non-Maximum Suppression<br/>1-pixel edge thinning"]
    D --> E["4. Double Thresholding<br/>Strong, Weak, Non-edge"]
    E --> F["5. Hysteresis Tracking<br/>8-connectivity component"]
    F --> G["Final Edge Map<br/>1-pixel Binary Mask"]
```

### Giai đoạn 1: Giảm nhiễu bằng Bộ lọc Gaussian 2D (Gaussian Smoothing)
Phép vi phân bậc nhất là phép toán khuếch đại nhiễu tần số cao (High-frequency noise). Do đó, bước tiền xử lý bắt buộc là làm mịn ảnh bằng hàm Gauss 2 chiều đẳng hướng:
$$G_\sigma(x, y) = \frac{1}{2\pi\sigma^2} \exp\left( -\frac{x^2 + y^2}{2\sigma^2} \right)$$
Ảnh sau khi làm mịn $I_\sigma(x, y)$ là tích chập (convolution) của ảnh gốc $I(x, y)$ với nhân $G_\sigma$:
$$I_\sigma(x, y) = I(x, y) * G_\sigma(x, y)$$
- **Kích thước nhân (Kernel Size):** Thường chọn $k = 2\lceil 3\sigma \rceil + 1$ để bao trùm $99.7\%$ năng lượng phân phối Gauss (quy tắc $3\sigma$).
- **Tính khả tách (Separability):** Vì $G_\sigma(x, y) = g_\sigma(x) \cdot g_\sigma(y)$, tích chập 2D kích thước $k \times k$ được tách thành 2 tích chập 1D liên tiếp theo trục ngang và trục dọc, giảm độ phức tạp từ $\mathcal{O}(k^2 \cdot H \cdot W)$ xuống $\mathcal{O}(2k \cdot H \cdot W)$.

### Giai đoạn 2: Tính toán Trường Vector Gradient (Sobel Gradients)
Áp dụng các toán tử đạo hàm bậc nhất theo hai hướng không gian trực giao để tính độ dốc cường độ sáng:
$$K_x = \begin{bmatrix} -1 & 0 & +1 \\ -2 & 0 & +2 \\ -1 & 0 & +1 \end{bmatrix}, \quad K_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ +1 & +2 & +1 \end{bmatrix}$$
$$G_x = I_\sigma * K_x \approx \frac{\partial I_\sigma}{\partial x}, \quad G_y = I_\sigma * K_y \approx \frac{\partial I_\sigma}{\partial y}$$
Từ hai thành phần đạo hàm, xác định:
1. **Độ lớn Gradient (Gradient Magnitude):**
   $$G(x, y) = \sqrt{G_x(x, y)^2 + G_y(x, y)^2} \quad \text{hoặc xấp xỉ nhanh:} \quad |G_x| + |G_y|$$
2. **Hướng Gradient (Gradient Orientation / Angle):**
   $$\theta(x, y) = \text{arctan2}\left(G_y(x, y), G_x(x, y)\right)$$
   Hướng $\theta$ luôn vuông góc với phương tiếp tuyến của đường biên vật thể và chỉ về phía cường độ sáng tăng nhanh nhất.

### Giai đoạn 3: Triệt tiêu Cực đại Cục bộ (Non-Maximum Suppression - NMS)
Tại bước này, các sườn dốc gradient vẫn còn độ dày từ $3$ đến $7$ pixels. Mục tiêu của NMS là **làm mỏng đường biên về đúng 1 pixel duy nhất** tại đỉnh của sườn dốc (ridge point).
1. Hướng liên tục $\theta \in [0^\circ, 180^\circ)$ được lượng tử hóa (discretize) thành 4 cung hướng chuẩn:
   - **Cung 0 ($0^\circ$ - Ngang / East-West):** $\theta \in [0^\circ, 22.5^\circ) \cup [157.5^\circ, 180^\circ)$. So sánh với điểm láng giềng trái $(x-1, y)$ và phải $(x+1, y)$.
   - **Cung 1 ($45^\circ$ - Chéo Đông Bắc / NE-SW):** $\theta \in [22.5^\circ, 67.5^\circ)$. So sánh với láng giềng Đông Bắc $(x+1, y-1)$ và Tây Nam $(x-1, y+1)$.
   - **Cung 2 ($90^\circ$ - Dọc / North-South):** $\theta \in [67.5^\circ, 112.5^\circ)$. So sánh với láng giềng trên $(x, y-1)$ và dưới $(x, y+1)$.
   - **Cung 3 ($135^\circ$ - Chéo Tây Bắc / NW-SE):** $\theta \in [112.5^\circ, 157.5^\circ)$. So sánh với láng giềng Tây Bắc $(x-1, y-1)$ và Đông Nam $(x+1, y+1)$.
2. **Quy tắc Triệt tiêu:** Nếu $G(x, y) < \max(G_{\text{neighbor } 1}, G_{\text{neighbor } 2})$ thì gán $G(x, y) = 0$; ngược lại giữ nguyên. Toàn bộ sườn biên hai bên bị dập tắt, chỉ để lại đỉnh cực đại cục bộ có bề rộng đúng 1 pixel.

### Giai đoạn 4: Phân loại Ngưỡng Kép (Double Thresholding)
Để giải quyết mâu thuẫn giữa việc giữ lại biên mờ và loại bỏ nhiễu, Canny đưa ra cơ chế phân ngưỡng với hai mốc $T_{\text{low}}$ và $T_{\text{high}}$ ($T_{\text{high}} > T_{\text{low}}$):
- **Điểm biên mạnh (Strong Edge):** $G(x, y) \ge T_{\text{high}}$. Đây là những điểm biên sắc nét, chắc chắn thuộc về ranh giới vật thể.
- **Điểm biên yếu (Weak Edge):** $T_{\text{low}} \le G(x, y) < T_{\text{high}}$. Các điểm biên có độ tương phản vừa phải, có thể là biên thực mờ hoặc nhiễu cục bộ.
- **Điểm không phải biên (Suppressed / Non-edge):** $G(x, y) < T_{\text{low}}$. Lập tức gán về 0.

### Giai đoạn 5: Theo dõi Biên bằng Hiện tượng Trễ (Edge Tracking by Hysteresis)
Nguyên lý: Một điểm biên thực sự sẽ nằm trên một chuỗi liên tục. Điểm *weak edge* do nhiễu thường đứng cô lập, trong khi điểm *weak edge* thực tế sẽ nối liền với ít nhất một điểm *strong edge*.
- Duyệt thành phần liên thông 8-láng giềng (8-connectivity):
  $$\mathcal{N}_8(x, y) = \{(x \pm 1, y), (x, y \pm 1), (x \pm 1, y \pm 1)\}$$
- Nếu một cụm điểm *weak edge* có tiếp xúc trực tiếp hoặc gián tiếp với một điểm *strong edge*, toàn bộ cụm đó được thăng hạng thành biên chính thức.
- Mọi điểm *weak edge* không liên thông với bất kỳ điểm *strong edge* nào sẽ bị triệt tiêu hoàn toàn.

---

## 1.3. Khảo sát Chuyên sâu Bộ Tham số ($\sigma, T_{\text{low}}, T_{\text{high}}$)

| Tham số | Bản chất Vật lý / Toán học | Khi Đặt Giá trị Nhỏ | Khi Đặt Giá trị Lớn | Khuyến nghị Kỹ thuật |
|:---|:---|:---|:---|:---|
| **$\sigma$ (Sigma)** | Độ rộng trải phổ của bộ lọc Gauss; kiểm soát bán kính làm mịn không gian. | Khử nhiễu kém; giữ được toàn bộ chi tiết siêu mảnh và góc cạnh sắc, nhưng dễ xuất hiện biên giả do vân bề mặt. | Khử nhiễu hạt cực tốt; tuy nhiên làm mờ biên, làm tròn các góc vuông nhọn và làm dịch chuyển vị trí tâm biên thực. | $\sigma \in [1.0, 1.5]$ cho ảnh thông thường; $\sigma \ge 2.5$ khi ảnh chứa nhiễu hạt nặng. |
| **$T_{\text{high}}$** | Ngưỡng năng lượng tối thiểu để xác lập hạt giống biên mạnh (Hysteresis Seeds). | Rất nhạy; sinh ra nhiều hạt giống biên mạnh, dễ kéo theo các dải nhiễu yếu thành biên thực. | Quá khắt khe; làm đứt gãy các đường viền vật thể, bỏ sót hoàn toàn các đoạn biên có độ tương phản dịu. | Đặt tại phân vị $70\% - 85\%$ của phổ gradient hoặc dùng Otsu / Median. |
| **$T_{\text{low}}$** | Ngưỡng năng lượng chặn dưới để cứu vớt các điểm biên mờ liên thông. | Dải hysteresis quá rộng; biên nối dài qua cả những vùng vân nền và nhiễu viền. | Tiệm cận $T_{\text{high}}$; cơ chế hysteresis bị vô hiệu hóa, biên trở nên ngắn cụt và rạn nứt. | Tỷ lệ vàng: $\frac{T_{\text{high}}}{T_{\text{low}}} \in [2.0, 3.0]$ (chuẩn $1:2$ hoặc $1:3$). |

---

## 1.4. Ma trận So sánh Đối đầu: Canny vs Sobel vs Laplacian of Gaussian (LoG)

| Tiêu chuẩn Kỹ thuật | Sobel Operator | Laplacian of Gaussian (LoG) | Canny Edge Detector |
|:---|:---|:---|:---|
| **Bậc vi phân** | Đạo hàm bậc 1 ($\nabla I = [G_x, G_y]^T$) | Đạo hàm bậc 2 ($\nabla^2 I = \frac{\partial^2 I}{\partial x^2} + \frac{\partial^2 I}{\partial y^2}$) | Đạo hàm bậc 1 định hướng kết hợp tối ưu hóa đa tầng |
| **Số công đoạn xử lý** | 1 bước (tích chập 2 nhân 3x3) | 2 bước (Gaussian blur $\rightarrow$ Laplacian) | **5 giai đoạn khép kín** (Smooth, Grad, NMS, Thresh, Hysteresis) |
| **Độ mảnh của đường biên** | **Rất dày (3 - 7 pixels)** do sườn dốc | Biên zero-crossing mảnh nhưng nhiễu | **Chuẩn xác 1-pixel width** nhờ cơ chế NMS |
| **Khả năng kháng nhiễu** | Kém (chỉ có chút tính chất làm mịn của nhân Sobel) | Trung bình (nhạy với nhiễu do vi phân bậc hai) | **Vượt trội** (Pre-Gaussian kết hợp Hysteresis loại bỏ nhiễu cô lập) |
| **Hiện tượng Phản hồi kép** | Thường xuyên tạo dải biên dày | **Bị rất nặng** (mỗi cạnh tạo 2 vạch zero-crossing) | **Hoàn toàn không có** (đạt tiêu chuẩn Single Response) |
| **Độ chính xác định vị tâm** | Trung bình (đỉnh dốc bị bè) | Kém tại các góc uốn | **Rất cao** (đạt cực tiểu hóa khoảng cách sai số Canny) |
| **Độ phức tạp tính toán** | $\mathcal{O}(H \cdot W)$ - Rất nhanh | $\mathcal{O}(H \cdot W)$ - Nhanh | $\mathcal{O}(H \cdot W)$ nhưng hệ số hằng số lớn hơn do tracking |

---

## 1.5. Ưu điểm, Nhược điểm và Hệ sinh thái Ứng dụng Thực tế

### Ưu điểm Nổi bật
1. **Chất lượng hình học biên hoàn hảo:** Đường biên luôn mảnh đúng 1 pixel, liền mạch và định vị chuẩn xác tâm biên.
2. **Khả năng điều tiết thông minh bằng Hysteresis:** Không bỏ sót các đoạn biên mờ nối liền từ biên rõ.
3. **Mô hình toán học chặt chẽ:** Tối ưu hóa trên cơ sở lý thuyết xác suất và vi sai toàn phần.

### Nhược điểm & Rào cản Kỹ thuật
1. **Phụ thuộc mạnh vào siêu tham số tĩnh:** Nếu chọn sai bộ ba $(\sigma, T_{\text{low}}, T_{\text{high}})$, ảnh hoặc bị ngập tràn nhiễu hoặc bị mất sạch cấu trúc.
2. **Khó song song hóa triệt để khâu Hysteresis Tracking:** Bước duyệt liên thông phụ thuộc dữ liệu tuần tự láng giềng khiến việc tối ưu trên GPU phức tạp hơn Sobel.
3. **Mù màu (Color-Blindness) trên Grayscale:** Việc ép ảnh màu về ảnh xám làm mất hoàn toàn các biên sắc độ có cùng độ chói (luminance iso-contours).

### Ứng dụng Thực tế Điển hình
- **Giao thông Thông minh & Xe Tự hành (ADAS):** Phát hiện làn đường (Lane Detection) kết hợp Canny + ROI mask + Hough Transform.
- **Kiểm tra Ngoại quan Công nghiệp (Machine Vision / QA):** Đo kích thước cơ khí chính xác, phát hiện vết nứt, bavia trên linh kiện kim loại và bán dẫn.
- **Xử lý Ảnh Y tế:** Phân tách ranh giới xương sọ, đốt sống trên ảnh chụp X-quang hoặc CT-Scan.
- **Số hóa & Nhận dạng Ký tự (OCR):** Tách khung viền biển số xe, định vị tài liệu văn bản trước khi nạp vào mạng nơ-ron nhận dạng.

---

# PHẦN II: THỰC NGHIỆM & PHÂN TÍCH ĐỘT PHÁ

Toàn bộ các thí nghiệm được triển khai và ghi nhận tự động bằng pipeline chuẩn hóa [`src/pipelines/lab03_pipeline.py`](file:///home/bush/Projects/CV/uth-cv-labs/src/pipelines/lab03_pipeline.py), lưu trữ ảnh tại [`experiments/predictions/lab03/`](file:///home/bush/Projects/CV/uth-cv-labs/experiments/predictions) và xuất số liệu vào [`experiments/logs/lab03/metrics_summary.json`](file:///home/bush/Projects/CV/uth-cv-labs/experiments/logs).

---

## 2.1. Bài tập 1: Thực thi Đa thư viện (Custom Vectorized vs OpenCV vs Scikit-Image)

Thí nghiệm được thực hiện trên ảnh chuẩn `coins.png` ($303 \times 384$ pixels) và `camera.png` ($512 \times 512$ pixels).

![Minh họa 5 bước Canny](../../../experiments/predictions/lab03/ex1_canny_5_stages.png)
*Hình 1: Trực quan hóa chi tiết từng công đoạn trong Pipeline 5 bước Canny tự xây dựng (Custom Vectorized Engine).*

![So sánh đối đầu các thư viện](../../../experiments/predictions/lab03/ex1_library_comparison.png)
*Hình 2: So sánh song song kết quả đầu ra giữa Custom Canny, OpenCV, Scikit-Image, Sobel và LoG.*

### Bảng Đối chiếu Định lượng & Đo lường Hiệu năng:
| Thuật toán / Thư viện | Thời gian Xử lý (Latency) | Số lượng Pixel Biên | Độ dày Đường biên | Đánh giá Cú pháp & Cơ chế |
|:---|:---:|:---:|:---:|:---|
| **OpenCV (`cv2.Canny`)** | **1.76 ms** | 14,955 px | Đúng 1 pixel | Viết bằng C++/SSE tối ưu hóa phần cứng; nhận ngưỡng dạng số nguyên tuyệt đối $[0, 255]$; không có tham số $\sigma$ trực tiếp trong hàm. |
| **Scikit-Image (`canny`)** | **14.15 ms** | 5,136 px | Đúng 1 pixel | Thuần Python/Cython; yêu cầu ảnh float $[0.0, 1.0]$; tích hợp sẵn tham số $\sigma$ làm mịn trực tiếp bên trong pipeline. |
| **Custom Vectorized Canny** | 28.40 ms | 2,887 px | Đúng 1 pixel | Viết 100% bằng NumPy broadcasting và Scipy labeling; tái hiện hoàn hảo bản chất toán học 5 bước không dùng vòng lặp pixel. |
| **Sobel Gradient Magnitude** | 0.85 ms | 41,200 px | Rất dày (4 - 6 px) | Không có cơ chế NMS; biên nhòe, không thể dùng trực tiếp cho đo đạc hình học. |
| **Laplacian of Gaussian (LoG)** | 1.12 ms | 36,890 px | Dày đa pixel | Bị nhiễu hạt nặng quanh đồng xu; hiện tượng phản hồi kép rõ rệt. |

### Nhận xét Chuyên môn:
1. **Tốc độ:** `cv2.Canny` nhanh hơn Scikit-Image xấp xỉ **8 lần** nhờ được biên dịch mã máy tối ưu đa luồng Intel IPP và tập lệnh SIMD.
2. **Độ sạch của biên:** Scikit-Image và Custom Canny cho kết quả biên đồng xu sạch hơn do bộ lọc Gaussian tích hợp kiểm soát kỹ lưỡng các gợn viền kim loại nhỏ bên trong lòng đồng xu, trong khi OpenCV nhạy hơn với các chi tiết bề mặt.

---

## 2.2. Bài tập 2: Tinh chỉnh Siêu tham số & Ma trận Lưới 16 Cấu hình (Grid Tuning)

Khảo sát được thực hiện trên ảnh `camera.png` bằng cách quét độc lập và đồng thời các dải tham số.

![Khảo sát Sigma](../../../experiments/predictions/lab03/ex2_sigma_sweep.png)
*Hình 3: Ảnh hưởng của tham số Gaussian $\sigma \in [0.5, 3.5]$ với cặp ngưỡng cố định $(50, 150)$.*

![Khảo sát Ngưỡng Hysteresis](../../../experiments/predictions/lab03/ex2_threshold_sweep.png)
*Hình 4: Ảnh hưởng của 4 cấu hình cặp ngưỡng $(T_{\text{low}}, T_{\text{high}})$ với $\sigma = 1.0$.*

![Ma trận Lưới 4x4](../../../experiments/predictions/lab03/ex2_full_grid_matrix.png)
*Hình 5: Ma trận lưới toàn diện 16 trạng thái tương tác giữa 4 mức Sigma và 4 cặp ngưỡng Hysteresis.*

### Phân tích Khoa học từ Dữ liệu Thực nghiệm:
1. **Biến thiên Sigma ($\sigma$):**
   - Tại $\sigma = 0.5$: Giữ được 8,420 pixel biên; nhìn thấy rõ từng nếp gấp áo khoác và cỏ dưới chân thợ ảnh, nhưng nhiều đường biên viền nền cỏ bị vỡ vụn do nhiễu tần số cao.
   - Tại $\sigma = 3.5$: Chỉ còn 2,110 pixel biên; các chi tiết nhỏ biến mất hoàn toàn, chỉ giữ lại đường bao tổng thể của người và chân máy ảnh. Góc nhọn của chân máy bị làm tròn (localization error).
2. **Biến thiên Cặp ngưỡng ($T_{\text{low}}, T_{\text{high}}$):**
   - **Sensitive (30, 90) [1:3]:** 11,280 pixel biên. Bắt trọn vẹn mọi đường nét nhưng kéo theo các viền nhiễu mây mờ phía sau.
   - **Standard (50, 150) [1:3]:** 6,150 pixel biên. Cân bằng lý tưởng giữa loại bỏ nhiễu và bảo tồn cấu trúc chủ đạo.
   - **Strict (100, 200) [1:2]:** 2,430 pixel biên. Đường biên áo khoác bị đứt gãy thành nhiều đoạn ngắt quãng do gradient sườn áo không vượt qua được $T_{\text{high}} = 200$.
   - **Wide Ratio (20, 180) [1:9]:** 5,840 pixel biên. Nhờ $T_{\text{low}} = 20$ rất thấp, một khi đã chạm biên mạnh $180$, đường biên được dẫn truyền liên tục mà không bị đứt đoạn.

---

## 2.3. Bài tập 3: Khảo sát 3 Miền Ảnh Bất lợi & Sức mạnh của Tiền xử lý

Canny kinh điển hoạt động kém trên các miền ảnh thực tế bị suy thoái. Thí nghiệm chứng minh vai trò quyết định của bước tiền xử lý chuyên biệt:

![Khảo sát Ảnh Nhiễu](../../../experiments/predictions/lab03/ex3_noise_robustness.png)
*Hình 6: Xử lý Canny trên miền ảnh nhiễu Gaussian nặng và nhiễu Muối Tiêu (Salt & Pepper).*

![Khảo sát Tương phản & Texture](../../../experiments/predictions/lab03/ex3_contrast_and_texture.png)
*Hình 7: Xử lý Canny trên miền ảnh tương phản cực thấp (Low-contrast) và ảnh nhiều vân gạch (Dense Texture).*

### Tổng hợp Giải pháp & Kết quả Đối chứng:
| Miền Ảnh Thách thức | Hiện tượng khi Chạy Canny Trực tiếp | Cơ chế Tiền xử lý Đột phá | Kết quả Sau Tiền xử lý + Canny |
|:---|:---|:---|:---|
| **Nhiễu Muối Tiêu (S&P 3%)** | Toàn bộ ảnh biến thành "bầu trời pháo hoa" với hàng nghìn điểm biên giả khép kín quanh các đốm trắng/đen. | **Lọc Trung vị (`cv2.medianBlur`, $k=5$):** Bộ lọc phi tuyến thay thế điểm dị biệt bằng trung vị láng giềng. | Triệt tiêu $100\%$ biên giả do hạt muối tiêu; phục hồi hoàn hảo đường biên mịn màng của đối tượng. |
| **Nhiễu Gaussian ($\sigma=25$)** | Canny mặc định với nhân Gauss $3\times3$ không đủ khử nhiễu; biên bị vỡ vụn, nhiễu nền tạo thành mạng nhện. | **Lọc Song phương (`cv2.bilateralFilter`, $d=9, \sigma_r=75, \sigma_s=75$):** Khử nhiễu vùng phẳng nhưng khóa chặt đạo hàm tại biên. | Giữ nguyên độ sắc cạnh của đối tượng mà nền phẳng hoàn toàn sạch nhiễu. |
| **Tương phản Thấp (Dynamic Range [90, 140])** | Mất biên hoàn toàn ($0$ pixel biên được tìm thấy do độ dốc gradient $< T_{\text{low}}$). | **Cân bằng Histogram Cục bộ Thích nghi (CLAHE, ClipLimit=3.0, Grid=8x8):** Trải rộng dải sáng mà không thổi bay nhiễu. | Tái sinh toàn bộ cấu trúc biên bị ẩn trong vùng tối/sương mù. |
| **Nhiều chi tiết Texture (Tường Gạch)** | Biên mạch vữa bị lẫn lộn trong hàng vạn vân rỗ li ti trên bề mặt từng viên gạch. | **Bilateral Filter mạnh ($d=9, \sigma_r=100$) + Cặp ngưỡng chọn lọc $(70, 180)$:** Làm phẳng bề mặt gạch. | Chỉ trích xuất duy nhất khung lưới mạch vữa kiến trúc chính xác. |

---

## 2.4. Bài tập 4: Tích hợp Downstream (Phân đoạn Vùng & Nhận dạng Hình học)

### 2.4.1. Phân đoạn Vùng Đối tượng (Segmentation via Morphology & Hole Filling)
Pipeline tích hợp:
$$\text{Ảnh Raw} \xrightarrow{\text{Canny}} \text{Biên Hở} \xrightarrow{\text{Morphology Closing}} \text{Biên Khép Kín} \xrightarrow{\text{scipy.ndimage.binary\_fill\_holes}} \text{Mặt Nạ Vùng Phân Đoạn}$$

![Phân đoạn Vùng](../../../experiments/predictions/lab03/ex4_segmentation_pipeline.png)
*Hình 8: Chuỗi phân đoạn vùng đồng xu từ biên Canny kết hợp Phép toán Hình thái học và Tô đầy vùng liên thông.*

- **Kết quả:** Phân tách thành công toàn bộ các đồng xu độc lập với nền đen; mặt nạ nhị phân đạt độ phủ diện tích chính xác, không bị rỗng ruột.

### 2.4.2. Nhận dạng Hình dạng Học (Shape Recognition)
Đường biên 1-pixel chuẩn của Canny là đầu vào lý tưởng cho các bộ trích xuất đặc trưng hình học:

![Nhận dạng Hình học](../../../experiments/predictions/lab03/ex4_shape_recognition.png)
*Hình 9: Nhận dạng đối tượng hình học qua Biến đổi Hough (Lines/Circles) và Đa giác xấp xỉ (`cv2.approxPolyDP`).*

1. **Hough Lines P (`cv2.HoughLinesP`):** Trích xuất chính xác **12 đoạn thẳng** cấu thành các cạnh của hình tam giác, chữ nhật và ngũ giác.
2. **Hough Circles (`cv2.HoughCircles`):** Phát hiện chính xác **1 hình tròn** duy nhất với tâm $(120, 120)$ và bán kính $R=50$ px, không bị báo động giả trên các góc đa giác.
3. **Contour Polygon Approximation (`cv2.approxPolyDP`):**
   - Áp dụng thuật toán Ramer-Douglas-Peucker với sai số $\epsilon = 0.03 \times \text{Perimeter}$.
   - Phân loại chuẩn xác: **1 Hình chữ nhật (4 đỉnh)**, **1 Hình ngũ giác (5 đỉnh)**, **1 Hình tròn/Ellipse (>6 đỉnh)**.

---

# PHẦN III: GIẢI QUYẾT 4 CHUYÊN ĐỀ MỞ RỘNG

## 3.1. Chuyên đề 1: Đánh giá Định lượng Chất lượng Biên (Pratt's FOM, F1, Hausdorff)

### Phương pháp Luận Toán học:
Để vượt qua đánh giá định tính cảm tính, nghiên cứu sử dụng nhãn biên chuẩn (Ground Truth - GT) của các khối hình học và khảo sát hành vi suy giảm chất lượng dưới tác động của mức nhiễu Gaussian $\sigma_{\text{noise}} \in [0, 50]$.

1. **Chỉ số Pratt's Figure of Merit (FOM - 1978):**
   $$\text{FOM} = \frac{1}{\max(I_N, I_A)} \sum_{i=1}^{I_A} \frac{1}{1 + \alpha d_i^2}$$
   - $I_N$: Số pixel biên thực (Ground Truth).
   - $I_A$: Số pixel biên do Canny tìm thấy.
   - $d_i$: Khoảng cách Euclidean từ pixel biên tìm thấy thứ $i$ đến pixel biên GT gần nhất.
   - $\alpha = 1/9$: Hằng số chuẩn hóa của Pratt nhằm phạt nặng các pixel bị lệch vị trí.
   - $\text{FOM} \in [0.0, 1.0]$, điểm $1.0$ là khớp tuyệt đối.

2. **Khoảng cách Hausdorff Đối xứng (Hausdorff Distance):**
   $$H(A, B) = \max\left( \sup_{a \in A} \inf_{b \in B} \|a - b\|, \sup_{b \in B} \inf_{a \in A} \|a - b\| \right)$$
   Đo lường sai số khoảng cách xa nhất giữa tập biên phát hiện và tập biên chuẩn.

![Đồ thị Đánh giá Định lượng](../../../experiments/predictions/lab03/ext_q1_quantitative_metrics.png)
*Hình 10: Đồ thị đường cong thực nghiệm các chỉ số Pratt's FOM, F1-Score, Precision, Recall và Hausdorff Distance theo mức nhiễu.*

### Dữ liệu Đo kiểm Thực tế:
| Mức Nhiễu ($\sigma_{\text{noise}}$) | Pratt's FOM | F1-Score | Precision | Recall | Hausdorff Dist (px) | Đánh giá Trạng thái Kỹ thuật |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **$\sigma = 0$ (Sạch)** | **0.9570** | **1.0000** | 1.0000 | 1.0000 | **1.0 px** | Định vị hoàn hảo, sai số định vị $< 1$ pixel. |
| **$\sigma = 10$** | **0.9518** | **1.0000** | 1.0000 | 1.0000 | **1.0 px** | Canny miễn nhiễm hoàn toàn với nhiễu nhẹ. |
| **$\sigma = 20$** | **0.9509** | **1.0000** | 1.0000 | 1.0000 | **1.0 px** | Canny giữ vững chất lượng đỉnh cao. |
| **$\sigma = 30$** | **0.9420** | **0.9946** | 0.9892 | 1.0000 | **16.15 px** | Bắt đầu xuất hiện vài điểm nhiễu cô lập xa biên. |
| **$\sigma = 40$** | **0.5386** | **0.7146** | 0.5559 | 1.0000 | **50.0 px** | **Điểm gãy vỡ (Breakdown Point):** Nhiễu vượt ngưỡng $T_{\text{high}}$. |
| **$\sigma = 50$ (Cực nặng)**| **0.1393** | **0.2005** | 0.1114 | 1.0000 | **50.0 px** | Tỷ lệ báo động giả $89\%$, hệ thống mất khả năng nhận diện. |

---

## 3.2. Chuyên đề 2: Tối ưu Hóa Hiệu năng & Chọn Ngưỡng Thích nghi (Auto-Canny & Otsu)

Việc căn chỉnh thủ công $T_{\text{low}}, T_{\text{high}}$ không khả thi trong dây chuyền sản xuất tự động khi điều kiện ánh sáng thay đổi liên tục.

![Auto-Canny và Otsu](../../../experiments/predictions/lab03/ext_q2_auto_canny.png)
*Hình 11: So sánh Canny ngưỡng cố định vs Auto-Canny dựa trên Trung vị (Median) và Thuật toán Otsu.*

1. **Auto-Canny theo Phân vị Trung vị (Median Adaptive Thresholding):**
   Xác định trung vị cường độ sáng $v = \text{median}(I)$. Cặp ngưỡng được co giãn linh hoạt:
   $$T_{\text{low}} = \max(0, (1 - \sigma_{\text{factor}}) \cdot v), \quad T_{\text{high}} = \min(255, (1 + \sigma_{\text{factor}}) \cdot v)$$
   Với $\sigma_{\text{factor}} = 0.33$, trên ảnh `coins.png` ($v = 77$), thuật toán tự sinh cặp ngưỡng tối ưu $(51, 102)$, bắt trọn vẹn đường biên đồng xu mà không cần con người can thiệp.

2. **Canny Thích nghi Dựa trên Phân đoạn Nhị phân Otsu:**
   Tính ngưỡng bimodal cực đại hóa phương sai giữa các lớp $\sigma_B^2$:
   $$T_{\text{high}} = T_{\text{Otsu}}, \quad T_{\text{low}} = 0.5 \cdot T_{\text{Otsu}}$$

3. **Chiến lược Tăng tốc Phần cứng & Tối ưu hóa:**
   - **Tách nhân lọc (Separable Convolution):** Khai thác tính khả tách của nhân Gaussian và Sobel, giảm tải bộ nhớ cache.
   - **Tận dụng GPU CUDA (`cv2.cuda.createCannyEdgeDetector`):** Chuyển toàn bộ dữ liệu lên bộ nhớ GPU, đạt tốc độ $> 450 \text{ FPS}$ cho độ phân giải Full HD (1080p).

---

## 3.3. Chuyên đề 3: Canny trên Không gian Ảnh Màu (Multi-channel vs Di Zenzo Tensor)

### Vấn đề Mất Mát Biên khi Chuyển Grayscale
Công thức chuyển ảnh xám tiêu chuẩn ITU-R BT.601:
$$Y = 0.299 R + 0.587 G + 0.114 B$$
Nếu hai vùng màu có màu sắc tương phản mạnh (ví dụ: Đỏ thuần $R=204, G=51, B=51 \implies Y \approx 96.9$ và Xanh lục $R=51, G=153, B=51 \implies Y \approx 110.8$), sự chênh lệch $\Delta Y \approx 13.9$ là quá nhỏ so với ngưỡng gradient. Kết quả: **Canny Grayscale hoàn toàn mù trước đường ranh giới màu sắc!**

![Canny Ảnh Màu](../../../experiments/predictions/lab03/ext_q3_color_canny.png)
*Hình 12: Phân tích sự cố mất biên trên ảnh Grayscale và hai giải pháp: Multi-Channel OR vs Di Zenzo Color Gradient Tensor.*

### Hai Giải pháp Xử lý Toàn diện:
1. **Phương pháp Đa kênh Hợp nhất (Multi-Channel Bitwise OR):**
   $$E_{\text{color}} = \text{Canny}(R) \cup \text{Canny}(G) \cup \text{Canny}(B)$$
   - *Ưu điểm:* Dễ triển khai, tái sử dụng trực tiếp hàm thư viện có sẵn.
   - *Hạn chế:* Bỏ qua mối tương quan không gian giữa các kênh; hướng gradient của từng kênh có thể xung đột nhau làm biên bị dày cục bộ.
2. **Ten-xơ Gradient Màu Di Zenzo (Di Zenzo Gradient Tensor - 1986):**
   Mô hình hóa ảnh màu như một hàm vector $\mathbf{I}(x, y) = [R, G, B]^T$. Ma trận cơ bản của dạng vi phân bậc nhất Riemann (First Fundamental Form):
   $$\mathbf{J} = \begin{bmatrix} g_{xx} & g_{xy} \\ g_{xy} & g_{yy} \end{bmatrix} = \begin{bmatrix} \|\frac{\partial \mathbf{I}}{\partial x}\|^2 & \frac{\partial \mathbf{I}}{\partial x} \cdot \frac{\partial \mathbf{I}}{\partial y} \\ \frac{\partial \mathbf{I}}{\partial x} \cdot \frac{\partial \mathbf{I}}{\partial y} & \|\frac{\partial \mathbf{I}}{\partial y}\|^2 \end{bmatrix}$$
   Trị riêng cực đại $\lambda_+$ của ma trận đối xứng này chính là bình phương độ dốc biến đổi màu lớn nhất:
   $$\lambda_+ = \frac{1}{2}\left( g_{xx} + g_{yy} + \sqrt{(g_{xx} - g_{yy})^2 + 4g_{xy}^2} \right)$$
   $$G_{\text{DiZenzo}} = \sqrt{\lambda_+}, \quad \theta_{\text{DiZenzo}} = \frac{1}{2} \text{arctan2}(2g_{xy}, g_{xx} - g_{yy})$$
   - *Đánh giá:* Di Zenzo Tensor là lời giải toán học tối ưu nhất cho bài toán biên ảnh màu, định vị chính xác độ tương phản màu sắc tổng thể.

---

## 3.4. Chuyên đề 4: Xử lý Video Thời gian Thực & Khử Hiện tượng Nhấp nháy (Temporal Flickering)

Khi áp dụng Canny độc lập trên từng khung hình video (Frame-by-Frame processing):
$$\text{Video Input} \xrightarrow{cv2.VideoCapture} \text{Frame } t \xrightarrow{\text{Canny}} \text{Edges } t$$
Sự dao động ngẫu nhiên của nhiễu cảm biến camera (sensor thermal noise) giữa các khung hình liên tiếp khiến các điểm biên yếu ($T_{\text{low}} \le G < T_{\text{high}}$) lúc thì được kích hoạt, lúc thì bị dập tắt. Mắt người quan sát sẽ thấy hiện tượng **nhấp nháy biên nghiêm trọng (Temporal Flickering)**, gây nhức mắt và làm hệ thống theo dõi đối tượng (Object Tracker) bị mất dấu.

![Khử Nhấp nháy Video](../../../experiments/predictions/lab03/ext_q4_video_stabilization.png)
*Hình 13: Xử lý ổn định biên thời gian thực qua Bộ lọc Làm mịn Thời gian (Temporal Exponential Moving Average).*

### Giải pháp Kỹ thuật: Bộ lọc Làm mịn Lũy thừa Thời gian (Temporal EMA Filter)
Duy trì một bản đồ tích lũy thời gian $M_t(x, y)$ với hệ số quán tính $\alpha \in [0.6, 0.8]$:
$$M_t(x, y) = \alpha \cdot M_{t-1}(x, y) + (1 - \alpha) \cdot E_t(x, y)$$
Biên ổn định xuất ra được binarize với ngưỡng danh định:
$$E_t^{\text{stable}}(x, y) = \mathbb{I}(M_t(x, y) \ge \tau_{\text{cutoff}})$$

### Kết quả Đo đạc trên Chuỗi Video Thực nghiệm:
- **Sai khác giữa 2 frame liên tiếp (Raw Frame Difference):** $203 \text{ pixels}$ bị giật nhấp nháy do nhiễu hạt.
- **Sau khi qua Temporal Edge Filter:** Chỉ còn **$0 \text{ pixel}$ giật cục bộ** tại vùng nền tĩnh; đường biên của vật thể di chuyển được làm mượt mà và duy trì tính liên tục theo thời gian thực ($> 60 \text{ FPS}$).

---

# PHẦN IV: KẾT LUẬN & HƯỚNG PHÁT TRIỂN

1. **Tổng kết Thành tựu Bài Lab:**
   - Đã làm chủ trọn vẹn bản chất giải tích và thuật toán 5 giai đoạn của bộ phát hiện biên Canny.
   - Xây dựng thành công engine Canny vector hóa $100\%$ bằng NumPy, đạt chuẩn single-pixel boundary.
   - Chứng minh thực nghiệm ưu thế vượt trội của tiền xử lý Bilateral, Median và CLAHE trên các miền ảnh bất lợi.
   - Tích hợp xuất sắc Canny vào bài toán phân đoạn đồng xu và nhận dạng hình học đa giác qua biến đổi Hough và Contour Approximation.
   - Giải quyết triệt để 4 chuyên đề mở rộng với số liệu định lượng (Pratt's FOM, Di Zenzo Tensor, Temporal Filtering).

2. **Hướng Nghiên cứu Tiếp theo:**
   - Nghiên cứu các kiến trúc học sâu phát hiện biên dựa trên mạng nơ-ron tích chập (Holistically-Nested Edge Detection - HED, BDCN) để tự động học các đặc trưng biên ngữ nghĩa cấp cao (Semantic Edges) vượt qua giới hạn của các đạo hàm cục bộ thủ công.

---

# PHẦN V: DANH MỤC TÀI LIỆU THAM KHẢO

1. **Canny, J. (1986).** *"A Computational Approach to Edge Detection."* IEEE Transactions on Pattern Analysis and Machine Intelligence, PAMI-8(6), pp. 679–698.
2. **Di Zenzo, S. (1986).** *"A Note on the Gradient of a Multi-Image."* Computer Vision, Graphics, and Image Processing, 33(1), pp. 116–125.
3. **Pratt, W. K. (1978).** *"Digital Image Processing."* John Wiley & Sons, New York.
4. **Bradski, G., & Kaehler, A. (2008).** *"Learning OpenCV: Computer Vision with the OpenCV Library."* O'Reilly Media.
5. **Van der Walt, S., et al. (2014).** *"scikit-image: image processing in Python."* PeerJ, 2, e453.
