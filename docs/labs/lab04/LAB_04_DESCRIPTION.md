# BÀI THỰC HÀNH CHƯƠNG 3: SO SÁNH SỰ TƯƠNG ĐỒNG CỦA CÁC HÌNH ẢNH SỬ DỤNG WAVELET, PYTHON

> **Học phần:** Thị giác Máy tính (Computer Vision)  
> **Lab:** 04 · **Chương:** 3  
> **Chủ đề:** Image Similarity Comparison using Wavelet Transform & Perceptual Hashing in Python  
> **Tài liệu nguồn:** Slide Bài giảng Chương 3 (Slide 17 & Slide 18: Bài thực hành 4)  

---

## 1. Mục tiêu

1. **Nắm vững cơ sở toán học** của Biến đổi Wavelet Rời rạc 2 chiều (2D Discrete Wavelet Transform - 2D DWT) và nguyên lý phân tích đa phân giải (Multi-Resolution Analysis - MRA).
2. **Làm chủ thuật toán băm hình ảnh Wavelet (Wavelet Perceptual Hashing - wHash)**:
   - Hiểu rõ bản chất 3 công đoạn trong slide: **(1) Biến đổi Wavelet**, **(2) Lượng tử hóa hệ số (Quantization)**, và **(3) Tạo mã băm nhị phân (Hash Code Generation)**.
   - Nắm vững ý nghĩa của các tham số: sóng mẹ (`wavelet='db4'`, `haar`, `sym4`), mức độ phân rã (`level=1` hoặc `level=3`), và kích thước chuỗi băm (`bits=8`).
3. **Thành thạo thư viện PyWavelets (`pywt`)** kết hợp với **NumPy**, **OpenCV**, **Matplotlib**, **Scikit-learn** và **Pillow** để xây dựng pipeline so sánh tương đồng hình ảnh hoàn chỉnh.
4. **Xây dựng hàm đo khoảng cách Hamming và ma trận khoảng cách đôi một (Pairwise Distance Matrix)**:
   - Triển khai thuật toán tính khoảng cách Hamming tối ưu hóa vector hóa.
   - Tính toán khoảng cách giữa tất cả các cặp ảnh $\frac{M(M-1)}{2}$ theo đúng khung chương trình mẫu trong slide bài giảng.
5. **Đánh giá tính bền vững (Robustness Analysis)** của mã băm Wavelet trước các biến đổi quang học (độ sáng, độ tương phản, làm mờ Gaussian, nén JPEG, nhiễu hạt) và biến đổi hình học (thu phóng, cắt xén, lật ảnh, xoay góc).
6. **Đo lường định lượng và tối ưu hóa hệ thống trên tập dữ liệu có nhãn Ground Truth**:
   - Tính toán đầy đủ các chỉ số thống kê từ ma trận nhầm lẫn: Độ chính xác (**Accuracy**), Độ nhạy (**Recall / Sensitivity**), Độ đặc hiệu (**Specificity**) và Độ xác thực (**Precision**), Điểm F1 (**F1-Score**).
   - Vẽ đường cong **ROC (Receiver Operating Characteristic)** và tính diện tích dưới đường cong **AUC (Area Under Curve)**.
   - Tìm kiếm ngưỡng quyết định phân lớp tối ưu $\tau^*$ (Optimal Similarity Threshold).
7. **So sánh đối đầu toàn diện** giữa Wavelet Hash (wHash) với các thuật toán băm cảm nhận kinh điển khác (aHash, dHash, pHash) và phân tích các bài toán ứng dụng thực tế quy mô lớn.

---

## 2. Môi trường & Thư viện yêu cầu

- **Ngôn ngữ:** Python 3.10+
- **Cài đặt thư viện bắt buộc:**
  ```bash
  pip install pywavelets opencv-python numpy matplotlib scikit-learn pillow scipy
  ```

---

## 3. Khung Mã Nguồn Chuẩn từ Slide Bài Giảng

Bản mô tả bài lab bám sát tuyệt đối 2 đoạn mã nguồn cốt lõi trong slide bài giảng của bộ môn:

### 3.1. Slide 17: Thuật Toán Wavelet Hash Cơ Bản & Tính Khoảng Cách Hamming
```python
import pywt
import numpy as np
import matplotlib.pyplot as plt

def wavelet_hash(image_path, wavelet='db4', level=3):
    """
    Calculates the wavelet hash of an image.

    Args:
        image_path: Path to the image file.
        wavelet: Wavelet type to use.
        level: Decomposition level.

    Returns:
        A list of bits representing the hash code.
    """
    # Load the image
    img = plt.imread(image_path)

    # Perform wavelet transform
    coeffs = pywt.wavedec2(img, wavelet, level=level)

    # Quantize coefficients
    coeffs = [pywt.quantize(c, 2) for c in coeffs]

    # Flatten the quantized coefficients
    flattened = np.concatenate([c.flatten() for c in coeffs])

    # Generate hash code
    hash_code = []
    for bit in flattened:
        hash_code.append(bit % 2)

    return hash_code

# Example usage:
hash1 = wavelet_hash("image1.jpg")
hash2 = wavelet_hash("image2.jpg")

# Compare hashes
hamming_distance = sum(abs(a - b) for a, b in zip(hash1, hash2))
print("Hamming distance:", hamming_distance)
```

> [!NOTE]
> **Lưu ý Kỹ thuật về `pywt.quantize` và cấu trúc `coeffs`:**
> 1. Hàm `pywt.wavedec2` trả về một danh sách có cấu trúc phân tầng: `[cA_n, (cH_n, cV_n, cD_n), ..., (cH_1, cV_1, cD_1)]`, trong đó phần tử đầu tiên là ma trận xấp xỉ 2D ($LL$), và các phần tử tiếp theo là **tuple gồm 3 ma trận chi tiết** ($LH, HL, HH$).
> 2. Trong PyWavelets chính thức không có hàm `pywt.quantize`. Do đó, dòng `pywt.quantize(c, 2)` trong slide đại diện cho **bước lượng tử hóa hệ số (Quantization Step)**. Trong thực hành, bước này được triển khai chuẩn hóa theo 2 cách tiếp cận kỹ thuật:
>    - **Cách 1 (Lượng tử hóa bước nhảy - Step Quantization):** Chia cho kích thước bước lượng tử `step=2` và làm tròn `np.round(c / 2).astype(int)`.
>    - **Cách 2 (Lượng tử hóa trung vị - Median Quantization):** So sánh hệ số với giá trị trung vị hoặc trung bình của dải tần xấp xỉ `(c > np.median(c)).astype(int)` (đây là cách tiếp cận chuẩn mực của thuật toán wHash công nghiệp).

---

### 3.2. Slide 18: Khung Chương Trình Đánh Giá So Sánh Tương Đồng & Đường Cong ROC
```python
import pywt
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def wavelet_hash(image_path, wavelet='db4', level=1, bits=8):
    # ... (tương tự như phần code đã cung cấp)

# Load dữ liệu
images = ["image1.jpg", "image2.jpg", ...]
hashes = [wavelet_hash(img) for img in images]

# Tính khoảng cách Hamming
def hamming_distance(hash1, hash2):
    # ...

# So sánh và đánh giá
distances = []
for i in range(len(hashes)):
    for j in range(i + 1, len(hashes)):
        distance = hamming_distance(hashes[i], hashes[j])
        distances.append(distance)
        # So sánh với nhãn ground truth để tính toán các chỉ số đánh giá

# Vẽ đường cong ROC
# ...

# Đánh giá kết quả
print("Độ chính xác:", accuracy)
print("Độ nhạy:", recall)
print("Độ đặc hiệu:", precision)
```

> [!IMPORTANT]
> **Lưu ý Kỹ thuật về Các Chỉ Số Đánh Giá:**
> Trong slide bài giảng, dòng in kết quả có đoạn `print("Độ đặc hiệu:", precision)`.
> Về mặt thuật ngữ chuẩn trong Khoa học Dữ liệu và Thị giác Máy tính:
> - **Độ chính xác (Accuracy):** Tỷ lệ phân loại đúng trên tổng số cặp so sánh: $\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$.
> - **Độ nhạy (Sensitivity / Recall):** Tỷ lệ phát hiện đúng các cặp ảnh thực sự giống nhau: $\text{Recall} = \frac{TP}{TP + FN}$.
> - **Độ đặc hiệu (Specificity / True Negative Rate):** Tỷ lệ nhận diện đúng các cặp ảnh thực sự khác nhau: $\text{Specificity} = \frac{TN}{TN + FP}$.
> - **Độ xác thực (Precision / Positive Predictive Value):** Tỷ lệ các cặp được dự đoán giống nhau mà thực tế đúng là giống nhau: $\text{Precision} = \frac{TP}{TP + FP}$.
> 
> Trong bài thực hành này, sinh viên và kỹ sư **bắt buộc phải tính toán và báo cáo đầy đủ cả 5 chỉ số trên** (Accuracy, Recall, Specificity, Precision, F1-Score) để bức tranh đánh giá hoàn toàn sáng tỏ và khoa học.

---

## 4. Nội dung thực hành chi tiết

### I> Lý thuyết:

#### 1. Tìm hiểu về Biến đổi Wavelet 2D và Kỹ thuật Băm hình ảnh (Wavelet Hash - wHash):
- **a. Giải thích chi tiết 3 bước trong quy trình băm hình ảnh Wavelet (Slide 17):**
  - **1. Biến đổi Wavelet (Wavelet Transform):** Phân tích hình ảnh từ miền không gian sang miền không gian - tần số đa phân giải. Giải thích ưu thế vượt trội của Wavelet so với biến đổi Fourier (FFT) và Cosin (DCT) trong việc định vị đồng thời cả đặc trưng tần số lẫn tọa độ không gian.
  - **2. Lượng tử hóa hệ số (Coefficient Quantization):** Giảm độ chính xác của các hệ số Wavelet thực thành các mức rời rạc. Phân tích cơ chế ngưỡng (ngưỡng trung bình hoặc trung vị) để nén dữ liệu và triệt tiêu các dao động tần số cao ngẫu nhiên (nhiễu).
  - **3. Tạo mã băm (Hash Code Generation):** Ghép nối các hệ số lượng tử hóa thành chuỗi bit nhị phân cố định $\mathbf{h} \in \{0, 1\}^B$ (ví dụ: $B = 64$ bits với `bits=8`, hoặc $B = 256$ bits với `bits=16`).
- **b. Cơ sở toán học của Biến đổi Wavelet Rời rạc 2 chiều (2D DWT):**
  - Cấu trúc cây lọc 2D (2D Filter Banks) qua cặp bộ lọc phân tách: Bộ lọc thông thấp $H$ (Low-pass) và Bộ lọc thông cao $G$ (High-pass).
  - Bản chất vật lý của 4 băng tần con (Subbands) tại mỗi mức phân rã ($level$):
    - **$LL$ (Approximation):** Băng xấp xỉ tần số thấp theo cả 2 trục; chứa năng lượng chủ đạo và cấu trúc thô của ảnh.
    - **$LH$ (Horizontal Details):** Tần số thấp trục $X$, tần số cao trục $Y$; làm nổi bật đường viền ngang.
    - **$HL$ (Vertical Details):** Tần số cao trục $X$, tần số thấp trục $Y$; làm nổi bật đường viền dọc.
    - **$HH$ (Diagonal Details):** Tần số cao cả 2 trục; chứa chi tiết chéo và nhiễu viền.
  - Khảo sát các họ Wavelet mẹ thông dụng:
    - **Haar (`haar`):** Đơn giản nhất, trực giao, đáp ứng xung ngắn, tính toán nhanh.
    - **Daubechies (`db4`):** Trực giao, khả vi, có độ phẳng tối đa (maximally flat), bảo toàn năng lượng xuất sắc.
    - **Symlets (`sym4`):** Phiên bản gần như đối xứng của Daubechies, hạn chế méo pha không gian.
    - **Coiflets (`coif2`):** Đối xứng hơn ở cả hàm tỷ lệ và sóng mẹ.
- **c. So sánh Wavelet Hash (wHash) với các thuật toán Perceptual Hashing khác:**
  - Đối chiếu toàn diện giữa 4 thuật toán:
    - **aHash (Average Hash):** Băm theo giá trị trung bình pixel miền không gian thô.
    - **dHash (Difference Hash):** Băm theo độ dốc vi phân cường độ sáng giữa các cột kề cận.
    - **pHash (DCT-based Hash):** Băm theo các hệ số tần số thấp của biến đổi Cosin rời rạc (DCT).
    - **wHash (Wavelet Hash):** Băm theo hệ số xấp xỉ đa phân giải của biến đổi Wavelet.
  - So sánh chi tiết về: Tốc độ tính toán, kích thước mã băm, độ bền trước nén JPEG, độ bền trước co giãn (scaling) và biến đổi hình học.

#### 2. Độ đo Tương đồng & Khoảng cách Hamming (Hamming Distance):
- **a. Bản chất toán học của khoảng cách Hamming:**
  - Khoảng cách Hamming giữa 2 chuỗi băm $B$-bit $\mathbf{h}_1$ và $\mathbf{h}_2$:
    $$D_H(\mathbf{h}_1, \mathbf{h}_2) = \sum_{k=1}^B |h_{1, k} - h_{2, k}| = \sum_{k=1}^B (h_{1, k} \oplus h_{2, k}) = \text{popcount}(\mathbf{h}_1 \oplus \mathbf{h}_2)$$
  - Khoảng cách Hamming chuẩn hóa (Normalized Hamming Distance):
    $$d_H(\mathbf{h}_1, \mathbf{h}_2) = \frac{D_H(\mathbf{h}_1, \mathbf{h}_2)}{B} \in [0.0, 1.0]$$
  - Độ tương đồng tương ứng: $\text{Similarity} = 1.0 - d_H(\mathbf{h}_1, \mathbf{h}_2)$.
- **b. Quy tắc phân lớp nhị phân và Ngưỡng quyết định ($\tau$):**
  - Quy tắc quyết định:
    $$\text{Phán quyết} = \begin{cases} \text{Tương đồng (Similar / Duplicate / Positive)}, & \text{nếu } D_H(\mathbf{h}_1, \mathbf{h}_2) \le \tau \\ \text{Khác biệt (Dissimilar / Negative)}, & \text{nếu } D_H(\mathbf{h}_1, \mathbf{h}_2) > \tau \end{cases}$$
  - Phân tích sự đánh đổi khi chọn $\tau$: $\tau$ nhỏ (chống báo động giả nhưng dễ sót biến thể) vs $\tau$ lớn (bắt trọn biến thể nhưng dễ nhận nhầm).

#### 3. Ứng dụng thực tế của Wavelet Image Hashing:
- **a. Truy vấn và Tìm kiếm Hình ảnh Dựa trên Nội dung (CBIR - Content-Based Image Retrieval).**
- **b. Bảo vệ Bản quyền Số & Phát hiện Vi phạm Sở hữu Trí tuệ (Copyright Infringement / Watermarking).**
- **c. Lọc và Loại bỏ Dữ liệu Trùng lặp (Dataset Deduplication) trong MLOps & Computer Vision.**
- **d. Giám định Kỹ thuật số & An ninh Thông tin (Digital Forensics).**

---

### II> Bài tập thực hành:

#### 1. Triển khai thuật toán Wavelet Hash bằng Python và PyWavelets:
- **Xây dựng hàm băm chuẩn hóa:**
  - Triển khai hàm `wavelet_hash(image, wavelet='db4', level=1, bits=8, mode='median')` chấp nhận đường dẫn ảnh hoặc mảng NumPy/PIL.
  - Xây dựng 2 biến thể:
    - **Biến thể Slide 17:** Phân rã $level=3$, làm phẳng toàn bộ hệ số và lượng tử hóa modulo 2.
    - **Biến thể Slide 18:** Phân rã $level=1$ với kích thước $bits=8$ ($64$-bit hash code) dựa trên dải tần xấp xỉ $LL$.
- **Trực quan hóa cấu trúc Wavelet 2D:**
  - Vẽ lưới ảnh trực quan hiển thị 4 dải tần $LL, LH, HL, HH$ ở các mức $level=1, 2, 3$.
  - Khảo sát sự khác biệt khi thay đổi các họ sóng mẹ: `haar`, `db2`, `db4`, `sym4`, `coif2`.

#### 2. Xây dựng Ma trận Khoảng cách Hamming và Tìm kiếm Ảnh tương đồng:
- **Hàm tính khoảng cách Hamming vector hóa:**
  - Triển khai `hamming_distance(hash1, hash2)` tối ưu hóa bằng NumPy bitwise XOR / popcount.
- **Ma trận khoảng cách đôi một (Pairwise Distance Matrix):**
  - Cho tập ảnh thử nghiệm $M$ ảnh, tính toán toàn bộ ma trận khoảng cách đối xứng $M \times M$ theo vòng lặp `for i in range(len(hashes)): for j in range(i+1, len(hashes)):`.
  - Vẽ biểu đồ nhiệt (Heatmap Matrix) thể hiện ma trận khoảng cách giữa các nhóm ảnh.
- **Truy vấn Top-k Ảnh tương đồng nhất (Top-k Retrieval):**
  - Cho ảnh truy vấn (Query Image), tìm kiếm và hiển thị trực quan Top-k ảnh có khoảng cách Hamming nhỏ nhất trong cơ sở dữ liệu.

#### 3. Khảo sát Tính Bền vững (Robustness Analysis) của Wavelet Hash:
- **Thử nghiệm trên 2 nhóm biến đổi ảnh suy thoái:**
  - **Biến đổi Quang học (Photometric Attacks):**
    - Độ sáng: Tăng/giảm độ sáng ($\pm 30\%$).
    - Độ tương phản: Tăng/giảm tương phản ($\times 0.6, \times 1.4$).
    - Làm mờ: Gaussian Blur ($\sigma = 1.0, 2.0$).
    - Nén JPEG: Giảm chất lượng nén ($Q = 10, 30, 50, 80$).
    - Nhiễu hạt: Nhiễu Gaussian ($\sigma_{\text{noise}} = 15, 30$) và nhiễu muối tiêu (S&P $3\%$).
  - **Biến đổi Hình học Nhẹ (Geometric Attacks):**
    - Thu phóng: Downscale $0.5\times$, Upscale $1.5\times$.
    - Cắt xén: Center Crop $90\%, 80\%$.
    - Xoay góc nhỏ: $\pm 2^\circ, \pm 5^\circ, \pm 10^\circ$.
    - Lật ảnh đối xứng ngang: Horizontal Flip.
- **Đánh giá và Đối chiếu:**
  - Đo khoảng cách Hamming giữa ảnh gốc và ảnh bị biến đổi.
  - So sánh tính bền vững của **wHash** so với **aHash**, **dHash**, **pHash (DCT)** trên cùng bộ biến đổi.

#### 4. Đánh giá Định lượng Hiệu năng trên Tập Dữ liệu Ground Truth (Slide 18):
- **Xây dựng bộ dữ liệu benchmark có nhãn:**
  - **Tập Dương bản (Positive Pairs - Cùng nguồn gốc):** Cặp gồm ảnh gốc và các biến thể bị chỉnh sửa ($y = 1$).
  - **Tập Âm bản (Negative Pairs - Khác nguồn gốc):** Cặp gồm 2 bức ảnh có nội dung hoàn toàn khác biệt ($y = 0$).
- **Quét ngưỡng $\tau \in [0, B]$ và Tính toán Toàn diện các Chỉ số:**
  - Xây dựng ma trận nhầm lẫn ($TP, FP, TN, FN$).
  - Tính toán: **Accuracy**, **Recall (Độ nhạy)**, **Specificity (Độ đặc hiệu)**, **Precision (Độ xác thực)**, **F1-Score**.
- **Vẽ và Phân tích các Đường cong Đặc trưng:**
  - **Đường cong ROC (Receiver Operating Characteristic):** Biểu diễn $\text{TPR}$ theo $\text{FPR} = 1 - \text{Specificity}$.
  - **Chỉ số AUC:** Tính diện tích dưới đường cong ROC.
  - **Đường cong Precision-Recall (PR Curve).**
  - **Xác định Ngưỡng Quyết định Tối ưu $\tau^*$:** Theo tiêu chí cực đại hóa Youden's J Index ($J = \text{TPR} - \text{FPR}$) và cực đại hóa $F_1$-Score.

---

### III> Các câu hỏi mở rộng:

1. **Làm thế nào để cải thiện tính bền vững của Wavelet Hash trước phép xoay góc lớn (Rotation > 30°, 90°, 180°)?**
   - Phân tích nguyên nhân tại sao 2D DWT dạng lưới Descartes chuẩn bị suy giảm nghiêm trọng khi ảnh bị xoay góc lớn.
   - Các giải pháp kỹ thuật: Chuẩn hóa hướng xoay dựa trên mô-men hình ảnh (Image Moments / Principal Orientation), Biến đổi Log-Polar Fourier-Mellin kết hợp Wavelet, hoặc cơ chế Multi-Hash xoay 4 hướng ($0^\circ, 90^\circ, 180^\circ, 270^\circ$).
2. **Làm thế nào để mở rộng thuật toán tìm kiếm ảnh tương đồng trong cơ sở dữ liệu hàng triệu ảnh (Large-Scale CBIR)?**
   - Phân tích rào cản thời gian của phương pháp quét tuyến tính $\mathcal{O}(N)$ khoảng cách Hamming khi $N \ge 10^7$.
   - Các cấu trúc dữ liệu và giải thuật lập chỉ mục không gian băm:
     - Cây điểm tựa (VP-Tree - Vantage Point Tree, BK-Tree - Burkhard-Keller Tree).
     - Băm phụ thuộc độ gần (LSH - Locality-Sensitive Hashing).
     - Multi-Index Hashing (MIH): Chia chuỗi băm $64$-bit thành $m$ khối con ngắn để tra cứu trực tiếp bảng băm $\mathcal{O}(1)$ theo nguyên lý Dirichlet (Pigeonhole Principle).
3. **Sự khác biệt cốt lõi giữa Wavelet Perceptual Hash (wHash) và các Vector Đặc trưng Học Sâu (Deep Visual Embeddings như ResNet, ViT, CLIP)?**
   - So sánh chi tiết về:
     - Bản chất biểu diễn: Đặc trưng tần số - không gian cục bộ (Low-level perception) vs Đặc trưng ngữ nghĩa trừu tượng cấp cao (High-level semantic concept).
     - Chi phí lưu trữ: Chuỗi bit siêu nén ($64 - 256$ bits) vs Vector thực dày đặc ($512 - 2048$ floats 32-bit).
     - Tài nguyên tính toán: Chạy nhẹ nhàng trên CPU (< 1 ms/ảnh) vs Đòi hỏi bộ tăng tốc GPU chuyên dụng.
     - Các tình huống sử dụng phù hợp (Khi nào dùng wHash, khi nào dùng Deep Embeddings, và mô hình Hybrid kết hợp cả hai).
4. **Wavelet Packet Transform (WPT) và Dual-Tree Complex Wavelet Transform (DTCWT) có thể khắc phục những hạn chế gì của DWT chuẩn trong bài toán băm hình ảnh?**
   - Phân tích hai nhược điểm cốt tử của 2D DWT truyền thống:
     - Hiện tượng nhạy dịch chuyển (Shift-Variance / Lack of Translation Invariance): Dịch ảnh 1 pixel có thể làm thay đổi đáng kể phân phối năng lượng hệ số con.
     - Độ định hướng nghèo nàn (Poor Directional Selectivity): Chỉ phân tách được 3 hướng không gian ($LH, HL, HH$).
   - Giải thích cơ chế của Dual-Tree Complex Wavelet Transform (DTCWT) với 6 hướng định hướng chọn lọc ($\pm 15^\circ, \pm 45^\circ, \pm 75^\circ$) và tính gần như bất biến dịch chuyển (Near Shift-Invariance) giúp mã băm ổn định vượt bậc.

---

## 5. Yêu cầu sản phẩm đầu ra (Deliverables)

Theo đúng quy chuẩn quản trị chất lượng kỹ thuật của bộ môn Thị giác Máy tính UTH, sản phẩm bài lab bao gồm **02 file báo cáo hoàn chỉnh**:

| # | Sản phẩm đầu ra | Tên file quy ước | Mô tả chi tiết yêu cầu |
|:---:|:---|:---|:---|
| 1 | **Báo cáo Kỹ thuật Toàn diện** | [`LAB_04_REPORT.md`](file:///home/bush/Projects/CV/uth-cv-labs/docs/labs/lab04/LAB_04_REPORT.md) | File Markdown báo cáo **chi tiết, toàn bộ và hoàn chỉnh nhất có thể**: trình bày cặn kẽ lý thuyết toán học 2D DWT, giải thuật lượng tử hóa sinh mã băm, phân tích định tính và định lượng cho cả 4 bài thực hành, đối chiếu so sánh wHash với aHash/dHash/pHash, vẽ và phân tích đường cong ROC/PR, giải quyết sâu sắc 4 câu hỏi mở rộng kèm công thức, bảng đối chiếu và chứng minh khoa học. |
| 2 | **Báo cáo Trực quan Đa phương tiện** | [`LAB_04_REPORT.html`](file:///home/bush/Projects/CV/uth-cv-labs/docs/labs/lab04/LAB_04_REPORT.html) | File HTML **trực quan nhất, show được chi tiết toàn bộ bài làm để báo cáo và thuyết trình**: giao diện hiện đại, responsive, trực quan hóa từng dải tần LL/LH/HL/HH trong cây wavelet, hiển thị so sánh ảnh kết quả và ma trận khoảng cách heatmap, biểu đồ tương tác ROC/AUC và bảng đối chiếu đa thuật toán, code snippets có tô màu cú pháp. |
