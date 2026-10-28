# BÀI THỰC HÀNH CHƯƠNG 3: SO SÁNH SỰ TƯƠNG ĐỒNG CỦA CÁC HÌNH ẢNH SỬ DỤNG WAVELET, PYTHON

> **Học phần:** Thị giác Máy tính (Computer Vision)  
> **Lab:** 04 · **Chương:** 3  
> **Chủ đề:** Image Similarity Comparison using Wavelet Transform & Perceptual Hashing in Python  
> **Tài liệu nguồn:** Chapter 3 Part 1, trang 16–18: trang 16 nêu yêu cầu; trang 17–18 giải thích và cung cấp code tham khảo.

**Phạm vi theo đề:** chuẩn bị dữ liệu → biến đổi Wavelet → lượng tử hóa và tạo mã băm → so sánh Hamming → đánh giá bằng Accuracy, Recall, Specificity và ROC. Hai bài nâng cao là khảo sát các phương pháp băm Wavelet và xây dựng ứng dụng tìm kiếm ảnh. Các nội dung bổ sung dưới đây được ghi rõ để phân biệt với yêu cầu gốc.

**Dữ liệu thực hành hiện tại:** xem [LAB_04_DATASET.md](LAB_04_DATASET.md) về 26 ảnh nguồn Chihuahua–muffin và mèo–kem, quy tắc gán nhãn, biến thể và cách chia nhóm chọn ngưỡng/đánh giá.

**Triển khai hiện tại:** [notebook Lab04](../../../notebooks/labs/lab_04.ipynb) thực hiện phần cơ bản, khảo sát độ bền được chọn và hai bài nâng cao. Xem [report](LAB_04_REPORT.md) để phân biệt kết quả đã chạy với các hướng thảo luận bổ sung.

---

## 1. Mục tiêu

1. **Nắm vững cơ sở toán học** của Biến đổi Wavelet Rời rạc 2 chiều (2D Discrete Wavelet Transform - 2D DWT) và nguyên lý phân tích đa phân giải (Multi-Resolution Analysis - MRA).
2. **Làm chủ thuật toán băm hình ảnh Wavelet (Wavelet Perceptual Hashing - wHash)**:
   - Hiểu rõ bản chất 3 công đoạn trong slide: **(1) Biến đổi Wavelet**, **(2) Lượng tử hóa hệ số (Quantization)**, và **(3) Tạo mã băm nhị phân (Hash Code Generation)**.
   - Hiểu tham số sóng mẹ và mức phân rã trong code tham khảo. Project chọn baseline **RGB, LL và ngưỡng trung vị (median) riêng cho R/G/B, ghép thành mã 192 bit**. Slide 18 có `bits=8` nhưng chưa giải thích cách sử dụng; quy ước `bits=8` là ma trận 8×8 *mỗi kênh* thuộc thiết kế của project.
3. **Thành thạo thư viện PyWavelets (`pywt`)** kết hợp với **NumPy**, **OpenCV**, **Matplotlib**, **Scikit-learn** và **Pillow** để xây dựng pipeline so sánh tương đồng hình ảnh hoàn chỉnh.
4. **Xây dựng hàm đo khoảng cách Hamming và ma trận khoảng cách đôi một (Pairwise Distance Matrix)**:
   - Triển khai thuật toán tính khoảng cách Hamming tối ưu hóa vector hóa.
   - Tính toán khoảng cách giữa tất cả các cặp ảnh $\frac{M(M-1)}{2}$ theo đúng khung chương trình mẫu trong slide bài giảng.
5. **Khảo sát bổ sung về tính bền vững:** quan sát ảnh hưởng của một số biến đổi ảnh; lựa chọn phạm vi thí nghiệm phù hợp dữ liệu.
6. **Đánh giá trên các cặp ảnh có nhãn:** báo cáo **Accuracy**, **Recall / Sensitivity**, **Specificity** và đường cong **ROC** theo trang 16. Precision, F1, AUC và tối ưu ngưỡng là nội dung bổ sung nếu triển khai.
7. **Thực hiện hai bài nâng cao:** khảo sát các phương pháp băm Wavelet và xây dựng tìm kiếm ảnh bằng mã băm. So sánh thêm aHash/dHash/pHash là mở rộng ngoài hai yêu cầu này.

---

## 2. Môi trường & Thư viện yêu cầu

- **Ngôn ngữ:** Python 3.10+
- **Các thư viện dự kiến sử dụng trong project** (slide không bắt buộc toàn bộ danh sách):
  ```bash
  pip install pywavelets opencv-python numpy matplotlib scikit-learn pillow scipy
  ```

---

## 3. Khung Mã Nguồn Tham Khảo từ Slide Bài Giảng

Hai đoạn dưới ghi lại ý tưởng và khung xử lý trong slide 17–18. Code tham khảo có chỗ chưa hoàn chỉnh; các quyết định sửa để chạy được phải được giải thích trong bài làm, không xem là yêu cầu bổ sung của slide.

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
> 1. `pywt.wavedec2` trả về `[cA_n, (cH_n, cV_n, cD_n), ..., (cH_1, cV_1, cD_1)]`. Các phần tử chi tiết là tuple, nên cần duyệt đúng cấu trúc để làm phẳng; không thể gọi `.flatten()` trực tiếp trên mọi phần tử của danh sách.
> 2. Code slide gọi `pywt.quantize`, nhưng đây không phải API có sẵn của PyWavelets. Khi triển khai phải chọn và giải thích một quy tắc lượng tử hóa. Ví dụ chia hệ số cho bước lượng tử rồi làm tròn là một lựa chọn cần thảo luận, không phải cách đã được slide xác định đầy đủ.
> 3. Trang 16–18 không quy định lấy riêng phần xấp xỉ, dùng trung vị (median), hay số bit của mã. Project đề xuất RGB–LL–median 192 bit làm baseline hiện tại; không gọi đó là thuật toán được slide 18 quy định đầy đủ.
> 4. Đoạn `zip(hash1, hash2)` có thể bỏ qua phần dư nếu hai mã khác độ dài. Bản triển khai cần kiểm tra hai mã nhị phân cùng độ dài trước khi so sánh.

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
> Theo trang 16, phần cơ bản cần Accuracy, Recall, Specificity và ROC. Precision và F1-Score có thể báo cáo bổ sung; không thay Precision cho Specificity. Khi tính chỉ số cần quy định cách xử lý mẫu số bằng 0.

---

## 4. Nội dung thực hành chi tiết

### I> Lý thuyết:

#### 1. Tìm hiểu về Biến đổi Wavelet 2D và Kỹ thuật Băm hình ảnh (Wavelet Hash - wHash):
- **a. Giải thích chi tiết 3 bước trong quy trình băm hình ảnh Wavelet (Slide 17):**
  - **1. Biến đổi Wavelet (Wavelet Transform):** Phân tích ảnh thành các hệ số ở những mức phân giải khác nhau. Giải thích vai trò của các hệ số đối với việc biểu diễn và so sánh ảnh.
  - **2. Lượng tử hóa hệ số (Coefficient Quantization):** Chuyển hệ số thực thành các mức rời rạc. Baseline của project so sánh hệ số LL đã chuẩn hóa kích thước với median: lớn hơn median thành 1, còn lại thành 0. Biến thể khảo sát dùng lượng tử hóa bước nhảy và modulo 2.
  - **3. Tạo mã băm (Hash Code Generation):** Làm phẳng ma trận bit từng kênh rồi ghép theo R–G–B. Với `bits=8`, mỗi kênh cho $8\times8=64$ bit, tổng $B=3\times64=192$ bit. Đây là quy ước thiết kế của project, không phải độ dài bắt buộc của đề.
- **b. Kiến thức giải thích thêm về 2D DWT (không phải yêu cầu lấy riêng một băng tần để băm):**
  - Cấu trúc cây lọc 2D (2D Filter Banks) qua cặp bộ lọc phân tách: Bộ lọc thông thấp $H$ (Low-pass) và Bộ lọc thông cao $G$ (High-pass).
  - Đọc cấu trúc kết quả PyWavelets: ma trận xấp xỉ `cA` (gọi là LL) và các ma trận chi tiết `cH`, `cV`, `cD`. Baseline dùng LL để biểu diễn cấu trúc tổng thể; biến thể khảo sát bên dưới dùng toàn bộ hệ số. LL không tự động có kích thước 8×8.
  - Có thể tìm hiểu thêm các họ `haar`, `db4`, `sym4`, `coif2`. Việc khảo sát nhiều họ thuộc bài nâng cao số 1.
- **c. So sánh bổ sung với các thuật toán Perceptual Hashing khác (tùy chọn):**
  - Đối chiếu toàn diện giữa 4 thuật toán:
    - **aHash (Average Hash):** Băm theo giá trị trung bình pixel miền không gian thô.
    - **dHash (Difference Hash):** Băm theo độ dốc vi phân cường độ sáng giữa các cột kề cận.
    - **pHash (DCT-based Hash):** Băm theo các hệ số tần số thấp của biến đổi Cosin rời rạc (DCT).
    - **Băm Wavelet:** Tạo mã từ hệ số Wavelet theo quy tắc lượng tử hóa và mã hóa được công bố.
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

Trang 16 có năm bước cơ bản. Để giữ cấu trúc tài liệu hiện có, phần dưới vẫn gồm bốn mục: chuẩn bị dữ liệu được nêu trong mục 1 và nhãn cặp trong mục 4; biến đổi/tạo băm ở mục 1; Hamming ở mục 2; đánh giá ở mục 4. Mục 3 là khảo sát bổ sung. Hai bài nâng cao được tách rõ sau phần này.

#### 1. Triển khai thuật toán Wavelet Hash bằng Python và PyWavelets:
- **Xây dựng hàm băm chuẩn hóa:**
  - Chuẩn bị ảnh có cặp tương tự và không tương tự; thống nhất kiểu ảnh, kích thước và cách chuẩn hóa trước khi so sánh.
  - Triển khai hàm nhận ảnh hoặc đường dẫn theo luồng: biến đổi Wavelet → lượng tử hóa hệ số → tạo mã băm nhị phân.
  - Dùng khung slide 17–18 làm tham khảo; giải thích các sửa đổi để xử lý đúng cấu trúc hệ số và thay lời gọi `pywt.quantize` chưa có sẵn.
  - Chọn **biến thể RGB–LL–median–192 bit bên dưới làm baseline mặc định của project**. Giữ biến thể lượng tử hóa bước nhảy trên ảnh xám để đối chiếu ở bài nâng cao 1. Công bố rõ tham số và các bước chuẩn hóa của từng biến thể.
- **Baseline — Biến thể RGB dùng LL và ngưỡng trung vị riêng từng kênh:**
  - Dùng `wavelet_hash_color(image, wavelet='db4', level=1, bits=8, mode='median')` với ảnh NumPy RGB `(H,W,3)`, ảnh PIL hoặc đường dẫn. `bits` là cạnh ma trận hệ số *mỗi kênh*, không phải tổng số bit.
  - Luồng chuẩn hóa: ảnh RGB → tách R/G/B → từng kênh float [0,1] và resize 256×256 → `pywt.wavedec2` với `wavelet='db4'`, `level=1`, chế độ biên `symmetric` → lấy `cA` (LL) của từng kênh. Không chuyển ảnh sang grayscale trong baseline.
  - Thu nhỏ mỗi LL về 8×8 bằng `cv2.INTER_AREA`, tính median trên 64 hệ số của *chính kênh đó*. Quy tắc `h_c = (LL_c,8x8 > median(LL_c,8x8)).astype(uint8)` tạo 64 bit/kênh; ghép R|G|B thành 192 bit.
  - Việc thu nhỏ LL là bước chuẩn hóa đặc trưng bổ sung của project, vì kích thước LL phụ thuộc ảnh, sóng mẹ, mức phân rã và chế độ biên; `level=1` không tự tạo LL 8×8.
  - Lý do chọn: người học muốn giữ thông tin màu; cách tách ba kênh giữ luồng Wavelet 2D dễ theo dõi. Median riêng từng kênh giữ thứ tự hệ số chứ không bảo đảm phản ánh chính xác cảm nhận màu. Baseline là mốc so sánh, chưa được chứng minh tối ưu hoặc bền vững với mọi biến đổi.
  - Với hệ số bằng median, gán 0; ảnh hằng có thể tạo mã toàn 0. Không coi số lượng bit 1 hay độ cân bằng bit là tiêu chí xác nhận hash đúng.
- **Biến thể khảo sát — Lượng tử hóa bước nhảy và modulo 2, lấy ý tưởng từ slide 17:**
  - Biến thể đối chiếu giữ ảnh grayscale cùng kích thước; công bố thang giá trị trước lượng tử hóa. Chọn `wavelet='db4'`, `level=3`, chế độ biên `symmetric` làm cấu hình khảo sát ban đầu. Nó không thay baseline RGB.
  - Duyệt đầy đủ ma trận xấp xỉ và từng ma trận chi tiết theo một thứ tự cố định. Với mỗi hệ số, chọn quy tắc đề xuất `q = np.rint(c / step).astype(int)`, rồi tạo bit `q % 2`; công bố `step` và thang hệ số cùng nhau.
  - `step=2` có thể được khảo sát trên thang ảnh 0–255 để gần ý tưởng code tham khảo; đây là lựa chọn triển khai thay cho lời gọi `pywt.quantize`, không khẳng định slide đã quy định phép làm tròn này.
  - Ghép các bit theo thứ tự đã chọn. Độ dài mã phụ thuộc cấu hình, không mặc định bằng 64. So sánh các cặp bằng cùng một biến thể/cấu hình; khi đối chiếu hai biến thể có độ dài khác nhau, dùng thêm Hamming chuẩn hóa và các chỉ số phân loại, không so Hamming thô như cùng thang đo.
- **Trực quan hóa bổ sung (tùy chọn):** hiển thị ảnh và hệ số Wavelet để giải thích bước trích xuất. Khảo sát nhiều sóng mẹ/mức phân rã đặt ở bài nâng cao số 1.

#### 2. So sánh mã băm bằng khoảng cách Hamming:
- **Hàm tính khoảng cách Hamming vector hóa:**
  - Triển khai `hamming_distance(hash1, hash2)` tối ưu hóa bằng NumPy bitwise XOR / popcount.
- **Ma trận khoảng cách đôi một (Pairwise Distance Matrix):**
  - Tính khoảng cách cho các cặp ảnh theo khung slide 18; kiểm tra mã nhị phân cùng độ dài. Nếu có $M$ ảnh và xét mọi cặp không lặp, có $M(M-1)/2$ cặp.
  - Có thể tổ chức thành ma trận đối xứng $M \times M$ và vẽ heatmap để quan sát; đây là cách trình bày bổ sung.
  - Việc xếp hạng Top-k để tìm kiếm ảnh thuộc bài nâng cao số 2 bên dưới.

#### 3. Khảo sát bổ sung về tính bền vững của băm Wavelet:

Các phép biến đổi và mức tham số dưới đây là gợi ý của project, không phải danh sách bắt buộc trong trang 16. Chọn một số trường hợp phù hợp, ghi rõ cấu hình và kết quả thực tế.

- **Hai nhóm biến đổi có thể thử:**
  - **Biến đổi Quang học (Photometric Attacks):**
    - Độ sáng: Tăng/giảm độ sáng ($\pm 30\%$).
    - Độ tương phản: Tăng/giảm tương phản ($\times 0.6, \times 1.4$).
    - Làm mờ: Gaussian Blur ($\sigma = 1.0, 2.0$).
    - Nén JPEG: Giảm chất lượng nén ($Q = 10, 30, 50, 80$).
    - Nhiễu hạt: Nhiễu Gaussian ($\sigma_{\text{noise}} = 15, 30$) và nhiễu muối tiêu (S&P $3\%$).
  - **Biến đổi Hình học (Geometric Changes):**
    - Thu phóng: Downscale $0.5\times$, Upscale $1.5\times$.
    - Cắt xén: Center Crop $90\%, 80\%$.
    - Xoay góc nhỏ: $\pm 2^\circ, \pm 5^\circ, \pm 10^\circ$.
    - Lật ảnh đối xứng ngang: Horizontal Flip.
- **Đánh giá và Đối chiếu:**
  - Đo khoảng cách Hamming giữa ảnh gốc và ảnh bị biến đổi.
  - Nếu mở rộng sang aHash/dHash/pHash, so sánh trên cùng dữ liệu và ghi rõ đây là phần ngoài yêu cầu khảo sát các phương pháp băm Wavelet.

#### 4. Đánh giá Định lượng Hiệu năng trên Tập Dữ liệu Ground Truth (Slide 18):
- **Xây dựng bộ dữ liệu benchmark có nhãn:**
  - **Cặp tương tự ($y=1$):** có thể là cùng đối tượng ở góc chụp khác nhau hoặc cùng ảnh với các mức nhiễu/biến đổi khác nhau, như ví dụ trang 16. Ghi rõ tiêu chí tương tự và loại cặp được chọn; không mặc định chỉ có ảnh gốc–bản chỉnh sửa.
  - **Cặp không tương tự ($y=0$):** gán theo cùng tiêu chí dữ liệu đã công bố. Có thể chọn ảnh khác đối tượng và thêm cặp dễ nhầm để khảo sát; không suy ra nhãn từ khoảng cách Hamming.
- **Chọn ngưỡng quyết định và đánh giá theo trang 16:**
  - Xây dựng ma trận nhầm lẫn ($TP, FP, TN, FN$).
  - Tính toán **Accuracy**, **Recall (Độ nhạy)** và **Specificity (Độ đặc hiệu)** tại ngưỡng được công bố.
  - Quét các ngưỡng để vẽ ROC. Nếu dùng Hamming thô, khoảng cách nhỏ hơn nghĩa là tương tự hơn; khi thư viện ROC yêu cầu điểm số lớn hơn cho lớp dương, dùng điểm như `-distance`.
- **Vẽ và Phân tích các Đường cong Đặc trưng:**
  - **Đường cong ROC (Receiver Operating Characteristic):** Biểu diễn $\text{TPR}$ theo $\text{FPR} = 1 - \text{Specificity}$.
  - **Bổ sung tùy chọn:** Precision, F1, AUC, PR curve; chọn ngưỡng theo Youden's J hoặc F1. Nếu tinh chỉnh ngưỡng, cần phân biệt dữ liệu chọn ngưỡng với dữ liệu đánh giá cuối.

### II.1> Hai bài tập nâng cao theo trang 16:

#### Nâng cao 1. Khảo sát các phương pháp băm Wavelet và so sánh hiệu suất

- Chọn các biến thể băm Wavelet để so sánh. Có thể thay sóng mẹ, mức phân rã hoặc cách lượng tử hóa; đây là các lựa chọn triển khai cho yêu cầu khảo sát.
- Dùng cùng các cặp dữ liệu và tiêu chí đánh giá. Công bố tham số, độ dài mã; nếu độ dài khác nhau, không so sánh trực tiếp Hamming thô như cùng thang đo.
- Có thể đo thêm thời gian và độ bền trước biến đổi. Không bắt buộc thử toàn bộ danh sách sóng mẹ, phép biến đổi hoặc các thuật toán aHash/dHash/pHash ở phần bổ sung.

#### Nâng cao 2. Xây dựng ứng dụng tìm kiếm ảnh dựa trên băm Wavelet

- Luồng: ảnh truy vấn → tạo mã băm → tính Hamming với các ảnh trong tập → xếp khoảng cách tăng dần → hiển thị ảnh gần nhất.
- Có thể dùng **Top-k**: lấy `k` ảnh có khoảng cách nhỏ nhất, ví dụ Top-3 trả ba ảnh gần nhất. Đây là cách triển khai ứng dụng tìm kiếm, không phải thuật toán băm mới.
- Nếu ảnh truy vấn cũng nằm trong tập, loại chính nó khỏi kết quả khi đánh giá tìm kiếm. Top-k vẫn có thể trả ảnh dù tất cả đều khác; thêm ngưỡng nếu muốn thông báo không có ảnh đủ tương tự.
- Một phần tìm kiếm tương tác trong notebook có thể minh họa luồng; trang 16 không ấn định phải xây website hay loại giao diện cụ thể.

---

### III> Các câu hỏi mở rộng (tùy chọn, ngoài hai bài nâng cao trang 16):

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
     - Tài nguyên tính toán: so sánh chi phí thực tế theo mô hình, kích thước ảnh và phần cứng; không khẳng định thời gian xử lý hoặc yêu cầu GPU khi chưa đo.
     - Các tình huống sử dụng phù hợp (Khi nào dùng wHash, khi nào dùng Deep Embeddings, và mô hình Hybrid kết hợp cả hai).
4. **Wavelet Packet Transform (WPT) và Dual-Tree Complex Wavelet Transform (DTCWT) có thể khắc phục những hạn chế gì của DWT chuẩn trong bài toán băm hình ảnh?**
   - Phân tích hai nhược điểm cốt tử của 2D DWT truyền thống:
     - Hiện tượng nhạy dịch chuyển (Shift-Variance / Lack of Translation Invariance): Dịch ảnh 1 pixel có thể làm thay đổi đáng kể phân phối năng lượng hệ số con.
     - Độ định hướng nghèo nàn (Poor Directional Selectivity): Chỉ phân tách được 3 hướng không gian ($LH, HL, HH$).
   - Giải thích cơ chế của Dual-Tree Complex Wavelet Transform (DTCWT) với 6 hướng định hướng chọn lọc ($\pm 15^\circ, \pm 45^\circ, \pm 75^\circ$) và tính gần như bất biến dịch chuyển (Near Shift-Invariance) giúp mã băm ổn định vượt bậc.

---

## 5. Yêu cầu sản phẩm đầu ra (Deliverables)

Theo cách tổ chức của project, bài làm được trình bày bằng **02 file báo cáo** dưới đây. Trang 16 không quy định định dạng Markdown/HTML. Báo cáo cần phân biệt phần cơ bản, nâng cao và bổ sung; chỉ công bố kết quả thí nghiệm đã thực hiện.

| # | Sản phẩm đầu ra | Tên file quy ước | Mô tả chi tiết yêu cầu |
|:---:|:---|:---|:---|
| 1 | **Báo cáo Kỹ thuật** | [`LAB_04_REPORT.md`](LAB_04_REPORT.md) | Giải thích Wavelet, lựa chọn lượng tử hóa/tạo mã, dữ liệu và nhãn cặp, Hamming, Accuracy/Recall/Specificity và ROC. Trình bày riêng hai bài nâng cao và các thí nghiệm bổ sung nếu đã làm; ghi rõ tham số, nhận xét và hạn chế. |
| 2 | **Báo cáo Trực quan** | [`LAB_04_REPORT.html`](LAB_04_REPORT.html) | Đồng bộ nội dung với Markdown; hiển thị ảnh, mã băm/khoảng cách, kết quả phân loại và ROC. Có thể thêm hệ số Wavelet, heatmap và kết quả Top-k theo phần đã triển khai. |
