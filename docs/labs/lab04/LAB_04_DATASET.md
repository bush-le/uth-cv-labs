# Dữ liệu cặp ảnh Lab04

## Phạm vi và quy tắc nhãn

Trạng thái: **Done — chuẩn bị dữ liệu**. [Notebook Lab04](../../../notebooks/labs/lab_04.ipynb) đã sử dụng bộ cặp này; số liệu đánh giá nằm trong [report](LAB_04_REPORT.md).

Dữ liệu nguồn do người học cung cấp: 14 ảnh trong `chihua_muffin` (7 chó, 7 muffin) và 12 ảnh trong `meo_kem` (6 mèo, 6 kem), tổng cộng 26 ảnh. Các ảnh mẫu có sẵn ở ngay thư mục `data/samples/lab04/` không tham gia bộ cặp này. Tên file không biểu thị loại đối tượng; ánh xạ loại được kiểm tra bằng mắt và ghi trong cấu hình.

Tiêu chí tương tự của bộ thử nghiệm này là **cùng ảnh nguồn**:

- Nhãn 1: ảnh gốc và một biến thể của chính nó.
- Nhãn 0: hai ảnh gốc khác loại đối tượng trong cùng chủ đề: chó–muffin hoặc mèo–kem. Xét tất cả các cặp khác loại trong từng nhóm dữ liệu, mỗi cặp một lần.

Hai ảnh cùng là chó/mèo không tự được gán nhãn 1. Bộ hiện tại không kiểm tra nhận diện cùng một vật ở các góc chụp khác nhau. Các cặp âm được chọn vì khác đối tượng nhưng có thể giống hình thức; không bảo đảm mọi cặp đều thực sự khó với thuật toán. Nhãn được xác định độc lập với Hamming.

## Luồng chuẩn bị

Ảnh nguồn → kiểm tra đọc được và trùng nội dung → chia nhóm theo loại đối tượng → tạo biến thể trong nhóm của ảnh nguồn → xuất danh sách ảnh và danh sách cặp.

Ảnh gốc giữ nguyên tại `data/samples/lab04/`. Biến thể là RGB uint8 `(H,W,3)`, giữ kích thước gốc. Chưa resize hoặc chuyển grayscale: các bước đầu vào của hash sẽ được thực hiện nhất quán khi triển khai baseline.

Mỗi ảnh nguồn có bốn biến thể:

- `bright`: nhân độ sáng 1.3 bằng Pillow, bão hòa giá trị theo thang uint8.
- `dark`: nhân độ sáng 0.7.
- `noise`: thêm Gaussian độc lập theo kênh, độ lệch chuẩn 10 trên thang 0–255; clip rồi chuyển uint8, lưu PNG.
- `jpeg`: mã hóa lại với chất lượng 50. Đây là thử nghiệm nén thêm vì ảnh nguồn có thể đã bị nén trước đó.

Các giá trị là lựa chọn minh họa ban đầu, không phải thông số tối ưu hoặc yêu cầu số lượng của đề. Seed 42 điều khiển chia nhóm và tạo nhiễu. Tên file được ánh xạ tường minh để không gán nhãn theo thứ tự chẵn/lẻ.

## Chia nhóm và số lượng

Chọn ngẫu nhiên 3 ảnh gốc mỗi loại làm `calibration` để khảo sát/chọn ngưỡng, còn lại làm `evaluation` để đánh giá cuối. Mọi biến thể đi cùng ảnh gốc của nó. Không có ảnh nguồn dùng chung giữa hai nhóm.

- **Calibration:** 12 ảnh gốc, 48 biến thể; 48 cặp dương và 18 cặp âm, tổng 66 cặp.
- **Evaluation:** 14 ảnh gốc, 56 biến thể; 56 cặp dương và 25 cặp âm, tổng 81 cặp.
- **Toàn bộ:** 26 ảnh gốc, 104 biến thể, 130 ảnh và 147 cặp.

Hai nhãn không cân bằng. Khi báo cáo, công bố số cặp và đọc Accuracy cùng Recall, Specificity, ROC. Các cặp dùng chung ảnh nên không phải 147 quan sát hoàn toàn độc lập. Không chọn ngưỡng trên nhóm evaluation rồi dùng kết quả đó làm đánh giá cuối.

Chia theo file ảnh nguồn chưa bảo đảm chia theo cá thể động vật, người chụp hoặc nguồn website: bộ ảnh chưa có metadata xác nhận các yếu tố này. Cần ghi nguồn tải/tác giả nếu có, và kiểm tra các ảnh gần trùng trước khi mở rộng bộ dữ liệu.

## File và cách chạy lại

- [Cấu hình](../../../configs/dataset/lab04_pairs.yaml): ánh xạ loại ảnh, seed, số ảnh chọn ngưỡng và tham số biến đổi.
- [Module chuẩn bị](../../../src/data/lab04_pairs.py): chạy từ thư mục project bằng `python -m src.data.lab04_pairs`.
- Đầu ra sinh tự động nằm tại `data/processed/lab04_pairs/`, được Git ignore theo quy tắc project.

Các file đầu ra:

- `sources.csv`: ID nguồn, chủ đề, loại, nhóm, đường dẫn, SHA-256 của file nguồn và kích thước.
- `images.csv`: danh sách ảnh gốc/biến thể, ID nguồn và phép biến đổi. Trường `sha256` kế thừa từ ảnh nguồn để truy vết, không phải checksum file biến thể.
- `pairs_calibration.csv`, `pairs_evaluation.csv`: hai đường dẫn ảnh, ID, nhãn và loại cặp.
- `summary.json`: số lượng thực tế.
- `generation_config.json`: cấu hình dùng để sinh dữ liệu.

Chạy lại cùng dữ liệu/cấu hình tái tạo các file cùng tên. Nếu thay bộ ảnh hoặc cấu hình, dùng thư mục đầu ra mới để tránh giữ nhầm biến thể cũ. Module dừng nếu danh sách file PNG trong hai folder khác ánh xạ cấu hình; cần kiểm tra và cập nhật ánh xạ khi thêm/xóa ảnh.

## Kiểm tra đã thực hiện

- Smoke test: tạo được ảnh và manifest với dữ liệu nhỏ.
- Normal tests: nhãn đúng nguồn/loại, nhóm không giao nhau, chạy lại cho cùng nội dung, ảnh nguồn không thay đổi.
- Chạy trên 26 ảnh thật: sinh đủ 104 biến thể và 147 cặp; không có file nguồn trùng SHA-256.

Notebook nạp các CSV để thực hiện Wavelet → hash → Hamming và đánh giá theo description. Các phép crop/xoay/lật khảo sát bổ sung trong notebook không được thêm vào manifest evaluation ban đầu.
