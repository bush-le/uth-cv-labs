# UTH Computer Vision Workspace (`uth-cv-labs`)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

> **Đơn vị**: Trường Đại học Giao thông Vận tải TP.HCM (UTH).

---

## 📌 Giới thiệu Tổng quan (Overview)

Branch `lab-04` chứa bài so sánh ảnh bằng Wavelet hash trên ảnh Chihuahua–muffin và mèo–kem do người học cung cấp.

- [Notebook Lab04](notebooks/labs/lab_04.ipynb): bản thực hành chính, chạy **Restart Kernel → Run All**.
- [Description](docs/labs/lab04/LAB_04_DESCRIPTION.md) · [Dữ liệu và nhãn](docs/labs/lab04/LAB_04_DATASET.md).
- [Report Markdown](docs/labs/lab04/LAB_04_REPORT.md) · [Report HTML](docs/labs/lab04/LAB_04_REPORT.html) (nhúng sẵn hình).
- [Cấu hình dữ liệu](configs/dataset/lab04_pairs.yaml) · [Cấu hình hash/notebook](configs/experiment/lab04_wavelet.yaml).

Cài `pip install -r requirements/base.txt` và `pip install jupyter nbformat nbclient ipykernel markdown`. Chạy `python -m src.data.lab04_pairs` để tạo CSV và biến thể tại `data/processed/lab04_pairs/` trước khi mở notebook. Ảnh nguồn vẫn giữ trong `data/samples/lab04/chihua_muffin/` và `meo_kem/`.

Có thể chạy `python scripts/lab04/run_notebook.py` để thực thi notebook, kiểm tra và xuất hình/số liệu vào `docs/assets/lab04/`. Nếu thay thí nghiệm, cập nhật nhận xét và số liệu trong report Markdown rồi chạy `python scripts/lab04/render_report.py` để tạo HTML có nhúng hình. Các file trong `experiments/` là kết quả tạm và không cần để chạy bài hiện tại.

Baseline hiện tại là RGB → băm từng kênh bằng db4 level 1 → LL 8×8 → median riêng từng kênh → ghép R|G|B thành 192 bit, do project lựa chọn. Biến thể bước nhảy trên ảnh xám chỉ dùng làm phép đối chiếu ở bài nâng cao. Notebook chọn ngưỡng trên calibration rồi giữ nguyên để đánh giá evaluation. Tham số sinh biến thể lấy từ cấu hình dữ liệu; tham số hash và khảo sát hình học lấy từ cấu hình thí nghiệm.

Luồng thực thi gồm `src/data/lab04_pairs.py`, các hàm hash trong `src/models/classical/wavelet_hash.py`, đánh giá trong `src/pipelines/lab04_evaluation.py` và các hàm vẽ trong `src/pipelines/lab04_visuals.py` để notebook tập trung vào các bước thực hành.

---

## 🗂️ Cấu trúc Thư mục (Directory Layout)

```text
.
├── agents/                            # Hệ thống quy trình kỹ thuật & Second Brain cho AI Agent
│   ├── README.md                      # Cổng nhận thức trung tâm & bản đồ điều phối
│   ├── WORKFLOW_OVERVIEW.md           # Tổng quan luồng thực thi khép kín
│   ├── AI_FIRST_LIFECYCLE.md          # 9 giai đoạn vĩ mô (Define -> Document)
│   ├── AGENT_INTERACTION_LOOP.md      # Chu trình tương tác vi mô 7 bước
│   ├── CONTEXT_MANAGEMENT.md          # Quản trị ngữ cảnh & anti-hallucination
│   ├── BEHAVIORAL_LAYER.md            # Đặc tả hành vi qua Prompting Schema
│   ├── TASK_DESCRIPTION.md            # Đặc tả tác vụ (Plan -> Pipeline -> DoD)
│   ├── PROGRESS_TRACKING.md           # Quản trị tiến độ Kanban & Single-WIP
│   ├── PIPELINE_SEPARATION.md         # Phân tách 5 pipelines Clean Architecture
│   ├── AI_REFERENCE.md                # Quy chuẩn, bất biến CV/DL & Conventional Commits
│   ├── CODEBASE_AUDIT.md              # Kiểm toán mã nguồn 3 tầng & an toàn CV
│   ├── TESTING_STRATEGY.md            # Chiến lược kiểm thử 2 chế độ (Smoke vs Normal)
│   └── TROUBLESHOOTING.md             # Quy trình xử lý sự cố & gỡ lỗi RCA 5 bước
├── configs/                           # Quản lý cấu hình bằng Hydra / YAML
│   ├── config.yaml                    # File cấu hình entrypoint mẫu
│   ├── dataset/
│   │   └── default.yaml               # Cấu hình dataset mẫu
│   ├── model/
│   │   └── template.yaml              # Cấu hình model mẫu
│   └── experiment/
│       └── template.yaml              # Cấu hình thực nghiệm mẫu
├── data/                              # Dữ liệu phân tầng (Ignored by Git)
│   ├── raw/
│   │   └── .gitkeep
│   ├── interim/
│   │   └── .gitkeep
│   ├── processed/
│   │   └── .gitkeep
│   ├── samples/
│   │   └── .gitkeep
│   └── README.md                      # Hướng dẫn tổ chức dữ liệu
├── docs/                              # Tài liệu & Báo cáo lab
│   ├── assets/
│   │   └── .gitkeep
│   └── labs/
│       └── template.md                # File mẫu báo cáo lab
├── experiments/                       # Nơi lưu trữ artifacts (Ignored by Git)
│   ├── checkpoints/
│   │   └── .gitkeep
│   ├── logs/
│   │   └── .gitkeep
│   └── predictions/
│       └── .gitkeep
├── notebooks/                         # Thư mục Jupyter Notebooks
│   ├── exploratory/
│   │   └── template.ipynb             # Notebook mẫu cho khảo sát dữ liệu (EDA)
│   ├── labs/
│   │   └── template.ipynb             # Notebook mẫu cho thực hành lab
│   └── README.md                      # Hướng dẫn sử dụng notebook
├── requirements/                      # Quản lý dependencies theo môi trường
│   ├── base.txt                       # Dependencies cơ bản
│   ├── deeplearning.txt               # Deep learning dependencies
│   ├── dev.txt                        # Development & testing dependencies
│   └── tracking.txt                   # Experiment tracking dependencies
├── scripts/                           # Shell scripts mẫu
│   ├── download_weights.sh            # Script tải pretrained weights mẫu
│   ├── setup_env.sh                   # Script khởi tạo môi trường
│   └── run_lab.sh                     # Script chạy lab mẫu
├── src/                               # Package mã nguồn chính (Clean Architecture)
│   ├── __init__.py
│   ├── common/                        # Tiện ích dùng chung
│   │   ├── __init__.py
│   │   ├── logger.py                  # Module logger mẫu
│   │   └── metrics.py                 # Module tính chỉ số mẫu
│   ├── data/                          # Dataset & DataModule
│   │   ├── __init__.py
│   │   ├── transforms.py              # Module tiền xử lý & transforms mẫu
│   │   ├── datamodule.py              # Module Dataset & DataLoader mẫu
│   │   └── lab04_pairs.py             # Chuẩn bị và nạp các cặp ảnh Lab04
│   ├── models/                        # Kiến trúc mô hình
│   │   ├── __init__.py
│   │   ├── classical/                 # Mô hình xử lý ảnh cổ điển
│   │   │   ├── __init__.py
│   │   │   └── wavelet_hash.py         # Wavelet hash, Hamming và Top-k
│   │   └── deep/                      # Mô hình học sâu
│   │       └── __init__.py
│   └── pipelines/                     # Pipeline thực thi
│       ├── __init__.py
│       ├── train.py                   # Script huấn luyện mẫu
│       ├── evaluate.py                # Script đánh giá mẫu
│       ├── infer.py                   # Script suy luận mẫu
│       ├── lab04_evaluation.py        # Chọn ngưỡng, đánh giá và khảo sát
│       └── lab04_visuals.py           # Hình minh họa của notebook
├── tests/                             # Bộ kiểm thử mẫu
│   ├── __init__.py
│   ├── conftest.py                    # Test fixtures mẫu
│   ├── test_data.py                   # Test data mẫu
│   └── test_models.py                 # Test model mẫu
├── .editorconfig
├── .gitattributes
├── .gitignore
├── .pre-commit-config.yaml
├── CHANGELOG.md
├── CONTRIBUTING.md
├── Makefile
├── pyproject.toml
├── README.md
└── requirements.txt
```

---

## 🚀 Hướng dẫn Cài đặt & Sử dụng (Setup)

### 1. Khởi tạo Môi trường (Environment Setup)

```bash
# 1. Tạo và kích hoạt virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Cài đặt dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements/dev.txt -r requirements/tracking.txt

# 3. Kích hoạt pre-commit hooks (nếu dùng git)
pre-commit install
```

---

## 🛠️ Makefile Commands

| Lệnh | Mô tả |
| :--- | :--- |
| `make help` | Hiển thị các lệnh hỗ trợ |
| `make install` | Cài đặt dependencies cơ bản |
| `make install-dev` | Cài đặt công cụ dev, test và tracking |
| `make format` | Định dạng code bằng Ruff và Black |
| `make lint` | Kiểm tra chất lượng code với Ruff |
| `make test` | Chạy bộ kiểm thử với Pytest |
| `make clean` | Dọn dẹp cache và file tạm |
