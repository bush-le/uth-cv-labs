# UTH Computer Vision Workspace (`uth-cv-labs`)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
---

## 📌 Giới thiệu Tổng quan (Overview)

Repository này là bộ **khung xương dự án mẫu (Pure Project Scaffolding & Template)**:
- Cung cấp cấu trúc thư mục phân tầng, modular và chuẩn hóa theo quy chuẩn kỹ thuật.
- Không chứa nội dung hoặc bài toán cụ thể nào, sẵn sàng để phục vụ cho các bài thực hành (Labs) trong tương lai.

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
│   │   └── sample_traffic.jpg         # Ảnh mẫu đầu vào thực hành
│   └── README.md                      # Hướng dẫn tổ chức dữ liệu
├── docs/                              # Tài liệu & Báo cáo lab
│   ├── assets/                        # Tài nguyên ảnh, sơ đồ workflow
│   └── labs/
│       ├── README.md                  # Danh mục tổng hợp tất cả bài lab
│       ├── TEMPLATE.md                # File mẫu báo cáo lab
│       └── lab01/                     # Không gian tài liệu độc lập Lab 01
│           ├── LAB_01_DESCRIPTION.md  # Đề bài & tiêu chí đánh giá
│           ├── LAB_01_REPORT.md       # Báo cáo kỹ thuật chi tiết
│           └── LAB_01_REPORT.html     # Báo cáo giao diện web tương tác
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
│   │   ├── metrics.py                 # Module tính chỉ số mẫu
│   │   └── visualizer.py              # Module trực quan hóa mẫu
│   ├── data/                          # Dataset & DataModule
│   │   ├── __init__.py
│   │   ├── transforms.py              # Module tiền xử lý & transforms mẫu
│   │   └── datamodule.py              # Module Dataset & DataLoader mẫu
│   ├── models/                        # Kiến trúc mô hình
│   │   ├── __init__.py
│   │   ├── classical/                 # Mô hình xử lý ảnh cổ điển
│   │   │   └── __init__.py
│   │   └── deep/                      # Mô hình học sâu
│   │       └── __init__.py
│   └── pipelines/                     # Pipeline thực thi
│       ├── __init__.py
│       ├── train.py                   # Script huấn luyện mẫu
│       ├── evaluate.py                # Script đánh giá mẫu
│       └── infer.py                   # Script suy luận mẫu
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
