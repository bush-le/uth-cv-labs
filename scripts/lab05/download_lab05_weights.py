"""Script tải và kiểm tra mô hình pretrained MTCNN & FaceNet (InceptionResnetV1)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.common.device import resolve_device
from src.common.logger import setup_logger
from src.labs.lab05_facenet.facenet_extractor import FaceNetExtractor
from src.labs.lab05_facenet.mtcnn_detector import MTCNNDetector
from src.labs.lab05_facenet.pipeline import load_yaml_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tải và xác thực trọng số MTCNN & FaceNet."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment/lab05_facenet.yaml",
        help="Đường dẫn tới file cấu hình thực nghiệm Lab05.",
    )
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    exp_cfg = cfg["experiment"]
    facenet_cfg = cfg["facenet"]
    mtcnn_cfg = cfg["mtcnn"]

    logger = setup_logger("lab05.weights", exp_cfg["log_level"], exp_cfg["log_file"])
    device = resolve_device(exp_cfg["device"])

    ckpt_dir = Path(facenet_cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Đang kiểm tra và khởi tạo MTCNN...")
    _ = MTCNNDetector(
        image_size=mtcnn_cfg["image_size"],
        margin=mtcnn_cfg["margin"],
        min_face_size=mtcnn_cfg["min_face_size"],
        thresholds=mtcnn_cfg["thresholds"],
        factor=mtcnn_cfg["factor"],
        keep_all=mtcnn_cfg["keep_all"],
        min_confidence=mtcnn_cfg["min_confidence"],
        device=device,
    )

    logger.info(
        "Đang tải/kiểm tra trọng số FaceNet (%s)...", facenet_cfg["pretrained_dataset"]
    )
    extractor = FaceNetExtractor(
        pretrained_dataset=facenet_cfg["pretrained_dataset"],
        checkpoint_dir=ckpt_dir,
        l2_normalize=facenet_cfg["l2_normalize"],
        device=device,
    )

    # Kiểm tra truyền một tensor giả lập qua mạng
    dummy_input = torch.zeros((1, 3, 160, 160), dtype=torch.float32, device=device)
    embedding = extractor.extract(dummy_input)
    logger.info(
        "Xác thực thành công! Kích thước vector embedding đầu ra: %s, chuẩn L2 = %.4f",
        embedding.shape,
        float((embedding**2).sum() ** 0.5),
    )


if __name__ == "__main__":
    main()
