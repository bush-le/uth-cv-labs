"""Script khởi chạy hệ thống nhận diện khuôn mặt thời gian thực với FaceNet & MTCNN."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.device import resolve_device
from src.common.logger import setup_logger
from src.labs.lab05_facenet.face_database import FaceDatabase
from src.labs.lab05_facenet.facenet_extractor import FaceNetExtractor
from src.labs.lab05_facenet.mtcnn_detector import MTCNNDetector
from src.labs.lab05_facenet.pipeline import load_yaml_config
from src.labs.lab05_facenet.similarity_matcher import SimilarityMatcher
from src.labs.lab05_facenet.webcam_recognition import (
    RealtimeRecognitionPipeline,
    WebcamManager,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chạy nhận diện khuôn mặt thời gian thực trên Webcam (Lab05)."
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="configs/dataset/lab05_faces.yaml",
        help="Đường dẫn file cấu hình dataset.",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        default="configs/experiment/lab05_facenet.yaml",
        help="Đường dẫn file cấu hình thực nghiệm.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=None,
        help="Ghi đè chỉ số camera (mặc định lấy từ YAML).",
    )
    args = parser.parse_args()

    ds_cfg = load_yaml_config(args.dataset_config)["dataset"]
    exp_full = load_yaml_config(args.exp_config)
    exp_cfg = exp_full["experiment"]
    webcam_cfg = exp_full["webcam"]
    mtcnn_cfg = exp_full["mtcnn"]
    facenet_cfg = exp_full["facenet"]
    matcher_cfg = exp_full["matcher"]

    logger = setup_logger("lab05.runner", exp_cfg["log_level"], exp_cfg["log_file"])
    device = resolve_device(exp_cfg["device"])

    db_path = Path(ds_cfg["database_path"])
    if db_path.exists():
        database = FaceDatabase.load(db_path)
    else:
        logger.warning(
            "Chưa tìm thấy file cơ sở dữ liệu tại %s. "
            "Khởi tạo Database rỗng (mọi khuôn mặt sẽ hiển thị là 'Unknown'). "
            "Hãy chạy scripts/register_faces.py trước để đăng ký danh tính!",
            db_path,
        )
        database = FaceDatabase(embedding_dim=facenet_cfg["embedding_dim"])

    detector = MTCNNDetector(
        image_size=mtcnn_cfg["image_size"],
        margin=mtcnn_cfg["margin"],
        min_face_size=mtcnn_cfg["min_face_size"],
        thresholds=mtcnn_cfg["thresholds"],
        factor=mtcnn_cfg["factor"],
        keep_all=mtcnn_cfg["keep_all"],
        min_confidence=mtcnn_cfg["min_confidence"],
        device=device,
    )
    extractor = FaceNetExtractor(
        pretrained_dataset=facenet_cfg["pretrained_dataset"],
        checkpoint_dir=facenet_cfg["checkpoint_dir"],
        l2_normalize=facenet_cfg["l2_normalize"],
        device=device,
    )
    matcher = SimilarityMatcher(
        similarity_threshold=float(matcher_cfg["similarity_threshold"])
    )

    cam_idx = (
        args.camera_index
        if args.camera_index is not None
        else int(webcam_cfg["camera_index"])
    )
    webcam_manager = WebcamManager(
        camera_index=cam_idx,
        frame_width=int(webcam_cfg["frame_width"]),
        frame_height=int(webcam_cfg["frame_height"]),
    )

    pipeline = RealtimeRecognitionPipeline(
        detector=detector,
        extractor=extractor,
        matcher=matcher,
        database=database,
        image_size=int(ds_cfg["image_size"]),
    )

    logger.info(
        "Bắt đầu nhận diện trên Webcam (Ngưỡng Cosine Similarity > %.2f -> Matched)...",
        matcher.similarity_threshold,
    )
    try:
        pipeline.run_webcam(
            webcam_manager=webcam_manager,
            window_name=webcam_cfg["window_name"],
            mirror_display=bool(webcam_cfg["mirror_display"]),
            output_dir=exp_cfg["output_dir"],
        )
    except RuntimeError as err:
        logger.error(
            "Lỗi khởi tạo Webcam: %s. Hãy kiểm tra kết nối camera hoặc quyền truy cập.",
            err,
        )


if __name__ == "__main__":
    main()
