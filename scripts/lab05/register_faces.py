"""Công cụ đăng ký khuôn mặt từ thư mục ảnh có sẵn hoặc chụp trực tiếp qua Webcam."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from src.common.lab05_utils import load_yaml_config, resolve_device, setup_logger
from src.data.face_database import FaceDatabase
from src.models.deep.facenet_extractor import FaceNetExtractor
from src.models.deep.mtcnn_detector import MTCNNDetector
from src.pipelines.lab05_face_registration import FaceRegistrationPipeline
from src.pipelines.lab05_webcam_recognition import WebcamManager


def capture_from_webcam(
    identity: str,
    num_samples: int,
    save_root: Path,
    camera_index: int,
) -> None:
    """Hỗ trợ người dùng chụp ảnh khuôn mặt trực tiếp từ webcam để lưu vào thư mục đăng ký."""
    person_dir = save_root / identity
    person_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    with WebcamManager(camera_index=camera_index) as cam:
        while count < num_samples:
            frame = cam.read_frame()
            display = cv2.flip(frame, 1)
            cv2.putText(
                display,
                f"Registering '{identity}': [{count}/{num_samples}] - Press SPACE to capture, Q to quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Capture Registration Faces", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                count += 1
                img_path = person_dir / f"{identity}_{count:02d}.jpg"
                cv2.imwrite(str(img_path), frame)
            elif key in (ord("q"), ord("Q"), 27):
                break
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Đăng ký khuôn mặt vào cơ sở dữ liệu Lab05."
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="configs/dataset/lab05_faces.yaml",
        help="Đường dẫn cấu hình dataset.",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        default="configs/experiment/lab05_facenet.yaml",
        help="Đường dẫn cấu hình thực nghiệm.",
    )
    parser.add_argument(
        "--mode",
        choices=["folder", "capture"],
        default="folder",
        help="'folder': quét ảnh trong data/raw/lab05/faces | 'capture': chụp mới từ webcam.",
    )
    parser.add_argument(
        "--identity",
        type=str,
        default="",
        help="Tên người cần chụp khi sử dụng --mode capture.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Số lượng ảnh cần chụp khi dùng --mode capture.",
    )
    args = parser.parse_args()

    ds_cfg = load_yaml_config(args.dataset_config)["dataset"]
    exp_full = load_yaml_config(args.exp_config)
    exp_cfg = exp_full["experiment"]
    mtcnn_cfg = exp_full["mtcnn"]
    facenet_cfg = exp_full["facenet"]
    webcam_cfg = exp_full["webcam"]

    logger = setup_logger("lab05.register", exp_cfg["log_level"], exp_cfg["log_file"])
    device = resolve_device(exp_cfg["device"])

    raw_faces_dir = Path(ds_cfg["raw_faces_dir"])
    raw_faces_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "capture":
        if not args.identity.strip():
            raise ValueError(
                "Phải cung cấp --identity <Tên_Người> khi ở chế độ capture."
            )
        capture_from_webcam(
            identity=args.identity.strip(),
            num_samples=args.num_samples,
            save_root=raw_faces_dir,
            camera_index=webcam_cfg["camera_index"],
        )

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
    database = FaceDatabase(embedding_dim=facenet_cfg["embedding_dim"])

    pipeline = FaceRegistrationPipeline(
        detector=detector,
        extractor=extractor,
        database=database,
        image_size=ds_cfg["image_size"],
        require_single_face=ds_cfg["require_single_face_on_registration"],
    )

    summary = pipeline.register_from_directory(
        faces_root_dir=raw_faces_dir,
        supported_extensions=tuple(ds_cfg["supported_extensions"]),
    )
    database.save(ds_cfg["database_path"])

    logger.info(
        "Hoàn tất đăng ký: %d danh tính | Đã xử lý %d ảnh (Chấp nhận: %d, Từ chối: %d).",
        summary.identities_registered,
        summary.images_processed,
        summary.images_accepted,
        summary.images_rejected,
    )


if __name__ == "__main__":
    main()
