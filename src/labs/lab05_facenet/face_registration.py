"""Pipeline đăng ký khuôn mặt từ thư mục ảnh hoặc chụp trực tiếp qua Webcam."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.labs.lab05_facenet.face_database import FaceDatabase
from src.labs.lab05_facenet.face_preprocessing import preprocess_face
from src.labs.lab05_facenet.facenet_extractor import FaceNetExtractor
from src.labs.lab05_facenet.mtcnn_detector import MTCNNDetector

logger = logging.getLogger("lab05.registration")


@dataclass
class RegistrationSummary:
    """Thống kê kết quả sau khi chạy quy trình đăng ký khuôn mặt."""

    identities_registered: int
    images_processed: int
    images_accepted: int
    images_rejected: int


class FaceRegistrationPipeline:
    """Xây dựng cơ sở dữ liệu nhận diện từ các ảnh khuôn mặt mẫu."""

    def __init__(
        self,
        detector: MTCNNDetector,
        extractor: FaceNetExtractor,
        database: FaceDatabase,
        image_size: int = 160,
        require_single_face: bool = True,
    ) -> None:
        self.detector = detector
        self.extractor = extractor
        self.database = database
        self.image_size = image_size
        self.require_single_face = require_single_face

    def register_single_frame(self, identity: str, frame_bgr: np.ndarray) -> bool:
        """Phát hiện khuôn mặt, kiểm tra hợp lệ và trích xuất embedding cho 1 khung hình."""
        if frame_bgr is None or frame_bgr.size == 0:
            logger.warning("Bỏ qua khung hình rỗng cho danh tính '%s'.", identity)
            return False

        detected_faces = self.detector.detect_faces(frame_bgr)
        if len(detected_faces) == 0:
            logger.warning("Không tìm thấy khuôn mặt nào cho danh tính '%s'.", identity)
            return False

        if self.require_single_face and len(detected_faces) > 1:
            logger.warning(
                "Phát hiện %d khuôn mặt trong ảnh đăng ký của '%s'. "
                "Từ chối để tránh gán nhầm danh tính.",
                len(detected_faces),
                identity,
            )
            return False

        primary_face = detected_faces[0]
        face_tensor = preprocess_face(
            frame_bgr=frame_bgr,
            bbox=primary_face.bbox,
            image_size=self.image_size,
            device=self.extractor.device,
        )
        embedding = self.extractor.extract(face_tensor)
        self.database.add_embedding(identity=identity, embedding=embedding)
        return True

    def register_from_directory(
        self,
        faces_root_dir: str | Path,
        supported_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp"),
    ) -> RegistrationSummary:
        """Quét thư mục data/raw/lab05/faces/<identity>/ và đăng ký toàn bộ ảnh hợp lệ."""
        root = Path(faces_root_dir)
        if not root.exists():
            raise FileNotFoundError(f"Thư mục dữ liệu khuôn mặt không tồn tại: {root}")

        processed = 0
        accepted = 0
        rejected = 0

        person_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
        for person_dir in person_dirs:
            identity = person_dir.name
            image_files = sorted(
                [
                    f
                    for f in person_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in supported_extensions
                ]
            )
            for img_path in image_files:
                processed += 1
                img_bgr = cv2.imread(str(img_path))
                if img_bgr is None:
                    logger.warning("Không thể đọc file ảnh lỗi: %s", img_path)
                    rejected += 1
                    continue

                if self.register_single_frame(identity=identity, frame_bgr=img_bgr):
                    accepted += 1
                    logger.info("Đã đăng ký thành công: %s <- %s", identity, img_path.name)
                else:
                    rejected += 1

        return RegistrationSummary(
            identities_registered=len(self.database.identities),
            images_processed=processed,
            images_accepted=accepted,
            images_rejected=rejected,
        )
