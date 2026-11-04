"""Pipeline nhận diện khuôn mặt thời gian thực qua Webcam bằng MTCNN và FaceNet."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from src.common.lab05_utils import FaceVisualizer, FPSMeter, RecognitionResult
from src.data.face_database import FaceDatabase
from src.data.face_preprocessing import preprocess_face
from src.models.deep.facenet_extractor import FaceNetExtractor
from src.models.deep.mtcnn_detector import MTCNNDetector
from src.models.similarity_matcher import SimilarityMatcher

logger = logging.getLogger("lab05.webcam")


class VideoCaptureProtocol(Protocol):
    """Giao diện trừu tượng hóa cv2.VideoCapture phục vụ Dependency Injection khi kiểm thử."""

    def isOpened(self) -> bool: ...
    def read(self) -> tuple[bool, np.ndarray | None]: ...
    def release(self) -> None: ...
    def set(self, propId: int, value: float) -> bool: ...


class WebcamManager:
    """Quản lý vòng đời kết nối và giải phóng tài nguyên Webcam an toàn bằng Context Manager."""

    def __init__(
        self,
        camera_index: int = 0,
        frame_width: int = 640,
        frame_height: int = 480,
        capture_backend: VideoCaptureProtocol | None = None,
    ) -> None:
        self.camera_index = camera_index
        self.frame_width = frame_width
        self.frame_height = frame_height
        self._injected_cap = capture_backend
        self.cap: VideoCaptureProtocol | None = None

    def open(self) -> None:
        """Mở kết nối tới camera và kiểm tra trạng thái hoạt động."""
        if self._injected_cap is not None:
            self.cap = self._injected_cap
        else:
            # Trên Windows, ưu tiên CAP_DSHOW để khởi tạo webcam nhanh và ổn định
            self.cap = cv2.VideoCapture(self.camera_index)

        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError(
                f"Không thể mở Webcam tại camera_index={self.camera_index}. "
                "Vui lòng kiểm tra kết nối camera hoặc quyền truy cập trên hệ điều hành."
            )

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.frame_width))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.frame_height))
        logger.info(
            "Đã mở Webcam index=%d (%dx%d).",
            self.camera_index,
            self.frame_width,
            self.frame_height,
        )

    def read_frame(self) -> np.ndarray:
        """Thu thập một khung hình từ webcam."""
        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError("Webcam chưa được khởi tạo hoặc đã bị ngắt kết nối.")
        ret, frame = self.cap.read()
        if not ret or frame is None or frame.size == 0:
            raise RuntimeError("Không thể thu thập khung hình từ Webcam.")
        return frame

    def close(self) -> None:
        """Giải phóng tài nguyên camera và đóng các cửa sổ OpenCV."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info("Đã giải phóng tài nguyên Webcam an toàn.")

    def __enter__(self) -> WebcamManager:
        self.open()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()


class RealtimeRecognitionPipeline:
    """Điều phối luồng nhận diện thời gian thực: Frame -> MTCNN -> FaceNet -> Matcher -> UI."""

    def __init__(
        self,
        detector: MTCNNDetector,
        extractor: FaceNetExtractor,
        matcher: SimilarityMatcher,
        database: FaceDatabase,
        image_size: int = 160,
    ) -> None:
        self.detector = detector
        self.extractor = extractor
        self.matcher = matcher
        self.database = database
        self.image_size = image_size
        self.fps_meter = FPSMeter()

    def process_frame(
        self, frame_bgr: np.ndarray
    ) -> tuple[np.ndarray, list[RecognitionResult]]:
        """Xử lý một khung hình đơn lẻ: phát hiện mặt, trích xuất đặc trưng, đối sánh và vẽ kết quả."""
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("Khung hình đầu vào không hợp lệ (None hoặc rỗng).")

        detected_faces = self.detector.detect_faces(frame_bgr)
        results: list[RecognitionResult] = []

        for det in detected_faces:
            face_tensor = preprocess_face(
                frame_bgr=frame_bgr,
                bbox=det.bbox,
                image_size=self.image_size,
                device=self.extractor.device,
            )
            embedding = self.extractor.extract(face_tensor)
            decision = self.matcher.match(embedding, self.database)

            results.append(
                RecognitionResult(
                    bbox=det.bbox,
                    detection_confidence=det.confidence,
                    status=decision.status,
                    identity=decision.identity,
                    similarity=decision.similarity,
                )
            )

        current_fps = self.fps_meter.update()
        annotated_frame = FaceVisualizer.draw_results(
            frame_bgr=frame_bgr,
            results=results,
            fps=current_fps,
            registered_count=len(self.database.identities),
            threshold=self.matcher.similarity_threshold,
        )
        return annotated_frame, results

    def run_webcam(
        self,
        webcam_manager: WebcamManager,
        window_name: str = "UTH CV Lab05 - FaceNet & MTCNN",
        mirror_display: bool = True,
        output_dir: str | Path = "experiments/predictions/lab05",
        max_frames: int | None = None,
        headless: bool = False,
    ) -> int:
        """Chạy vòng lặp nhận diện khuôn mặt liên tục trên Webcam."""
        save_dir = Path(output_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        frame_count = 0

        try:
            with webcam_manager as cam:
                while True:
                    frame = cam.read_frame()
                    if mirror_display:
                        frame = cv2.flip(frame, 1)

                    annotated, results = self.process_frame(frame)
                    frame_count += 1

                    if not headless:
                        cv2.imshow(window_name, annotated)
                        key = cv2.waitKey(1) & 0xFF
                        if key in (ord("q"), ord("Q"), 27):  # Phím Q hoặc ESC để thoát
                            logger.info("Người dùng nhấn phím thoát chương trình.")
                            break
                        if key in (ord("s"), ord("S")):  # Phím S để lưu ảnh minh chứng
                            snap_path = (
                                save_dir / f"webcam_snapshot_{frame_count:04d}.png"
                            )
                            cv2.imwrite(str(snap_path), annotated)
                            logger.info("Đã lưu ảnh chụp màn hình tại: %s", snap_path)

                    if max_frames is not None and frame_count >= max_frames:
                        break
        finally:
            if not headless:
                cv2.destroyAllWindows()

        return frame_count
