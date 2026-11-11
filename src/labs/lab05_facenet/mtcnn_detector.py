"""Module phát hiện khuôn mặt sử dụng mạng MTCNN (Multi-task Cascaded Convolutional Networks)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN

from src.labs.lab05_facenet.face_preprocessing import validate_and_clip_bbox

logger = logging.getLogger("lab05.mtcnn")


@dataclass(frozen=True)
class DetectedFace:
    """Thông tin một khuôn mặt được MTCNN phát hiện."""

    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2) đã kẹp biên hợp lệ
    confidence: float


class MTCNNDetector:
    """Trình phát hiện khuôn mặt thời gian thực sử dụng MTCNN."""

    def __init__(
        self,
        image_size: int = 160,
        margin: int = 14,
        min_face_size: int = 40,
        thresholds: list[float] | tuple[float, float, float] = (0.6, 0.7, 0.7),
        factor: float = 0.709,
        keep_all: bool = True,
        min_confidence: float = 0.90,
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.margin = margin
        self.min_confidence = float(min_confidence)
        self.detector = MTCNN(
            image_size=image_size,
            margin=margin,
            min_face_size=min_face_size,
            thresholds=list(thresholds),
            factor=factor,
            keep_all=keep_all,
            post_process=True,
            device=self.device,
        )
        logger.info("Đã khởi tạo MTCNNDetector trên thiết bị: %s", self.device)

    def detect_faces(self, frame_bgr: np.ndarray) -> list[DetectedFace]:
        """Phát hiện tất cả khuôn mặt hợp lệ trong khung hình BGR.

        Args:
            frame_bgr: Ảnh đầu vào định dạng BGR (H, W, 3) từ OpenCV.

        Returns:
            Danh sách các đối tượng DetectedFace sắp xếp theo diện tích giảm dần.
        """
        if frame_bgr is None or frame_bgr.size == 0 or frame_bgr.ndim != 3:
            return []

        height, width = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        with torch.no_grad():
            boxes, probs = self.detector.detect(frame_rgb, landmarks=False)

        if boxes is None or probs is None:
            return []

        detected_list: list[DetectedFace] = []
        for raw_box, prob in zip(boxes, probs, strict=False):
            if prob is None or float(prob) < self.min_confidence:
                continue
            valid_box = validate_and_clip_bbox(
                raw_box,
                frame_width=width,
                frame_height=height,
                margin=self.margin,
            )
            if valid_box is not None:
                detected_list.append(
                    DetectedFace(bbox=valid_box, confidence=float(prob))
                )

        # Sắp xếp ưu tiên khuôn mặt có diện tích lớn nhất lên đầu
        detected_list.sort(
            key=lambda item: (
                (item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1])
            ),
            reverse=True,
        )
        return detected_list
