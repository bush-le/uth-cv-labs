"""Tiện ích dùng chung cho Lab05: Đọc cấu hình YAML, Logging, đo FPS và hiển thị OpenCV."""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml


@dataclass(frozen=True)
class RecognitionResult:
    """Cấu trúc dữ liệu lưu kết quả nhận diện của một khuôn mặt trên khung hình."""

    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    detection_confidence: float
    status: str  # "Matched" hoặc "Unknown"
    identity: str  # Tên người nếu Matched, hoặc "Unknown"
    similarity: float


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """Đọc file cấu hình YAML một cách an toàn."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình tại: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Nội dung YAML không hợp lệ (phải là dictionary): {path}")
    return data


def resolve_device(device_cfg: str = "auto") -> torch.device:
    """Xác định thiết bị tính toán (CPU hoặc CUDA)."""
    normalized = device_cfg.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cuda" and not torch.cuda.is_available():
        logging.getLogger("lab05").warning(
            "Yêu cầu CUDA nhưng hệ thống không hỗ trợ GPU CUDA. Chuyển về CPU."
        )
        return torch.device("cpu")
    return torch.device(normalized)


def setup_logger(
    name: str = "lab05",
    level: str = "INFO",
    log_file: str | Path | None = None,
) -> logging.Logger:
    """Thiết lập logger chuẩn ghi ra console và file log."""
    logger = logging.getLogger(name)
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class FPSMeter:
    """Bộ tính toán tốc độ khung hình (Frames Per Second) sử dụng trung bình trượt."""

    def __init__(self, alpha: float = 0.9) -> None:
        self.alpha = float(np.clip(alpha, 0.0, 0.99))
        self._last_time: float | None = None
        self._fps: float = 0.0

    def update(self) -> float:
        """Cập nhật mốc thời gian mới và trả về giá trị FPS hiện tại."""
        now = time.perf_counter()
        if self._last_time is not None:
            dt = now - self._last_time
            if dt > 1e-6:
                instant_fps = 1.0 / dt
                if self._fps == 0.0:
                    self._fps = instant_fps
                else:
                    self._fps = self.alpha * self._fps + (1.0 - self.alpha) * instant_fps
        self._last_time = now
        return self._fps

    @property
    def fps(self) -> float:
        """Trả về FPS hiện tại."""
        return self._fps


class FaceVisualizer:
    """Chịu trách nhiệm vẽ bounding box, nhãn Matched/Unknown, điểm similarity và FPS."""

    COLOR_MATCHED = (0, 200, 0)    # Xanh lá (BGR) cho trạng thái Matched
    COLOR_UNKNOWN = (0, 0, 220)    # Đỏ (BGR) cho trạng thái Unknown
    COLOR_HUD_BG = (25, 25, 25)    # Nền đen xám cho thanh thông tin hệ thống
    COLOR_TEXT = (255, 255, 255)   # Chữ trắng

    @classmethod
    def draw_results(
        cls,
        frame_bgr: np.ndarray,
        results: list[RecognitionResult],
        fps: float = 0.0,
        registered_count: int = 0,
        threshold: float = 0.7,
    ) -> np.ndarray:
        """Vẽ toàn bộ kết quả nhận diện và thông số trạng thái lên bản sao của frame."""
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("Khung hình đầu vào rỗng, không thể hiển thị.")

        canvas = frame_bgr.copy()
        height, width = canvas.shape[:2]

        # 1. Vẽ thanh trạng thái hệ thống ở góc trên
        cv2.rectangle(canvas, (0, 0), (width, 36), cls.COLOR_HUD_BG, thickness=-1)
        hud_text = (
            f"FPS: {fps:5.1f} | DB Identities: {registered_count} "
            f"| Faces: {len(results)} | Threshold: {threshold:.2f} | [Q] Quit [S] Save"
        )
        cv2.putText(
            canvas,
            hud_text,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            cls.COLOR_TEXT,
            1,
            cv2.LINE_AA,
        )

        # 2. Vẽ từng khuôn mặt được phát hiện
        for res in results:
            x1, y1, x2, y2 = res.bbox
            is_matched = res.status == "Matched"
            box_color = cls.COLOR_MATCHED if is_matched else cls.COLOR_UNKNOWN

            # Vẽ khung chữ nhật quanh khuôn mặt
            cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, thickness=2)

            # Định dạng nhãn hiển thị đúng yêu cầu: "Matched: <Tên> (0.85)" hoặc "Unknown (0.52)"
            if is_matched:
                label = f"Matched: {res.identity} (sim={res.similarity:.2f})"
            else:
                label = f"Unknown (sim={res.similarity:.2f})"

            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
            )
            label_y1 = max(38, y1 - text_h - baseline - 8)
            label_y2 = label_y1 + text_h + baseline + 6
            label_x2 = min(width - 1, x1 + text_w + 10)

            cv2.rectangle(
                canvas, (x1, label_y1), (label_x2, label_y2), box_color, thickness=-1
            )
            cv2.putText(
                canvas,
                label,
                (x1 + 5, label_y2 - baseline - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                cls.COLOR_TEXT,
                2,
                cv2.LINE_AA,
            )

        return canvas
