"""Smoke Test cho Lab05: Kiểm tra import, đọc config, khởi tạo Database, Matcher và Visualizer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.common.config import load_yaml_config
from src.common.logger import setup_logger
from src.labs.lab05_facenet.face_database import FaceDatabase, l2_normalize_numpy
from src.labs.lab05_facenet.face_preprocessing import preprocess_face, validate_and_clip_bbox
from src.labs.lab05_facenet.similarity_matcher import SimilarityMatcher
from src.labs.lab05_facenet.visuals import (
    FaceVisualizer,
    FPSMeter,
    RecognitionResult,
)


@pytest.mark.smoke
def test_smoke_configs_loadable() -> None:
    """Kiểm tra cả hai file YAML cấu hình của Lab05 đọc được và đủ khóa chính."""
    ds_cfg = load_yaml_config("configs/dataset/lab05_faces.yaml")
    exp_cfg = load_yaml_config("configs/experiment/lab05_facenet.yaml")

    assert "dataset" in ds_cfg
    assert ds_cfg["dataset"]["image_size"] == 160
    assert "matcher" in exp_cfg
    assert exp_cfg["matcher"]["similarity_threshold"] == 0.7


@pytest.mark.smoke
def test_smoke_preprocessing_and_visualizer() -> None:
    """Kiểm tra luồng tiền xử lý tensor và vẽ kết quả trên khung hình giả lập."""
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    bbox = validate_and_clip_bbox(
        (100, 100, 260, 260), frame_width=640, frame_height=480
    )
    assert bbox == (100, 100, 260, 260)

    face_tensor = preprocess_face(dummy_frame, bbox=bbox, image_size=160, device="cpu")
    assert tuple(face_tensor.shape) == (1, 3, 160, 160)

    res = [
        RecognitionResult(
            bbox=bbox,
            detection_confidence=0.99,
            status="Matched",
            identity="SinhVien_01",
            similarity=0.85,
        )
    ]
    canvas = FaceVisualizer.draw_results(dummy_frame, res, fps=30.0, registered_count=1)
    assert canvas.shape == dummy_frame.shape


@pytest.mark.smoke
def test_smoke_database_and_matcher_end_to_end(tmp_path: Path) -> None:
    """Kiểm tra khởi tạo FaceDatabase, lưu/nạp .npz và đối sánh ngưỡng 0.7."""
    db = FaceDatabase(embedding_dim=512)
    vec = np.ones(512, dtype=np.float32)
    db.add_embedding("Person_A", vec)

    saved_file = db.save(tmp_path / "test_db.npz")
    reloaded_db = FaceDatabase.load(saved_file)

    matcher = SimilarityMatcher(similarity_threshold=0.7)
    decision = matcher.match(l2_normalize_numpy(vec), reloaded_db)
    assert decision.status == "Matched"
    assert decision.identity == "Person_A"
    assert decision.similarity > 0.99


@pytest.mark.smoke
def test_smoke_logger_and_fps_meter(tmp_path: Path) -> None:
    """Kiểm tra setup_logger và FPSMeter hoạt động chính xác."""
    log_file = tmp_path / "smoke.log"
    logger = setup_logger("lab05_test", level="DEBUG", log_file=log_file)
    assert logger.name == "lab05_test"
    logger.info("Test message from smoke test")
    assert log_file.exists()

    fps_meter = FPSMeter(alpha=0.8)
    assert fps_meter.fps == 0.0
    val = fps_meter.update()
    assert val == 0.0
