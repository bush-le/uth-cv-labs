"""Unit Tests kiểm thử toàn diện các trường hợp biên của Lab05 theo TESTING_STRATEGY.md."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data.face_database import FaceDatabase, l2_normalize_numpy
from src.data.face_preprocessing import preprocess_face, validate_and_clip_bbox
from src.models.deep.mtcnn_detector import DetectedFace
from src.models.similarity_matcher import SimilarityMatcher
from src.pipelines.lab05_face_registration import FaceRegistrationPipeline
from src.pipelines.lab05_webcam_recognition import (
    RealtimeRecognitionPipeline,
    WebcamManager,
)


# =====================================================================
# 1. KIỂM THỬ BOUNDING BOX & TIỀN XỬ LÝ
# =====================================================================
def test_bbox_validation_clips_out_of_bounds() -> None:
    """Đảm bảo tọa độ âm hoặc vượt quá kích thước khung hình được kẹp về biên hợp lệ."""
    clipped = validate_and_clip_bbox((-20.5, -10.2, 700.0, 500.0), 640, 480)
    assert clipped == (0, 0, 640, 480)


def test_bbox_validation_rejects_degenerate_boxes() -> None:
    """Từ chối bounding box có x2 <= x1, kích thước quá nhỏ (<10px) hoặc chứa NaN."""
    assert validate_and_clip_bbox((200, 200, 100, 300), 640, 480) is None
    assert validate_and_clip_bbox((10, 10, 15, 15), 640, 480) is None
    assert validate_and_clip_bbox((float("nan"), 10, 100, 100), 640, 480) is None


def test_preprocess_face_invalid_image_raises() -> None:
    """Báo lỗi ValueError khi truyền ảnh rỗng hoặc ảnh xám 2 chiều."""
    gray_img = np.zeros((100, 100), dtype=np.uint8)
    with pytest.raises(ValueError, match="3 kênh"):
        preprocess_face(gray_img, (10, 10, 50, 50))


# =====================================================================
# 2. KIỂM THỬ COSINE SIMILARITY & NGƯỠNG 0.7
# =====================================================================
def test_cosine_similarity_math_accuracy() -> None:
    """Kiểm tra công thức Cosine Similarity: trùng hướng = 1.0, vuông góc = 0.0, ngược hướng = -1.0."""
    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    v3 = np.array([-1.0, 0.0, 0.0], dtype=np.float32)

    assert SimilarityMatcher.compute_cosine_similarity(v1, v1) == pytest.approx(1.0)
    assert SimilarityMatcher.compute_cosine_similarity(v1, v2) == pytest.approx(0.0)
    assert SimilarityMatcher.compute_cosine_similarity(v1, v3) == pytest.approx(-1.0)


def test_threshold_strictly_greater_than_0_7_is_matched() -> None:
    """Khi similarity > 0.7 (ví dụ 0.7001), kết quả phải là 'Matched'."""
    matcher = SimilarityMatcher(similarity_threshold=0.7)
    decision = matcher.evaluate_threshold(0.7001, "Alice")
    assert decision.status == "Matched"
    assert decision.identity == "Alice"


def test_threshold_strictly_less_than_0_7_is_unknown() -> None:
    """Khi similarity < 0.7 (ví dụ 0.6999), kết quả phải là 'Unknown'."""
    matcher = SimilarityMatcher(similarity_threshold=0.7)
    decision = matcher.evaluate_threshold(0.6999, "Alice")
    assert decision.status == "Unknown"
    assert decision.identity == "Unknown"


def test_threshold_boundary_exact_0_7_is_unknown() -> None:
    """Trường hợp biên similarity == 0.7 phải trả về 'Unknown' để tuân thủ điều kiện > 0.7."""
    matcher = SimilarityMatcher(similarity_threshold=0.7)
    decision = matcher.evaluate_threshold(0.7, "Alice")
    assert decision.status == "Unknown"
    assert decision.identity == "Unknown"


# =====================================================================
# 3. KIỂM THỬ FACE DATABASE & TRƯỜNG HỢP RỖNG / LỖI
# =====================================================================
def test_empty_database_returns_unknown() -> None:
    """Khi Database rỗng, mọi truy vấn phải trả về 'Unknown' với similarity = 0.0."""
    db = FaceDatabase(embedding_dim=4)
    matcher = SimilarityMatcher(similarity_threshold=0.7)
    query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    decision = matcher.match(query, db)
    assert decision.status == "Unknown"
    assert decision.identity == "Unknown"
    assert decision.similarity == 0.0


def test_empty_embedding_raises_value_error() -> None:
    """Kiểm tra lỗi khi truyền embedding rỗng vào Matcher hoặc Database."""
    db = FaceDatabase(embedding_dim=4)
    matcher = SimilarityMatcher(similarity_threshold=0.7)
    with pytest.raises(ValueError):
        matcher.match(np.array([], dtype=np.float32), db)
    with pytest.raises(ValueError):
        l2_normalize_numpy(np.array([], dtype=np.float32))


# =====================================================================
# 4. KIỂM THỬ MOCK PIPELINE: 0 MẶT, 1 MẶT, NHIỀU MẶT & LỖI WEBCAM
# =====================================================================
class DummyDetector:
    """Giả lập MTCNNDetector phục vụ kiểm thử đơn vị không cần tải trọng số."""

    def __init__(self, faces_to_return: list[DetectedFace]) -> None:
        self.faces = faces_to_return

    def detect_faces(self, frame_bgr: np.ndarray) -> list[DetectedFace]:
        return self.faces


class DummyExtractor:
    """Giả lập FaceNetExtractor trả về vector cố định."""

    device = torch.device("cpu")

    def __init__(self, output_vec: np.ndarray) -> None:
        self.output_vec = l2_normalize_numpy(output_vec)

    def extract(self, face_tensor: torch.Tensor) -> np.ndarray:
        return self.output_vec


def test_registration_rejects_no_face_and_multiple_faces() -> None:
    """Khi đăng ký khuôn mặt: ảnh không có mặt hoặc có >1 mặt phải bị từ chối."""
    db = FaceDatabase(embedding_dim=4)
    extractor = DummyExtractor(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    frame = np.zeros((200, 200, 3), dtype=np.uint8)

    # Trường hợp 0 khuôn mặt
    reg_zero = FaceRegistrationPipeline(DummyDetector([]), extractor, db)  # type: ignore[arg-type]
    assert reg_zero.register_single_frame("User1", frame) is False

    # Trường hợp 2 khuôn mặt trong cùng 1 ảnh đăng ký
    two_faces = [
        DetectedFace((10, 10, 60, 60), 0.99),
        DetectedFace((80, 80, 140, 140), 0.95),
    ]
    reg_multi = FaceRegistrationPipeline(
        DummyDetector(two_faces), extractor, db, require_single_face=True  # type: ignore[arg-type]
    )
    assert reg_multi.register_single_frame("User1", frame) is False
    assert db.is_empty


def test_realtime_pipeline_handles_zero_one_and_multiple_faces() -> None:
    """Kiểm tra RealtimeRecognitionPipeline xử lý 0 mặt, 1 mặt và nhiều mặt trong cùng frame."""
    db = FaceDatabase(embedding_dim=4)
    target_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    db.add_embedding("Hanh_Nguyen", target_vec)

    matcher = SimilarityMatcher(similarity_threshold=0.7)
    extractor = DummyExtractor(target_vec)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # 0 mặt
    pipe_0 = RealtimeRecognitionPipeline(DummyDetector([]), extractor, matcher, db)  # type: ignore[arg-type]
    _, res_0 = pipe_0.process_frame(frame)
    assert len(res_0) == 0

    # 2 mặt cùng lúc
    faces_2 = [
        DetectedFace((20, 20, 120, 120), 0.99),
        DetectedFace((200, 200, 320, 320), 0.96),
    ]
    pipe_2 = RealtimeRecognitionPipeline(DummyDetector(faces_2), extractor, matcher, db)  # type: ignore[arg-type]
    _, res_2 = pipe_2.process_frame(frame)
    assert len(res_2) == 2
    assert all(r.status == "Matched" and r.identity == "Hanh_Nguyen" for r in res_2)


def test_webcam_manager_raises_when_camera_unavailable() -> None:
    """Kiểm tra xử lý ngoại lệ RuntimeError khi Webcam không thể truy cập."""

    class BrokenCapture:
        def isOpened(self) -> bool:
            return False

        def read(self) -> tuple[bool, np.ndarray | None]:
            return False, None

        def release(self) -> None:
            pass

        def set(self, propId: int, value: float) -> bool:
            return False

    manager = WebcamManager(camera_index=99, capture_backend=BrokenCapture())
    with pytest.raises(RuntimeError, match="Không thể mở Webcam"):
        manager.open()
