"""Module đối sánh độ tương đồng đặc trưng khuôn mặt bằng Cosine Similarity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.labs.lab05_facenet.face_database import FaceDatabase, l2_normalize_numpy


@dataclass(frozen=True)
class MatchDecision:
    """Kết quả đối sánh một vector khuôn mặt từ camera với cơ sở dữ liệu."""

    status: str      # "Matched" hoặc "Unknown"
    identity: str    # Tên danh tính nếu Matched, hoặc "Unknown"
    similarity: float


class SimilarityMatcher:
    """Thực hiện phép tính Cosine Similarity và phân ngưỡng theo yêu cầu bài thực hành.

    Quy tắc từ Slide 9 (Chương 3 Part 2):
    - Nếu similarity > 0.7  -> Hiển thị "Matched"
    - Nếu similarity < 0.7  -> Hiển thị "Unknown"
    - Trường hợp biên similarity == 0.7 -> Quy định rõ là "Unknown" (không tự ý mở rộng điều kiện Matched).
    """

    def __init__(self, similarity_threshold: float = 0.7) -> None:
        if not (-1.0 <= similarity_threshold <= 1.0):
            raise ValueError("Ngưỡng Cosine Similarity phải nằm trong khoảng [-1.0, 1.0].")
        self.similarity_threshold = float(similarity_threshold)

    @staticmethod
    def compute_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Tính chính xác Cosine Similarity giữa hai vector đặc trưng 1 chiều.

        Công thức: cos(theta) = (a . b) / (||a||_2 * ||b||_2)
        Lưu ý: Đây là Cosine Similarity trong miền [-1, 1], KHÔNG phải Cosine Distance (1 - cos)
        và KHÔNG phải khoảng cách Euclidean.
        """
        a = np.asarray(vec_a, dtype=np.float32).reshape(-1)
        b = np.asarray(vec_b, dtype=np.float32).reshape(-1)

        if a.size == 0 or b.size == 0:
            raise ValueError("Vector embedding đầu vào không được rỗng.")
        if a.shape != b.shape:
            raise ValueError(f"Hai vector không cùng số chiều: {a.shape} vs {b.shape}.")
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            raise ValueError("Vector chứa giá trị NaN hoặc Inf.")

        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a < 1e-12 or norm_b < 1e-12:
            raise ValueError("Không thể tính Cosine Similarity cho vector không (zero vector).")

        sim = float(np.dot(a, b) / (norm_a * norm_b))
        return float(np.clip(sim, -1.0, 1.0))

    def evaluate_threshold(self, similarity: float, candidate_identity: str) -> MatchDecision:
        """Áp dụng điều kiện ngưỡng nghiêm ngặt > 0.7 của giảng viên."""
        if similarity > self.similarity_threshold:
            return MatchDecision(
                status="Matched",
                identity=candidate_identity,
                similarity=float(similarity),
            )
        return MatchDecision(
            status="Unknown",
            identity="Unknown",
            similarity=float(similarity),
        )

    def match(self, query_embedding: np.ndarray, database: FaceDatabase) -> MatchDecision:
        """So sánh embedding thu từ webcam với toàn bộ cơ sở dữ liệu đã đăng ký."""
        query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        if query.size == 0:
            raise ValueError("Query embedding bị rỗng.")

        if database is None or database.is_empty:
            return MatchDecision(status="Unknown", identity="Unknown", similarity=0.0)

        labels, db_matrix = database.get_all_embeddings()
        if len(labels) == 0 or db_matrix.shape[0] == 0:
            return MatchDecision(status="Unknown", identity="Unknown", similarity=0.0)

        query_norm = l2_normalize_numpy(query)
        db_norm = l2_normalize_numpy(db_matrix)

        # Phép nhân vô hướng của các vector đơn vị chính là Cosine Similarity
        similarities = np.dot(db_norm, query_norm)
        best_idx = int(np.argmax(similarities))
        best_similarity = float(np.clip(similarities[best_idx], -1.0, 1.0))
        best_identity = labels[best_idx]

        return self.evaluate_threshold(
            similarity=best_similarity,
            candidate_identity=best_identity,
        )
