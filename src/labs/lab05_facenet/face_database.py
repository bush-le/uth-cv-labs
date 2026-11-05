"""Module quản lý cơ sở dữ liệu Face Embeddings: Đăng ký, chuẩn hóa L2, lưu và nạp file .npz."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("lab05.database")


def l2_normalize_numpy(vector: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Chuẩn hóa vector hoặc ma trận các vector theo chuẩn L2 (độ dài đơn vị = 1)."""
    arr = np.asarray(vector, dtype=np.float32)
    if arr.size == 0:
        raise ValueError("Không thể chuẩn hóa vector rỗng.")
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / np.maximum(norms, eps)


class FaceDatabase:
    """Cơ sở dữ liệu lưu trữ các vector đặc trưng (embeddings) của người đã đăng ký."""

    def __init__(self, embedding_dim: int = 512) -> None:
        if embedding_dim <= 0:
            raise ValueError("Chiều của vector embedding phải lớn hơn 0.")
        self.embedding_dim = embedding_dim
        self._records: dict[str, list[np.ndarray]] = {}

    @property
    def is_empty(self) -> bool:
        """Kiểm tra cơ sở dữ liệu có đang trống hay không."""
        return len(self._records) == 0

    @property
    def identities(self) -> list[str]:
        """Danh sách tên các danh tính đã đăng ký."""
        return sorted(self._records.keys())

    @property
    def total_samples(self) -> int:
        """Tổng số vector mẫu đã lưu trong toàn bộ cơ sở dữ liệu."""
        return sum(len(v) for v in self._records.values())

    def add_embedding(self, identity: str, embedding: np.ndarray) -> None:
        """Thêm một vector đặc trưng mới cho một danh tính."""
        clean_name = identity.strip()
        if not clean_name:
            raise ValueError("Tên danh tính (identity) không được để trống.")

        vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vec.shape[0] != self.embedding_dim:
            raise ValueError(
                f"Kích thước embedding không khớp: nhận {vec.shape[0]}, "
                f"kỳ vọng {self.embedding_dim}."
            )
        if not np.isfinite(vec).all():
            raise ValueError("Vector embedding chứa giá trị NaN hoặc Inf không hợp lệ.")

        norm_vec = l2_normalize_numpy(vec)
        if clean_name not in self._records:
            self._records[clean_name] = []
        self._records[clean_name].append(norm_vec)

    def get_all_embeddings(self) -> tuple[list[str], np.ndarray]:
        """Lấy toàn bộ danh sách nhãn và ma trận embeddings (N, D) để so sánh."""
        if self.is_empty:
            return [], np.empty((0, self.embedding_dim), dtype=np.float32)

        labels: list[str] = []
        vectors: list[np.ndarray] = []
        for identity in self.identities:
            samples = self._records[identity]
            for vec in samples:
                labels.append(identity)
                vectors.append(vec)
            # Nếu một người có nhiều ảnh đăng ký, bổ sung thêm vector trung bình (centroid) đã chuẩn hóa L2
            if len(samples) > 1:
                centroid = l2_normalize_numpy(np.mean(np.stack(samples, axis=0), axis=0))
                labels.append(identity)
                vectors.append(centroid)

        return labels, np.stack(vectors, axis=0).astype(np.float32)

    def save(self, file_path: str | Path) -> Path:
        """Lưu cơ sở dữ liệu embedding xuống file định dạng nén .npz."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        all_labels: list[str] = []
        all_vectors: list[np.ndarray] = []
        for identity, samples in self._records.items():
            for vec in samples:
                all_labels.append(identity)
                all_vectors.append(vec)

        if all_vectors:
            embeddings_matrix = np.stack(all_vectors, axis=0).astype(np.float32)
        else:
            embeddings_matrix = np.empty((0, self.embedding_dim), dtype=np.float32)

        np.savez_compressed(
            path,
            identities=np.array(all_labels, dtype=np.str_),
            embeddings=embeddings_matrix,
            embedding_dim=np.array(self.embedding_dim, dtype=np.int32),
        )
        logger.info(
            "Đã lưu FaceDatabase tại %s (%d người, %d mẫu).",
            path,
            len(self.identities),
            len(all_labels),
        )
        return path

    @classmethod
    def load(cls, file_path: str | Path) -> FaceDatabase:
        """Nạp lại cơ sở dữ liệu từ file .npz."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file FaceDatabase tại: {path}")

        data = np.load(path, allow_pickle=False)
        embedding_dim = int(data["embedding_dim"])
        db = cls(embedding_dim=embedding_dim)

        identities = data["identities"]
        embeddings = data["embeddings"]
        for name, vec in zip(identities, embeddings, strict=True):
            db.add_embedding(str(name), vec)

        logger.info(
            "Đã nạp FaceDatabase từ %s (%d người, %d mẫu).",
            path,
            len(db.identities),
            db.total_samples,
        )
        return db
