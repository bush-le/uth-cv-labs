"""Visual presentation utilities for Lab 05 FaceNet & MTCNN."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class RecognitionResult:
    """Structure storing face recognition result for a bounding box."""

    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    detection_confidence: float
    status: str  # "Matched" or "Unknown"
    identity: str  # Identity name or "Unknown"
    similarity: float


class FPSMeter:
    """Calculates frame rate with running exponential average."""

    def __init__(self, alpha: float = 0.9) -> None:
        self.alpha = float(np.clip(alpha, 0.0, 0.99))
        self._last_time: float | None = None
        self._fps: float = 0.0

    def update(self) -> float:
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
        return self._fps


class FaceVisualizer:
    """Draws bounding boxes, identity labels, similarity metrics, and HUD."""

    COLOR_MATCHED = (0, 200, 0)
    COLOR_UNKNOWN = (0, 0, 220)
    COLOR_HUD_BG = (25, 25, 25)
    COLOR_TEXT = (255, 255, 255)

    def __init__(self, matching_threshold: float = 0.7) -> None:
        self.matching_threshold = matching_threshold

    def draw_recognition(
        self,
        frame_bgr: np.ndarray,
        result: RecognitionResult,
    ) -> np.ndarray:
        """Render a single recognition result onto an image copy."""
        return self.draw_results(frame_bgr, [result], threshold=self.matching_threshold)

    @classmethod
    def draw_results(
        cls,
        frame_bgr: np.ndarray,
        results: list[RecognitionResult],
        fps: float = 0.0,
        registered_count: int = 0,
        threshold: float = 0.7,
    ) -> np.ndarray:
        """Render detection boxes and HUD on a frame copy."""
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("Empty frame provided for visualization.")

        canvas = frame_bgr.copy()
        height, width = canvas.shape[:2]

        # HUD bar
        cv2.rectangle(canvas, (0, 0), (width, 36), cls.COLOR_HUD_BG, thickness=-1)
        hud_text = (
            f"FPS: {fps:5.1f} | DB Identities: {registered_count} "
            f"| Faces: {len(results)} | Threshold: {threshold:.2f}"
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

        for res in results:
            x1, y1, x2, y2 = res.bbox
            is_matched = res.status == "Matched"
            box_color = cls.COLOR_MATCHED if is_matched else cls.COLOR_UNKNOWN

            cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, thickness=2)
            label = (
                f"Matched: {res.identity} ({res.similarity:.2f})"
                if is_matched
                else f"Unknown ({res.similarity:.2f})"
            )

            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
            )
            label_y1 = max(38, y1 - text_h - baseline - 8)
            label_y2 = label_y1 + text_h + baseline + 6
            label_x2 = min(width - 1, x1 + text_w + 10)

            cv2.rectangle(canvas, (x1, label_y1), (label_x2, label_y2), box_color, thickness=-1)
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


def plot_detection_result(
    image_bgr: np.ndarray,
    bboxes: Sequence[tuple[int, int, int, int]],
    landmarks: Sequence[np.ndarray] | None = None,
    figsize: tuple[int, int] = (10, 8),
) -> plt.Figure:
    """Plot MTCNN bounding boxes and 5 facial landmarks for notebook visualization."""
    vis = image_bgr.copy()
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if landmarks is not None:
        for pts in landmarks:
            if pts is not None:
                for pt in pts:
                    cv2.circle(vis, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)

    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(vis_rgb)
    ax.set_title(f"Phát hiện {len(bboxes)} khuôn mặt qua MTCNN", fontsize=12)
    ax.axis("off")
    plt.tight_layout()
    return fig


def plot_similarity_matrix(
    matrix: np.ndarray,
    labels: list[str],
    figsize: tuple[int, int] = (8, 6),
) -> plt.Figure:
    """Plot pairwise embedding similarity heatmap."""
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    fig.colorbar(im, ax=ax, label="Độ tương đồng (Cosine Similarity)")
    ax.set_title("Ma trận độ tương đồng FaceNet Embeddings", fontsize=12)
    plt.tight_layout()
    return fig
