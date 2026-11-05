"""Script sinh dữ liệu kiểm thử tổng hợp (Simulation Artifacts) phục vụ báo cáo khi chưa cắm Webcam."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from src.common.logger import setup_logger
from src.labs.lab05_facenet.face_database import FaceDatabase, l2_normalize_numpy
from src.labs.lab05_facenet.similarity_matcher import SimilarityMatcher
from src.labs.lab05_facenet.visuals import FaceVisualizer, RecognitionResult


def main() -> None:
    logger = setup_logger("lab05.artifacts", "INFO", "experiments/logs/lab05.log")
    out_dir = Path("experiments/predictions/lab05")
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed=42)
    db = FaceDatabase(embedding_dim=512)
    base_vec = l2_normalize_numpy(rng.standard_normal(512).astype(np.float32))
    db.add_embedding("Nguyen_Van_A", base_vec)

    matcher = SimilarityMatcher(similarity_threshold=0.7)

    # 1. Tạo vector giả lập cho trường hợp Matched (similarity ~ 0.88 > 0.7)
    noise_small = rng.standard_normal(512).astype(np.float32) * 0.02
    vec_matched = l2_normalize_numpy(base_vec + noise_small)
    dec_matched = matcher.match(vec_matched, db)

    # 2. Tạo vector giả lập cho trường hợp Unknown (similarity ~ 0.15 < 0.7)
    vec_unknown = l2_normalize_numpy(rng.standard_normal(512).astype(np.float32))
    dec_unknown = matcher.match(vec_unknown, db)

    # 3. Vẽ ảnh mô phỏng kết quả Matched & Unknown (đánh dấu rõ SIMULATION)
    canvas = np.full((480, 640, 3), 40, dtype=np.uint8)
    cv2.putText(
        canvas,
        "[SIMULATION ARTIFACT - SYNTHETIC TEST FRAME]",
        (15, 460),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 215, 255),
        1,
        cv2.LINE_AA,
    )

    sim_results = [
        RecognitionResult(
            bbox=(70, 110, 270, 360),
            detection_confidence=0.99,
            status=dec_matched.status,
            identity=dec_matched.identity,
            similarity=dec_matched.similarity,
        ),
        RecognitionResult(
            bbox=(360, 120, 560, 370),
            detection_confidence=0.97,
            status=dec_unknown.status,
            identity=dec_unknown.identity,
            similarity=dec_unknown.similarity,
        ),
    ]
    annotated = FaceVisualizer.draw_results(
        frame_bgr=canvas,
        results=sim_results,
        fps=28.5,
        registered_count=1,
        threshold=0.7,
    )

    sim_img_path = out_dir / "simulated_matched_ vs_unknown.png"
    cv2.imwrite(str(sim_img_path), annotated)

    report_metrics = {
        "note": "SIMULATION_DATA_ONLY - Chạy run_lab05_webcam.py trên máy thật để thu ảnh Webcam thực tế",
        "threshold": 0.7,
        "matched_case": {
            "identity": dec_matched.identity,
            "status": dec_matched.status,
            "cosine_similarity": round(dec_matched.similarity, 4),
        },
        "unknown_case": {
            "identity": dec_unknown.identity,
            "status": dec_unknown.status,
            "cosine_similarity": round(dec_unknown.similarity, 4),
        },
    }
    json_path = out_dir / "simulation_evaluation.json"
    json_path.write_text(
        json.dumps(report_metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Đã sinh artifact mô phỏng tại: %s và %s", sim_img_path, json_path)


if __name__ == "__main__":
    main()
