"""Tests for Lab 05 FaceNet & MTCNN atomic modules."""

import numpy as np

from src.labs.lab05_facenet import (
    face_database,
    face_preprocessing,
    similarity_matcher,
    visuals,
)


def test_l2_normalize_numpy():
    vec = np.array([[3.0, 4.0]])
    normed = face_database.l2_normalize_numpy(vec)
    assert np.isclose(np.linalg.norm(normed), 1.0)


def test_similarity_matcher_metrics():
    matcher = similarity_matcher.SimilarityMatcher(similarity_threshold=0.6)
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([1.0, 0.0, 0.0])
    v3 = np.array([0.0, 1.0, 0.0])

    sim_same = matcher.compute_cosine_similarity(v1, v2)
    assert np.isclose(sim_same, 1.0)

    sim_diff = matcher.compute_cosine_similarity(v1, v3)
    assert np.isclose(sim_diff, 0.0)


def test_face_preprocessing_crop():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cropped = face_preprocessing.crop_face_with_margin(
        img, (20, 20, 60, 60), margin=0.1
    )
    assert cropped.ndim == 3
    assert cropped.shape[0] > 0 and cropped.shape[1] > 0


def test_visuals_face_visualizer():
    vis = visuals.FaceVisualizer(matching_threshold=0.6)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = visuals.RecognitionResult(
        bbox=(10, 10, 50, 50),
        detection_confidence=0.95,
        status="Matched",
        identity="Alice",
        similarity=0.85,
    )
    rendered = vis.draw_recognition(frame, result)
    assert rendered.shape == frame.shape
