"""Unit tests for Lab 05 FaceNet biometric, manifold, and geometric visual functions."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.labs.lab05_facenet.visuals import (
    plot_biometric_verification_suite,
    plot_feature_signature_profile,
    plot_gallery_probe_matrix,
    plot_hyperspherical_tsne_clusters,
    plot_margin_sensitivity_curve,
    visualize_mtcnn_alignment_and_margin,
)


@pytest.fixture
def sample_face_detection_data() -> tuple[
    np.ndarray, tuple[int, int, int, int], np.ndarray
]:
    """Generate a synthetic 240x320 BGR image, bounding box, and 5 facial landmarks."""
    img = np.full((240, 320, 3), 128, dtype=np.uint8)
    # Add face-like patch
    img[50:190, 80:240] = 200
    bbox = (80, 50, 240, 190)
    # 5 landmarks: left eye, right eye, nose, left mouth, right mouth
    landmarks = np.array(
        [
            [120.0, 95.0],
            [200.0, 98.0],
            [160.0, 130.0],
            [130.0, 160.0],
            [190.0, 162.0],
        ],
        dtype=np.float32,
    )
    return img, bbox, landmarks


@pytest.fixture
def sample_biometric_scores() -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic genuine and impostor cosine similarity score distributions."""
    rng = np.random.default_rng(42)
    # Genuine pairs: high similarity ~ [0.75, 0.96]
    genuine = np.clip(rng.normal(0.88, 0.04, 150), 0.0, 1.0)
    # Impostor pairs: low similarity ~ [0.05, 0.45]
    impostor = np.clip(rng.normal(0.22, 0.08, 300), -0.2, 0.65)
    return genuine, impostor


@pytest.fixture
def sample_embedding_clusters() -> tuple[np.ndarray, list[str]]:
    """Generate synthetic 512-D L2-normalized embeddings for 3 identities."""
    rng = np.random.default_rng(42)
    embeddings = []
    identities = []
    for id_idx in range(3):
        base = rng.standard_normal(512).astype(np.float32)
        base = base / np.linalg.norm(base)
        for _ in range(10):
            noise = rng.standard_normal(512).astype(np.float32) * 0.015
            vec = base + noise
            vec = vec / np.linalg.norm(vec)
            embeddings.append(vec)
            identities.append(f"Person_{id_idx + 1}")
    return np.array(embeddings), identities


def test_visualize_mtcnn_alignment_and_margin(sample_face_detection_data):
    """Verify 3-panel geometric diagnostic: detection+landmarks, tight crop, and margin-expanded crop."""
    img, bbox, landmarks = sample_face_detection_data
    fig = visualize_mtcnn_alignment_and_margin(
        img, bbox, landmarks=landmarks, margin=14
    )
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 3
    plt.close(fig)

    # Test with landmarks=None
    fig_no_lm = visualize_mtcnn_alignment_and_margin(
        img, bbox, landmarks=None, margin=10
    )
    assert isinstance(fig_no_lm, plt.Figure)
    assert len(fig_no_lm.axes) == 3
    plt.close(fig_no_lm)


def test_plot_gallery_probe_matrix():
    """Verify annotated gallery-probe similarity matrix with green match borders."""
    sim_matrix = np.array(
        [
            [0.98, 0.21, 0.15, 0.08],
            [0.21, 0.94, 0.32, 0.12],
            [0.15, 0.32, 0.96, 0.28],
            [0.08, 0.12, 0.28, 0.91],
        ],
        dtype=np.float32,
    )
    labels = ["Person_A", "Person_B", "Person_C", "Person_D"]
    fig = plot_gallery_probe_matrix(sim_matrix, row_labels=labels, threshold=0.70)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) >= 1
    plt.close(fig)


def test_plot_biometric_verification_suite(sample_biometric_scores):
    """Verify 3-panel biometric suite: Dual KDE error tails, FAR/FRR trade-off with EER, and ROC curve."""
    genuine, impostor = sample_biometric_scores
    fig = plot_biometric_verification_suite(genuine, impostor, threshold=0.70)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 3
    plt.close(fig)


def test_plot_hyperspherical_tsne_clusters(sample_embedding_clusters):
    """Verify 2D t-SNE scatter with identity centroid stars and radial link lines."""
    embeddings, identities = sample_embedding_clusters
    fig = plot_hyperspherical_tsne_clusters(embeddings, identities, perplexity=5.0)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) >= 1
    plt.close(fig)


def test_plot_feature_signature_profile():
    """Verify 512-D waveform sparklines and channel difference bars."""
    rng = np.random.default_rng(42)
    base_a = rng.standard_normal(512).astype(np.float32)
    base_a = base_a / np.linalg.norm(base_a)

    vec_a1 = base_a
    noise = rng.standard_normal(512).astype(np.float32) * 0.015
    vec_a2 = base_a + noise
    vec_a2 = vec_a2 / np.linalg.norm(vec_a2)

    base_b = rng.standard_normal(512).astype(np.float32)
    vec_b = base_b / np.linalg.norm(base_b)

    fig = plot_feature_signature_profile(vec_a1, vec_a2, vec_b, sample_dims=64)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) >= 2
    plt.close(fig)


def test_plot_margin_sensitivity_curve():
    """Verify Cosine similarity sensitivity curve across crop margins with threshold basin."""
    # Test default synthesize mode
    fig_def = plot_margin_sensitivity_curve()
    assert isinstance(fig_def, plt.Figure)
    assert len(fig_def.axes) >= 1
    plt.close(fig_def)

    # Test custom margins and similarities
    margins = [0, 5, 10, 14, 20, 25, 30, 40]
    sims = [0.72, 0.83, 0.91, 0.95, 0.92, 0.85, 0.76, 0.62]
    fig_custom = plot_margin_sensitivity_curve(
        margins=margins, similarities=sims, threshold=0.70, optimal_margin=14
    )
    assert isinstance(fig_custom, plt.Figure)
    assert len(fig_custom.axes) >= 1
    plt.close(fig_custom)
