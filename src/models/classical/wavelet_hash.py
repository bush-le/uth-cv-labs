"""Wavelet Perceptual Hashing (wHash) and Image Similarity Evaluation Facade.

Consolidates and re-exports:
1. Wavelet Hash algorithm adhering to Slide 17 & Slide 18 (`wavelet_core`).
2. RGB 192-bit Wavelet Hash (`wavelet_core`).
3. Classic perceptual hashes (aHash, dHash, pHash) (`wavelet_core`).
4. Vectorized Hamming distance and top-k retrieval (`wavelet_distance`).
5. Ground truth classification evaluation and ROC analysis (`wavelet_evaluation`).
6. Controlled photometric and geometric distortion suite (`wavelet_distortions`).
"""

from __future__ import annotations

from src.models.classical.wavelet_core import (
    _load_image_grayscale,
    average_hash,
    difference_hash,
    phash_dct,
    wavelet_hash,
    wavelet_hash_color,
    wavelet_hash_slide17,
)
from src.models.classical.wavelet_distance import (
    calculate_pairwise_hamming_matrix,
    find_top_k_similar,
    hamming_distance,
    normalized_hamming_distance,
)
from src.models.classical.wavelet_distortions import (
    apply_geometric_distortion,
    apply_photometric_distortion,
)
from src.models.classical.wavelet_evaluation import (
    evaluate_ground_truth_pairs,
)

__all__ = [
    # Core Hashes
    "_load_image_grayscale",
    "wavelet_hash",
    "wavelet_hash_color",
    "wavelet_hash_slide17",
    "average_hash",
    "difference_hash",
    "phash_dct",
    # Distance & Search
    "hamming_distance",
    "normalized_hamming_distance",
    "calculate_pairwise_hamming_matrix",
    "find_top_k_similar",
    # Evaluation & ROC
    "evaluate_ground_truth_pairs",
    # Distortions
    "apply_photometric_distortion",
    "apply_geometric_distortion",
]
