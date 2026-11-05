"""Wavelet Perceptual Hashing (wHash) and Image Similarity Evaluation Facade.

Backward-compatibility shim re-exporting from src.labs.lab04_wavelet.
"""

from __future__ import annotations

from src.labs.lab04_wavelet.wavelet_core import (
    _load_image_grayscale,
    average_hash,
    difference_hash,
    phash_dct,
    wavelet_hash,
    wavelet_hash_color,
    wavelet_hash_slide17,
)
from src.labs.lab04_wavelet.wavelet_distance import (
    calculate_pairwise_hamming_matrix,
    find_top_k_similar,
    hamming_distance,
    normalized_hamming_distance,
)
from src.labs.lab04_wavelet.wavelet_distortions import (
    apply_geometric_distortion,
    apply_photometric_distortion,
)
from src.labs.lab04_wavelet.wavelet_evaluation import (
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
