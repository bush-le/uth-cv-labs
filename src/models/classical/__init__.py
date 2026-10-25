"""Classical computer vision models and perceptual hashing algorithms."""

from src.models.classical.wavelet_hash import (
    apply_geometric_distortion,
    apply_photometric_distortion,
    average_hash,
    calculate_pairwise_hamming_matrix,
    difference_hash,
    evaluate_ground_truth_pairs,
    find_top_k_similar,
    hamming_distance,
    normalized_hamming_distance,
    phash_dct,
    wavelet_hash,
    wavelet_hash_color,
    wavelet_hash_slide17,
)

__all__ = [
    "wavelet_hash",
    "wavelet_hash_color",
    "wavelet_hash_slide17",
    "average_hash",
    "difference_hash",
    "phash_dct",
    "hamming_distance",
    "normalized_hamming_distance",
    "calculate_pairwise_hamming_matrix",
    "evaluate_ground_truth_pairs",
    "apply_photometric_distortion",
    "apply_geometric_distortion",
    "find_top_k_similar",
]
