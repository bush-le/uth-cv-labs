"""Lab 04 Wavelet Package: 2D DWT, Perceptual Hashes, Distance Metrics, Evaluation, and Pipeline."""

from src.labs.lab04_wavelet import (
    pipeline,
    visuals,
    wavelet_core,
    wavelet_distance,
    wavelet_distortions,
    wavelet_evaluation,
    wavelet_hash,
    wavelet_pairs,
)
from src.labs.lab04_wavelet.pipeline import Lab04Pipeline
from src.labs.lab04_wavelet.wavelet_core import (
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
    compare_hash_methods,
    evaluate_ground_truth_pairs,
    measure_robustness,
    retrieve_top_k,
    score_pairs,
    select_threshold,
)
from src.labs.lab04_wavelet.wavelet_pairs import load_pair_dataset

__all__ = [
    "wavelet_core",
    "wavelet_hash",
    "wavelet_distance",
    "wavelet_distortions",
    "wavelet_pairs",
    "wavelet_evaluation",
    "visuals",
    "pipeline",
    "wavelet_hash",
    "wavelet_hash_color",
    "wavelet_hash_slide17",
    "average_hash",
    "difference_hash",
    "phash_dct",
    "hamming_distance",
    "normalized_hamming_distance",
    "calculate_pairwise_hamming_matrix",
    "find_top_k_similar",
    "apply_photometric_distortion",
    "apply_geometric_distortion",
    "evaluate_ground_truth_pairs",
    "score_pairs",
    "select_threshold",
    "compare_hash_methods",
    "retrieve_top_k",
    "measure_robustness",
    "load_pair_dataset",
    "Lab04Pipeline",
]
