"""Hamming distance metrics, pairwise distance matrix, and top-k retrieval."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.common.metrics import (
    hamming_distance,
    normalized_hamming_distance,
)


def calculate_pairwise_hamming_matrix(
    hashes: list[np.ndarray] | np.ndarray,
) -> np.ndarray:
    """Compute the full symmetric pairwise Hamming distance matrix for M images.

    Args:
        hashes: Sequence of M hash codes, each of length B.

    Returns:
        np.ndarray: Symmetric matrix of shape (M, M) with zeros on the diagonal.
    """
    hashes_arr = np.asarray(hashes, dtype=np.uint8)
    diff_matrix = hashes_arr[:, np.newaxis, :] != hashes_arr[np.newaxis, :, :]
    matrix = np.sum(diff_matrix, axis=2, dtype=np.int32)
    return matrix


def find_top_k_similar(
    query_hash: np.ndarray,
    database_hashes: Sequence[np.ndarray],
    k: int = 5,
) -> list[tuple[int, int]]:
    """Retrieve the top-k most similar items from a database based on Hamming distance.

    Args:
        query_hash: Query binary hash code.
        database_hashes: List of candidate hash codes in database.
        k: Number of nearest items to retrieve.

    Returns:
        list[tuple[int, int]]: List of (database_index, hamming_distance) pairs sorted ascending by distance.
    """
    distances = [hamming_distance(query_hash, cand) for cand in database_hashes]
    sorted_pairs = sorted(enumerate(distances), key=lambda x: x[1])
    return sorted_pairs[:k]


__all__ = [
    "hamming_distance",
    "normalized_hamming_distance",
    "calculate_pairwise_hamming_matrix",
    "find_top_k_similar",
]
