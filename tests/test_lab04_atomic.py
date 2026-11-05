"""Tests for Lab 04 Wavelet Image Hashing atomic modules."""

import numpy as np
import pytest

from src.labs.lab04_wavelet import (
    wavelet_core,
    wavelet_distance,
    wavelet_distortions,
    wavelet_hash,
    visuals,
)


def test_hamming_distance():
    h1 = np.array([1, 0, 1, 1, 0], dtype=np.uint8)
    h2 = np.array([1, 1, 0, 1, 0], dtype=np.uint8)
    dist = wavelet_distance.hamming_distance(h1, h2)
    assert dist == 2


def test_pairwise_hamming_matrix():
    h1 = np.array([1, 0, 1], dtype=np.uint8)
    h2 = np.array([1, 1, 0], dtype=np.uint8)
    matrix = wavelet_distance.calculate_pairwise_hamming_matrix([h1, h2])
    assert matrix.shape == (2, 2)
    assert matrix[0, 0] == 0
    assert matrix[0, 1] == 2
    assert matrix[1, 0] == 2


def test_wavelet_hash_generation():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[16:48, 16:48] = 200
    h = wavelet_core.wavelet_hash(img, bits=8)
    assert isinstance(h, np.ndarray)
    assert h.dtype in (np.uint8, bool, int, np.int64)


def test_wavelet_distortions():
    img = np.full((32, 32, 3), 100, dtype=np.uint8)
    noisy = wavelet_distortions.apply_photometric_distortion(img, "gaussian_noise", 15.0)
    assert noisy.shape == img.shape
