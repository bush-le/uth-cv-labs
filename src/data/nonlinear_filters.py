"""Non-linear spatial filters (median, bilateral) and synthetic noise corruption."""

from __future__ import annotations

import cv2
import numpy as np

from src.data.filter_utils import _validate_image


def apply_median_filter(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Denoise with a median filter (non-linear, per-channel).

    Each output pixel is the median of its neighborhood, effective against
    salt-and-pepper noise while preserving edges.

    Args:
        image: Input image array of shape (H, W, 3) or (H, W), dtype uint8.
        ksize: Odd aperture size (> 1).

    Returns:
        np.ndarray: Denoised image, same shape as input.

    Raises:
        ValueError: If ksize is not an odd integer greater than 1.
    """
    _validate_image(image)
    if ksize <= 1 or ksize % 2 == 0:
        raise ValueError(f"Median ksize must be an odd integer > 1, got {ksize}")
    if image.ndim == 3 and image.shape[2] == 3:
        channels = cv2.split(image)
        filtered = [cv2.medianBlur(ch, ksize) for ch in channels]
        return cv2.merge(filtered)
    return cv2.medianBlur(image, ksize)


def apply_bilateral_filter(
    image: np.ndarray,
    d: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
) -> np.ndarray:
    """Smooth while preserving edges with a bilateral filter (non-linear).

    Combines spatial proximity and intensity similarity, so homogeneous
    regions are blurred while boundaries stay sharp.

    Args:
        image: Input image array of shape (H, W, 3) or (H, W), dtype uint8.
        d: Pixel neighborhood diameter (> 0).
        sigma_color: Filter sigma in the color space (> 0).
        sigma_space: Filter sigma in the coordinate space (> 0).

    Returns:
        np.ndarray: Filtered image, same shape as input.

    Raises:
        ValueError: If any parameter is non-positive.
    """
    _validate_image(image)
    if d <= 0 or sigma_color <= 0 or sigma_space <= 0:
        raise ValueError(
            f"Bilateral params must be positive, got d={d}, "
            f"sigma_color={sigma_color}, sigma_space={sigma_space}"
        )
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def add_salt_pepper_noise(
    image: np.ndarray, amount: float = 0.02, seed: int = 42
) -> np.ndarray:
    """Corrupt an image with salt-and-pepper noise for demo purposes.

    A reproducible RNG is used so experiments stay repeatable.

    Args:
        image: Input image array of shape (H, W, 3) or (H, W), dtype uint8.
        amount: Fraction of pixels to corrupt, in (0, 1).
        seed: Random seed for reproducibility.

    Returns:
        np.ndarray: Noisy image, same shape and dtype uint8.

    Raises:
        ValueError: If amount is not in (0, 1).
    """
    if not (0 < amount < 1):
        raise ValueError(f"Noise amount must be in (0, 1), got {amount}")
    _validate_image(image)
    rng = np.random.default_rng(seed)
    noisy = image.copy()
    num_pixels = image.shape[0] * image.shape[1]
    num_corrupt = int(amount * num_pixels)

    rows = rng.integers(0, image.shape[0], size=num_corrupt)
    cols = rng.integers(0, image.shape[1], size=num_corrupt)
    is_salt = rng.random(num_corrupt) > 0.5

    if noisy.ndim == 3:
        noisy[rows[is_salt], cols[is_salt]] = (255, 255, 255)
        noisy[rows[~is_salt], cols[~is_salt]] = (0, 0, 0)
    else:
        noisy[rows[is_salt], cols[is_salt]] = 255
        noisy[rows[~is_salt], cols[~is_salt]] = 0
    return noisy
