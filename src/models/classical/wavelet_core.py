"""Core wavelet perceptual hashing algorithms (Slide 17 & Slide 18) and baseline hashes."""

from __future__ import annotations

import cv2
import numpy as np
import pywt
from PIL import Image


def _load_image_grayscale(image_input: str | np.ndarray | Image.Image) -> np.ndarray:
    """Normalize various image input formats into a 2D float32 grayscale NumPy array.

    Args:
        image_input: File path string, NumPy array (H, W) or (H, W, C), or PIL Image.

    Returns:
        np.ndarray: Grayscale float32 image of shape (H, W) normalized to [0.0, 255.0].
    """
    if isinstance(image_input, str):
        pil_img = Image.open(image_input).convert("L")
        return np.array(pil_img, dtype=np.float32)

    if isinstance(image_input, Image.Image):
        pil_img = image_input.convert("L")
        return np.array(pil_img, dtype=np.float32)

    if isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:
            return image_input.astype(np.float32)
        if image_input.ndim == 3:
            if image_input.shape[2] == 3:
                return cv2.cvtColor(
                    image_input.astype(np.uint8), cv2.COLOR_BGR2GRAY
                ).astype(np.float32)
            if image_input.shape[2] == 4:
                return cv2.cvtColor(
                    image_input.astype(np.uint8), cv2.COLOR_BGRA2GRAY
                ).astype(np.float32)

    raise ValueError(f"Unsupported image input type: {type(image_input)}")


def wavelet_hash(
    image: str | np.ndarray | Image.Image,
    wavelet: str = "db4",
    level: int = 1,
    bits: int = 8,
    mode: str = "median",
) -> np.ndarray:
    """Calculate the Wavelet Hash (wHash) of an image based on Slide 18 specification.

    Pipeline:
        1. Preprocess: Convert image to grayscale and resize to (bits*4, bits*4).
        2. Transform: Compute 2D Discrete Wavelet Transform (pywt.wavedec2).
        3. Extract: Isolate the approximation subband LL containing structural energy.
        4. Quantize: Binarize against the median (or mean) coefficient value.
        5. Encode: Return 1D binary bit array of length (bits * bits).

    Args:
        image: Path to image file, NumPy array, or PIL Image.
        wavelet: Mother wavelet family ('db4', 'haar', 'sym4', 'coif2'). Default 'db4'.
        level: Decomposition level (default 1).
        bits: Hash resolution along each axis (default 8 -> 64 bits).
        mode: Quantization threshold mode ('median' or 'mean'). Default 'median'.

    Returns:
        np.ndarray: 1D uint8 array of 0s and 1s representing the binary hash code.
    """
    img_gray = _load_image_grayscale(image)

    target_scale = max(32, bits * 4)
    img_resized = cv2.resize(
        img_gray, (target_scale, target_scale), interpolation=cv2.INTER_AREA
    )

    coeffs = pywt.wavedec2(img_resized, wavelet=wavelet, level=level)
    ll_band = coeffs[0]
    ll_resized = cv2.resize(ll_band, (bits, bits), interpolation=cv2.INTER_AREA)

    if mode == "mean":
        thresh = np.mean(ll_resized)
    else:
        thresh = np.median(ll_resized)

    hash_matrix = (ll_resized > thresh).astype(np.uint8)
    return hash_matrix.flatten()


def wavelet_hash_slide17(
    image: str | np.ndarray | Image.Image,
    wavelet: str = "db4",
    level: int = 3,
    step: float = 2.0,
) -> list[int]:
    """Calculate wavelet hash following the exact literal implementation in Slide 17.

    Flattens all approximation and detail coefficients, quantizes via integer division,
    and returns bits modulo 2.

    Args:
        image: Path to image file or array.
        wavelet: Wavelet type (default 'db4').
        level: Decomposition level (default 3).
        step: Step size for coefficient quantization.

    Returns:
        list[int]: List of integer bits {0, 1}.
    """
    img_gray = _load_image_grayscale(image)
    img_resized = cv2.resize(img_gray, (64, 64), interpolation=cv2.INTER_AREA)

    coeffs = pywt.wavedec2(img_resized, wavelet=wavelet, level=level)

    flat_parts = []
    cA_quant = np.round(coeffs[0] / step).astype(np.int64)
    flat_parts.append(cA_quant.flatten())

    for detail_tuple in coeffs[1:]:
        for subband in detail_tuple:
            sub_quant = np.round(subband / step).astype(np.int64)
            flat_parts.append(sub_quant.flatten())

    flattened = np.concatenate(flat_parts)
    return [int(abs(bit) % 2) for bit in flattened]


def average_hash(
    image: str | np.ndarray | Image.Image,
    hash_size: int = 8,
) -> np.ndarray:
    """Calculate Average Hash (aHash) based on spatial mean pixel intensity.

    Args:
        image: Image input.
        hash_size: Size of square thumbnail (default 8 -> 64 bits).

    Returns:
        np.ndarray: 1D uint8 array of 0s and 1s of length (hash_size * hash_size).
    """
    img = _load_image_grayscale(image)
    resized = cv2.resize(img, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    mean_val = np.mean(resized)
    return (resized > mean_val).astype(np.uint8).flatten()


def difference_hash(
    image: str | np.ndarray | Image.Image,
    hash_size: int = 8,
) -> np.ndarray:
    """Calculate Difference Hash (dHash) based on horizontal adjacent pixel gradients.

    Args:
        image: Image input.
        hash_size: Square output resolution (default 8 -> 64 bits).

    Returns:
        np.ndarray: 1D uint8 array of 0s and 1s of length (hash_size * hash_size).
    """
    img = _load_image_grayscale(image)
    resized = cv2.resize(img, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    return diff.astype(np.uint8).flatten()


def phash_dct(
    image: str | np.ndarray | Image.Image,
    hash_size: int = 8,
    highfreq_factor: int = 4,
) -> np.ndarray:
    """Calculate Perceptual Hash (pHash) based on 2D Discrete Cosine Transform (DCT).

    Args:
        image: Image input.
        hash_size: Hash code dimension (default 8 -> 64 bits).
        highfreq_factor: Scaling factor for intermediate DCT image (default 4 -> 32x32).

    Returns:
        np.ndarray: 1D uint8 array of 0s and 1s.
    """
    img = _load_image_grayscale(image)
    img_size = hash_size * highfreq_factor
    resized = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)

    dct_block = cv2.dct(resized)
    dct_low = dct_block[:hash_size, :hash_size]
    median_val = np.median(dct_low)

    return (dct_low > median_val).astype(np.uint8).flatten()
