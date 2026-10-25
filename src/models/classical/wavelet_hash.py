"""Wavelet hashes, binary Hamming distances, geometric edits and Top-k retrieval."""

from __future__ import annotations

from collections.abc import Sequence

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
        # PIL handles Unicode paths and makes the color conversion explicit.
        with Image.open(image_input) as source:
            return np.array(source.convert("L"), dtype=np.float32)

    if isinstance(image_input, Image.Image):
        pil_img = image_input.convert("L")
        return np.array(pil_img, dtype=np.float32)

    if isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:
            return image_input.astype(np.float32)
        if image_input.ndim == 3:
            # NumPy color arrays follow the OpenCV BGR/BGRA convention.
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
    image_size: tuple[int, int] = (256, 256),
) -> np.ndarray:
    """Calculate the project's LL-based hash; slide 18 does not define this design.

    Pipeline:
        1. Preprocess: Convert to grayscale float32 [0,1] and resize to image_size.
        2. Transform: Compute 2D Discrete Wavelet Transform (pywt.wavedec2).
        3. Extract: Isolate the approximation subband LL containing structural energy.
        4. Quantize: Binarize against the median (or mean) coefficient value.
        5. Encode: Return 1D binary bit array of length (bits * bits).

    Args:
        image: Path to image file, NumPy array, or PIL Image.
        wavelet: Mother wavelet family ('db4', 'haar', 'sym4', 'coif2'). Default 'db4'.
        level: Decomposition level (default 1).
        bits: Hash resolution along each axis, producing a total of (bits * bits) bits (default 8 -> 64 bits).
        mode: Quantization threshold mode ('median' or 'mean'). Default 'median'.
        image_size: Resize target (width, height), default (256,256).

    Returns:
        np.ndarray: 1D uint8 array of 0s and 1s representing the binary hash code.
    """
    if bits < 1 or level < 1 or mode not in {"median", "mean"}:
        raise ValueError("Positive bits/level and median/mean mode are required.")
    if len(image_size) != 2 or min(image_size) < bits:
        raise ValueError("Image dimensions must be at least the hash side length.")
    img_gray = _load_image_grayscale(image)
    if not img_gray.size or not np.isfinite(img_gray).all():
        raise ValueError("Image must be nonempty and finite.")
    if img_gray.min() < 0 or img_gray.max() > 255:
        raise ValueError("Grayscale image values must be in [0,255] or [0,1].")
    if img_gray.max() > 1:
        img_gray = img_gray / 255.0
    img_resized = cv2.resize(img_gray, tuple(image_size), interpolation=cv2.INTER_AREA)
    if level > pywt.dwtn_max_level(img_resized.shape, wavelet):
        raise ValueError("Decomposition level exceeds the useful image/wavelet limit.")

    # 2D DWT Decomposition
    coeffs = pywt.wavedec2(img_resized, wavelet=wavelet, level=level, mode="symmetric")

    # coeffs[0] is the approximation subband cA_level (LL)
    ll_band = coeffs[0]

    # Resize LL to the configured bit-grid dimensions.
    ll_resized = cv2.resize(ll_band, (bits, bits), interpolation=cv2.INTER_AREA)

    # Quantization
    if mode == "mean":
        thresh = np.mean(ll_resized)
    else:
        thresh = np.median(ll_resized)

    # Generate binary hash code
    hash_matrix = (ll_resized > thresh).astype(np.uint8)
    hash_code = hash_matrix.flatten()

    return hash_code


def wavelet_hash_color(
    image: str | np.ndarray | Image.Image,
    wavelet: str = "db4",
    level: int = 1,
    bits: int = 8,
    mode: str = "median",
    image_size: tuple[int, int] = (256, 256),
) -> np.ndarray:
    """Hash an RGB image by concatenating independent R, G and B LL hashes.

    Args:
        image: RGB file/PIL image or RGB NumPy array shaped (H, W, 3).
        wavelet: Wavelet family passed to each 2D channel hash.
        level: Decomposition level for each channel.
        bits: Side length of each channel's bit matrix.
        mode: Per-channel LL threshold, ``median`` or ``mean``.
        image_size: Resize target as (width, height).

    Returns:
        Binary uint8 vector shaped (3 * bits * bits,), ordered R then G then B.
    """
    if isinstance(image, str):
        with Image.open(image) as source:
            rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
    elif isinstance(image, Image.Image):
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    elif isinstance(image, np.ndarray):
        rgb = image
    else:
        raise ValueError("Expected an RGB image path, PIL image, or NumPy array.")
    if rgb.ndim != 3 or rgb.shape[2] != 3 or not rgb.size:
        raise ValueError("RGB array must have shape (H, W, 3) and be nonempty.")
    if not np.issubdtype(rgb.dtype, np.number) or not np.isfinite(rgb).all():
        raise ValueError("RGB image values must be finite numbers.")
    if rgb.min() < 0 or rgb.max() > 255:
        raise ValueError("RGB image values must be in [0,255] or [0,1].")
    return np.concatenate(
        [
            wavelet_hash(
                rgb[:, :, channel],
                wavelet=wavelet,
                level=level,
                bits=bits,
                mode=mode,
                image_size=image_size,
            )
            for channel in range(3)
        ]
    )


def wavelet_hash_slide17(
    image: str | np.ndarray | Image.Image,
    wavelet: str = "db4",
    level: int = 3,
    step: float = 2.0,
    image_size: tuple[int, int] = (256, 256),
) -> list[int]:
    """Calculate a step-quantized variant inspired by the incomplete slide 17 code.

    Flattens approximation/detail coefficients, divides by step and rounds,
    and returns bits modulo 2.

    Args:
        image: Path to image file or array.
        wavelet: Wavelet type (default 'db4').
        level: Decomposition level (default 3).
        step: Step size for coefficient quantization.
        image_size: Resize target (width,height), default (256,256).

    Returns:
        list[int]: List of integer bits {0, 1}.
    """
    if step <= 0 or not np.isfinite(step) or level < 1 or min(image_size) < 1:
        raise ValueError("Positive finite step, level and image dimensions required.")
    img_gray = _load_image_grayscale(image)
    if not img_gray.size or not np.isfinite(img_gray).all():
        raise ValueError("Image must be nonempty and finite.")
    if img_gray.min() < 0 or img_gray.max() > 255:
        raise ValueError("Image values must be in [0,255] or [0,1].")
    if img_gray.max() <= 1:
        img_gray = img_gray * 255.0
    img_resized = cv2.resize(img_gray, tuple(image_size), interpolation=cv2.INTER_AREA)
    if level > pywt.dwtn_max_level(img_resized.shape, wavelet):
        raise ValueError("Decomposition level exceeds the useful image/wavelet limit.")

    coeffs = pywt.wavedec2(img_resized, wavelet=wavelet, level=level, mode="symmetric")

    # Handle the structure of wavedec2: [cA_n, (cH_n, cV_n, cD_n), ...]
    flat_parts = []
    # cA_n
    cA_quant = np.round(coeffs[0] / step).astype(np.int64)
    flat_parts.append(cA_quant.flatten())

    # Details tuples
    for detail_tuple in coeffs[1:]:
        for subband in detail_tuple:
            sub_quant = np.round(subband / step).astype(np.int64)
            flat_parts.append(sub_quant.flatten())

    flattened = np.concatenate(flat_parts)

    hash_code = (flattened % 2).astype(np.uint8).tolist()
    return hash_code


def hamming_distance(
    hash1: Sequence[int] | np.ndarray, hash2: Sequence[int] | np.ndarray
) -> int:
    """Calculate the exact Hamming distance between two binary hash codes.

    Args:
        hash1: First binary hash code of length B.
        hash2: Second binary hash code of length B.

    Returns:
        int: Total number of bit positions where hash1 and hash2 differ.
    """
    h1 = np.asarray(hash1)
    h2 = np.asarray(hash2)

    if h1.shape != h2.shape:
        raise ValueError(f"Hash lengths mismatch: {h1.shape} vs {h2.shape}")
    if h1.ndim != 1 or not h1.size:
        raise ValueError("Hashes must be nonempty one-dimensional arrays.")
    if not np.isin(h1, [0, 1]).all() or not np.isin(h2, [0, 1]).all():
        raise ValueError("Hashes must contain only binary values.")

    return int(np.count_nonzero(h1 != h2))


def normalized_hamming_distance(
    hash1: Sequence[int] | np.ndarray, hash2: Sequence[int] | np.ndarray
) -> float:
    """Calculate the normalized Hamming distance in range [0.0, 1.0].

    Args:
        hash1: First binary hash code.
        hash2: Second binary hash code.

    Returns:
        float: Normalized distance (Hamming Distance / Bit Length).
    """
    h1 = np.asarray(hash1)
    dist = hamming_distance(h1, hash2)
    return float(dist / len(h1))


def calculate_pairwise_hamming_matrix(
    hashes: list[np.ndarray] | np.ndarray,
) -> np.ndarray:
    """Compute the full symmetric pairwise Hamming distance matrix for M images.

    Follows the nested comparison structure in Slide 18:
        for i in range(len(hashes)):
            for j in range(i+1, len(hashes)):
                distance = hamming_distance(...)

    Args:
        hashes: Sequence of M hash codes, each of length B.

    Returns:
        np.ndarray: Symmetric matrix of shape (M, M) with zeros on the diagonal.
    """
    hashes_arr = np.asarray(hashes, dtype=np.uint8)
    # Vectorized broadcasting: (M, 1, B) != (1, M, B) -> sum along axis=2
    diff_matrix = hashes_arr[:, np.newaxis, :] != hashes_arr[np.newaxis, :, :]
    matrix = np.sum(diff_matrix, axis=2, dtype=np.int32)
    return matrix


def apply_geometric_distortion(
    image: np.ndarray,
    distortion_type: str,
    param: float | int | bool,
) -> np.ndarray:
    """Apply controlled geometric transformation to an image.

    Args:
        image: Input image array uint8.
        distortion_type: Type of geometric distortion ('scale', 'crop', 'rotate', 'flip').
        param: Distortion magnitude parameter.

    Returns:
        np.ndarray: Transformed image matching original dimensions.
    """
    h, w = image.shape[:2]

    if distortion_type == "scale":
        # param: scale factor e.g. 0.5 or 1.5
        factor = float(param)
        new_w, new_h = max(8, int(w * factor)), max(8, int(h * factor))
        scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        # Resize back to original size for uniform comparison
        return cv2.resize(scaled, (w, h), interpolation=cv2.INTER_LINEAR)

    if distortion_type == "crop":
        # param: crop ratio e.g. 0.8
        ratio = float(param)
        cw, ch = int(w * ratio), int(h * ratio)
        x1 = (w - cw) // 2
        y1 = (h - ch) // 2
        cropped = image[y1 : y1 + ch, x1 : x1 + cw]
        return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    if distortion_type == "rotate":
        # param: angle in degrees
        angle = float(param)
        matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)

    if distortion_type == "flip":
        # Horizontal flip used by the notebook experiment.
        return cv2.flip(image, 1)

    raise ValueError(f"Unknown geometric distortion: {distortion_type}")


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
