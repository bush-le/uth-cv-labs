"""Wavelet Perceptual Hashing (wHash) and Image Similarity Evaluation Suite.

Implements:
1. Wavelet Hash algorithm adhering to Slide 17 and Slide 18 specifications.
2. Classic perceptual hashes (aHash, dHash, pHash) for comparative benchmarking.
3. Fast vectorized Hamming distance calculation and pairwise distance matrices.
4. Ground truth classification evaluation (Accuracy, Sensitivity/Recall, Specificity, Precision, F1-Score).
5. ROC curve generation, AUC computation, and optimal threshold selection.
6. Comprehensive image distortion suite for robustness analysis.
7. Top-k nearest neighbor image retrieval.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np
import pywt
from PIL import Image

from src.common.logger import get_logger

logger = get_logger(__name__)


# ==============================================================================
# 1. Wavelet Hash Implementations (Slide 17 & Slide 18 Variants)
# ==============================================================================


def _load_image_grayscale(image_input: str | np.ndarray | Image.Image) -> np.ndarray:
    """Normalize various image input formats into a 2D float32 grayscale NumPy array.

    Args:
        image_input: File path string, NumPy array (H, W) or (H, W, C), or PIL Image.

    Returns:
        np.ndarray: Grayscale float32 image of shape (H, W) normalized to [0.0, 255.0].
    """
    if isinstance(image_input, str):
        # Read with PIL for robust Unicode path handling and match Slide 18
        pil_img = Image.open(image_input).convert("L")
        return np.array(pil_img, dtype=np.float32)

    if isinstance(image_input, Image.Image):
        pil_img = image_input.convert("L")
        return np.array(pil_img, dtype=np.float32)

    if isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:
            return image_input.astype(np.float32)
        if image_input.ndim == 3:
            # Assume RGB / BGR
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
        1. Preprocess: Convert image to grayscale and resize to (bits*4, bits*4) or (64, 64).
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

    Returns:
        np.ndarray: 1D uint8 array of 0s and 1s representing the binary hash code.
    """
    img_gray = _load_image_grayscale(image)

    # Standardize input scale (scale proportional to target hash bits)
    target_scale = max(32, bits * 4)
    img_resized = cv2.resize(
        img_gray, (target_scale, target_scale), interpolation=cv2.INTER_AREA
    )

    # 2D DWT Decomposition
    coeffs = pywt.wavedec2(img_resized, wavelet=wavelet, level=level)

    # coeffs[0] is the approximation subband cA_level (LL)
    ll_band = coeffs[0]

    # Resize/crop LL band to exactly (bits, bits)
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
    # Ensure dimensions are powers of 2 for clean multi-level DWT
    img_resized = cv2.resize(img_gray, (64, 64), interpolation=cv2.INTER_AREA)

    coeffs = pywt.wavedec2(img_resized, wavelet=wavelet, level=level)

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

    hash_code = [int(abs(bit) % 2) for bit in flattened]
    return hash_code


# ==============================================================================
# 2. Baseline Perceptual Hashes (aHash, dHash, pHash)
# ==============================================================================


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
    # Difference between column P[x] and P[x+1]
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

    # 2D DCT via OpenCV
    dct_block = cv2.dct(resized)

    # Extract top-left low frequencies (excluding DC term at [0, 0])
    dct_low = dct_block[:hash_size, :hash_size]
    median_val = np.median(dct_low)

    return (dct_low > median_val).astype(np.uint8).flatten()


# ==============================================================================
# 3. Vectorized Hamming Distance & Pairwise Matrix
# ==============================================================================


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
    h1 = np.asarray(hash1, dtype=np.uint8)
    h2 = np.asarray(hash2, dtype=np.uint8)

    if h1.shape != h2.shape:
        raise ValueError(f"Hash lengths mismatch: {h1.shape} vs {h2.shape}")

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
    h1 = np.asarray(hash1, dtype=np.uint8)
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
    m, b = hashes_arr.shape

    # Vectorized broadcasting: (M, 1, B) != (1, M, B) -> sum along axis=2
    diff_matrix = hashes_arr[:, np.newaxis, :] != hashes_arr[np.newaxis, :, :]
    matrix = np.sum(diff_matrix, axis=2, dtype=np.int32)
    return matrix


# ==============================================================================
# 4. Ground Truth Evaluation, Metrics & ROC / AUC Analysis
# ==============================================================================


def evaluate_ground_truth_pairs(
    distances: Sequence[int | float] | np.ndarray,
    ground_truth_labels: Sequence[int | bool] | np.ndarray,
    thresholds: Sequence[int | float] | np.ndarray | None = None,
    max_distance: int = 64,
) -> dict[str, Any]:
    """Perform comprehensive statistical classification evaluation across distance thresholds.

    Evaluates:
        - Accuracy, Recall (Độ nhạy), Specificity (Độ đặc hiệu), Precision, F1-Score
        - ROC curve coordinates (FPR vs TPR) and AUC score
        - Precision-Recall curve
        - Optimal threshold according to Youden's J statistic

    Args:
        distances: 1D array of pairwise distances between image pairs.
        ground_truth_labels: 1D array of binary labels (1 = Similar/Match, 0 = Dissimilar/Non-match).
        thresholds: Sequence of thresholds to evaluate. If None, sweeps range [0, max_distance].
        max_distance: Maximum possible Hamming distance (bit length).

    Returns:
        dict[str, Any]: Detailed evaluation metrics and curve coordinates.
    """
    dist_arr = np.asarray(distances, dtype=np.float32)
    gt_arr = np.asarray(ground_truth_labels, dtype=np.int32)

    if thresholds is None:
        thresh_list = np.arange(0, max_distance + 1)
    else:
        thresh_list = np.asarray(thresholds)

    tpr_list: list[float] = []
    fpr_list: list[float] = []
    precision_list: list[float] = []
    recall_list: list[float] = []
    specificity_list: list[float] = []
    accuracy_list: list[float] = []
    f1_list: list[float] = []

    pos_count = int(np.sum(gt_arr == 1))
    neg_count = int(np.sum(gt_arr == 0))

    if pos_count == 0 or neg_count == 0:
        raise ValueError(
            "Ground truth labels must contain both positive (1) and negative (0) pairs."
        )

    best_j = -1.0
    best_thresh_j = 0
    best_f1 = -1.0
    best_thresh_f1 = 0

    for tau in thresh_list:
        # Prediction: True (1) if distance <= tau; else False (0)
        preds = (dist_arr <= tau).astype(np.int32)

        tp = int(np.sum((preds == 1) & (gt_arr == 1)))
        fp = int(np.sum((preds == 1) & (gt_arr == 0)))
        tn = int(np.sum((preds == 0) & (gt_arr == 0)))
        fn = int(np.sum((preds == 0) & (gt_arr == 1)))

        acc = (tp + tn) / (tp + tn + fp + fn + 1e-7)
        recall = tp / (tp + fn + 1e-7)  # Sensitivity / TPR
        specificity = tn / (tn + fp + 1e-7)  # TNR
        fpr = fp / (fp + tn + 1e-7)  # 1 - Specificity
        precision = tp / (tp + fp + 1e-7)  # PPV
        f1 = 2.0 * precision * recall / (precision + recall + 1e-7)

        tpr_list.append(recall)
        fpr_list.append(fpr)
        precision_list.append(precision)
        recall_list.append(recall)
        specificity_list.append(specificity)
        accuracy_list.append(acc)
        f1_list.append(f1)

        # Youden's J statistic = TPR - FPR
        j_stat = recall - fpr
        if j_stat > best_j:
            best_j = j_stat
            best_thresh_j = int(tau)

        if f1 > best_f1:
            best_f1 = f1
            best_thresh_f1 = int(tau)

    # Compute AUC using trapezoidal rule
    # Sort by FPR for proper integration
    sorted_indices = np.argsort(fpr_list)
    fpr_sorted = np.array(fpr_list)[sorted_indices]
    tpr_sorted = np.array(tpr_list)[sorted_indices]

    # Ensure endpoints [0, 0] and [1, 1] are anchored
    fpr_anchored = np.concatenate([[0.0], fpr_sorted, [1.0]])
    tpr_anchored = np.concatenate([[0.0], tpr_sorted, [1.0]])
    auc_val = float(np.trapezoid(tpr_anchored, fpr_anchored))

    # Metrics at optimal threshold
    opt_idx = int(np.where(thresh_list == best_thresh_f1)[0][0])

    return {
        "thresholds": thresh_list.tolist(),
        "fpr": fpr_list,
        "tpr": tpr_list,
        "auc": float(np.clip(auc_val, 0.0, 1.0)),
        "precision": precision_list,
        "recall": recall_list,
        "specificity": specificity_list,
        "accuracy": accuracy_list,
        "f1": f1_list,
        "optimal_threshold_f1": best_thresh_f1,
        "optimal_threshold_youden": best_thresh_j,
        "optimal_metrics": {
            "threshold": best_thresh_f1,
            "accuracy": accuracy_list[opt_idx],
            "recall": recall_list[opt_idx],
            "specificity": specificity_list[opt_idx],
            "precision": precision_list[opt_idx],
            "f1": f1_list[opt_idx],
        },
    }


# ==============================================================================
# 5. Image Distortion Suite for Robustness Analysis
# ==============================================================================


def apply_photometric_distortion(
    image: np.ndarray,
    distortion_type: str,
    param: float | int,
) -> np.ndarray:
    """Apply controlled photometric degradation to an image.

    Args:
        image: Input RGB or Grayscale image array uint8.
        distortion_type: Type of distortion ('brightness', 'contrast', 'blur', 'jpeg', 'gaussian_noise', 'salt_pepper').
        param: Distortion magnitude parameter.

    Returns:
        np.ndarray: Distorted image of the same shape and uint8 dtype.
    """
    img_float = image.astype(np.float32)

    if distortion_type == "brightness":
        # param: multiplier, e.g., 0.7 or 1.3
        distorted = np.clip(img_float * float(param), 0, 255).astype(np.uint8)
        return distorted

    if distortion_type == "contrast":
        # param: contrast multiplier, e.g., 0.6 or 1.4
        mean_val = np.mean(img_float)
        distorted = np.clip(
            (img_float - mean_val) * float(param) + mean_val, 0, 255
        ).astype(np.uint8)
        return distorted

    if distortion_type == "blur":
        # param: gaussian sigma, e.g., 1.5 or 3.0
        sigma = float(param)
        k = int(np.ceil(3.0 * sigma)) * 2 + 1
        return cv2.GaussianBlur(image, (k, k), sigmaX=sigma, sigmaY=sigma)

    if distortion_type == "jpeg":
        # param: quality factor 0 - 100
        quality = int(np.clip(param, 1, 100))
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encimg = cv2.imencode(".jpg", image, encode_param)
        return cv2.imdecode(encimg, cv2.IMREAD_UNCHANGED)

    if distortion_type == "gaussian_noise":
        # param: noise sigma in pixel units
        sigma = float(param)
        noise = np.random.normal(0, sigma, image.shape)
        return np.clip(img_float + noise, 0, 255).astype(np.uint8)

    if distortion_type == "salt_pepper":
        # param: noise ratio, e.g., 0.03
        ratio = float(param)
        distorted = image.copy()
        mask = np.random.rand(*image.shape[:2])
        distorted[mask < ratio / 2.0] = 0
        distorted[mask > 1.0 - (ratio / 2.0)] = 255
        return distorted

    raise ValueError(f"Unknown photometric distortion: {distortion_type}")


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
        # param: True for horizontal flip
        return cv2.flip(image, 1)

    raise ValueError(f"Unknown geometric distortion: {distortion_type}")


# ==============================================================================
# 6. Top-K Similar Image Retrieval
# ==============================================================================


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
