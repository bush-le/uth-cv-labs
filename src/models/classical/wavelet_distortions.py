"""Photometric and geometric distortions for perceptual hash robustness evaluation."""

from __future__ import annotations

import cv2
import numpy as np


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
        distorted = np.clip(img_float * float(param), 0, 255).astype(np.uint8)
        return distorted

    if distortion_type == "contrast":
        mean_val = np.mean(img_float)
        distorted = np.clip(
            (img_float - mean_val) * float(param) + mean_val, 0, 255
        ).astype(np.uint8)
        return distorted

    if distortion_type == "blur":
        sigma = float(param)
        k = int(np.ceil(3.0 * sigma)) * 2 + 1
        return cv2.GaussianBlur(image, (k, k), sigmaX=sigma, sigmaY=sigma)

    if distortion_type == "jpeg":
        quality = int(np.clip(param, 1, 100))
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encimg = cv2.imencode(".jpg", image, encode_param)
        return cv2.imdecode(encimg, cv2.IMREAD_UNCHANGED)

    if distortion_type == "gaussian_noise":
        sigma = float(param)
        noise = np.random.normal(0, sigma, image.shape)
        return np.clip(img_float + noise, 0, 255).astype(np.uint8)

    if distortion_type == "salt_pepper":
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
        factor = float(param)
        new_w, new_h = max(8, int(w * factor)), max(8, int(h * factor))
        scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(scaled, (w, h), interpolation=cv2.INTER_LINEAR)

    if distortion_type == "crop":
        ratio = float(param)
        cw, ch = int(w * ratio), int(h * ratio)
        x1 = (w - cw) // 2
        y1 = (h - ch) // 2
        cropped = image[y1 : y1 + ch, x1 : x1 + cw]
        return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    if distortion_type == "rotate":
        angle = float(param)
        matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)

    if distortion_type == "flip":
        return cv2.flip(image, 1)

    raise ValueError(f"Unknown geometric distortion: {distortion_type}")
