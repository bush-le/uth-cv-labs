"""Point operators and spatial filtering module for Lab 02.

Provides standardized functions for:
1. Point operations: brightness, contrast, negative, thresholding.
2. Linear filters: mean, Gaussian, sharpening via convolution kernels.
3. Edge detection: Sobel, Prewitt, Laplacian (first and second derivatives).
4. Non-linear filters: median and bilateral.
"""

from pathlib import Path

import cv2
import numpy as np

from src.common.logger import get_logger

logger = get_logger(__name__)


# ==============================================================================
# 1. Point Operations (Yêu cầu 1-4)
# ==============================================================================


def adjust_brightness(image: np.ndarray, beta: float) -> np.ndarray:
    """Adjust image brightness by adding a constant to every pixel.

    Formula: g(x, y) = f(x, y) + beta, with saturation clipping to [0, 255].

    Args:
        image: Input image array of shape (H, W, 3) or (H, W), dtype uint8.
        beta: Brightness offset. Positive values brighten, negative darken.

    Returns:
        np.ndarray: Brightness-adjusted image, same shape and dtype uint8.
    """
    _validate_image(image)
    adjusted = image.astype(np.float32) + beta
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def adjust_contrast(image: np.ndarray, alpha: float) -> np.ndarray:
    """Adjust image contrast by multiplying every pixel with a constant.

    Formula: g(x, y) = alpha * f(x, y), with saturation clipping to [0, 255].

    Args:
        image: Input image array of shape (H, W, 3) or (H, W), dtype uint8.
        alpha: Contrast gain. alpha > 1 increases contrast,
            0 < alpha < 1 washes the image out.

    Returns:
        np.ndarray: Contrast-adjusted image, same shape and dtype uint8.

    Raises:
        ValueError: If alpha is non-positive.
    """
    if alpha <= 0:
        raise ValueError(f"Contrast gain alpha must be positive, got {alpha}")
    _validate_image(image)
    adjusted = image.astype(np.float32) * alpha
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def negative_transform(image: np.ndarray) -> np.ndarray:
    """Compute the photographic negative of an 8-bit image.

    Formula: g(x, y) = 255 - f(x, y).

    Args:
        image: Input image array of shape (H, W, 3) or (H, W), dtype uint8.

    Returns:
        np.ndarray: Negative image, same shape and dtype uint8.
    """
    _validate_image(image)
    return (255 - image).astype(np.uint8)


def apply_threshold(
    image: np.ndarray, threshold: int, max_value: int = 255
) -> np.ndarray:
    """Binarize an image with a fixed global threshold.

    Formula: g(x, y) = max_value if f(x, y) > threshold, else 0.

    Args:
        image: Input image array of shape (H, W, 3) or (H, W), dtype uint8.
        threshold: Cutoff intensity in [0, 255].
        max_value: Value assigned to foreground pixels (default 255).

    Returns:
        np.ndarray: Binary image, same shape and dtype uint8.

    Raises:
        ValueError: If threshold is outside [0, 255].
    """
    if not (0 <= threshold <= 255):
        raise ValueError(f"Threshold must be in [0, 255], got {threshold}")
    _validate_image(image)
    _, binary = cv2.threshold(image, threshold, max_value, cv2.THRESH_BINARY)
    return binary


# ==============================================================================
# 2. Linear Filters (Yêu cầu 5-7)
# ==============================================================================


def get_mean_kernel(ksize: int = 3) -> np.ndarray:
    """Build a normalized box (mean) kernel.

    All coefficients are positive; the kernel sum equals 1, preserving
    the average brightness (DC response gain = 1).

    Args:
        ksize: Odd kernel size (e.g., 3, 5, 7).

    Returns:
        np.ndarray: Normalized mean kernel of shape (ksize, ksize), float32.

    Raises:
        ValueError: If ksize is not a positive odd integer.
    """
    _validate_kernel_size(ksize)
    return np.full((ksize, ksize), 1.0 / (ksize * ksize), dtype=np.float32)


def get_gaussian_kernel(ksize: int = 3, sigma: float = 1.0) -> np.ndarray:
    """Build a normalized 2D Gaussian kernel.

    Coefficients decrease with distance from the center; the kernel sum
    equals 1, preserving the average brightness.

    Args:
        ksize: Odd kernel size (e.g., 3, 5, 7).
        sigma: Standard deviation of the Gaussian distribution. Must be > 0.

    Returns:
        np.ndarray: Normalized Gaussian kernel of shape (ksize, ksize).

    Raises:
        ValueError: If ksize is not a positive odd integer or sigma <= 0.
    """
    _validate_kernel_size(ksize)
    if sigma <= 0:
        raise ValueError(f"Sigma must be positive, got {sigma}")
    kernel_1d = cv2.getGaussianKernel(ksize, sigma)
    return (kernel_1d @ kernel_1d.T).astype(np.float32)


def get_sharpen_kernel() -> np.ndarray:
    """Build the classic 3x3 sharpening kernel (Identity - Laplacian).

    Because Identity sums to 1 and Laplacian sums to 0, the resulting
    kernel sums to 1. The large positive center (+5) with negative
    neighbors (-1) amplifies center-surround differences.

    Returns:
        np.ndarray: Sharpen kernel of shape (3, 3), float32.
    """
    return np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)


def get_laplacian_kernel() -> np.ndarray:
    """Build the 4-neighbor Laplacian kernel (second derivative).

    The kernel measures curvature (rate of slope change) and sums to 0,
    so flat regions map to zero. This is an EDGE DETECTION kernel,
    not a sharpening kernel.

    Returns:
        np.ndarray: Laplacian kernel of shape (3, 3), float32.
    """
    return np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)


def apply_custom_filter(
    image: np.ndarray, kernel: np.ndarray, ddepth: int = -1
) -> np.ndarray:
    """Apply a custom convolution kernel with cv2.filter2D.

    Args:
        image: Input image array of shape (H, W, 3) or (H, W), dtype uint8.
        kernel: Convolution kernel of shape (kH, kW), float32.
        ddepth: Output depth flag for cv2.filter2D (default -1 = same as input).

    Returns:
        np.ndarray: Filtered image, same shape as input.

    Raises:
        ValueError: If kernel is not a 2D array.
    """
    _validate_image(image)
    if kernel.ndim != 2:
        raise ValueError(f"Kernel must be 2D, got shape {kernel.shape}")
    return cv2.filter2D(image, ddepth, kernel)


def apply_mean_filter(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Blur an image with a normalized mean (box) kernel.

    Args:
        image: Input image array of shape (H, W, 3) or (H, W), dtype uint8.
        ksize: Odd kernel size.

    Returns:
        np.ndarray: Blurred image, same shape as input.
    """
    return apply_custom_filter(image, get_mean_kernel(ksize))


def apply_gaussian_filter(
    image: np.ndarray, ksize: int = 3, sigma: float = 1.0
) -> np.ndarray:
    """Blur an image with a Gaussian kernel (weighted averaging).

    Args:
        image: Input image array of shape (H, W, 3) or (H, W), dtype uint8.
        ksize: Odd kernel size.
        sigma: Gaussian standard deviation (> 0).

    Returns:
        np.ndarray: Blurred image, same shape as input.
    """
    return apply_custom_filter(image, get_gaussian_kernel(ksize, sigma))


def apply_sharpen_filter(image: np.ndarray) -> np.ndarray:
    """Sharpen an image with the classic 3x3 kernel (Identity - Laplacian).

    Args:
        image: Input image array of shape (H, W, 3) or (H, W), dtype uint8.

    Returns:
        np.ndarray: Sharpened image, same shape as input.
    """
    return apply_custom_filter(image, get_sharpen_kernel())


# ==============================================================================
# 3. Edge Detection (Yêu cầu 8)
# ==============================================================================


def get_sobel_kernels() -> tuple[np.ndarray, np.ndarray]:
    """Return the Sobel kernels for x and y directions.

    Both kernels are first-derivative operators (kernel sum = 0).
    The middle row/column is emphasized (weight 2) for noise robustness.

    Returns:
        tuple[np.ndarray, np.ndarray]: (sobel_x, sobel_y) kernels.
    """
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    return sobel_x, sobel_y


def get_prewitt_kernels() -> tuple[np.ndarray, np.ndarray]:
    """Return the Prewitt kernels for x and y directions.

    Both kernels are first-derivative operators (kernel sum = 0).
    Unlike Sobel, coefficients are uniform (no emphasis on the middle).

    Returns:
        tuple[np.ndarray, np.ndarray]: (prewitt_x, prewitt_y) kernels.
    """
    prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    return prewitt_x, prewitt_y


def apply_sobel(
    image: np.ndarray, direction: str = "x", ksize: int = 3
) -> np.ndarray:
    """Compute the first derivative with a Sobel operator.

    Args:
        image: Input image array of shape (H, W, 3) or (H, W), dtype uint8.
        direction: Gradient direction, "x" (vertical edges) or "y"
            (horizontal edges).
        ksize: Sobel kernel size (1, 3, 5, or 7).

    Returns:
        np.ndarray: Gradient image rescaled to uint8 via convertScaleAbs.

    Raises:
        ValueError: If direction is not "x" or "y".
    """
    _validate_image(image)
    if direction not in {"x", "y"}:
        raise ValueError(f"Direction must be 'x' or 'y', got {direction!r}")
    dx, dy = (1, 0) if direction == "x" else (0, 1)
    gradient = cv2.Sobel(image, cv2.CV_64F, dx, dy, ksize=ksize)
    return cv2.convertScaleAbs(gradient)


def apply_prewitt(image: np.ndarray, direction: str = "x") -> np.ndarray:
    """Compute the first derivative with a Prewitt kernel via filter2D.

    Args:
        image: Input image array of shape (H, W, 3) or (H, W), dtype uint8.
        direction: Gradient direction, "x" (vertical edges) or "y"
            (horizontal edges).

    Returns:
        np.ndarray: Gradient image rescaled to uint8 via convertScaleAbs.

    Raises:
        ValueError: If direction is not "x" or "y".
    """
    _validate_image(image)
    prewitt_x, prewitt_y = get_prewitt_kernels()
    kernel = prewitt_x if direction == "x" else prewitt_y
    gradient = cv2.filter2D(image.astype(np.float32), cv2.CV_32F, kernel)
    return cv2.convertScaleAbs(gradient)


def apply_laplacian(image: np.ndarray) -> np.ndarray:
    """Compute the second derivative with a Laplacian operator.

    Returns an edge-detection response (kernel sum = 0), not a sharpened
    image. Flat regions map to zero, boundaries produce positive and
    negative lobes with a zero-crossing at the edge.

    Args:
        image: Input image array of shape (H, W, 3) or (H, W), dtype uint8.

    Returns:
        np.ndarray: Laplacian response rescaled to uint8 via convertScaleAbs.
    """
    _validate_image(image)
    response = cv2.Laplacian(image, cv2.CV_64F)
    return cv2.convertScaleAbs(response)


def gradient_magnitude(grad_x: np.ndarray, grad_y: np.ndarray) -> np.ndarray:
    """Combine x and y gradients into a magnitude map: G = sqrt(Gx^2 + Gy^2).

    Args:
        grad_x: Gradient response along x, shape (H, W, C) or (H, W).
        grad_y: Gradient response along y, same shape as grad_x.

    Returns:
        np.ndarray: Magnitude map rescaled to uint8 via convertScaleAbs.

    Raises:
        ValueError: If grad_x and grad_y shapes do not match.
    """
    if grad_x.shape != grad_y.shape:
        raise ValueError(
            f"Gradient shapes must match, got {grad_x.shape} and {grad_y.shape}"
        )
    magnitude = np.sqrt(
        grad_x.astype(np.float64) ** 2 + grad_y.astype(np.float64) ** 2
    )
    return cv2.convertScaleAbs(magnitude)


# ==============================================================================
# 4. Non-linear Filters (Yêu cầu 10)
# ==============================================================================


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


def kernel_sum(kernel: np.ndarray) -> float:
    """Return the sum of kernel coefficients (DC gain indicator).

    A sum of 1 preserves average brightness; a sum of 0 removes flat
    regions and keeps only boundaries (derivative behavior).

    Args:
        kernel: Convolution kernel of shape (kH, kW).

    Returns:
        float: Sum of all kernel coefficients.
    """
    return float(np.sum(kernel))


# ==============================================================================
# 5. Internal Validators
# ==============================================================================


def _validate_image(image: np.ndarray) -> None:
    """Validate that the input is a non-empty uint8 image array.

    Args:
        image: Candidate image array.

    Raises:
        ValueError: If the array is invalid (wrong ndim, empty, or dtype).
    """
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Input must be a non-empty numpy array.")
    if image.ndim not in {2, 3}:
        raise ValueError(f"Expected 2D or 3D image array, got shape {image.shape}")
    if image.ndim == 3 and image.shape[2] not in {1, 3, 4}:
        raise ValueError(f"Expected 1, 3 or 4 channels, got shape {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"Expected dtype uint8, got {image.dtype}")


def _validate_kernel_size(ksize: int) -> None:
    """Validate that a kernel size is a positive odd integer.

    Args:
        ksize: Candidate kernel size.

    Raises:
        ValueError: If ksize is not a positive odd integer.
    """
    if not isinstance(ksize, int) or ksize <= 0 or ksize % 2 == 0:
        raise ValueError(f"Kernel size must be a positive odd integer, got {ksize}")


def find_project_root(start: Path | None = None) -> Path:
    """Locate the repository root by searching for pyproject.toml upward.

    Args:
        start: Starting directory (defaults to this file's location).

    Returns:
        Path: Repository root directory.
    """
    current = Path(start or __file__).resolve()
    if current.is_file():
        current = current.parent
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return current
