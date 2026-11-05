"""Module tiền xử lý khuôn mặt: kiểm tra tọa độ BBox, cắt ảnh, chuẩn hóa kích thước và Tensor."""

from __future__ import annotations

import cv2
import numpy as np
import torch


def validate_and_clip_bbox(
    bbox: tuple[float, float, float, float] | list[float] | np.ndarray,
    frame_width: int,
    frame_height: int,
    margin: int = 0,
) -> tuple[int, int, int, int] | None:
    """Kiểm tra tính hợp lệ của bounding box, cộng lề (margin) và kẹp trong biên ảnh.

    Args:
        bbox: Tọa độ (x1, y1, x2, y2) từ MTCNN.
        frame_width: Chiều rộng ảnh gốc.
        frame_height: Chiều cao ảnh gốc.
        margin: Số pixel mở rộng thêm quanh vùng mặt.

    Returns:
        Tuple (x1, y1, x2, y2) kiểu int nếu hợp lệ, hoặc None nếu suy biến/nằm ngoài ảnh.
    """
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("Kích thước khung hình phải lớn hơn 0.")
    if len(bbox) != 4:
        return None

    x1_f, y1_f, x2_f, y2_f = [float(v) for v in bbox]
    if not np.isfinite([x1_f, y1_f, x2_f, y2_f]).all():
        return None

    half_margin = max(0, int(margin)) // 2
    x1 = max(0, int(np.floor(x1_f)) - half_margin)
    y1 = max(0, int(np.floor(y1_f)) - half_margin)
    x2 = min(frame_width, int(np.ceil(x2_f)) + half_margin)
    y2 = min(frame_height, int(np.ceil(y2_f)) + half_margin)

    if x2 <= x1 or y2 <= y1:
        return None
    if (x2 - x1) < 10 or (y2 - y1) < 10:
        return None

    return (x1, y1, x2, y2)


def fixed_image_standardization(image_tensor: torch.Tensor) -> torch.Tensor:
    """Chuẩn hóa giá trị pixel từ dải [0, 255] sang [-1, 1] theo chuẩn FaceNet pretrained.

    Công thức: (x - 127.5) / 128.0
    """
    return (image_tensor - 127.5) / 128.0


def crop_face_with_margin(
    frame_bgr: np.ndarray,
    bbox: tuple[float, float, float, float],
    margin: int | float = 0,
) -> np.ndarray:
    """Crop face region from BGR image with an optional margin."""
    h, w = frame_bgr.shape[:2]
    int_margin = int(margin * min(w, h)) if isinstance(margin, float) and margin < 1.0 else int(margin)
    clipped = validate_and_clip_bbox(bbox, frame_width=w, frame_height=h, margin=int_margin)
    if clipped is None:
        raise ValueError(f"Invalid bounding box: {bbox}")
    x1, y1, x2, y2 = clipped
    return frame_bgr[y1:y2, x1:x2].copy()


def preprocess_face(
    frame_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
    image_size: int = 160,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Cắt vùng khuôn mặt từ ảnh BGR, chuyển sang RGB, resize và tạo Tensor chuẩn hóa cho FaceNet.

    Args:
        frame_bgr: Ảnh đầu vào định dạng BGR (H, W, 3) kiểu uint8 từ OpenCV.
        bbox: Tọa độ (x1, y1, x2, y2) hợp lệ.
        image_size: Kích thước đầu vào của FaceNet (mặc định 160x160).
        device: Thiết bị lưu trữ tensor ('cpu' hoặc 'cuda').

    Returns:
        Tensor có kích thước (1, 3, image_size, image_size), kiểu float32, đã chuẩn hóa.
    """
    if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("Ảnh đầu vào phải là ảnh màu 3 kênh (H, W, 3).")

    h, w = frame_bgr.shape[:2]
    clipped = validate_and_clip_bbox(bbox, frame_width=w, frame_height=h, margin=0)
    if clipped is None:
        raise ValueError(f"Bounding box không hợp lệ: {bbox} trên khung hình ({w}x{h}).")

    x1, y1, x2, y2 = clipped
    face_crop_bgr = frame_bgr[y1:y2, x1:x2]
    if face_crop_bgr.size == 0:
        raise ValueError("Vùng ảnh khuôn mặt sau khi cắt bị rỗng.")

    # Chuyển hệ màu BGR (OpenCV) sang RGB (PyTorch/FaceNet)
    face_crop_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)

    # Chuẩn hóa kích thước về 160x160
    face_resized = cv2.resize(
        face_crop_rgb,
        (image_size, image_size),
        interpolation=cv2.INTER_AREA,
    )

    # Chuyển từ (H, W, C) sang (1, C, H, W) kiểu float32
    tensor = torch.from_numpy(face_resized).permute(2, 0, 1).unsqueeze(0).float()
    tensor = fixed_image_standardization(tensor)
    return tensor.to(device)