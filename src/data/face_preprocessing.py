"""Face preprocessing shim.

Backward-compatibility shim re-exporting from src.labs.lab05_facenet.face_preprocessing.
"""

from src.labs.lab05_facenet.face_preprocessing import (
    crop_face_with_margin,
    fixed_image_standardization,
    preprocess_face,
    validate_and_clip_bbox,
)

__all__ = [
    "validate_and_clip_bbox",
    "fixed_image_standardization",
    "preprocess_face",
    "crop_face_with_margin",
]
