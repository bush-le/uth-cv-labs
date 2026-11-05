"""Face registration pipeline shim.

Backward-compatibility shim re-exporting from src.labs.lab05_facenet.face_registration.
"""

from src.labs.lab05_facenet.face_registration import (
    FaceRegistrationPipeline,
    RegistrationSummary,
)

__all__ = ["FaceRegistrationPipeline", "RegistrationSummary"]
