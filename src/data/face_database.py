"""Face database shim.

Backward-compatibility shim re-exporting from src.labs.lab05_facenet.face_database.
"""

from src.labs.lab05_facenet.face_database import FaceDatabase, l2_normalize_numpy

__all__ = ["FaceDatabase", "l2_normalize_numpy"]
