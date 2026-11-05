"""MTCNN face detector shim.

Backward-compatibility shim re-exporting from src.labs.lab05_facenet.mtcnn_detector.
"""

from src.labs.lab05_facenet.mtcnn_detector import DetectedFace, MTCNNDetector

__all__ = ["DetectedFace", "MTCNNDetector"]
