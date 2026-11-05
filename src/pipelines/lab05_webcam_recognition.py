"""Webcam recognition pipeline shim.

Backward-compatibility shim re-exporting from src.labs.lab05_facenet.webcam_recognition.
"""

from src.labs.lab05_facenet.webcam_recognition import (
    RealtimeRecognitionPipeline,
    WebcamManager,
)

__all__ = ["RealtimeRecognitionPipeline", "WebcamManager"]
