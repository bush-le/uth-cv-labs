"""Lab 05 FaceNet & MTCNN Package: Face Detection, Embedding, Similarity, Database, and Webcam Recognition."""

from src.labs.lab05_facenet import (
    face_database,
    face_preprocessing,
    face_registration,
    facenet_extractor,
    mtcnn_detector,
    similarity_matcher,
    visuals,
    webcam_recognition,
)
from src.labs.lab05_facenet.face_database import FaceDatabase, l2_normalize_numpy
from src.labs.lab05_facenet.face_preprocessing import (
    crop_face_with_margin,
    fixed_image_standardization,
    preprocess_face,
    validate_and_clip_bbox,
)
from src.labs.lab05_facenet.face_registration import (
    FaceRegistrationPipeline,
    RegistrationSummary,
)
from src.labs.lab05_facenet.facenet_extractor import FaceNetExtractor
from src.labs.lab05_facenet.mtcnn_detector import MTCNNDetector
from src.labs.lab05_facenet.similarity_matcher import SimilarityMatcher
from src.labs.lab05_facenet.visuals import (
    FaceVisualizer,
    FPSMeter,
    RecognitionResult,
    plot_detection_result,
    plot_similarity_matrix,
)
from src.labs.lab05_facenet.webcam_recognition import (
    RealtimeRecognitionPipeline,
    WebcamManager,
)

__all__ = [
    "mtcnn_detector",
    "facenet_extractor",
    "similarity_matcher",
    "face_database",
    "face_preprocessing",
    "face_registration",
    "webcam_recognition",
    "visuals",
    "MTCNNDetector",
    "FaceNetExtractor",
    "SimilarityMatcher",
    "FaceDatabase",
    "l2_normalize_numpy",
    "crop_face_with_margin",
    "fixed_image_standardization",
    "preprocess_face",
    "validate_and_clip_bbox",
    "FaceRegistrationPipeline",
    "RegistrationSummary",
    "WebcamManager",
    "RealtimeRecognitionPipeline",
    "FaceVisualizer",
    "FPSMeter",
    "RecognitionResult",
    "plot_detection_result",
    "plot_similarity_matrix",
]
