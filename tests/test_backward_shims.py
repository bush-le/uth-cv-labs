"""Tests for backward compatibility shims."""



def test_legacy_transforms_and_filters_shims():
    from src.data.filters import adjust_brightness, apply_sobel, get_mean_kernel
    from src.data.transforms import bgr_to_grayscale, crop_roi, load_image_cv

    assert callable(bgr_to_grayscale)
    assert callable(crop_roi)
    assert callable(load_image_cv)
    assert callable(adjust_brightness)
    assert callable(apply_sobel)
    assert callable(get_mean_kernel)


def test_legacy_models_shims():
    from src.models.classical.canny import custom_canny_pipeline, run_opencv_canny
    from src.models.classical.wavelet_hash import hamming_distance, wavelet_hash
    from src.models.deep.facenet_extractor import FaceNetExtractor
    from src.models.deep.mtcnn_detector import MTCNNDetector
    from src.models.similarity_matcher import SimilarityMatcher

    assert callable(custom_canny_pipeline)
    assert callable(run_opencv_canny)
    assert callable(wavelet_hash)
    assert callable(hamming_distance)
    assert MTCNNDetector is not None
    assert FaceNetExtractor is not None
    assert SimilarityMatcher is not None


def test_legacy_pipelines_shims():
    from src.pipelines.lab01_pipeline import run_lab01
    from src.pipelines.lab02_pipeline import run_lab02
    from src.pipelines.lab03_pipeline import Lab03Pipeline
    from src.pipelines.lab04_pipeline import Lab04Pipeline
    from src.pipelines.lab05_pipeline import run_lab05_pipeline

    assert callable(run_lab01)
    assert callable(run_lab02)
    assert Lab03Pipeline is not None
    assert Lab04Pipeline is not None
    assert callable(run_lab05_pipeline)
