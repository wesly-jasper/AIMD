import pytest
from PIL import Image
import numpy as np
from app.services.detection.detectors.face_detector import (
    BaselineFaceDetector,
    DeepLearningFaceDetector,
    BaseFaceDetector,
)
from app.schemas.detection import DetectionResult

def test_face_detector_missing_file():
    detector = BaselineFaceDetector()
    with pytest.raises(FileNotFoundError):
        detector.detect("non_existent_file.jpg")

def test_face_detector_success(tmp_path):
    detector = BaselineFaceDetector()
    assert isinstance(detector, BaseFaceDetector)
    
    img_file = tmp_path / "sample.jpg"
    img = Image.new("RGB", (64, 64), color="red")
    img.save(str(img_file))
    
    result = detector.detect(str(img_file))
    assert isinstance(result, DetectionResult)
    assert result.detector == "BaselineFaceDetector"
    assert result.media_type == "image"
    assert result.manipulation_type == "face"
    assert "status" in result.metadata

def test_deep_learning_face_detector_missing_file():
    detector = DeepLearningFaceDetector()
    with pytest.raises(FileNotFoundError):
        detector.detect("missing_deepfake_file.jpg")

def test_deep_learning_face_detector_execution(tmp_path):
    detector = DeepLearningFaceDetector()
    assert isinstance(detector, BaseFaceDetector)

    img_file = tmp_path / "face_sample.jpg"
    img = Image.new("RGB", (100, 100), color="blue")
    img.save(str(img_file))

    result = detector.detect(str(img_file))
    assert isinstance(result, DetectionResult)
    assert result.media_type == "image"
    assert result.manipulation_type == "face"
    assert result.confidence >= 0.0
