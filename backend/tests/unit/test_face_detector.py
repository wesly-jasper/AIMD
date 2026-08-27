import pytest
from app.services.detection.detectors.face_detector import BaselineFaceDetector, BaseFaceDetector
from app.schemas.detection import DetectionResult

def test_face_detector_missing_file():
    detector = BaselineFaceDetector()
    with pytest.raises(FileNotFoundError):
        detector.detect("non_existent_file.jpg")

def test_face_detector_success(tmp_path):
    detector = BaselineFaceDetector()
    assert isinstance(detector, BaseFaceDetector)
    
    img_file = tmp_path / "sample.jpg"
    img_file.write_bytes(b"dummy image bytes")
    
    result = detector.detect(str(img_file))
    assert isinstance(result, DetectionResult)
    assert result.detector == "BaselineFaceDetector"
    assert result.media_type == "image"
    assert result.manipulation_type == "face"
    assert "status" in result.metadata
