import pytest
from app.services.detection.detectors.object_detector import BaselineObjectDetector, BaseObjectDetector
from app.schemas.detection import DetectionResult

def test_object_detector_missing_file():
    detector = BaselineObjectDetector()
    with pytest.raises(FileNotFoundError):
        detector.detect("non_existent_file.jpg")

def test_object_detector_success(tmp_path):
    detector = BaselineObjectDetector()
    assert isinstance(detector, BaseObjectDetector)
    
    img_file = tmp_path / "sample.jpg"
    img_file.write_bytes(b"dummy image bytes")
    
    result = detector.detect(str(img_file))
    assert isinstance(result, DetectionResult)
    assert result.detector == "BaselineObjectDetector"
    assert result.media_type == "image"
    assert result.manipulation_type == "object"
    assert result.metadata.get("status") == "baseline"
