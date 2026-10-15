import pytest
from PIL import Image
from app.services.detection.detectors.object_detector import (
    BaselineObjectDetector,
    YOLOv8ObjectDetector,
    BaseObjectDetector,
)
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

def test_yolov8_detector_missing_file():
    detector = YOLOv8ObjectDetector()
    with pytest.raises(FileNotFoundError):
        detector.detect("missing_yolo.jpg")

def test_yolov8_detector_execution(tmp_path):
    detector = YOLOv8ObjectDetector()
    assert isinstance(detector, BaseObjectDetector)

    img_file = tmp_path / "yolo_sample.png"
    img = Image.new("RGB", (128, 128), color="white")
    img.save(str(img_file))

    result = detector.detect(str(img_file))
    assert isinstance(result, DetectionResult)
    assert result.media_type == "image"
    assert result.manipulation_type == "object"
