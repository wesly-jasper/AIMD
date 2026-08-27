import pytest

from app.schemas.detection import DetectionResult

from app.services.detection.engine.image_detection_engine import (
    ImageDetectionEngine
)


class MockDetector:

    def detect(self,file_path):
        return DetectionResult(
            detector="MockDetector",
            media_type="image",
            manipulation_detected=True,
            confidence=0.95,
            manipulation_type="face"
        )


class InvalidDetector:

    def detect(self,file_path):
        return {
            "detector":"InvalidDetector",
            "manipulation_detected":True
        }


def test_engine_without_detector(tmp_path):

    image=tmp_path/"test.jpg"
    image.write_bytes(b"test")

    engine=ImageDetectionEngine()

    with pytest.raises(RuntimeError):
        engine.detect(image)


def test_engine_file_not_found():

    engine=ImageDetectionEngine(
        detectors=[MockDetector()]
    )

    with pytest.raises(FileNotFoundError):
        engine.detect(
            "missing.jpg"
        )


def test_engine_calls_detector(tmp_path):

    image=tmp_path/"test.jpg"
    image.write_bytes(b"test")

    engine=ImageDetectionEngine(
        detectors=[MockDetector()]
    )

    result=engine.detect(image)

    assert isinstance(result, dict)
    assert result["media_type"] == "image"
    assert len(result["detections"]) == 1
    
    det = result["detections"][0]

    assert isinstance(
        det,
        DetectionResult
    )

    assert det.detector=="MockDetector"
    assert det.manipulation_detected is True
    assert det.confidence==0.95


def test_engine_invalid_detector_result(tmp_path):

    image=tmp_path/"test.jpg"
    image.write_bytes(b"test")

    engine=ImageDetectionEngine(
        detectors=[InvalidDetector()]
    )

    with pytest.raises(TypeError):
        engine.detect(image)