import pytest

from app.schemas.detection import DetectionResult
from app.services.detection.detection_service import DetectionService


class MockDetector:

    def __init__(self,name="MockDetector"):
        self.name=name

    def detect(self,file_path):
        return DetectionResult(
            detector=self.name,
            media_type="image",
            manipulation_detected=True,
            confidence=0.95,
            manipulation_type="face"
        )


class InvalidDetector:

    def detect(self,file_path):
        return {
            "detector":"InvalidDetector"
        }


def test_detection_service():

    service=DetectionService(
        detectors=[MockDetector()]
    )

    result=service.analyze(
        "test.jpg"
    )

    assert "detections" in result
    assert len(result["detections"])==1

    detection=result["detections"][0]

    assert isinstance(
        detection,
        DetectionResult
    )

    assert detection.detector=="MockDetector"
    assert detection.confidence==0.95


def test_detection_service_without_detectors():

    service=DetectionService()

    result=service.analyze(
        "test.jpg"
    )

    assert result["detections"]==[]


def test_detection_service_invalid_result():

    service=DetectionService(
        detectors=[InvalidDetector()]
    )

    with pytest.raises(TypeError):
        service.analyze(
            "test.jpg"
        )


def test_detection_service_multiple_detectors():

    service=DetectionService(
        detectors=[
            MockDetector("Detector1"),
            MockDetector("Detector2")
        ]
    )

    result=service.analyze(
        "test.jpg"
    )

    assert len(result["detections"])==2

    assert result["detections"][0].detector=="Detector1"
    assert result["detections"][1].detector=="Detector2"