from app.schemas.detection import DetectionResult

from app.services.detection.detection_service import (
    DetectionService
)

from app.services.detection.engine.image_detection_engine import (
    ImageDetectionEngine
)


class MockImageDetector:

    def detect(self,file_path):

        return DetectionResult(
            detector="MockImageDetector",
            media_type="image",
            manipulation_detected=True,
            confidence=0.91,
            manipulation_type="image",
            regions=[],
            metadata={
                "score":0.91
            }
        )


def test_image_detection_service_flow(tmp_path):

    image=tmp_path/"test.jpg"
    image.write_bytes(b"test image")

    image_engine=ImageDetectionEngine(
        detectors=[MockImageDetector()]
    )

    result=image_engine.detect(image)

    service=DetectionService(
        detectors=[
            MockImageDetector()
        ]
    )

    service_result=service.analyze(image)

    # In integration it should aggregate results.
    # The image engine now returns {"media_type": "image", "detections": [DetectionResult, ...]}
    assert result["media_type"] == "image"
    assert len(result["detections"]) == 1
    
    det = result["detections"][0]

    assert det.detector=="MockImageDetector"
    assert det.manipulation_detected is True
    assert det.confidence==0.91

    assert "detections" in service_result
    assert len(service_result["detections"])==1

    detection=service_result["detections"][0]

    assert isinstance(
        detection,
        DetectionResult
    )

    assert detection.detector=="MockImageDetector"
    assert detection.confidence==0.91