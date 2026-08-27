from app.schemas.detection import (
    DetectionRegion,
    DetectionRequest,
    DetectionResult
)


def test_detection_region():
    region=DetectionRegion(
        type="face",
        confidence=0.95,
        bbox=[10,20,100,120]
    )

    assert region.type=="face"
    assert region.confidence==0.95
    assert region.bbox==[10,20,100,120]


def test_detection_result():
    result=DetectionResult(
        detector="TestDetector",
        media_type="image",
        manipulation_detected=True,
        confidence=0.92,
        manipulation_type="face",
        regions=[
            DetectionRegion(
                type="face",
                confidence=0.95,
                bbox=[10,20,100,120]
            )
        ]
    )

    assert result.detector=="TestDetector"
    assert result.media_type=="image"
    assert result.manipulation_detected is True
    assert result.confidence==0.92
    assert len(result.regions)==1


def test_detection_result_without_regions():
    result=DetectionResult(
        detector="TestDetector",
        media_type="image",
        manipulation_detected=False,
        confidence=0.12,
        manipulation_type="unknown"
    )

    assert result.regions==[]


def test_invalid_confidence():
    try:
        DetectionResult(
            detector="TestDetector",
            media_type="image",
            manipulation_detected=True,
            confidence=1.5,
            manipulation_type="face"
        )
        assert False
    except ValueError:
        assert True

def test_detection_request():

    request=DetectionRequest(
        file_path="uploads/test.jpg",
        media_type="image"
    )

    assert request.file_path=="uploads/test.jpg"
    assert request.media_type=="image"