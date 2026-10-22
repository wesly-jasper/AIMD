import pytest
from PIL import Image
from app.services.detection.detectors.image.forensic_image_detector import ForensicImageDetector
from app.schemas.detection import (
    DetectionResult,
    DetectorAvailability,
    DetectorExecutionStatus,
    ScoreSemantics,
)

def test_forensic_image_detector_missing_file():
    detector = ForensicImageDetector()
    with pytest.raises(FileNotFoundError):
        detector.detect("missing_image.jpg")

def test_forensic_image_detector_execution(tmp_path):
    detector = ForensicImageDetector()
    
    img_path = tmp_path / "sample_test.jpg"
    img = Image.new("RGB", (256, 256), color="cyan")
    img.save(str(img_path))
    
    result = detector.detect(str(img_path))
    assert isinstance(result, DetectionResult)
    assert result.detector == "ForensicImageDetector"
    assert result.media_type == "image"
    assert 0.0 <= result.confidence <= 1.0
    assert "ela_score" in result.metadata
    assert "fft_spectral_score" in result.metadata
    assert "noise_inconsistency_score" in result.metadata
    assert result.assessment.value == (
        "SUSPICIOUS" if result.manipulation_detected else "CLEAN"
    )


def test_invalid_image_is_failed_not_unavailable(tmp_path):
    """
    An undecodable file is a failed run, not an absent capability.

    This detector needs nothing beyond OpenCV and PIL, so it is never genuinely
    unavailable. Recording an unreadable image as UNAVAILABLE (as it did) blamed
    the capability for a problem with the evidence, which points an examiner at
    installing a dependency they already have instead of at the file.
    """
    img_path = tmp_path / "invalid.jpg"
    img_path.write_bytes(b"not an image")

    result = ForensicImageDetector().detect(str(img_path))

    assert result.availability == DetectorAvailability.AVAILABLE
    assert result.execution_status == DetectorExecutionStatus.FAILED
    assert result.status_reason
    # The zero must not read as "measured, and found almost no anomaly".
    assert result.score_semantics == ScoreSemantics.NOT_APPLICABLE
    assert result.produced_observations is False
    assert result.manipulation_detected is False
    assert result.confidence == 0.0
    assert result.regions == []
