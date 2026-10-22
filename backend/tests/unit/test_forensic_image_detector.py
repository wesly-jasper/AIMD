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


def test_heuristic_scores_are_inconclusive_not_a_determination(tmp_path):
    """
    Regression: every ordinary camera JPEG used to be reported as manipulated.

    The hand-weighted ELA/FFT/noise composite exceeded its 0.50 threshold on
    clean photographs, because all three cues respond to normal JPEG
    compression. Without calibration it may not decide either way.
    """
    import numpy as np

    rng = np.random.default_rng(7)
    base = rng.normal(128, 30, (256, 256, 3)).clip(0, 255).astype("uint8")
    img_path = tmp_path / "camera_like.jpg"
    Image.fromarray(base).save(str(img_path), quality=90)

    result = ForensicImageDetector().detect(str(img_path))

    assert result.execution_status == DetectorExecutionStatus.INCONCLUSIVE
    assert result.manipulation_detected is False
    assert result.score_semantics == ScoreSemantics.NOT_APPLICABLE
    assert result.assessment.value == "INCONCLUSIVE"
    assert "composite_severity" in result.metadata
    assert "uncalibrated" in result.status_reason.lower()


def test_generator_named_in_metadata_is_a_positive_observation(tmp_path):
    from PIL import PngImagePlugin

    info = PngImagePlugin.PngInfo()
    info.add_text("parameters", "Steps: 30, Sampler: Euler, Model: Stable Diffusion XL")
    img_path = tmp_path / "generated.png"
    Image.new("RGB", (128, 128), color="cyan").save(str(img_path), pnginfo=info)

    result = ForensicImageDetector().detect(str(img_path))

    assert result.execution_status == DetectorExecutionStatus.COMPLETED
    assert result.manipulation_detected is True
    assert result.manipulation_type == "ai_generated_synthetic"
    assert "stable diffusion" in result.status_reason.lower()


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
