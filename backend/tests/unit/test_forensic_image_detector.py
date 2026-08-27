import pytest
from PIL import Image
from app.services.detection.detectors.image.forensic_image_detector import ForensicImageDetector
from app.schemas.detection import DetectionResult

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
