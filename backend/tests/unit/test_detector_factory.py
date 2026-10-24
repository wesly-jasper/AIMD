from app.core.config import settings
from app.services.detection.detector_factory import (
    DetectorFactory
)


def test_trufor_disabled(monkeypatch):
    # Set explicitly: a developer's backend/.env may enable TruFor.
    monkeypatch.setattr(settings, "trufor_enabled", False)

    result=DetectorFactory.create_trufor()

    assert result is None


def test_object_detection_uses_the_configured_yolov8_weights(monkeypatch, tmp_path):
    weights = tmp_path / "yolo.pt"
    weights.write_bytes(b"")
    monkeypatch.setattr(settings, "object_detection_enabled", True)
    monkeypatch.setattr(settings, "object_detection_model_path", weights)

    detector = DetectorFactory.create_object_detector()

    assert type(detector).__name__ == "YOLOv8ObjectDetector"
    assert detector.model_name == str(weights)


def test_object_detection_disabled_is_the_unconfigured_placeholder(monkeypatch):
    monkeypatch.setattr(settings, "object_detection_enabled", False)
    assert type(DetectorFactory.create_object_detector()).__name__ == "BaselineObjectDetector"
