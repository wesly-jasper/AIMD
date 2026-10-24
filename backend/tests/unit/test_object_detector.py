import pytest
from types import SimpleNamespace
from PIL import Image
from app.services.detection.detectors.object_detector import (
    BaselineObjectDetector,
    YOLOv8ObjectDetector,
    BaseObjectDetector,
)
from app.schemas.detection import (
    DetectionResult,
    DetectorAvailability,
    DetectorExecutionStatus,
    ScoreSemantics,
)

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
    # No model is configured, so the capability is genuinely absent and the run
    # was never attempted -- recorded as two separate facts rather than one
    # "status" string that could not distinguish them.
    assert result.availability == DetectorAvailability.UNAVAILABLE
    assert result.execution_status == DetectorExecutionStatus.SKIPPED
    assert "no object detection model" in result.status_reason.lower()
    assert result.score_semantics == ScoreSemantics.NOT_APPLICABLE
    assert result.produced_observations is False
    assert result.manipulation_detected is False

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


def test_object_localization_does_not_claim_clean_authenticity(tmp_path):
    img_file = tmp_path / "object-only.png"
    Image.new("RGB", (64, 64), color="white").save(img_file)
    detector = YOLOv8ObjectDetector(model_name="not-loaded.pt")
    detector._loaded = True
    detector._model = SimpleNamespace(
        names={},
        predict=lambda **_kwargs: [SimpleNamespace(boxes=[])],
    )

    result = detector.detect(str(img_file))

    assert result.assessment.value == "INCONCLUSIVE"
    assert result.manipulation_detected is False
    assert result.confidence == 0.0
    # It ran, so this is not an absence of evidence -- but object localisation
    # cannot support a determination about manipulation in either direction.
    assert result.execution_status == DetectorExecutionStatus.INCONCLUSIVE
    assert result.supports_determination is False
    # The result carries no manipulation score, so the 0.0 must not be readable
    # as "measured, and found almost no anomaly".
    assert result.score_semantics == ScoreSemantics.NOT_APPLICABLE


def test_unloadable_yolo_does_not_return_a_baseline_result(tmp_path):
    """
    A detector must never report under another detector's identity.

    ``YOLOv8ObjectDetector.detect`` used to ``return
    BaselineObjectDetector().detect(file_path)`` when the model had not loaded,
    so the stored record named a detector the pipeline never asked for and
    omitted the one it did. Nothing recorded that YOLOv8 had been requested, or
    why it was not used.
    """
    img_file = tmp_path / "unloadable.png"
    Image.new("RGB", (32, 32), color="white").save(img_file)
    detector = YOLOv8ObjectDetector(model_name="definitely-not-a-model.pt")
    # Stand in for a failed load without depending on whether ultralytics is
    # installed in the test environment.
    detector._load_attempted = True
    detector._loaded = False
    detector._load_error = "FileNotFoundError: definitely-not-a-model.pt"

    result = detector.detect(str(img_file))

    assert result.detector == "YOLOv8ObjectDetector"
    assert result.availability == DetectorAvailability.UNAVAILABLE
    assert result.execution_status == DetectorExecutionStatus.SKIPPED
    assert "definitely-not-a-model.pt" in result.status_reason
    assert result.produced_observations is False


def test_yolo_inference_crash_is_failed_not_unavailable(tmp_path):
    """
    A loaded model that crashes is a defect, not a missing capability.

    Collapsing the two would tell an operator to install weights they already
    have, and it previously also mislabelled the result as a baseline run.
    """
    img_file = tmp_path / "crash.png"
    Image.new("RGB", (32, 32), color="white").save(img_file)
    detector = YOLOv8ObjectDetector(model_name="loaded.pt")
    detector._loaded = True

    def _explode(**_kwargs):
        raise RuntimeError("forced inference failure")

    detector._model = SimpleNamespace(names={}, predict=_explode)

    result = detector.detect(str(img_file))

    assert result.detector == "YOLOv8ObjectDetector"
    assert result.availability == DetectorAvailability.AVAILABLE
    assert result.execution_status == DetectorExecutionStatus.FAILED
    assert "RuntimeError" in result.status_reason
    assert result.produced_observations is False


def test_missing_weights_are_unavailable_and_never_downloaded(monkeypatch, tmp_path):
    """A bare 'yolov8n.pt' would make Ultralytics fetch weights from the internet."""
    import ultralytics

    def no_network(*_a, **_k):
        raise AssertionError("YOLO must not be constructed without a local weights file")

    monkeypatch.setattr(ultralytics, "YOLO", no_network)
    detector = YOLOv8ObjectDetector(model_name=str(tmp_path / "yolov8n.pt"))

    availability, reason = detector.availability()

    assert availability == DetectorAvailability.UNAVAILABLE
    assert "weights file not found" in reason
    assert str(tmp_path) not in reason  # no host path in a stored reason
