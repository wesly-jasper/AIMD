from types import SimpleNamespace

import numpy as np
from PIL import Image

from app.core.config import settings
from app.schemas.detection import (
    Assessment,
    DetectionResult,
    DetectorAvailability,
    DetectorExecutionStatus,
)
from app.services.detection.detector_factory import DetectorFactory
from app.services.detection.engine.image_detection_engine import ImageDetectionEngine
from app.services.orchestration import analysis_pipeline as pipeline_module
from app.services.orchestration.analysis_pipeline import AnalysisPipeline


def test_enabled_trufor_with_missing_weights_is_explicitly_unavailable(
    tmp_path, monkeypatch
):
    image_path = tmp_path / "sample.png"
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(image_path)
    monkeypatch.setattr(settings, "trufor_enabled", True)
    monkeypatch.setattr(settings, "trufor_model_path", tmp_path / "missing.pth")

    detector = DetectorFactory.create_trufor()
    result = detector.detect(image_path)

    assert result.assessment == Assessment.UNAVAILABLE
    # Availability and execution are recorded as two separate facts: the
    # capability is absent, so the run was never attempted. A single "status"
    # string could not express both, which is why it is gone.
    assert result.availability == DetectorAvailability.UNAVAILABLE
    assert result.execution_status == DetectorExecutionStatus.SKIPPED
    assert result.status_reason
    assert result.manipulation_detected is False
    assert result.confidence == 0.0
    assert result.regions == []


def test_missing_detector_does_not_make_final_assessment_clean(monkeypatch):
    """
    A detector that never ran cannot help establish that nothing was found.

    Previously the pipeline averaged every detector's confidence, so an
    UNAVAILABLE detector contributing 0.0 dragged the mean below the "clean"
    threshold -- letting an absent check support a negative finding.
    """
    pipeline = AnalysisPipeline.__new__(AnalysisPipeline)
    detections = [
        SimpleNamespace(
            detector="TruForDetector", assessment="UNAVAILABLE",
            confidence=0.0, regions=[],
        ),
        SimpleNamespace(
            detector="ForensicImageDetector", assessment="CLEAN",
            confidence=0.1, regions=[],
        ),
    ]
    monkeypatch.setattr(
        pipeline_module.repo,
        "get_detections_for_analysis",
        lambda _db, _analysis_id: detections,
    )

    assessment, evidence_strength = pipeline._compute_final_assessment(
        "analysis-id", object()
    )

    assert assessment == "INCONCLUSIVE"
    # Ordinal, and explicitly NONE -- not a number that could be read as a
    # probability of authenticity.
    assert evidence_strength == "NONE"


def test_final_assessment_returns_no_numeric_confidence(monkeypatch):
    """The blended numeric confidence is gone from the pipeline too."""
    pipeline = AnalysisPipeline.__new__(AnalysisPipeline)
    detections = [
        SimpleNamespace(
            detector="ForensicImageDetector", assessment="CLEAN",
            confidence=0.1, regions=[],
        ),
    ]
    monkeypatch.setattr(
        pipeline_module.repo,
        "get_detections_for_analysis",
        lambda _db, _analysis_id: detections,
    )

    assessment, evidence_strength = pipeline._compute_final_assessment(
        "analysis-id", object()
    )

    # Full coverage, no anomalies -> the strongest negative statement allowed,
    # which is still not a finding of authenticity.
    assert assessment == "NO_SIGNIFICANT_ANOMALIES_DETECTED"
    assert isinstance(evidence_strength, str)
    assert not isinstance(evidence_strength, float)


def test_object_detector_alone_cannot_drive_an_assessment(monkeypatch):
    """YOLO reporting an object is not evidence of manipulation."""
    pipeline = AnalysisPipeline.__new__(AnalysisPipeline)
    detections = [
        SimpleNamespace(
            detector="YOLOv8ObjectDetector", assessment="SUSPICIOUS",
            confidence=0.99, regions=[{"bbox": [0, 0, 5, 5]}],
        ),
    ]
    monkeypatch.setattr(
        pipeline_module.repo,
        "get_detections_for_analysis",
        lambda _db, _analysis_id: detections,
    )

    assessment, _ = pipeline._compute_final_assessment("analysis-id", object())

    assert assessment == "INCONCLUSIVE"


def test_health_reports_trufor_model_as_configured_not_operational(
    tmp_path, monkeypatch
):
    from app import main
    from app.db import database

    class FakeSession:
        def execute(self, _statement):
            return None

        def close(self):
            return None

    model_path = tmp_path / "weights.pth"
    model_path.write_bytes(b"not validated model data")
    monkeypatch.setattr(database, "SessionLocal", FakeSession)
    monkeypatch.setattr(settings, "trufor_enabled", True)
    monkeypatch.setattr(settings, "trufor_model_path", model_path)

    response = main.health()

    # "configured" means a model path is set and the file exists -- it does NOT
    # mean the checkpoint was validated or that inference would succeed.
    assert response["detectors"]["trufor"] == "configured"

    # /health must not assert per-detector availability it cannot introspect.
    # It previously returned a hardcoded map that reported detectors as
    # "available" without ever consulting the registry.
    assert response["detectors"]["availability_introspection"] == "NOT_IMPLEMENTED"
    assert "models" not in response


def test_health_reports_univfd_configuration(tmp_path, monkeypatch):
    from app.core.health import univfd_configuration_status
    from app.services.detection.models import univfd_release as release

    monkeypatch.setattr(settings, "univfd_enabled", False)
    assert univfd_configuration_status() == "disabled"

    monkeypatch.setattr(settings, "univfd_enabled", True)
    monkeypatch.setattr(settings, "univfd_model_path", tmp_path)
    (tmp_path / release.HEAD_FILE).write_bytes(b"head")
    # The head alone is not enough: the CLIP backbone is the other half.
    assert univfd_configuration_status() == "model_missing"

    (tmp_path / release.CLIP_FILE).write_bytes(b"not validated")
    assert univfd_configuration_status() == "configured"


def test_health_does_not_fabricate_detector_availability(monkeypatch):
    """Registration is reported; availability is not invented."""
    from app import main
    from app.db import database

    class FakeSession:
        def execute(self, _statement):
            return None

        def close(self):
            return None

    monkeypatch.setattr(database, "SessionLocal", FakeSession)

    response = main.health()
    detectors = response["detectors"]

    registered = detectors["registered_image_detectors"]
    assert isinstance(registered, list)
    # Values are real class names of constructed detectors, not status strings.
    assert "available" not in registered
    assert all(isinstance(name, str) and name for name in registered)


def test_health_does_not_echo_database_errors(monkeypatch):
    """/health is unauthenticated; driver errors carry paths and connection strings."""
    from app import main
    from app.db import database

    def _broken_session():
        raise RuntimeError("unable to open database file C:\\secret\\path\\aimd.db")

    monkeypatch.setattr(database, "SessionLocal", _broken_session)
    response = main.health()

    assert response["database"] == "error"
    assert response["status"] == "degraded"
    assert "secret" not in str(response)


def test_trufor_initialization_failure_is_partial_and_keeps_other_detector(
    tmp_path, monkeypatch
):
    image_path = tmp_path / "input.png"
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(image_path)
    monkeypatch.setattr(settings, "trufor_enabled", True)
    monkeypatch.setattr(settings, "trufor_model_path", tmp_path / "missing.pth")
    trufor = DetectorFactory.create_trufor()
    other = DetectionResult(
        detector="ForensicImageDetector",
        media_type="image",
        manipulation_detected=False,
        confidence=0.1,
        manipulation_type="image",
        assessment=Assessment.INCONCLUSIVE,
        metadata={"assessment": "INCONCLUSIVE"},
    )

    class OtherDetector:
        def detect(self, _path):
            return other

    engine = ImageDetectionEngine(detectors=[trufor, OtherDetector()])
    pipeline = AnalysisPipeline.__new__(AnalysisPipeline)
    pipeline._detection_container = SimpleNamespace(
        router=SimpleNamespace(route=lambda *_args: engine.detect(image_path))
    )
    stored = []
    monkeypatch.setattr(
        pipeline_module.repo,
        "create_detection",
        lambda _db, data: stored.append(data),
    )
    failures = []

    pipeline._run_detection(
        image_path, "media-id", "analysis-id", "image", [], object(), failures
    )

    assert len(stored) == 2
    # The absent capability is persisted as the two separate facts, so a reader
    # of the stored record can tell "never ran" from "ran and found nothing".
    assert stored[0]["availability"] == "UNAVAILABLE"
    assert stored[0]["execution_status"] == "SKIPPED"
    assert stored[1]["detector"] == "ForensicImageDetector"
    # "TruForDetector", not "TruFor": the detector now reports the identity the
    # engine's signal-family map keys on.
    assert any("TruForDetector incomplete" in failure for failure in failures)
