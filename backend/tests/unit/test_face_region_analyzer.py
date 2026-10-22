import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from app.schemas.detection import (
    Assessment,
    DetectionResult,
    DetectorAvailability,
    DetectorExecutionStatus,
    ScoreSemantics,
)
from app.services.detection.detectors.image.face_region_analyzer import FaceRegionAnalyzer


class FakeLandmarker:
    def __init__(self, result=None, error=None):
        self.result = result or SimpleNamespace(face_landmarks=[])
        self.error = error
        self.closed = False

    def detect(self, image):
        if self.error:
            raise self.error
        return self.result

    def close(self):
        self.closed = True


class FakeMediaPipeImage:
    def __init__(self, image_format, data):
        self.image_format = image_format
        self.data = data


def fake_mediapipe(landmarker):
    vision = SimpleNamespace(
        RunningMode=SimpleNamespace(IMAGE="IMAGE"),
        FaceLandmarkerOptions=lambda **kwargs: SimpleNamespace(**kwargs),
        FaceLandmarker=SimpleNamespace(
            create_from_options=lambda options: landmarker,
        ),
    )
    return SimpleNamespace(
        tasks=SimpleNamespace(
            BaseOptions=lambda **kwargs: SimpleNamespace(**kwargs),
            vision=vision,
        ),
        ImageFormat=SimpleNamespace(SRGB="SRGB"),
        Image=FakeMediaPipeImage,
    )


def make_landmark_result():
    landmarks = [
        SimpleNamespace(
            x=0.2 + ((index * 37) % 478) / 478 * 0.6,
            y=0.1 + ((index * 71) % 478) / 478 * 0.8,
        )
        for index in range(478)
    ]
    return SimpleNamespace(face_landmarks=[landmarks])


def make_image(path: Path):
    Image.new("RGB", (160, 160), color="gray").save(path)
    return path


def test_face_landmarker_tasks_detects_face_when_available(tmp_path, monkeypatch):
    model = tmp_path / "face_landmarker.task"
    model.write_bytes(b"test model placeholder")
    landmarker = FakeLandmarker(make_landmark_result())
    monkeypatch.setitem(sys.modules, "mediapipe", fake_mediapipe(landmarker))
    image_path = make_image(tmp_path / "face.png")

    analyzer = FaceRegionAnalyzer(model_path=model)
    analyzer._analyze_face_regions = lambda *args: []
    result = analyzer.detect(str(image_path))

    assert result.metadata["faces_detected"] == 1
    assert result.metadata["status"] == "active_face_region_analysis"
    assert result.assessment == Assessment.INCONCLUSIVE
    assert result.manipulation_detected is False
    assert result.regions == []


def test_close_releases_tasks_landmarker(tmp_path, monkeypatch):
    model = tmp_path / "face_landmarker.task"
    model.write_bytes(b"test model placeholder")
    landmarker = FakeLandmarker()
    monkeypatch.setitem(sys.modules, "mediapipe", fake_mediapipe(landmarker))

    analyzer = FaceRegionAnalyzer(model_path=model)
    analyzer.close()

    assert landmarker.closed is True
    assert analyzer._face_landmarker is None


def test_no_detectable_faces_is_inconclusive(tmp_path, monkeypatch):
    model = tmp_path / "face_landmarker.task"
    model.write_bytes(b"test model placeholder")
    monkeypatch.setitem(sys.modules, "mediapipe", fake_mediapipe(FakeLandmarker()))
    image_path = make_image(tmp_path / "no_face.png")

    result = FaceRegionAnalyzer(model_path=model).detect(str(image_path))

    # The landmarker ran and found nothing to analyse. That is a real outcome and
    # not a negative one: no face was examined, so nothing was cleared.
    assert result.availability == DetectorAvailability.AVAILABLE
    assert result.execution_status == DetectorExecutionStatus.INCONCLUSIVE
    assert result.manipulation_detected is False
    assert result.regions == []
    assert result.status_reason.startswith("No faces detected")
    assert result.supports_determination is False


def test_invalid_image_is_failed_not_unavailable(tmp_path, monkeypatch):
    """
    Undecodable evidence is a FAILED run, not an absent capability.

    This asserted UNAVAILABLE. MediaPipe was loaded and ready; the image bytes
    could not be decoded. Recording that as "we do not have this capability"
    points an examiner at the deployment instead of at the evidence file.
    """
    model = tmp_path / "face_landmarker.task"
    model.write_bytes(b"test model placeholder")
    monkeypatch.setitem(sys.modules, "mediapipe", fake_mediapipe(FakeLandmarker()))
    image_path = tmp_path / "invalid.jpg"
    image_path.write_bytes(b"not an image")

    result = FaceRegionAnalyzer(model_path=model).detect(str(image_path))

    assert result.availability == DetectorAvailability.AVAILABLE
    assert result.execution_status == DetectorExecutionStatus.FAILED
    assert result.status_reason
    assert result.manipulation_detected is False
    assert result.score_semantics == ScoreSemantics.NOT_APPLICABLE
    assert result.regions == []
    assert result.produced_observations is False


def test_face_region_runtime_failure_is_failed_not_unavailable(tmp_path, monkeypatch):
    model = tmp_path / "face_landmarker.task"
    model.write_bytes(b"test model placeholder")
    landmarker = FakeLandmarker(make_landmark_result())
    monkeypatch.setitem(sys.modules, "mediapipe", fake_mediapipe(landmarker))
    image_path = make_image(tmp_path / "valid.png")
    analyzer = FaceRegionAnalyzer(model_path=model)

    def fail_regions(*_args):
        raise RuntimeError("forced region analysis failure")

    analyzer._analyze_face_regions = fail_regions
    result = analyzer.detect(str(image_path))

    # A crash in region analysis is a defect in this code. It must stay
    # distinguishable from a detector that was never installed, because the two
    # have different remedies.
    assert result.availability == DetectorAvailability.AVAILABLE
    assert result.execution_status == DetectorExecutionStatus.FAILED
    assert "RuntimeError" in result.status_reason
    assert result.manipulation_detected is False
    assert result.confidence == 0.0
    assert result.score_semantics == ScoreSemantics.NOT_APPLICABLE
    assert result.regions == []
    assert result.produced_observations is False


def test_detector_initialization_failure_is_unavailable(tmp_path, monkeypatch):
    model = tmp_path / "face_landmarker.task"
    model.write_bytes(b"test model placeholder")
    unavailable_landmarker = FakeLandmarker(error=RuntimeError("initialization failed"))
    fake_mp = fake_mediapipe(unavailable_landmarker)
    fake_mp.tasks.vision.FaceLandmarker.create_from_options = lambda options: (_ for _ in ()).throw(
        RuntimeError("model initialization failed")
    )
    monkeypatch.setitem(sys.modules, "mediapipe", fake_mp)
    image_path = make_image(tmp_path / "valid.png")

    analyzer = FaceRegionAnalyzer(model_path=model)
    result = analyzer.detect(str(image_path))

    # The Tasks runtime could not be constructed, so the capability really is
    # absent: UNAVAILABLE + SKIPPED, with the reason recorded.
    assert result.availability == DetectorAvailability.UNAVAILABLE
    assert result.execution_status == DetectorExecutionStatus.SKIPPED
    assert "initialization failed" in result.status_reason.lower()
    assert result.manipulation_detected is False
    assert result.confidence == 0.0
    assert result.score_semantics == ScoreSemantics.NOT_APPLICABLE
    assert result.regions == []
    assert result.produced_observations is False


def test_missing_mediapipe_is_unavailable(tmp_path, monkeypatch):
    model = tmp_path / "face_landmarker.task"
    model.write_bytes(b"test model placeholder")
    monkeypatch.setitem(sys.modules, "mediapipe", None)
    image_path = make_image(tmp_path / "valid.png")

    result = FaceRegionAnalyzer(model_path=model).detect(str(image_path))

    assert result.availability == DetectorAvailability.UNAVAILABLE
    assert result.execution_status == DetectorExecutionStatus.SKIPPED
    assert result.status_reason
    assert result.manipulation_detected is False
    assert result.regions == []
    assert result.produced_observations is False


def test_disabled_by_configuration_is_skipped_not_unavailable(tmp_path):
    """
    An operator switching a detector off is not a missing capability.

    ``use_mediapipe=False`` used to report UNAVAILABLE, which told a reader the
    deployment was incomplete when in fact the analysis was configured not to
    run. AVAILABLE + SKIPPED says what actually happened.
    """
    image_path = make_image(tmp_path / "valid.png")

    result = FaceRegionAnalyzer(use_mediapipe=False).detect(str(image_path))

    assert result.availability == DetectorAvailability.AVAILABLE
    assert result.execution_status == DetectorExecutionStatus.SKIPPED
    assert "disabled" in result.status_reason.lower()
    assert result.produced_observations is False
    assert result.supports_determination is False


def test_real_mediapipe_landmarker_detects_public_sample_when_configured():
    sample_path = os.getenv("FACE_LANDMARKER_TEST_IMAGE")
    model_path = Path(__file__).resolve().parents[2] / "models" / "face_landmarker.task"
    if not sample_path or not model_path.is_file():
        pytest.skip("Set FACE_LANDMARKER_TEST_IMAGE and install the model asset for real inference")

    analyzer = FaceRegionAnalyzer()
    if analyzer._face_landmarker is None:
        pytest.fail(f"Supported MediaPipe Tasks detector unavailable: {analyzer._initialization_error}")

    try:
        result = analyzer.detect(sample_path)
        assert result.metadata["faces_detected"] == 1
        assert result.metadata["regions_analyzed"] > 0
        assert result.assessment in (Assessment.CLEAN, Assessment.SUSPICIOUS)
        assert not result.metadata.get("reason")
    finally:
        analyzer.close()


def test_unavailable_face_stage_marks_pipeline_partial_and_keeps_other_results(monkeypatch):
    from app.services.orchestration import analysis_pipeline as pipeline_module
    from app.services.orchestration.analysis_pipeline import AnalysisPipeline

    unavailable = DetectionResult(
        detector="FaceRegionAnalyzer",
        media_type="image",
        manipulation_detected=False,
        confidence=0.0,
        manipulation_type="deepfake",
        assessment=Assessment.UNAVAILABLE,
        metadata={
            "assessment": "UNAVAILABLE",
            "status": "unavailable",
            "reason": "Face Landmarker model asset is missing",
        },
    )
    other_result = DetectionResult(
        detector="ForensicImageDetector",
        media_type="image",
        manipulation_detected=False,
        confidence=0.1,
        manipulation_type="image_manipulation",
        assessment=Assessment.CLEAN,
        metadata={"assessment": "CLEAN"},
    )
    router = SimpleNamespace(
        route=lambda *args: {"detections": [unavailable, other_result]}
    )
    pipeline = AnalysisPipeline.__new__(AnalysisPipeline)
    pipeline._detection_container = SimpleNamespace(router=router)
    stored = []
    monkeypatch.setattr(pipeline_module.repo, "create_detection", lambda db, data: stored.append(data))
    failures = []

    pipeline._run_detection(
        Path("unused.jpg"), "media-id", "analysis-id", "image", [],
        object(), failures,
    )

    assert len(stored) == 2
    assert any("FaceRegionAnalyzer" in failure for failure in failures)
    assert stored[0]["assessment"] == "UNAVAILABLE"
    assert stored[1]["detector"] == "ForensicImageDetector"
