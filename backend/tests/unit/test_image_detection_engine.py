import pytest

from app.schemas.detection import DetectionRegion, DetectionResult

from app.services.detection.engine.image_detection_engine import (
    ImageDetectionEngine
)


class MockDetector:

    def detect(self,file_path):
        return DetectionResult(
            detector="MockDetector",
            media_type="image",
            manipulation_detected=True,
            confidence=0.95,
            manipulation_type="face"
        )


class InvalidDetector:

    def detect(self,file_path):
        return {
            "detector":"InvalidDetector",
            "manipulation_detected":True
        }


def test_engine_without_detector(tmp_path):

    image=tmp_path/"test.jpg"
    image.write_bytes(b"test")

    engine=ImageDetectionEngine()

    with pytest.raises(RuntimeError):
        engine.detect(image)


def test_engine_file_not_found():

    engine=ImageDetectionEngine(
        detectors=[MockDetector()]
    )

    with pytest.raises(FileNotFoundError):
        engine.detect(
            "missing.jpg"
        )


def test_engine_calls_detector(tmp_path):

    image=tmp_path/"test.jpg"
    image.write_bytes(b"test")

    engine=ImageDetectionEngine(
        detectors=[MockDetector()]
    )

    result=engine.detect(image)

    assert isinstance(result, dict)
    assert result["media_type"] == "image"
    assert len(result["detections"]) == 1
    
    det = result["detections"][0]

    assert isinstance(
        det,
        DetectionResult
    )

    assert det.detector=="MockDetector"
    assert det.manipulation_detected is True
    assert det.confidence==0.95


def test_engine_invalid_detector_result(tmp_path):
    """A detector returning the wrong type is a FAILURE, not an absence."""
    image=tmp_path/"test.jpg"
    image.write_bytes(b"test")

    engine=ImageDetectionEngine(
        detectors=[InvalidDetector()]
    )

    result = engine.detect(image)

    assert len(result["detections"]) == 1
    det = result["detections"][0]

    # The detector was present and was invoked; it misbehaved. Reporting this as
    # UNAVAILABLE would disguise a defect in our code as a missing dependency.
    assert det.availability.value == "AVAILABLE"
    assert det.execution_status.value == "FAILED"
    assert det.produced_observations is False

    assert result["assessment"] == "INCONCLUSIVE"
    assert result["analysis_complete"] is False


def test_engine_runtime_failure_is_failed_not_unavailable(tmp_path):
    """
    AVAILABLE + FAILED is a real state and must be representable.

    Collapsing a crash into UNAVAILABLE loses the distinction between "we do not
    have this capability" and "our capability is broken", which have different
    remedies and different implications for the investigation.
    """
    image = tmp_path / "test.jpg"
    image.write_bytes(b"test")

    class FailingDetector:
        def detect(self, _file_path):
            raise RuntimeError("model inference failed")

    engine = ImageDetectionEngine(detectors=[FailingDetector(), MockDetector()])
    result = engine.detect(image)

    assert len(result["detections"]) == 2
    failed = result["detections"][0]

    assert failed.availability.value == "AVAILABLE"
    assert failed.execution_status.value == "FAILED"
    assert "RuntimeError" in failed.status_reason

    # A failed detector has no score; its 0.0 must not read as "measured zero".
    assert failed.score_semantics.value == "NOT_APPLICABLE"

    assert result["detections"][1].detector == "MockDetector"

    # The positive observation stands despite the failure: missing data cannot
    # retract an anomaly that was actually measured.
    assert result["assessment"] == "SUSPICIOUS"
    assert result["analysis_complete"] is False
    assert result["incomplete_detectors"][0]["detector"] == "FailingDetector"


def test_a_failed_detector_cannot_establish_absence_of_anomalies(tmp_path):
    """The asymmetry: absence is not reportable without full coverage."""
    image = tmp_path / "test.jpg"
    image.write_bytes(b"test")

    class FailingDetector:
        def detect(self, _file_path):
            raise RuntimeError("model inference failed")

    class QuietDetector:
        def detect(self, _file_path):
            return DetectionResult(
                detector="ForensicImageDetector",
                media_type="image",
                manipulation_detected=False,
                confidence=0.05,
                manipulation_type="image",
            )

    engine = ImageDetectionEngine(detectors=[FailingDetector(), QuietDetector()])
    result = engine.detect(image)

    # Must NOT be NO_SIGNIFICANT_ANOMALIES_DETECTED: coverage was incomplete.
    assert result["assessment"] == "INCONCLUSIVE"
    assert "did not run" in result["assessment_rationale"]


def test_engine_reports_no_blended_confidence(tmp_path):
    """
    The composite confidence number is gone and must not come back.

    It averaged incomparable, non-independent scores and let detectors that never
    ran contribute 0.0 toward a "clean" verdict.
    """
    image = tmp_path / "test.jpg"
    image.write_bytes(b"test")

    engine = ImageDetectionEngine(detectors=[MockDetector()])
    result = engine.detect(image)

    assert "overall_confidence" not in result

    # Scores remain available, but attributed and tagged with their semantics.
    scores = result["detector_scores"]
    assert len(scores) == 1
    assert scores[0]["detector"] == "MockDetector"
    assert scores[0]["score"] == 0.95
    assert scores[0]["score_semantics"] == "HEURISTIC_SEVERITY"
    assert result["evidence_strength"] in {
        "NONE", "WEAK", "MODERATE", "STRONG", "CONTRADICTORY"
    }


def test_object_detection_is_never_manipulation_evidence(tmp_path):
    """YOLO finding an object is not evidence the image is fake."""
    image = tmp_path / "test.jpg"
    image.write_bytes(b"test")

    class ObjectDetector:
        def detect(self, _file_path):
            return DetectionResult(
                detector="YOLOv8ObjectDetector",
                media_type="image",
                manipulation_detected=True,   # even if it claims this
                confidence=0.99,
                manipulation_type="object",
            )

    engine = ImageDetectionEngine(detectors=[ObjectDetector()])
    result = engine.detect(image)

    # Object presence carries no evidential weight in either direction, so there
    # is nothing to conclude from.
    assert result["assessment"] == "INCONCLUSIVE"
    assert result["supporting_observations"] == []
    assert result["detector_scores"] == []


def test_contradicting_observations_are_reported_not_averaged(tmp_path):
    image = tmp_path / "test.jpg"
    image.write_bytes(b"test")

    class Positive:
        def detect(self, _file_path):
            return DetectionResult(
                detector="CopyMoveDetector",
                media_type="image",
                manipulation_detected=True,
                confidence=0.8,
                manipulation_type="copy_move",
            )

    class Negative:
        def detect(self, _file_path):
            return DetectionResult(
                detector="ForensicImageDetector",
                media_type="image",
                manipulation_detected=False,
                confidence=0.1,
                manipulation_type="image",
            )

    engine = ImageDetectionEngine(detectors=[Positive(), Negative()])
    result = engine.detect(image)

    assert result["evidence_strength"] == "CONTRADICTORY"
    assert result["supporting_observations"] == ["CopyMoveDetector"]
    assert result["contradicting_observations"] == ["ForensicImageDetector"]
    # Uncertainty is retained rather than resolved into a conclusion.
    assert result["assessment"] == "SUSPICIOUS"


def test_same_family_detectors_are_not_independent_corroboration(tmp_path):
    """
    Two compression-based detectors agreeing is one signal, not two.

    Both read JPEG block structure, so a single recompression makes both fire.
    Treating that as corroboration counts one piece of evidence twice.
    """
    image = tmp_path / "test.jpg"
    image.write_bytes(b"test")

    def _positive(name):
        class D:
            def detect(self, _file_path):
                return DetectionResult(
                    detector=name,
                    media_type="image",
                    manipulation_detected=True,
                    confidence=0.9,
                    manipulation_type="image",
                    regions=[
                        DetectionRegion(type="region", confidence=0.9,
                                        bbox=[0, 0, 10, 10])
                    ],
                )
        return D()

    engine = ImageDetectionEngine(detectors=[
        _positive("ForensicImageDetector"),
        _positive("ManipulationMaskGenerator"),
    ])
    result = engine.detect(image)

    # Despite two high-confidence localised positives, this stays SUSPICIOUS
    # because they belong to the same signal family.
    assert result["assessment"] == "SUSPICIOUS"
    assert "single signal family" in result["assessment_rationale"]

    # Across families, the same evidence does escalate.
    engine = ImageDetectionEngine(detectors=[
        _positive("ForensicImageDetector"),
        _positive("CopyMoveDetector"),
    ])
    assert engine.detect(image)["assessment"] == "LIKELY_MANIPULATED"


def test_no_significant_anomalies_is_not_a_finding_of_authenticity(tmp_path):
    image = tmp_path / "test.jpg"
    image.write_bytes(b"test")

    class QuietDetector:
        def detect(self, _file_path):
            return DetectionResult(
                detector="ForensicImageDetector",
                media_type="image",
                manipulation_detected=False,
                confidence=0.02,
                manipulation_type="image",
            )

    engine = ImageDetectionEngine(detectors=[QuietDetector()])
    result = engine.detect(image)

    assert result["assessment"] == "NO_SIGNIFICANT_ANOMALIES_DETECTED"
    # The rationale must state the limitation, not imply authenticity.
    rationale = result["assessment_rationale"]
    assert "not a finding of authenticity" in rationale
    assert "authentic" not in rationale.replace("not a finding of authenticity", "")


def test_unavailable_detector_cannot_make_low_confidence_result_clean():
    from app.schemas.detection import Assessment

    engine = ImageDetectionEngine()
    unavailable = DetectionResult(
        detector="UnavailableDetector",
        media_type="image",
        manipulation_detected=False,
        confidence=0.0,
        manipulation_type="unknown",
        assessment=Assessment.UNAVAILABLE,
    )
    low_confidence_clean = DetectionResult(
        detector="ForensicImageDetector",
        media_type="image",
        manipulation_detected=False,
        confidence=0.1,
        manipulation_type="image",
        assessment=Assessment.CLEAN,
    )

    result = engine._aggregate([unavailable, low_confidence_clean])

    assert result["assessment"] == "INCONCLUSIVE"