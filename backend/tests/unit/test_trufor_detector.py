import pytest

from app.schemas.detection import (
    DetectionResult,
    DetectorAvailability,
    DetectorExecutionStatus,
    ScoreSemantics,
)
from app.services.detection.detectors.image.trufor_detector import (
    TruForDetector
)
from app.services.detection.engine.image_detection_engine import ImageDetectionEngine


class MockRunner:

    def predict(self,file_path):
        return {
            "manipulation_detected":True,
            "confidence":0.92,
            "regions":[],
            "metadata":{
                "score":0.92
            }
        }


class InvalidRunner:

    def predict(self,file_path):
        return {
            "invalid":"result"
        }


class ExplodingRunner:

    def predict(self, file_path):
        raise RuntimeError("forced inference failure")


def test_trufor_detector_file_not_found():

    detector=TruForDetector(
        MockRunner()
    )

    with pytest.raises(FileNotFoundError):
        detector.detect(
            "missing.jpg"
        )


def test_trufor_detector(tmp_path):

    image=tmp_path/"test.jpg"
    image.write_bytes(b"test")

    detector=TruForDetector(
        MockRunner()
    )

    result=detector.detect(image)

    assert isinstance(
        result,
        DetectionResult
    )

    assert result.detector=="TruForDetector"
    assert result.media_type=="image"
    assert result.manipulation_detected is True
    assert result.confidence==0.92
    assert result.manipulation_type=="image_manipulation"
    assert result.regions==[]
    assert result.metadata["score"]==0.92
    assert result.availability == DetectorAvailability.AVAILABLE
    assert result.execution_status == DetectorExecutionStatus.COMPLETED
    # The only learned detector in the suite, and the only one entitled to call
    # its score a model output -- raw, because no calibration curve was fitted.
    assert result.score_semantics == ScoreSemantics.RAW_MODEL_SCORE


def test_trufor_identity_matches_the_engine_signal_family_map(tmp_path):
    """
    The reported identity must be the one the fusion layer keys on.

    Results went out as ``"TruFor"`` while the engine's ``_SIGNAL_FAMILY`` map,
    the factory and the class were all named ``"TruForDetector"``. The "learned"
    family therefore never matched a result, so the one learned detector in the
    suite was silently filed as "unclassified" and could not contribute to
    cross-family corroboration as designed.
    """
    image = tmp_path / "test.jpg"
    image.write_bytes(b"test")

    result = TruForDetector(MockRunner()).detect(image)

    assert ImageDetectionEngine._family(result.detector) == "learned"


def test_trufor_unavailable_runner_is_unavailable_not_clean(tmp_path):
    """
    The TruFor checkpoint is not redistributable, so absence is the normal state.

    It must be recorded as an absent capability and never as a completed run:
    an assessment reached without the learned family has narrower coverage, and
    the record has to show that.
    """
    image = tmp_path / "test.jpg"
    image.write_bytes(b"test")

    result = TruForDetector(None, "TruFor checkpoint is not installed").detect(image)

    assert result.detector == "TruForDetector"
    assert result.availability == DetectorAvailability.UNAVAILABLE
    assert result.execution_status == DetectorExecutionStatus.SKIPPED
    assert "checkpoint" in result.status_reason
    assert result.confidence == 0.0
    assert result.score_semantics == ScoreSemantics.NOT_APPLICABLE
    assert result.produced_observations is False


def test_trufor_inference_crash_is_failed_not_unavailable(tmp_path):
    """
    A loaded checkpoint that crashes is a defect, not a missing capability.

    Collapsing the two would tell an operator to install a model they already
    have.
    """
    image = tmp_path / "test.jpg"
    image.write_bytes(b"test")

    result = TruForDetector(ExplodingRunner()).detect(image)

    assert result.availability == DetectorAvailability.AVAILABLE
    assert result.execution_status == DetectorExecutionStatus.FAILED
    assert "RuntimeError" in result.status_reason
    assert result.produced_observations is False


def test_trufor_detector_malformed_runner_result_is_failed(tmp_path):
    """
    A malformed runner response is recorded, not raised.

    Raising surfaced as an engine-level error with no detector record at all, so
    the run left no trace of having been attempted. A FAILED result keeps the
    reason in the evidence where an examiner will see it.
    """
    image=tmp_path/"test.jpg"
    image.write_bytes(b"test")

    result = TruForDetector(InvalidRunner()).detect(image)

    assert result.availability == DetectorAvailability.AVAILABLE
    assert result.execution_status == DetectorExecutionStatus.FAILED
    assert "manipulation_detected" in result.status_reason
    assert result.produced_observations is False
