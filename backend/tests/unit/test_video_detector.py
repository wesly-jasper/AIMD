import pytest
from types import SimpleNamespace
from app.services.detection.detectors.video_detector import BaselineVideoDetector, BaseVideoDetector
from app.schemas.detection import (
    Assessment,
    DetectionResult,
    DetectorAvailability,
    DetectorExecutionStatus,
)
from app.services.detection.engine.video_detection_engine import VideoDetectionEngine

def test_video_detector_missing_file():
    detector = BaselineVideoDetector()
    with pytest.raises(FileNotFoundError):
        detector.detect("non_existent_file.mp4")

def test_undecodable_video_is_failed_not_unavailable(tmp_path):
    """
    A file with no decodable frames is a failed run, not an absent capability.

    OpenCV is installed and the temporal engine imports, so the capability is
    present -- the problem is this file. Reporting UNAVAILABLE (as it did) told
    an operator to fix their installation and, worse, made "nothing was
    examined" indistinguishable from "the detector is not deployed".
    """
    detector = BaselineVideoDetector()
    assert isinstance(detector, BaseVideoDetector)

    vid_file = tmp_path / "sample.mp4"
    vid_file.write_bytes(b"dummy video bytes")

    result = detector.detect(str(vid_file))
    assert isinstance(result, DetectionResult)
    assert result.detector == "BaselineVideoDetector"
    assert result.media_type == "video"
    assert result.manipulation_type == "temporal"
    assert result.availability == DetectorAvailability.AVAILABLE
    assert result.execution_status == DetectorExecutionStatus.FAILED
    assert result.status_reason
    assert result.produced_observations is False
    assert result.manipulation_detected is False
    assert result.confidence == 0.0


def test_wrapper_records_which_detector_actually_measured(tmp_path, monkeypatch):
    """
    The wrapper re-reports somebody else's numbers, so it must say whose.

    It stamps its own identity on the result (the pipeline addresses it by that
    name), which means the underlying detector's identity has to be preserved
    somewhere or the record implies this wrapper performed the measurement. It
    must also carry the underlying run's status fields rather than only the
    derived ``assessment`` label, which discarded ``status_reason``.
    """
    from app.services.detection.detectors import video_detector as module

    vid_file = tmp_path / "sample.mp4"
    vid_file.write_bytes(b"dummy video bytes")

    underlying = DetectionResult(
        detector="VideoTemporalEngine",
        detector_version="9.9.9",
        media_type="video",
        manipulation_detected=False,
        confidence=0.0,
        manipulation_type="temporal",
        availability=DetectorAvailability.AVAILABLE,
        execution_status=DetectorExecutionStatus.INCONCLUSIVE,
        status_reason="frame detector failures left coverage incomplete",
    )

    class FakeEngine:
        def detect(self, _path):
            return {"detections": [underlying], "suspicious_segments": []}

    monkeypatch.setattr(module, "VideoDetectionEngine", FakeEngine, raising=False)
    monkeypatch.setattr(
        "app.services.detection.engine.video_detection_engine.VideoDetectionEngine",
        FakeEngine,
    )

    result = BaselineVideoDetector().detect(str(vid_file))

    assert result.detector == "BaselineVideoDetector"
    assert result.metadata["underlying_detector"] == "VideoTemporalEngine"
    assert result.metadata["underlying_detector_version"] == "9.9.9"
    # The underlying run's status survives the hand-off, reason included.
    assert result.execution_status == DetectorExecutionStatus.INCONCLUSIVE
    assert result.status_reason == "frame detector failures left coverage incomplete"
    assert result.supports_determination is False


def _stub_engine(image_failures=0, face_failures=0):
    engine = VideoDetectionEngine()
    engine._image_detector_failures = image_failures
    engine._face_detector_failures = face_failures
    engine._spatial_tracker = SimpleNamespace(
        tracked_regions=[],
        get_tracked_regions_for_frame=lambda _index: [],
    )
    engine._temporal_anomaly_detector = SimpleNamespace(analyze_tracks=lambda _tracks: [])
    return engine


def _frames(scores, tracked_regions=None):
    return [
        {
            "frame_index": i,
            "timestamp_seconds": float(i),
            "combined_score": score,
            "regions": [],
            "tracked_regions": list(tracked_regions or []),
        }
        for i, score in enumerate(scores)
    ]


def test_video_with_frame_detector_failure_is_not_reported_clean():
    engine = _stub_engine(image_failures=1)

    result = engine._aggregate(_frames([0.0]), native_fps=24.0, total_frames=24)

    assert result["detections"][0].assessment == Assessment.INCONCLUSIVE
    assert result["detections"][0].metadata["assessment"] == "INCONCLUSIVE"


def test_negative_result_requires_full_coverage_not_a_low_score():
    """
    The negative statement is made from coverage, not from a threshold.

    ``video_confidence < 0.20`` used to produce a CLEAN label directly. That
    value blends the *proportion* of anomalous frames with the *peak* frame
    score -- a count and a magnitude -- over per-frame scores that are themselves
    blends of ELA/FFT/noise severity and optical flow, so a low value does not
    mean "no anomalies"; it can equally mean "a short clip with one mild
    elevation". The only negative result the engine may now reach requires that
    every sampled frame was analysed by every frame detector with no failures and
    that no frame exceeded threshold -- a fact about what ran.
    """
    detection = _stub_engine()._aggregate(
        _frames([0.05, 0.10, 0.08]), native_fps=24.0, total_frames=72
    )["detections"][0]

    assert detection.assessment == Assessment.NO_SIGNIFICANT_ANOMALIES_DETECTED
    assert detection.execution_status == DetectorExecutionStatus.COMPLETED
    assert detection.manipulation_detected is False
    # Still not a finding of authenticity: it is bounded by what was sampled and
    # by what these detectors can measure.
    assert detection.metadata["frames_analysed"] == 3
    assert detection.metadata["total_video_frames"] == 72

    # One elevated frame, well under the suspicious proportion, is INCONCLUSIVE
    # rather than negative -- an anomaly was measured, it is simply not
    # widespread, and the low composite score may not retract it.
    one_hot = _stub_engine()._aggregate(
        _frames([0.0] * 9 + [0.95]), native_fps=24.0, total_frames=240
    )["detections"][0]
    assert one_hot.metadata["suspicious_frame_proportion"] == 0.1
    assert one_hot.assessment == Assessment.INCONCLUSIVE


def test_tracked_regions_do_not_inflate_the_anomaly_score():
    """
    Tracking stability is not an anomaly measure and is excluded from the score.

    A ``tracked_confidence_boost`` of up to +0.15 was folded into
    ``video_confidence``: it added *how stably a region was tracked* to *how
    anomalous the frames were*, so the sum had no referent, and because it could
    only ever move the score upward, every video with trackable regions read as
    more anomalous than its frame evidence supported.
    """
    scores = [0.0, 0.10, 0.05]
    stable = [SimpleNamespace(temporal_consistency=1.0, confidence=0.9)] * 3

    without = _stub_engine()._aggregate(
        _frames(scores), native_fps=24.0, total_frames=72
    )["detections"][0]
    with_tracking = _stub_engine()._aggregate(
        _frames(scores, tracked_regions=stable), native_fps=24.0, total_frames=72
    )["detections"][0]

    assert with_tracking.confidence == without.confidence
    assert with_tracking.assessment == without.assessment
    # Reported, under its own name, so an examiner still sees it.
    assert with_tracking.metadata["avg_tracked_temporal_consistency"] == 1.0
    # One region per frame across three frames; the count is per-frame
    # occurrences, not distinct tracks.
    assert with_tracking.metadata["tracked_regions_count"] == 3 * len(stable)
    assert "tracked_confidence_boost" not in with_tracking.metadata


def _write_video(path, frames=6):
    import cv2
    import numpy as np

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 1.0, (64, 48))
    rng = np.random.default_rng(0)
    for _ in range(frames):
        writer.write(rng.integers(0, 256, (48, 64, 3), dtype=np.uint8))
    writer.release()


class _ReturningDetector:
    """A frame detector that reports its outcome instead of raising."""

    def __init__(self, execution_status, availability=DetectorAvailability.AVAILABLE):
        self.execution_status = execution_status
        self.availability = availability

    def detect(self, _path):
        return DetectionResult(
            detector="Stub", media_type="image", manipulation_detected=False,
            confidence=0.0, manipulation_type="unknown",
            availability=self.availability, execution_status=self.execution_status,
            status_reason="stub",
        )


def _engine_with(image_status, face_status):
    engine = VideoDetectionEngine()
    engine._image_detector = _ReturningDetector(image_status)
    engine._face_detector = _ReturningDetector(face_status)
    return engine


def test_failed_frame_results_are_failures_not_clean_frames(tmp_path):
    """Regression: a returned FAILED result was counted as a CLEAN frame."""
    video = tmp_path / "v.avi"
    _write_video(video)
    engine = _engine_with(DetectorExecutionStatus.FAILED, DetectorExecutionStatus.COMPLETED)

    detection = engine.detect(video)["detections"][0]

    assert detection.assessment != Assessment.NO_SIGNIFICANT_ANOMALIES_DETECTED
    failures = detection.metadata["detector_failures"]
    assert failures["image_detector_failures"] == failures["total_frames_analyzed"] > 0


def test_failure_counters_reset_between_videos(tmp_path):
    """Regression: counters were per-engine, so one bad video tainted the next."""
    video = tmp_path / "v.avi"
    _write_video(video)
    engine = _engine_with(DetectorExecutionStatus.FAILED, DetectorExecutionStatus.COMPLETED)
    engine.detect(video)

    engine._image_detector = _ReturningDetector(DetectorExecutionStatus.COMPLETED)
    detection = engine.detect(video)["detections"][0]

    assert detection.metadata["detector_failures"]["image_detector_failures"] == 0
    assert detection.metadata["failure_messages"] == []
    assert detection.assessment == Assessment.NO_SIGNIFICANT_ANOMALIES_DETECTED


def test_inconclusive_frame_detector_cannot_support_a_negative(tmp_path):
    video = tmp_path / "v.avi"
    _write_video(video)
    engine = _engine_with(DetectorExecutionStatus.INCONCLUSIVE, DetectorExecutionStatus.COMPLETED)

    detection = engine.detect(video)["detections"][0]

    assert detection.assessment == Assessment.INCONCLUSIVE
    assert detection.metadata["detector_failures"]["image_detector_failures"] == 0


def test_majority_frame_detector_failure_outranks_a_positive_proportion():
    """
    A positive proportion computed from a minority of frames is not reported.

    This is the one place the engine's deliberate asymmetry is overridden: a
    measured anomaly normally stands even when other frames failed, but when over
    half of all frame-detector invocations failed the proportion itself is not
    representative of the video, so there is no reliable positive to stand on.
    """
    engine = _stub_engine(image_failures=4, face_failures=3)

    detection = engine._aggregate(
        _frames([0.9, 0.9, 0.0, 0.0]), native_fps=24.0, total_frames=96
    )["detections"][0]

    assert detection.metadata["suspicious_frame_proportion"] == 0.5
    assert detection.assessment == Assessment.INCONCLUSIVE
    assert detection.execution_status == DetectorExecutionStatus.INCONCLUSIVE
    assert detection.manipulation_detected is False
    # The reason reaches the field an examiner reads, not just metadata.
    assert "image detector failed on 4 of 4 frames" in detection.status_reason
    assert "face detector failed on 3 of 4 frames" in detection.status_reason
