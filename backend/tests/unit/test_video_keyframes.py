"""
Keyframe analysis of video: cuts are found, frames either side of a cut are
examined, flagged frames are reported with timestamps, and a clip on which no
keyframe was flagged is INCONCLUSIVE -- never "no anomalies" for the whole video.
"""
import cv2
import numpy as np
import pytest
from PIL import Image

from app.schemas.detection import (
    DetectionRegion,
    DetectionResult,
    DetectorAvailability,
    DetectorExecutionStatus,
    ScoreSemantics,
)
from app.services.detection.detectors.base_detector import BaseDetector
from app.services.detection.engine.video_keyframes import (
    VideoKeyframeAnalyzer,
    scan_shots,
    select_keyframes,
)

FPS = 25


def _video(path, shots):
    """``shots``: list of (seconds, BGR colour). Noise keeps frames non-identical."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (96, 64))
    rng = np.random.default_rng(0)
    for seconds, colour in shots:
        for _ in range(int(seconds * FPS)):
            frame = np.full((64, 96, 3), colour, np.uint8)
            frame = np.clip(frame + rng.integers(-6, 7, frame.shape), 0, 255).astype(np.uint8)
            writer.write(frame)
    writer.release()
    return path


class _ByColour(BaseDetector):
    """Flags frames whose mean is red-dominant: stands in for TruFor."""
    detector_id = "Fake"
    manipulation_type = "image_manipulation"

    def __init__(self):
        self.seen = 0

    def detect(self, file_path):
        self.seen += 1
        with Image.open(file_path) as img:
            r, g, b = np.asarray(img.convert("RGB")).reshape(-1, 3).mean(0)
        flagged = r > 150 and g < 100
        return DetectionResult(
            detector=self.detector_id, media_type="image", manipulation_detected=flagged,
            confidence=0.9 if flagged else 0.1, manipulation_type=self.manipulation_type,
            availability=DetectorAvailability.AVAILABLE,
            execution_status=DetectorExecutionStatus.COMPLETED,
            score_semantics=ScoreSemantics.RAW_MODEL_SCORE,
            regions=[DetectionRegion(type="forgery_localization", confidence=0.9, bbox=[0, 0, 8, 8])] if flagged else [],
        )


class _Missing(BaseDetector):
    detector_id = "Missing"

    def availability(self):
        return DetectorAvailability.UNAVAILABLE, "weights not installed"

    def detect(self, file_path):  # pragma: no cover - never called
        raise AssertionError


def test_an_inserted_shot_is_found_and_its_frames_examined(tmp_path, monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "video_keyframe_verdicts_enabled", True)
    # 4 s of grey, a 1.5 s red insert, 4 s of grey: the reliability set's
    # "insertion" case, which 1 fps sampling could miss entirely.
    path = _video(tmp_path / "v.mp4", [(4, (128, 128, 128)), (1.5, (30, 30, 220)), (4, (128, 128, 128))])
    scan = scan_shots(path)

    cut_times = [c["timestamp"] for c in scan["cuts"]]
    assert len(cut_times) == 2
    assert cut_times[0] == pytest.approx(4.0, abs=0.15)
    assert cut_times[1] == pytest.approx(5.5, abs=0.15)
    assert len(scan["short_shots"]) == 1

    frames = select_keyframes(scan)
    reasons = {f.reason for f in frames}
    assert {"before_cut", "after_cut"} <= reasons

    detector = _ByColour()
    out = VideoKeyframeAnalyzer({"VideoKeyframeTruFor": detector}).analyse(path)
    (result,) = out["detections"]
    assert result.media_type == "video"
    assert result.execution_status is DetectorExecutionStatus.COMPLETED
    assert result.manipulation_detected is True
    flagged = result.metadata["flagged_timestamps"]
    assert flagged and all(4.0 <= t <= 5.6 for t in flagged)
    assert all(r.start_timestamp is not None for r in result.regions)
    assert "Frames between them were not" in result.metadata["coverage"]["statement"]


def test_no_flagged_keyframe_is_inconclusive_not_clean(tmp_path):
    path = _video(tmp_path / "v.mp4", [(6, (128, 128, 128))])
    out = VideoKeyframeAnalyzer({"VideoKeyframeTruFor": _ByColour()}).analyse(path)
    (result,) = out["detections"]
    assert result.execution_status is DetectorExecutionStatus.INCONCLUSIVE
    assert result.manipulation_detected is False
    assert "not a finding about the whole video" in result.status_reason
    # A single shot still gets several frames, not one.
    assert result.metadata["frames_analysed"] >= 3


def test_an_unavailable_model_is_reported_not_run(tmp_path):
    path = _video(tmp_path / "v.mp4", [(2, (128, 128, 128))])
    out = VideoKeyframeAnalyzer({"VideoKeyframeTruFor": _Missing()}).analyse(path)
    (result,) = out["detections"]
    assert result.availability is DetectorAvailability.UNAVAILABLE
    assert result.status_reason == "weights not installed"


def test_video_engine_keeps_the_temporal_result_when_keyframes_fail(tmp_path):
    from app.services.detection.engine.video_detection_engine import VideoDetectionEngine

    class Exploding:
        detectors = {"VideoKeyframeTruFor": None}

        def analyse(self, path):
            raise RuntimeError("decoder crashed")

    path = _video(tmp_path / "v.mp4", [(2, (128, 128, 128))])
    out = VideoDetectionEngine(keyframe_analyzer=Exploding()).detect(path)
    names = [d.detector for d in out["detections"]]
    assert names[0] == "VideoTemporalAnalysisEngine"
    failed = out["detections"][-1]
    assert failed.detector == "VideoKeyframeTruFor"
    assert failed.execution_status is DetectorExecutionStatus.FAILED


def test_flagged_keyframes_do_not_set_the_verdict_by_default(tmp_path, monkeypatch):
    """Off by default: measured to flag genuine clips as often as manipulated ones."""
    from app.core.config import settings

    monkeypatch.setattr(settings, "video_keyframe_verdicts_enabled", False)
    path = _video(tmp_path / "v.mp4", [(4, (128, 128, 128)), (1.5, (30, 30, 220)), (4, (128, 128, 128))])
    (result,) = VideoKeyframeAnalyzer({"VideoKeyframeTruFor": _ByColour()}).analyse(path)["detections"]
    assert result.execution_status is DetectorExecutionStatus.INCONCLUSIVE
    assert result.manipulation_detected is False
    # The frames stay visible for review, with their timestamps.
    assert result.metadata["flagged_timestamps"]
    assert result.regions and all(r.start_timestamp is not None for r in result.regions)
    assert "shown for review" in result.status_reason
