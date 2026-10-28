"""
Learned image detectors on selected video frames, with shot boundaries.

The temporal engine samples at 1 fps and runs only classical heuristics; in the
reliability evaluation it flagged 0 of 60 manipulated clips, and no learned
model ever saw a video frame. This stage adds three things beside it:

  1. **Shot boundaries.** Frames are sampled at up to ``SCAN_FPS`` and compared
     by colour histogram; a jump well above the clip's own frame-to-frame
     variation is a cut. Cuts and very short shots are where an insertion or a
     deleted span (jump cut) sits, so they are reported with timestamps. A cut
     is an observation for review, not evidence of manipulation: edited footage
     has cuts.
  2. **Keyframes.** One frame per shot, plus the frames on both sides of every
     cut, up to ``MAX_KEYFRAMES``. Choosing them by content rather than by clock
     is what lets a 1.5 s insertion be looked at at all.
  3. **Learned detectors per keyframe.** The same TruFor and UnivFD instances
     the image engine uses (one copy of each model per process), each call
     bounded by the detector timeout. Every flagged frame is reported with its
     timestamp and region.

COVERAGE RULE
-------------
A few sampled frames are never proof about the whole video. When no keyframe is
flagged the result is INCONCLUSIVE with the coverage stated, never "no
anomalies": frames between the keyframes were not examined. A flag stands, and
says which frames.
"""
from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from app.core.config import settings
from app.schemas.detection import (
    Assessment,
    DetectionRegion,
    DetectionResult,
    DetectorAvailability,
    DetectorExecutionStatus,
    ScoreSemantics,
)
from app.services.detection.runtime import run_detector

logger = logging.getLogger(__name__)

#: Frames per second examined for cuts.
SCAN_FPS = 10.0
#: Upper bound on frames decoded for the cut scan, whatever the clip length.
MAX_SCAN_FRAMES = 3000
#: Upper bound on frames given to the learned detectors.
MAX_KEYFRAMES = 24
#: A cut needs a histogram distance at least this far above the clip's median...
CUT_MAD_FACTOR = 8.0
#: ...and at least this large in absolute terms (Bhattacharyya distance, 0-1).
CUT_MIN_DISTANCE = 0.35
#: A shot shorter than this, between two cuts, is reported as a short insert.
SHORT_SHOT_SECONDS = 2.0
#: Video-level flag: at least this many flagged keyframes...
MIN_FLAGGED_FRAMES = 2
#: ...making up at least this share of the keyframes analysed. Hand-chosen, not
#: calibrated; recorded with every result.
MIN_FLAGGED_SHARE = 0.2

_DECISION_RULE = (
    f"flagged when at least {MIN_FLAGGED_FRAMES} keyframes and at least "
    f"{MIN_FLAGGED_SHARE:.0%} of those analysed are flagged; hand-chosen, not calibrated"
)


@dataclass
class _Frame:
    index: int
    timestamp: float
    reason: str


def _histogram(frame: np.ndarray) -> np.ndarray:
    small = cv2.resize(frame, (160, max(1, int(160 * frame.shape[0] / frame.shape[1]))))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [16, 8, 8], [0, 180, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()


def scan_shots(path: Path) -> dict[str, Any]:
    """Cuts and shots from frames sampled at up to ``SCAN_FPS``."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("unable to open video")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, round(fps / SCAN_FPS))
        if total // step > MAX_SCAN_FRAMES:
            step = max(1, total // MAX_SCAN_FRAMES)
        indices, distances = [], []
        previous = None
        index = 0
        while True:
            ok = cap.grab()
            if not ok:
                break
            if index % step == 0:
                ok, frame = cap.retrieve()
                if ok:
                    hist = _histogram(frame)
                    if previous is not None:
                        distances.append(
                            float(cv2.compareHist(previous, hist, cv2.HISTCMP_BHATTACHARYYA))
                        )
                    indices.append(index)
                    previous = hist
            index += 1
    finally:
        cap.release()

    total = max(total, index)
    cuts: list[dict[str, Any]] = []
    if distances:
        d = np.asarray(distances)
        median = float(np.median(d))
        mad = float(np.median(np.abs(d - median))) or 1e-3
        threshold = max(CUT_MIN_DISTANCE, median + CUT_MAD_FACTOR * mad)
        for i, value in enumerate(distances):
            if value >= threshold:
                before, after = indices[i], indices[i + 1]
                cuts.append({
                    "frame_before": before,
                    "frame_after": after,
                    "timestamp": round(after / fps, 3),
                    "distance": round(value, 4),
                })
    else:
        threshold = None

    bounds = [0] + [c["frame_after"] for c in cuts] + [total]
    shots = [
        {"start_frame": a, "end_frame": max(a, b - 1),
         "start": round(a / fps, 3), "end": round(b / fps, 3)}
        for a, b in zip(bounds[:-1], bounds[1:]) if b > a
    ]
    short = [
        s for s in shots[1:-1] if (s["end"] - s["start"]) < SHORT_SHOT_SECONDS
    ]
    return {
        "fps": fps,
        "total_frames": total,
        "scan_step": step,
        "scanned_frames": len(indices),
        "cut_threshold": round(threshold, 4) if threshold is not None else None,
        "cuts": cuts,
        "shots": shots,
        "short_shots": short,
    }


def select_keyframes(scan: dict[str, Any]) -> list[_Frame]:
    """Shot midpoints, then the frames either side of each cut, capped."""
    fps = scan["fps"]
    chosen: dict[int, str] = {}
    for shot in scan["shots"]:
        chosen.setdefault((shot["start_frame"] + shot["end_frame"]) // 2, "shot_midpoint")
    for cut in scan["cuts"]:
        chosen.setdefault(cut["frame_before"], "before_cut")
        chosen.setdefault(cut["frame_after"], "after_cut")
    if len(chosen) < 3 and scan["total_frames"] > 0:
        # A single long shot: spread a few frames over it instead of one.
        for q in (0.1, 0.3, 0.5, 0.7, 0.9):
            chosen.setdefault(int(q * (scan["total_frames"] - 1)), "uniform")
    frames = sorted(chosen.items())
    if len(frames) > MAX_KEYFRAMES:
        # Keep every cut-adjacent frame first; thin the rest evenly.
        priority = [f for f in frames if f[1] != "shot_midpoint" and f[1] != "uniform"]
        rest = [f for f in frames if f not in priority]
        room = max(0, MAX_KEYFRAMES - len(priority))
        if room and rest:
            picks = np.linspace(0, len(rest) - 1, num=min(room, len(rest))).round().astype(int)
            rest = [rest[i] for i in sorted(set(picks))]
        else:
            rest = []
        frames = sorted(priority[:MAX_KEYFRAMES] + rest)
    return [_Frame(i, round(i / fps, 3), why) for i, why in frames]


def _write_frames(path: Path, frames: list[_Frame], out_dir: Path) -> dict[int, Path]:
    cap = cv2.VideoCapture(str(path))
    written: dict[int, Path] = {}
    try:
        for frame in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame.index)
            ok, pixels = cap.read()
            if not ok:
                continue
            target = out_dir / f"frame_{frame.index:07d}.png"
            ok, encoded = cv2.imencode(".png", pixels)  # lossless: no new JPEG traces
            if ok:
                target.write_bytes(encoded.tobytes())
                written[frame.index] = target
    finally:
        cap.release()
    return written


#: The image detector each keyframe detector runs. A keyframe result is that
#: detector's output on a frame, so any policy set for the image detector (an
#: advisory one never votes) holds for its keyframe results too.
KEYFRAME_SOURCE = {
    "VideoKeyframeTruFor": "TruForDetector",
    "VideoKeyframeUnivFD": "UnivFDDetector",
}


def with_keyframe_variants(detectors) -> set[str]:
    """The given detector names plus the keyframe detectors that run them."""
    names = set(detectors or ())
    return names | {k for k, source in KEYFRAME_SOURCE.items() if source in names}


class VideoKeyframeAnalyzer:
    """
    Runs learned image detectors on a video's keyframes.

    ``detectors`` maps an output detector name to the image detector instance,
    e.g. ``{"VideoKeyframeTruFor": trufor, "VideoKeyframeUnivFD": univfd}``.
    """

    version = "1.0.0"

    def __init__(self, detectors: dict[str, Any]):
        self.detectors = {name: d for name, d in detectors.items() if d is not None}

    def analyse(self, path: Path) -> dict[str, Any]:
        scan = scan_shots(path)
        frames = select_keyframes(scan)
        coverage = {
            "keyframes_selected": len(frames),
            "total_frames": scan["total_frames"],
            "share_of_frames": round(len(frames) / max(1, scan["total_frames"]), 4),
            "statement": (
                f"{len(frames)} of {scan['total_frames']} frames were examined by the "
                "learned detectors. Frames between them were not, so this result "
                "says nothing about them."
            ),
        }
        temporal = {k: scan[k] for k in ("cuts", "short_shots", "cut_threshold", "scan_step", "scanned_frames")}

        detections: list[DetectionResult] = []
        with tempfile.TemporaryDirectory(prefix="aimd-keyframes-") as tmp:
            written = _write_frames(path, frames, Path(tmp))
            for name, detector in self.detectors.items():
                detections.append(
                    self._run_on_frames(name, detector, frames, written, coverage, temporal)
                )
        return {"detections": detections, "shot_scan": scan, "coverage": coverage}

    def _run_on_frames(self, name, detector, frames, written, coverage, temporal) -> DetectionResult:
        family_ai = getattr(detector, "manipulation_type", "") == "ai_generation"
        version = f"{self.version}+{getattr(detector, 'version', 'unknown')}"
        common = dict(
            detector=name, detector_version=version, media_type="video",
            manipulation_type=getattr(detector, "manipulation_type", "unknown"),
        )

        availability, reason = detector.availability()
        if availability is DetectorAvailability.UNAVAILABLE:
            return DetectionResult(
                **common, manipulation_detected=False, confidence=0.0,
                availability=DetectorAvailability.UNAVAILABLE,
                execution_status=DetectorExecutionStatus.SKIPPED,
                score_semantics=ScoreSemantics.NOT_APPLICABLE,
                status_reason=reason, metadata={"coverage": coverage},
            )

        per_frame: list[dict[str, Any]] = []
        regions: list[DetectionRegion] = []
        failures = 0
        for frame in frames:
            frame_path = written.get(frame.index)
            if frame_path is None:
                failures += 1
                per_frame.append({"frame_index": frame.index, "timestamp": frame.timestamp,
                                  "status": "UNREADABLE"})
                continue
            result = run_detector(
                detector, frame_path,
                timeout_s=settings.detector_timeout_seconds,
                retry_on_memory_error=settings.detector_retry_on_memory_error,
            )
            entry = {
                "frame_index": frame.index,
                "timestamp": frame.timestamp,
                "selected_because": frame.reason,
                "status": result.execution_status.value,
                "score": result.confidence if result.produced_observations else None,
                "score_semantics": result.score_semantics.value,
                "flagged": bool(result.manipulation_detected) and result.supports_determination,
            }
            if result.execution_status is DetectorExecutionStatus.FAILED:
                failures += 1
                entry["reason"] = result.status_reason
            per_frame.append(entry)
            if entry["flagged"]:
                for r in result.regions or [DetectionRegion(type=result.manipulation_type, confidence=result.confidence)]:
                    regions.append(DetectionRegion(
                        type=r.type, confidence=r.confidence, bbox=r.bbox, mask_path=r.mask_path,
                        start_frame=frame.index, end_frame=frame.index,
                        start_timestamp=frame.timestamp, end_timestamp=frame.timestamp,
                        extra={**(r.extra or {}), "selected_because": frame.reason},
                    ))

        analysed = [f for f in per_frame if f["status"] in ("COMPLETED", "INCONCLUSIVE")]
        flagged = [f for f in per_frame if f["flagged"]]
        scores = [f["score"] for f in analysed if f["score"] is not None]
        metadata = {
            "frames": per_frame,
            "frames_analysed": len(analysed),
            "frames_flagged": len(flagged),
            "frames_failed": failures,
            "flagged_timestamps": [f["timestamp"] for f in flagged],
            "peak_frame_score": max(scores) if scores else None,
            "coverage": coverage,
            "temporal": temporal,
            "decision_rule": _DECISION_RULE,
            "limitations": [
                coverage["statement"],
                "Each keyframe is analysed as a still image; motion and audio are not.",
                "The video-level flag rule is hand-chosen and has not been calibrated.",
            ],
        }
        if family_ai:
            metadata["threshold"] = getattr(detector, "threshold", None)

        if not analysed:
            return DetectionResult(
                **common, manipulation_detected=False, confidence=0.0,
                availability=DetectorAvailability.AVAILABLE,
                execution_status=DetectorExecutionStatus.FAILED,
                score_semantics=ScoreSemantics.NOT_APPLICABLE,
                status_reason=f"no keyframe could be analysed ({failures} failed)",
                metadata=metadata,
            )

        share = len(flagged) / len(analysed)
        semantics = ScoreSemantics(analysed[0]["score_semantics"]) if scores else ScoreSemantics.NOT_APPLICABLE
        meets_rule = len(flagged) >= MIN_FLAGGED_FRAMES and share >= MIN_FLAGGED_SHARE
        if meets_rule and not settings.video_keyframe_verdicts_enabled:
            # Measured: the rule does not separate manipulated from genuine
            # clips (see config.video_keyframe_verdicts_enabled). The frames
            # stay in the record for an examiner; the verdict does not move.
            return DetectionResult(
                **common, manipulation_detected=False, confidence=0.0,
                availability=DetectorAvailability.AVAILABLE,
                execution_status=DetectorExecutionStatus.INCONCLUSIVE,
                score_semantics=ScoreSemantics.NOT_APPLICABLE,
                status_reason=(
                    f"{len(flagged)} of {len(analysed)} keyframes were flagged (at "
                    + ", ".join(f"{t}s" for t in metadata["flagged_timestamps"][:8])
                    + "); shown for review. Keyframe flags do not set the video "
                    "verdict: in evaluation they flagged genuine clips as often as "
                    "manipulated ones"
                ),
                regions=regions[:30], metadata=metadata,
            )
        if meets_rule:
            return DetectionResult(
                **common, manipulation_detected=True,
                confidence=float(max(f["score"] for f in flagged)),
                availability=DetectorAvailability.AVAILABLE,
                execution_status=DetectorExecutionStatus.COMPLETED,
                score_semantics=semantics,
                assessment=Assessment.LIKELY_AI_GENERATED if family_ai else Assessment.SUSPICIOUS,
                regions=regions[:30], metadata=metadata,
            )
        why = (
            f"{len(flagged)} of {len(analysed)} keyframes were flagged, below the "
            "video-level rule" if flagged else
            f"none of the {len(analysed)} keyframes was flagged"
        )
        return DetectionResult(
            **common, manipulation_detected=False, confidence=0.0,
            availability=DetectorAvailability.AVAILABLE,
            execution_status=DetectorExecutionStatus.INCONCLUSIVE,
            score_semantics=ScoreSemantics.NOT_APPLICABLE,
            status_reason=f"{why}; frames between keyframes were not examined, so "
                          "this is not a finding about the whole video",
            # Flagged frames below the video-level rule stay visible.
            regions=regions[:30], metadata=metadata,
        )
