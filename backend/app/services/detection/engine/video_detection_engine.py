"""
Video Detection Engine -- real temporal forensic analysis with spatial-temporal tracking.

Implements frame-level analysis:
  1. Samples frames at configurable FPS.
  2. Runs ForensicImageDetector on each sampled frame.
  3. Runs BaselineFaceDetector for temporal face consistency.
  4. Computes frame-to-frame optical flow to detect motion discontinuities.
  5. Tracks suspicious regions across frames using spatial-temporal tracker.
  6. Aggregates frame scores into suspicious segments.
  7. Returns structured result with temporal AND spatial localization.

Assessment values:
  SUSPICIOUS                      -- anomalies on a material proportion of frames
  NO_SIGNIFICANT_ANOMALIES_DETECTED
                                  -- every sampled frame was analysed by every
                                     frame detector and none exceeded threshold.
                                     Records what was looked for and not found;
                                     NOT a finding of authenticity.
  INCONCLUSIVE                    -- insufficient frames, incomplete coverage, or
                                     a result the thresholds cannot classify

WHY THERE IS NO "CLEAN FROM A LOW SCORE" BRANCH
-----------------------------------------------
``video_confidence`` used to drive a negative conclusion directly:

    elif video_confidence < 0.20: assessment = "CLEAN"

That is invalid. ``video_confidence`` blends the *proportion* of anomalous frames
with the *peak* frame score -- a count and a magnitude, which are not
measurements of one quantity -- and each frame's own ``combined_score`` is itself
a blend of ELA/FFT/noise severity, optical-flow magnitude and a face-boundary
heuristic. A low value on that composite does not mean "no anomalies"; it can
equally mean "a short clip with one mild elevation". The negative statement is
now made from *coverage* instead: it requires that every sampled frame was
analysed by every frame detector with no failures, which is a fact about what
ran rather than a threshold on a number with no referent.

The composite is still reported, under its own name, with HEURISTIC_SEVERITY
semantics. Decomposing it into attributed per-signal scores is fusion-layer work
(see ``ImageDetectionEngine`` for the shape that work takes).
"""
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.schemas.detection import (
    Assessment,
    DetectionRegion,
    DetectionResult,
    DetectorAvailability,
    DetectorExecutionStatus,
    ScoreSemantics,
)
from app.services.detection.detectors.image.forensic_image_detector import ForensicImageDetector
from app.services.detection.detectors.face_detector import BaselineFaceDetector
from app.services.detection.engine.video_spatial_tracker import VideoSpatialTemporalTracker, TemporalAnomalyDetector

logger = logging.getLogger(__name__)

# Threshold above which a frame is considered anomalous
FRAME_ANOMALY_THRESHOLD = 0.45
# Minimum proportion of suspicious frames to flag video as SUSPICIOUS
SUSPICIOUS_PROPORTION_THRESHOLD = 0.20
# Optical flow magnitude above which a motion discontinuity is flagged
FLOW_DISCONTINUITY_THRESHOLD = 8.0
# Maximum frames to analyse (to limit processing time)
MAX_ANALYSIS_FRAMES = 120


class VideoDetectionEngine:

    def __init__(self, detectors=None):
        self._image_detector = ForensicImageDetector()
        self._face_detector = BaselineFaceDetector()
        self._spatial_tracker = VideoSpatialTemporalTracker()
        self._temporal_anomaly_detector = TemporalAnomalyDetector()
        self.detectors = detectors or [self._image_detector, self._face_detector]
        # Per-video counters for UNCERTAINTY evidence; reset by _analyse_frames.
        self._reset_counters()

    def _reset_counters(self) -> None:
        # The engine is a long-lived singleton. These counters used to be set
        # once in __init__ and never reset, so one bad video made every later
        # video report failures it never had.
        self._image_detector_failures = 0
        self._face_detector_failures = 0
        # Frames on which a detector ran but could not support a determination.
        self._image_detector_undetermined = 0
        self._face_detector_undetermined = 0

    def detect(self, file_path: Path | str) -> dict[str, Any]:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        cap = cv2.VideoCapture(str(file_path))
        if not cap.isOpened():
            raise ValueError(f"Unable to open video: {file_path}")

        try:
            native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()

        if total_frames == 0 or native_fps == 0:
            return self._failed(
                "Could not read video properties (frame count or FPS reported "
                "as zero); the container opened but carries no readable video "
                "stream metadata"
            )

        # Sample at ~1 FPS or native if short video
        analysis_fps = min(1.0, native_fps)
        step = max(1, round(native_fps / analysis_fps))

        # Cap total frames analysed
        frames_to_analyse = min(
            MAX_ANALYSIS_FRAMES,
            max(1, total_frames // step),
        )
        # Recalculate step to respect cap
        if frames_to_analyse < total_frames // step:
            step = max(1, total_frames // frames_to_analyse)

        logger.info(
            "Video analysis: %s -- %d total frames, step=%d, analysing ~%d frames",
            file_path.name, total_frames, step, frames_to_analyse,
        )

        frame_results = self._analyse_frames(file_path, step, native_fps)

        if not frame_results:
            return self._failed(
                "No frames could be decoded and analysed; nothing was examined"
            )

        return self._aggregate(frame_results, native_fps, total_frames)

    # ── Frame-level analysis ──────────────────────────────────────────────────

    def _analyse_frames(
        self,
        file_path: Path,
        step: int,
        native_fps: float,
    ) -> list[dict[str, Any]]:
        """
        Extract and analyse frames. Returns list of per-frame result dicts.
        """
        import tempfile, os

        cap = cv2.VideoCapture(str(file_path))
        if not cap.isOpened():
            return []

        results: list[dict] = []
        prev_gray: np.ndarray | None = None
        frame_index = 0
        analysed = 0

        # Reset per-video state
        self._spatial_tracker.reset()
        self._reset_counters()

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                while analysed < MAX_ANALYSIS_FRAMES:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_index % step == 0:
                        timestamp = round(frame_index / native_fps, 3)
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                        # ── Optical flow discontinuity ────────────────────────
                        flow_score = 0.0
                        if prev_gray is not None and prev_gray.shape == gray.shape:
                            flow_score = self._compute_flow_score(prev_gray, gray)
                        prev_gray = gray.copy()

                        # ── Forensic image analysis ───────────────────────────
                        # Write frame to temp file for detector
                        tmp_path = os.path.join(tmpdir, f"f_{frame_index:07d}.jpg")
                        cv2.imwrite(tmp_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 92])

                        image_confidence = 0.0
                        image_assessment = "INCONCLUSIVE"
                        regions: list[DetectionRegion] = []
                        try:
                            img_result: DetectionResult = self._image_detector.detect(tmp_path)
                            # A detector that returns FAILED/UNAVAILABLE instead of
                            # raising used to be read as a CLEAN frame.
                            if not img_result.produced_observations:
                                self._image_detector_failures += 1
                            else:
                                if img_result.execution_status is DetectorExecutionStatus.INCONCLUSIVE:
                                    self._image_detector_undetermined += 1
                                image_confidence = float(img_result.confidence)
                                image_assessment = img_result.assessment.value
                                regions = img_result.regions
                        except Exception as exc:
                            logger.warning("ForensicImageDetector failed on frame %d: %s", frame_index, exc, exc_info=True)
                            self._image_detector_failures += 1

                        # ── Face forensics ────────────────────────────────────
                        face_score = 0.0
                        face_regions: list[DetectionRegion] = []
                        try:
                            face_result: DetectionResult = self._face_detector.detect(tmp_path)
                            if not face_result.produced_observations:
                                self._face_detector_failures += 1
                            else:
                                if face_result.execution_status is DetectorExecutionStatus.INCONCLUSIVE:
                                    self._face_detector_undetermined += 1
                                face_score = float(face_result.confidence)
                                face_regions = face_result.regions
                        except Exception as exc:
                            logger.warning("FaceDetector failed on frame %d: %s", frame_index, exc, exc_info=True)
                            self._face_detector_failures += 1

                        # Combined frame anomaly score
                        # Weighted: image forensics 50%, optical flow 30%, face 20%
                        combined_score = (
                            image_confidence * 0.50
                            + min(flow_score / FLOW_DISCONTINUITY_THRESHOLD, 1.0) * 0.30
                            + face_score * 0.20
                        )
                        combined_score = round(float(np.clip(combined_score, 0.0, 1.0)), 4)

                        # Track regions spatially across frames
                        all_frame_regions = regions + face_regions
                        tracked_regions = self._spatial_tracker.process_frame(
                            frame, all_frame_regions, analysed  # Use analysed count as frame index for tracking
                        )

                        results.append({
                            "frame_index": frame_index,
                            "timestamp_seconds": timestamp,
                            "analysed_index": analysed,
                            "image_forensic_score": round(image_confidence, 4),
                            "flow_discontinuity_score": round(flow_score, 4),
                            "face_anomaly_score": round(face_score, 4),
                            "combined_score": combined_score,
                            "assessment": image_assessment,
                            "regions": regions,
                            "face_regions": face_regions,
                            "tracked_regions": tracked_regions,
                        })
                        analysed += 1

                    frame_index += 1

        finally:
            cap.release()

        return results

    def _compute_flow_score(
        self, prev_gray: np.ndarray, curr_gray: np.ndarray
    ) -> float:
        """
        Compute optical flow magnitude between two frames.
        High magnitude indicates a large sudden motion (possible cut/splice).
        Returns mean flow magnitude.
        """
        try:
            # Resize for speed
            h, w = prev_gray.shape
            scale = min(1.0, 320 / max(h, w))
            if scale < 1.0:
                size = (int(w * scale), int(h * scale))
                p = cv2.resize(prev_gray, size)
                c = cv2.resize(curr_gray, size)
            else:
                p, c = prev_gray, curr_gray

            flow = cv2.calcOpticalFlowFarneback(
                p, c, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )
            magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            return float(np.mean(magnitude))
        except Exception:
            return 0.0

    # ── Aggregation ───────────────────────────────────────────────────────────

    def _aggregate(
        self,
        frame_results: list[dict],
        native_fps: float,
        total_frames: int,
    ) -> dict[str, Any]:
        """
        Aggregate frame-level results into a video-level detection result.
        Identifies suspicious segments (contiguous anomalous frames).
        Includes spatial-temporal tracking results.
        """
        scores = [r["combined_score"] for r in frame_results]
        suspicious_frames = [r for r in frame_results if r["combined_score"] >= FRAME_ANOMALY_THRESHOLD]
        proportion = len(suspicious_frames) / len(frame_results)
        peak_score = max(scores)
        mean_score = float(np.mean(scores))

        # Get all tracked regions from all frames
        all_tracked_regions = []
        for r in frame_results:
            all_tracked_regions.extend(r.get("tracked_regions", []))

        # Analyze temporal anomalies in tracked regions
        temporal_anomalies = self._temporal_anomaly_detector.analyze_tracks(
            self._spatial_tracker.tracked_regions
        )

        # Temporal consistency of tracked regions. Reported, NOT added to the
        # anomaly score.
        #
        # This used to be a `tracked_confidence_boost` of up to +0.15 folded into
        # `video_confidence`. Two things were wrong with that. It added a measure
        # of *how stably a region was tracked* to a measure of *how anomalous the
        # frames were* -- different quantities, so the sum has no referent. And it
        # could only ever move the score upward, so every video with trackable
        # regions looked more anomalous than the frame evidence supported. A
        # stably tracked region is a reason to show an examiner where to look, not
        # evidence that something was manipulated.
        avg_temporal_consistency = (
            float(np.mean([r.temporal_consistency for r in all_tracked_regions]))
            if all_tracked_regions else 0.0
        )

        # Composite of a proportion and a peak. Kept for continuity of the
        # reported number and tagged HEURISTIC_SEVERITY; it no longer decides any
        # negative conclusion (see the module docstring).
        video_confidence = round(
            float(np.clip(proportion * 0.5 + peak_score * 0.3, 0.0, 1.0)), 4
        )

        # Calculate failure rates
        frames_analysed = len(frame_results)
        image_failure_rate = self._image_detector_failures / max(frames_analysed, 1)
        face_failure_rate = self._face_detector_failures / max(frames_analysed, 1)
        total_failure_rate = (self._image_detector_failures + self._face_detector_failures) / max(frames_analysed * 2, 1)

        # Assessment ladder, in order of precedence.
        #
        # Note the asymmetry, which is deliberate and matches the image engine: a
        # positive observation stands even when some frames failed (missing data
        # cannot retract an anomaly that was measured), but an *absence* of
        # anomalies does not stand when coverage was incomplete -- "no anomalies
        # found" cannot be reported for frames that were never analysed.
        if total_failure_rate > 0.5:
            # Over half the frame-detector invocations failed. Even a positive
            # proportion computed from the remainder is not representative.
            assessment = "INCONCLUSIVE"
        elif proportion >= SUSPICIOUS_PROPORTION_THRESHOLD:
            assessment = "SUSPICIOUS"
        elif self._image_detector_failures or self._face_detector_failures:
            assessment = "INCONCLUSIVE"
        elif suspicious_frames:
            # Some frames were anomalous, just not a material proportion of them.
            # Not a negative result.
            assessment = "INCONCLUSIVE"
        elif self._image_detector_undetermined or self._face_detector_undetermined:
            # A frame detector ran but could not support a determination (the
            # forensic heuristic is uncalibrated). Its silence is not "no
            # anomaly found".
            assessment = "INCONCLUSIVE"
        else:
            # Full coverage, every frame analysed by both frame detectors, and no
            # frame exceeded threshold. This is the strongest negative statement
            # the engine is entitled to make, and it is still not a finding of
            # authenticity: manipulation these detectors cannot measure, or
            # confined to frames between the sampled ones, would not appear here.
            assessment = "NO_SIGNIFICANT_ANOMALIES_DETECTED"

        # ── Identify suspicious segments ──────────────────────────────────────
        segments = self._identify_segments(frame_results)

        # ── Build tracked regions for final result ────────────────────────────
        # Get tracked regions from last analyzed frame
        last_frame_idx = len(frame_results) - 1 if frame_results else 0
        final_tracked_regions = self._spatial_tracker.get_tracked_regions_for_frame(last_frame_idx)

        # Collect top suspicious regions from flagged frames + tracked regions
        top_regions: list[DetectionRegion] = []
        for r in sorted(suspicious_frames, key=lambda x: x["combined_score"], reverse=True)[:5]:
            for reg in r["regions"][:2]:
                top_regions.append(reg)

        # Add top tracked regions
        for tr in sorted(final_tracked_regions, key=lambda x: x.confidence, reverse=True)[:5]:
            top_regions.append(tr)

        # Create failure info strings for UNCERTAINTY evidence
        image_failure_msg = f"image detector failed on {self._image_detector_failures} of {len(frame_results)} frames" if self._image_detector_failures > 0 else None
        face_failure_msg = f"face detector failed on {self._face_detector_failures} of {len(frame_results)} frames" if self._face_detector_failures > 0 else None

        result = DetectionResult(
            detector="VideoTemporalAnalysisEngine",
            detector_version="1.1",
            media_type="video",
            manipulation_detected=assessment == "SUSPICIOUS",
            confidence=video_confidence,
            manipulation_type="temporal",
            # Both are supplied: the status fields are authoritative, and the
            # engine's own label is more specific than the one that would be
            # derived from them (NO_SIGNIFICANT_ANOMALIES_DETECTED vs CLEAN).
            availability=DetectorAvailability.AVAILABLE,
            execution_status=(
                DetectorExecutionStatus.INCONCLUSIVE
                if assessment == "INCONCLUSIVE"
                else DetectorExecutionStatus.COMPLETED
            ),
            # A proportion blended with a peak, over per-frame scores that are
            # themselves blends of hand-thresholded signals. Ordinal, not a
            # probability.
            score_semantics=ScoreSemantics.HEURISTIC_SEVERITY,
            status_reason=(
                "; ".join(msg for msg in [image_failure_msg, face_failure_msg] if msg)
                or None
            ),
            assessment=Assessment(assessment),
            regions=top_regions[:15],
            metadata={
                "status": "active_temporal_analysis",
                "detector_version": "1.1",
                "assessment": assessment,
                "frames_analysed": len(frame_results),
                "total_video_frames": total_frames,
                "suspicious_frame_count": len(suspicious_frames),
                "suspicious_frame_proportion": round(proportion, 4),
                "peak_frame_score": round(peak_score, 4),
                "mean_frame_score": round(mean_score, 4),
                "video_confidence": video_confidence,
                "video_confidence_provenance": (
                    "0.5 * proportion of anomalous frames + 0.3 * peak frame "
                    "score; hand-chosen weights, not calibrated"
                ),
                # Reported separately because it is not an anomaly measure and is
                # deliberately excluded from the score above.
                "avg_tracked_temporal_consistency": round(avg_temporal_consistency, 4),
                "tracked_regions_count": len(all_tracked_regions),
                "temporal_anomalies": temporal_anomalies,
                "frame_scores": [
                    {
                        "frame_index": r["frame_index"],
                        "timestamp_seconds": r["timestamp_seconds"],
                        "score": r["combined_score"],
                    }
                    for r in frame_results
                ],
                "suspicious_segments": segments,
                "detector_failures": {
                    "image_detector_failures": self._image_detector_failures,
                    "face_detector_failures": self._face_detector_failures,
                    "image_detector_undetermined": self._image_detector_undetermined,
                    "face_detector_undetermined": self._face_detector_undetermined,
                    "total_frames_analyzed": len(frame_results)
                },
                "failure_messages": [msg for msg in [image_failure_msg, face_failure_msg] if msg is not None],
                "note": (
                    "OBSERVATION: Temporal anomaly scores derived from ELA, FFT, "
                    "noise residual, optical flow discontinuity, and face boundary analysis. "
                    "Spatial-temporal tracking applied for region consistency across frames. "
                    "These are forensic signals -- not proof of manipulation."
                ),
            },
        )

        return {
            "media_type": "video",
            "detections": [result],
            "suspicious_segments": segments,
            "temporal_anomalies": temporal_anomalies,
            "frame_level_scores": [
                {
                    "frame_index": r["frame_index"],
                    "timestamp_seconds": r["timestamp_seconds"],
                    "combined_score": r["combined_score"],
                }
                for r in frame_results
            ],
        }

    def _identify_segments(self, frame_results: list[dict]) -> list[dict]:
        """
        Identify contiguous runs of anomalous frames as suspicious segments.
        """
        segments: list[dict] = []
        in_segment = False
        seg_start: dict | None = None

        for r in frame_results:
            is_anomalous = r["combined_score"] >= FRAME_ANOMALY_THRESHOLD
            if is_anomalous and not in_segment:
                in_segment = True
                seg_start = r
            elif not is_anomalous and in_segment:
                in_segment = False
                if seg_start is not None:
                    segments.append({
                        "start_frame": seg_start["frame_index"],
                        "end_frame": r["frame_index"],
                        "start_timestamp": seg_start["timestamp_seconds"],
                        "end_timestamp": r["timestamp_seconds"],
                        "peak_score": max(
                            f["combined_score"]
                            for f in frame_results
                            if seg_start["frame_index"] <= f["frame_index"] <= r["frame_index"]
                        ),
                    })

        # Close any open segment at end of video
        if in_segment and seg_start is not None and frame_results:
            last = frame_results[-1]
            segments.append({
                "start_frame": seg_start["frame_index"],
                "end_frame": last["frame_index"],
                "start_timestamp": seg_start["timestamp_seconds"],
                "end_timestamp": last["timestamp_seconds"],
                "peak_score": max(
                    f["combined_score"]
                    for f in frame_results
                    if f["frame_index"] >= seg_start["frame_index"]
                ),
            })

        return segments

    def _non_result(
        self,
        execution_status: DetectorExecutionStatus,
        reason: str,
    ) -> dict[str, Any]:
        """
        A run that produced no observation, with the reason recorded on the field
        an examiner reads rather than buried in ``metadata``.

        ``confidence`` is 0.0 with NOT_APPLICABLE semantics so the zero cannot be
        read as "analysed, and found almost no anomaly".

        The predecessors of this helper passed neither ``assessment`` nor the
        status fields, so the schema derived AVAILABLE + COMPLETED + no anomaly
        and the result came back labelled **CLEAN**. A method named
        ``_inconclusive`` was reporting insufficient data as a negative finding;
        only ``metadata["assessment"]`` said otherwise, and nothing read it.
        Passing ``execution_status`` explicitly is what prevents that.
        """
        result = DetectionResult(
            detector="VideoTemporalAnalysisEngine",
            detector_version="1.1",
            media_type="video",
            manipulation_detected=False,
            confidence=0.0,
            manipulation_type="temporal",
            availability=DetectorAvailability.AVAILABLE,
            execution_status=execution_status,
            score_semantics=ScoreSemantics.NOT_APPLICABLE,
            status_reason=reason,
            regions=[],
            metadata={
                "status_reason": reason,
                "availability": DetectorAvailability.AVAILABLE.value,
                "execution_status": execution_status.value,
            },
        )
        return {
            "media_type": "video",
            "detections": [result],
            "suspicious_segments": [],
            "frame_level_scores": [],
        }

    # There is deliberately no ``_inconclusive`` helper. Every early return from
    # ``detect`` is a FAILED run -- a container that would not yield frames. The
    # engine's genuine inconclusive case is reached only after frames *were*
    # analysed, so it is expressed in ``_aggregate`` as a full result carrying
    # frame scores and regions, not as an empty non-result.

    def _failed(self, reason: str) -> dict[str, Any]:
        """
        Decoding or analysis did not complete.

        AVAILABLE + FAILED, not UNAVAILABLE: OpenCV is installed and opened the
        container, so the capability is present and the problem is this file or
        this code. Reporting UNAVAILABLE pointed an operator at their
        installation, and made an undecodable file indistinguishable from a
        missing dependency.
        """
        return self._non_result(DetectorExecutionStatus.FAILED, reason)
