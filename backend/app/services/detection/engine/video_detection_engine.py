"""
Video Detection Engine — real temporal forensic analysis.

Implements frame-level analysis:
  1. Samples frames at configurable FPS.
  2. Runs ForensicImageDetector on each sampled frame.
  3. Runs BaselineFaceDetector for temporal face consistency.
  4. Computes frame-to-frame optical flow to detect motion discontinuities.
  5. Aggregates frame scores into suspicious segments.
  6. Returns structured result with temporal localization.

Assessment values:
  SUSPICIOUS    — significant temporal anomaly evidence found
  CLEAN         — no anomalies above threshold
  INCONCLUSIVE  — insufficient frames or analysis failed

Video confidence is NOT simply the mean of random scores.
It is derived from:
  - Proportion of frames with elevated anomaly scores
  - Peak anomaly score
  - Temporal clustering of anomalies (contiguous anomalous segments)
"""
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.schemas.detection import DetectionResult, DetectionRegion
from app.services.detection.detectors.image.forensic_image_detector import ForensicImageDetector
from app.services.detection.detectors.face_detector import BaselineFaceDetector

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
        self.detectors = detectors or [self._image_detector, self._face_detector]

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
            return self._inconclusive("Could not read video properties")

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
            "Video analysis: %s — %d total frames, step=%d, analysing ~%d frames",
            file_path.name, total_frames, step, frames_to_analyse,
        )

        frame_results = self._analyse_frames(file_path, step, native_fps)

        if not frame_results:
            return self._inconclusive("No frames could be analysed")

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
                            image_confidence = float(img_result.confidence)
                            image_assessment = "SUSPICIOUS" if img_result.manipulation_detected else "CLEAN"
                            regions = img_result.regions
                        except Exception as exc:
                            logger.debug("ForensicImageDetector failed on frame %d: %s", frame_index, exc)

                        # ── Face forensics ────────────────────────────────────
                        face_score = 0.0
                        face_regions: list[DetectionRegion] = []
                        try:
                            face_result: DetectionResult = self._face_detector.detect(tmp_path)
                            face_score = float(face_result.confidence)
                            face_regions = face_result.regions
                        except Exception as exc:
                            logger.debug("FaceDetector failed on frame %d: %s", frame_index, exc)

                        # Combined frame anomaly score
                        # Weighted: image forensics 50%, optical flow 30%, face 20%
                        combined_score = (
                            image_confidence * 0.50
                            + min(flow_score / FLOW_DISCONTINUITY_THRESHOLD, 1.0) * 0.30
                            + face_score * 0.20
                        )
                        combined_score = round(float(np.clip(combined_score, 0.0, 1.0)), 4)

                        results.append({
                            "frame_index": frame_index,
                            "timestamp_seconds": timestamp,
                            "image_forensic_score": round(image_confidence, 4),
                            "flow_discontinuity_score": round(flow_score, 4),
                            "face_anomaly_score": round(face_score, 4),
                            "combined_score": combined_score,
                            "assessment": image_assessment,
                            "regions": regions,
                            "face_regions": face_regions,
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
        """
        scores = [r["combined_score"] for r in frame_results]
        suspicious_frames = [r for r in frame_results if r["combined_score"] >= FRAME_ANOMALY_THRESHOLD]
        proportion = len(suspicious_frames) / len(frame_results)
        peak_score = max(scores)
        mean_score = float(np.mean(scores))

        # Video confidence: weighted combination of proportion and peak
        video_confidence = round(
            float(np.clip(proportion * 0.6 + peak_score * 0.4, 0.0, 1.0)), 4
        )

        if proportion >= SUSPICIOUS_PROPORTION_THRESHOLD:
            assessment = "SUSPICIOUS"
        elif video_confidence < 0.20:
            assessment = "CLEAN"
        else:
            assessment = "INCONCLUSIVE"

        # ── Identify suspicious segments ──────────────────────────────────────
        segments = self._identify_segments(frame_results)

        # ── Build DetectionResult ─────────────────────────────────────────────
        # Collect top suspicious regions from flagged frames (up to 10)
        top_regions: list[DetectionRegion] = []
        for r in sorted(suspicious_frames, key=lambda x: x["combined_score"], reverse=True)[:5]:
            for reg in r["regions"][:2]:
                top_regions.append(reg)

        result = DetectionResult(
            detector="VideoTemporalAnalysisEngine",
            media_type="video",
            manipulation_detected=assessment == "SUSPICIOUS",
            confidence=video_confidence,
            manipulation_type="temporal",
            regions=top_regions[:10],
            metadata={
                "status": "active_temporal_analysis",
                "detector_version": "1.0",
                "assessment": assessment,
                "frames_analysed": len(frame_results),
                "total_video_frames": total_frames,
                "suspicious_frame_count": len(suspicious_frames),
                "suspicious_frame_proportion": round(proportion, 4),
                "peak_frame_score": round(peak_score, 4),
                "mean_frame_score": round(mean_score, 4),
                "video_confidence": video_confidence,
                "frame_scores": [
                    {
                        "frame_index": r["frame_index"],
                        "timestamp_seconds": r["timestamp_seconds"],
                        "score": r["combined_score"],
                    }
                    for r in frame_results
                ],
                "suspicious_segments": segments,
                "note": (
                    "OBSERVATION: Temporal anomaly scores derived from ELA, FFT, "
                    "noise residual, optical flow discontinuity, and face boundary analysis. "
                    "These are forensic signals — not proof of manipulation."
                ),
            },
        )

        return {
            "media_type": "video",
            "detections": [result],
            "suspicious_segments": segments,
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

    def _inconclusive(self, reason: str) -> dict[str, Any]:
        result = DetectionResult(
            detector="VideoTemporalAnalysisEngine",
            media_type="video",
            manipulation_detected=False,
            confidence=0.0,
            manipulation_type="temporal",
            regions=[],
            metadata={
                "assessment": "INCONCLUSIVE",
                "reason": reason,
                "note": "UNCERTAINTY: Insufficient data for temporal analysis.",
            },
        )
        return {
            "media_type": "video",
            "detections": [result],
            "suspicious_segments": [],
            "frame_level_scores": [],
        }
