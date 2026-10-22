"""
Video Temporal Forensics Detector.

Analyzes video streams for temporal anomalies, motion discontinuities (optical flow),
and frame-level forensic inconsistencies.
"""
from pathlib import Path
from typing import Optional

import cv2  # noqa: F401  -- imported to establish decoding capability at import time

from app.schemas.detection import DetectionResult, DetectorAvailability
from app.services.detection.detectors.base_detector import BaseDetector


class BaseVideoDetector(BaseDetector):
    manipulation_type = "temporal"
    supported_media_types = frozenset({"video"})


class BaselineVideoDetector(BaseVideoDetector):
    """
    Forensic video temporal consistency detector.

    A thin wrapper over ``VideoDetectionEngine``: it computes nothing itself, it
    delegates and re-reports. That makes the identity question sharp, so it is
    answered explicitly -- the result carries this wrapper's identity *and*
    records the underlying detector that produced the numbers in
    ``metadata["underlying_detector"]``, so the record never implies that a
    measurement came from code that did not make it.
    """

    detector_id = "BaselineVideoDetector"
    version = "1.0.0"

    def availability(self) -> tuple[DetectorAvailability, Optional[str]]:
        """
        Whether the temporal engine can be constructed at all.

        Says nothing about whether any given file is decodable -- that is a
        property of the evidence and is reported per-run as FAILED, not as an
        absent capability.
        """
        try:
            from app.services.detection.engine.video_detection_engine import (  # noqa: F401
                VideoDetectionEngine,
            )
        except Exception as exc:
            return (
                DetectorAvailability.UNAVAILABLE,
                f"Video temporal engine could not be imported "
                f"({type(exc).__name__}: {exc})",
            )
        return DetectorAvailability.AVAILABLE, None

    def detect(self, file_path: str) -> DetectionResult:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        availability, reason = self.availability()
        if availability is not DetectorAvailability.AVAILABLE:
            return self.unavailable(reason or "Video temporal engine is unavailable")

        from app.services.detection.engine.video_detection_engine import (
            VideoDetectionEngine,
        )

        try:
            engine_res = VideoDetectionEngine().detect(path)
        except Exception as exc:
            # AVAILABLE + FAILED. The engine exists and the run broke, or the file
            # could not be decoded. Neither is an absent capability, and recording
            # it as UNAVAILABLE (as this did) blamed the installation for a problem
            # with this code or with the evidence.
            return self.failed(f"Video analysis failed: {type(exc).__name__}")

        detections = engine_res.get("detections", [])
        if not detections:
            # Nothing was examined: no frame was decoded, so there is no
            # observation in either direction. This used to be reported as
            # UNAVAILABLE, which described a decodable-file problem as a missing
            # capability and told an operator to fix their installation.
            return self.failed(
                "Video decoding produced no analysable frames; the stream "
                "contained no decodable video or the container is unsupported"
            )

        res = detections[0]

        # Propagate the underlying run's *status fields*, not just its derived
        # ``assessment`` label. Copying the label alone silently converted a
        # FAILED engine run into INCONCLUSIVE and discarded ``status_reason``,
        # so the reason a temporal analysis had not completed was lost here.
        return DetectionResult(
            detector=self.detector_id,
            detector_version=self.version,
            media_type="video",
            manipulation_detected=res.manipulation_detected,
            confidence=res.confidence,
            manipulation_type=self.manipulation_type,
            availability=res.availability,
            execution_status=res.execution_status,
            score_semantics=res.score_semantics,
            status_reason=res.status_reason,
            regions=res.regions,
            metadata={
                **res.metadata,
                # Which code actually produced the numbers above.
                "underlying_detector": res.detector,
                "underlying_detector_version": res.detector_version,
                "suspicious_segments": engine_res.get("suspicious_segments", []),
            },
        )
