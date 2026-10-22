import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.schemas.detection import (
    DetectionRegion,
    DetectionResult,
    DetectorAvailability,
    DetectorExecutionStatus,
    ScoreSemantics,
)
from app.services.detection.detectors.base_detector import BaseDetector

logger = logging.getLogger(__name__)


class TruForDetector(BaseDetector):
    """
    TruFor Deep-Learning Forensic Detector.
    Integrates TruFor neural network weights for RGB-noise forgery detection,
    confidence estimation, and localization mask extraction.

    This is the only detector in the suite whose score is a model output rather
    than a hand-thresholded signal heuristic, which is why it declares
    RAW_MODEL_SCORE semantics. Raw, not calibrated: no reliability curve has been
    fitted against a labelled corpus, so the number orders evidence and is not a
    probability that the image was manipulated.
    """

    #: Must match the key in the engine's ``_SIGNAL_FAMILY`` map and the class
    #: name. Results previously went out as ``"TruFor"`` while the map, the
    #: factory and the engine all said ``"TruForDetector"``, so the "learned"
    #: signal family never matched a single result.
    detector_id = "TruForDetector"
    version = "1.0.0"
    supported_media_types = frozenset({"image"})
    manipulation_type = "image_manipulation"
    score_semantics = ScoreSemantics.RAW_MODEL_SCORE

    def __init__(self, runner: Any | None, initialization_error: str | None = None):
        self.runner = runner
        self.initialization_error = initialization_error

    def availability(self) -> tuple[DetectorAvailability, Optional[str]]:
        """
        Whether the TruFor runner and its checkpoint are actually present.

        The checkpoint is not redistributable with this repository, so UNAVAILABLE
        is the normal state of this detector in a fresh checkout. That must be
        visible in the record: an absent learned detector means the "learned"
        signal family contributed nothing, and an assessment reached without it
        has narrower coverage than one reached with it.
        """
        if self.initialization_error:
            return DetectorAvailability.UNAVAILABLE, self.initialization_error
        if self.runner is None:
            return DetectorAvailability.UNAVAILABLE, "TruFor runner is unavailable"
        return DetectorAvailability.AVAILABLE, None

    def detect(self, file_path: str) -> DetectionResult:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        availability, reason = self.availability()
        if availability is not DetectorAvailability.AVAILABLE:
            return self.unavailable(reason or "TruFor runner is unavailable")

        try:
            result = self.runner.predict(path)
        except Exception as exc:
            logger.warning("TruFor inference failed: %s", exc)
            # AVAILABLE + FAILED. The checkpoint loaded and inference broke, which
            # is a defect to fix -- not the same thing as the checkpoint being
            # absent, and it must not be recorded as though it were.
            return self.failed(f"TruFor inference failed: {type(exc).__name__}")

        if "manipulation_detected" not in result or "confidence" not in result:
            # A malformed runner response cannot be guessed at. Raising here would
            # surface as an engine-level error with no detector record, so report a
            # failed run instead and keep the reason in the evidence.
            logger.warning("TruFor runner returned a malformed result: %r", sorted(result))
            return self.failed(
                "TruFor runner result missing 'manipulation_detected' or 'confidence'"
            )

        # Parse regions into DetectionRegion instances
        regions: List[DetectionRegion] = []
        raw_regions = result.get("regions", [])
        for r in raw_regions:
            if isinstance(r, DetectionRegion):
                regions.append(r)
            elif isinstance(r, dict):
                regions.append(
                    DetectionRegion(
                        type=r.get("type", "forgery_localization"),
                        confidence=r.get("confidence", result.get("confidence", 0.0)),
                        bbox=r.get("bbox"),
                        mask_path=r.get("mask_path"),
                    )
                )

        # The runner may report that it ran but could not reach a determination.
        # That is INCONCLUSIVE -- a completed run with no finding -- and is not the
        # same as a negative result. The old code derived this from an
        # ``assessment`` string the runner supplied, which let a runner declare
        # "CLEAN" or even "UNAVAILABLE" for a run that had demonstrably happened.
        declared = result.get("assessment")
        if declared == "INCONCLUSIVE" or (
            declared not in ("SUSPICIOUS", "CLEAN")
            and result.get("manipulation_detected") is not True
            and declared is not None
        ):
            return self.inconclusive(
                f"TruFor reported no determination (runner assessment: {declared!r})",
                regions=regions,
                metadata=result.get("metadata", {}),
            )

        return DetectionResult(
            detector=self.detector_id,
            detector_version=self.version,
            media_type="image",
            manipulation_detected=bool(result["manipulation_detected"]),
            confidence=float(result["confidence"]),
            manipulation_type=self.manipulation_type,
            availability=DetectorAvailability.AVAILABLE,
            execution_status=DetectorExecutionStatus.COMPLETED,
            score_semantics=self.score_semantics,
            regions=regions,
            metadata=result.get("metadata", {}),
        )
