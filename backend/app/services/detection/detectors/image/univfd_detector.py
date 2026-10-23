import logging
import time
from pathlib import Path
from typing import Any, Optional

from app.schemas.detection import (
    Assessment,
    DetectionResult,
    DetectorAvailability,
    DetectorExecutionStatus,
    ScoreSemantics,
)
from app.services.detection.detectors.base_detector import BaseDetector
from app.services.detection.models.univfd_runner import LIMITATIONS, SEMANTICS

logger = logging.getLogger(__name__)


class UnivFDDetector(BaseDetector):
    """
    Whole-image AI-generation detector (UnivFD on CLIP ViT-L/14).

    Its own signal family, ``ai_generation``: it answers "was this image
    generated?", a different question from every other image detector's "was
    this image edited?". Its result is reported on its own branch of the
    assessment and never raises, lowers or blends into the manipulation verdict.
    """

    detector_id = "UnivFDDetector"
    version = "1.0.0"
    supported_media_types = frozenset({"image"})
    manipulation_type = "ai_generation"
    score_semantics = ScoreSemantics.RAW_MODEL_SCORE

    def __init__(
        self,
        runner: Any | None,
        threshold: float = 0.5,
        initialization_error: Optional[str] = None,
    ):
        self.runner = runner
        self.threshold = threshold
        self.initialization_error = initialization_error
        tag = getattr(runner, "version_tag", None)
        if isinstance(tag, str) and tag:
            self.version = f"{type(self).version}+{tag}"

    def availability(self) -> tuple[DetectorAvailability, Optional[str]]:
        if self.initialization_error:
            return DetectorAvailability.UNAVAILABLE, self.initialization_error
        if self.runner is None:
            return DetectorAvailability.UNAVAILABLE, "UnivFD runner is unavailable"
        return DetectorAvailability.AVAILABLE, None

    def detect(self, file_path: str | Path) -> DetectionResult:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        availability, reason = self.availability()
        if availability is not DetectorAvailability.AVAILABLE:
            return self.unavailable(reason or "UnivFD runner is unavailable")

        started = time.perf_counter()
        try:
            result = self.runner.predict(path)
            score = float(result["score"])
            logit = float(result["logit"])
        except Exception as exc:
            logger.warning("UnivFD inference failed: %s", exc)
            return self.failed(f"UnivFD inference failed: {type(exc).__name__}")

        metadata = {
            **dict(result.get("metadata", {})),
            "branch": "ai_generation",
            "logit": round(logit, 6),
            "threshold": self.threshold,
            "semantics": SEMANTICS,
            "limitations": list(LIMITATIONS),
            "execution_time_ms": round((time.perf_counter() - started) * 1000.0, 1),
        }
        indicated = score >= self.threshold
        return DetectionResult(
            detector=self.detector_id,
            detector_version=self.version,
            media_type="image",
            # ``manipulation_detected`` carries "generation indicated" here; the
            # family map keeps it out of the manipulation verdict.
            manipulation_detected=indicated,
            # Set explicitly so the stored label names the question asked,
            # rather than the derived SUSPICIOUS / CLEAN of an editing detector.
            assessment=(
                Assessment.LIKELY_AI_GENERATED if indicated
                else Assessment.NO_SIGNIFICANT_ANOMALIES_DETECTED
            ),
            confidence=round(min(max(score, 0.0), 1.0), 6),
            manipulation_type=self.manipulation_type,
            availability=DetectorAvailability.AVAILABLE,
            execution_status=DetectorExecutionStatus.COMPLETED,
            score_semantics=self.score_semantics,
            metadata=metadata,
        )
