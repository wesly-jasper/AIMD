import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.schemas.detection import (
    DetectionRegion,
    DetectionResult,
    DetectorAvailability,
    DetectorExecutionStatus,
    ScoreSemantics,
)
from app.services.detection.calibration import calibrate_score
from app.services.detection.detectors.base_detector import BaseDetector
from app.services.evidence.heatmap_artefacts import MASK_THRESHOLD, write_heatmap_artefacts

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
    #: Version of this wrapper. A loaded runner extends it per instance with the
    #: upstream commit and the checkpoint hash, so a result names the exact
    #: weights that produced it.
    version = "2.0.0"
    supported_media_types = frozenset({"image"})
    manipulation_type = "image_manipulation"
    score_semantics = ScoreSemantics.RAW_MODEL_SCORE

    def __init__(
        self,
        runner: Any | None,
        initialization_error: str | None = None,
        artefacts_dir: Optional[Path] = None,
    ):
        self.runner = runner
        self.initialization_error = initialization_error
        self.artefacts_dir = artefacts_dir
        tag = getattr(runner, "version_tag", None)
        if isinstance(tag, str) and tag:
            self.version = f"{type(self).version}+{tag}"

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

        started = time.perf_counter()
        try:
            result = self.runner.predict(path)
        except Exception as exc:
            logger.warning("TruFor inference failed", exc_info=True)
            # AVAILABLE + FAILED. The checkpoint loaded and inference broke, which
            # is a defect to fix -- not the same thing as the checkpoint being
            # absent, and it must not be recorded as though it were.
            return self.failed_from("TruFor inference failed", exc)

        if "manipulation_detected" not in result or "confidence" not in result:
            # A malformed runner response cannot be guessed at. Raising here would
            # surface as an engine-level error with no detector record, so report a
            # failed run instead and keep the reason in the evidence.
            logger.warning("TruFor runner returned a malformed result: %r", sorted(result))
            return self.failed(
                "TruFor runner result missing 'manipulation_detected' or 'confidence'"
            )

        metadata: Dict[str, Any] = dict(result.get("metadata", {}))
        # The maps are arrays: they become files, never metadata.
        localization_map = result.get("localization_map")
        reliability_map = result.get("reliability_map")
        mask_path: Optional[str] = None
        if localization_map is not None:
            try:
                artefacts = write_heatmap_artefacts(
                    path,
                    localization_map,
                    reliability_map,
                    producer="trufor",
                    model_tag=(metadata.get("checkpoint_sha256") or "unknown")[:12],
                    root=self.artefacts_dir,
                    # The cut the runner cut its regions at, so mask and regions agree.
                    mask_threshold=float(metadata.get("threshold", MASK_THRESHOLD)),
                )
            except Exception as exc:
                # The score is still a valid observation without its heatmap, so
                # the run stays COMPLETED -- but the missing evidence is recorded.
                logger.warning("TruFor heatmap could not be saved: %s", exc)
                metadata["artefacts_error"] = type(exc).__name__
            else:
                metadata["artefacts"] = artefacts
                mask_path = artefacts["mask"]["path"]

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
                        # Every region points at the full-image mask it is one
                        # area of; the bbox says where to look in it.
                        mask_path=r.get("mask_path") or mask_path,
                        extra=r.get("extra"),
                    )
                )

        metadata["execution_time_ms"] = round((time.perf_counter() - started) * 1000.0, 1)

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
                metadata=metadata,
            )

        confidence = float(result["confidence"])
        detected = bool(result["manipulation_detected"])
        semantics = self.score_semantics
        # A calibration fitted for exactly these weights turns the integrity
        # score into a probability with a recorded procedure, and replaces the
        # fixed 0.5 cut with the threshold chosen for its target FPR. Otherwise
        # the score stays RAW, and a version mismatch is stated, not hidden.
        calibration = calibrate_score(self.detector_id, self.version, confidence)
        if calibration is not None:
            metadata["calibration"] = calibration
            if calibration["applied"]:
                confidence = calibration["probability"]
                detected = calibration["flagged"]
                semantics = ScoreSemantics.CALIBRATED_PROBABILITY

        return DetectionResult(
            detector=self.detector_id,
            detector_version=self.version,
            media_type="image",
            manipulation_detected=detected,
            confidence=confidence,
            manipulation_type=self.manipulation_type,
            availability=DetectorAvailability.AVAILABLE,
            execution_status=DetectorExecutionStatus.COMPLETED,
            score_semantics=semantics,
            regions=regions,
            metadata=metadata,
        )
