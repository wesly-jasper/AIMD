"""
Image Detection Engine — aggregates results from multiple image detectors.

Aggregation logic:
  - Runs all configured detectors.
  - Computes weighted composite confidence.
  - Determines overall assessment based on:
      * Any SUSPICIOUS detector result → overall SUSPICIOUS
      * All INCONCLUSIVE → overall INCONCLUSIVE
      * No manipulation signals → CLEAN
  - Records which detector drove the result (dominant_signal).
  - Collects all localization regions.

Assessment: SUSPICIOUS | CLEAN | INCONCLUSIVE | UNAVAILABLE
"""
import logging
from pathlib import Path
from typing import Any

from app.schemas.detection import Assessment, DetectionResult

logger = logging.getLogger(__name__)

# Detector weights for composite confidence scoring
_DETECTOR_WEIGHTS: dict[str, float] = {
    "ForensicImageDetector": 0.60,   # ELA + FFT + noise — most reliable baseline
    "BaselineFaceDetector": 0.25,
    "BaselineObjectDetector": 0.00,  # UNAVAILABLE — excluded from scoring
    "TruForDetector": 0.80,          # Model-based — high weight if enabled
}
_DEFAULT_WEIGHT = 0.30


class ImageDetectionEngine:

    def __init__(self, detectors=None):
        self.detectors = detectors or []

    def detect(self, file_path: str | Path) -> dict[str, Any]:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not self.detectors:
            raise RuntimeError("No image detectors are configured")

        results: list[DetectionResult] = []
        for detector in self.detectors:
            try:
                result = detector.detect(file_path)
            except Exception as exc:
                logger.warning("Detector %s failed: %s", type(detector).__name__, exc)
                results.append(DetectionResult(
                    detector=type(detector).__name__,
                    media_type="image",
                    manipulation_detected=False,
                    confidence=0.0,
                    manipulation_type="unknown",
                    assessment=Assessment.INCONCLUSIVE,
                    metadata={"error": str(exc), "assessment": "INCONCLUSIVE"},
                ))
                continue

            if not isinstance(result, DetectionResult):
                raise TypeError(
                    f"Detector {type(detector).__name__} must return DetectionResult"
                )
            results.append(result)

        aggregate = self._aggregate(results)

        return {
            "media_type": "image",
            "detections": results,
            "assessment": aggregate["assessment"],
            "overall_confidence": aggregate["overall_confidence"],
            "dominant_signal": aggregate["dominant_signal"],
            "all_regions": aggregate["all_regions"],
        }

    def _aggregate(self, results: list[DetectionResult]) -> dict[str, Any]:
        if not results:
            return {
                "assessment": "INCONCLUSIVE",
                "overall_confidence": 0.0,
                "dominant_signal": None,
                "all_regions": [],
            }

        # Filter out UNAVAILABLE detectors from scoring
        scorable = [
            r for r in results
            if r.metadata.get("assessment") not in ("UNAVAILABLE",)
               and r.metadata.get("status") not in ("no_model_configured", "unavailable")
        ]

        if not scorable:
            return {
                "assessment": "INCONCLUSIVE",
                "overall_confidence": 0.0,
                "dominant_signal": None,
                "all_regions": [],
            }

        # ── Weighted confidence ───────────────────────────────────────────────
        total_weight = 0.0
        weighted_confidence = 0.0
        dominant_signal: str | None = None
        max_confidence = 0.0

        for r in scorable:
            w = _DETECTOR_WEIGHTS.get(r.detector, _DEFAULT_WEIGHT)
            weighted_confidence += r.confidence * w
            total_weight += w
            if r.confidence > max_confidence:
                max_confidence = r.confidence
                dominant_signal = r.detector

        overall_confidence = round(
            (weighted_confidence / total_weight) if total_weight > 0 else 0.0, 4
        )

        # ── Assessment ───────────────────────────────────────────────────────
        any_suspicious = any(r.manipulation_detected for r in scorable)
        all_inconclusive = all(
            r.metadata.get("assessment") in ("INCONCLUSIVE", None)
            for r in scorable
        )

        if any_suspicious and overall_confidence >= 0.35:
            assessment = "SUSPICIOUS"
        elif all_inconclusive:
            assessment = "INCONCLUSIVE"
        elif overall_confidence < 0.20:
            assessment = "CLEAN"
        else:
            assessment = "INCONCLUSIVE"

        # ── Collect all regions ───────────────────────────────────────────────
        all_regions = []
        for r in results:
            all_regions.extend(r.regions)

        return {
            "assessment": assessment,
            "overall_confidence": overall_confidence,
            "dominant_signal": dominant_signal,
            "all_regions": all_regions,
        }