import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.schemas.detection import Assessment, DetectionRegion, DetectionResult
from app.services.detection.detectors.base_detector import BaseDetector

logger = logging.getLogger(__name__)


class TruForDetector(BaseDetector):
    """
    TruFor Deep-Learning Forensic Detector.
    Integrates TruFor neural network weights for RGB-noise forgery detection,
    confidence estimation, and localization mask extraction.
    """

    def __init__(self, runner: Any):
        self.runner = runner

    def detect(self, file_path: str) -> DetectionResult:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        result = self.runner.predict(path)

        if "manipulation_detected" not in result or "confidence" not in result:
            raise KeyError("Invalid runner result: missing 'manipulation_detected' or 'confidence'")

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

        assessment_val = result.get("assessment")
        if assessment_val == "SUSPICIOUS" or result.get("manipulation_detected") is True:
            assessment = Assessment.SUSPICIOUS
        elif assessment_val == "CLEAN":
            assessment = Assessment.CLEAN
        elif assessment_val == "UNAVAILABLE":
            assessment = Assessment.UNAVAILABLE
        else:
            assessment = Assessment.INCONCLUSIVE

        return DetectionResult(
            detector="TruFor",
            media_type="image",
            manipulation_detected=result["manipulation_detected"],
            confidence=float(result["confidence"]),
            manipulation_type="image",
            assessment=assessment,
            regions=regions,
            metadata=result.get("metadata", {}),
        )
