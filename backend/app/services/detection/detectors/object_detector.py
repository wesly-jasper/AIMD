"""
Object detector.

The current implementation does not include a real object detection model.
AIMD does not pretend to detect manipulation through object detection
without an actual model.

Assessment: UNAVAILABLE

To add real object detection, implement a subclass of BaseObjectDetector
and register it in the DetectorFactory. Suitable options:
  - YOLOv8 (ultralytics) for general object detection
  - A fine-tuned model for detecting manipulation-specific objects

NOTE: Object detection and manipulation detection are different tasks.
      Do not conflate them. A model that detects "person, car, table"
      cannot determine whether an image has been manipulated.
"""
import logging
from pathlib import Path

from app.schemas.detection import DetectionResult, Assessment
from app.services.detection.detectors.base_detector import BaseDetector

logger = logging.getLogger(__name__)


class BaseObjectDetector(BaseDetector):
    pass


class BaselineObjectDetector(BaseObjectDetector):
    """
    Object detector placeholder.

    Returns UNAVAILABLE — no object detection model is configured.
    This is the honest response: we cannot determine manipulation
    from object presence/absence without an actual model.
    """

    DETECTOR_VERSION = "1.0-unavailable"

    def detect(self, file_path: str) -> DetectionResult:
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        return DetectionResult(
            detector="BaselineObjectDetector",
            media_type="image",
            manipulation_detected=False,
            confidence=0.0,
            manipulation_type="object",
            assessment=Assessment.UNAVAILABLE,
            regions=[],
            metadata={
                "assessment": "UNAVAILABLE",
                "status": "baseline",
                "detector_version": self.DETECTOR_VERSION,
                "note": (
                    "UNCERTAINTY: No object detection model is configured. "
                    "Object anomaly detection requires a trained model. "
                    "Configure a YOLOv8 or compatible detector via the plugin interface."
                ),
            },
        )
