from pathlib import Path
from app.schemas.detection import DetectionResult
from app.services.detection.detectors.base_detector import BaseDetector

class BaseVideoDetector(BaseDetector):
    pass

class BaselineVideoDetector(BaseVideoDetector):
    """
    Baseline implementation for video analysis.
    Placeholder for checking temporal inconsistencies.
    """
    def __init__(self):
        self.status = "baseline"

    def detect(self, file_path: str) -> DetectionResult:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        return DetectionResult(
            detector="BaselineVideoDetector",
            media_type="video",
            manipulation_detected=False,
            confidence=0.0,
            manipulation_type="temporal",
            regions=[],
            metadata={"status": self.status, "message": "Baseline video analysis used. No temporal inconsistencies detected."}
        )
