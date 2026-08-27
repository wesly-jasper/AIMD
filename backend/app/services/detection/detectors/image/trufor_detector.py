from pathlib import Path

from app.schemas.detection import DetectionResult
from app.services.detection.detectors.base_detector import BaseDetector


class TruForDetector(BaseDetector):

    def __init__(self,runner):
        self.runner=runner

    def detect(self,file_path):
        file_path=Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"File not found: {file_path}"
            )

        result=self.runner.predict(
            file_path
        )

        return DetectionResult(
            detector="TruFor",
            media_type="image",
            manipulation_detected=result["manipulation_detected"],
            confidence=result["confidence"],
            manipulation_type="image",
            regions=result.get("regions",[]),
            metadata=result.get("metadata",{})
        )