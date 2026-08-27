from pathlib import Path
from app.schemas.detection import DetectionResult

class VideoDetectionEngine:

    def __init__(self, detectors=None):
        self.detectors = detectors or []

    def detect(self, file_path):
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"File not found: {file_path}"
            )

        if not self.detectors:
            raise RuntimeError(
                "No video detectors are configured"
            )

        results = []
        for detector in self.detectors:
            result = detector.detect(file_path)
            if not isinstance(result, DetectionResult):
                raise TypeError(
                    "Video detector must return DetectionResult"
                )
            results.append(result)

        return {
            "media_type": "video",
            "detections": results
        }
