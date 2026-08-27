from pathlib import Path

from app.schemas.detection import DetectionResult


class ImageDetectionEngine:

    def __init__(self, detectors=None):
        self.detectors = detectors or []

    def detect(self, file_path):
        file_path=Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"File not found: {file_path}"
            )

        if not self.detectors:
            raise RuntimeError(
                "No image detectors are configured"
            )

        results = []
        for detector in self.detectors:
            result = detector.detect(file_path)
            if not isinstance(result, DetectionResult):
                raise TypeError(
                    "Image detector must return DetectionResult"
                )
            results.append(result)

        # We return a single aggregated DetectionResult or just a dict with all results.
        # However, looking at the schema, we might want to return the first or aggregate them.
        # But wait, MediaDetectionRouter returns it directly. Let's return a dict with a list, or an aggregate.
        # Wait, the API schema might expect a specific return. Let's look at what MediaDetectionRouter returns to its caller.
        return {
            "media_type": "image",
            "detections": results
        }