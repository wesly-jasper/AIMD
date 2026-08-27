from app.schemas.detection import DetectionResult


class DetectionService:

    def __init__(self,detectors=None):
        self.detectors=detectors or []

    def analyze(self,file_path):
        results=[]

        for detector in self.detectors:
            result=detector.detect(file_path)

            if not isinstance(result,DetectionResult):
                raise TypeError(
                    "Detector must return DetectionResult"
                )

            results.append(result)

        return {
            "detections":results
        }