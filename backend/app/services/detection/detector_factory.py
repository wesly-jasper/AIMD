from app.core.config import settings
from app.services.detection.detectors.image.trufor_detector import (
    TruForDetector
)
from app.services.detection.models.trufor_runner import (
    TruForRunner
)
from app.services.detection.detectors.face_detector import BaselineFaceDetector
from app.services.detection.detectors.object_detector import BaselineObjectDetector
from app.services.detection.detectors.audio_detector import BaselineAudioAnalyzer
from app.services.detection.detectors.video_detector import BaselineVideoDetector


from app.services.detection.detectors.image.forensic_image_detector import ForensicImageDetector

class DetectorFactory:

    @staticmethod
    def create_trufor():

        if not settings.trufor_enabled:
            return None

        if not settings.trufor_model_path:
            raise ValueError(
                "TruFor is enabled but model path is not configured"
            )

        runner=TruForRunner(
            settings.trufor_model_path
        )

        runner.load()

        return TruForDetector(
            runner
        )

    @staticmethod
    def create_forensic_image_detector():
        return ForensicImageDetector()

    @staticmethod
    def create_face_detector():
        return BaselineFaceDetector()

    @staticmethod
    def create_object_detector():
        return BaselineObjectDetector()

    @staticmethod
    def create_audio_detector():
        return BaselineAudioAnalyzer()

    @staticmethod
    def create_video_detector():
        return BaselineVideoDetector()