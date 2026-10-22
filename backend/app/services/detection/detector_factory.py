import logging

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
from app.services.detection.detectors.image.copy_move_detector import CopyMoveDetector
from app.services.detection.detectors.image.manipulation_mask_generator import ManipulationMaskGenerator
from app.services.detection.detectors.image.face_region_analyzer import FaceRegionAnalyzer

logger = logging.getLogger(__name__)

class DetectorFactory:

    @staticmethod
    def create_trufor():
        if not settings.trufor_enabled:
            return None

        if not settings.trufor_model_path:
            return TruForDetector(
                runner=None,
                initialization_error="TruFor model path is not configured",
            )

        try:
            runner = TruForRunner(settings.trufor_model_path, device=settings.device)
            runner.load()
            return TruForDetector(runner)
        except Exception as exc:
            logger.warning("TruFor initialization unavailable: %s", type(exc).__name__)
            return TruForDetector(
                runner=None,
                initialization_error=f"TruFor initialization failed: {type(exc).__name__}",
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

    @staticmethod
    def create_copy_move_detector():
        return CopyMoveDetector(detector_type="SIFT")

    @staticmethod
    def create_manipulation_mask_generator():
        return ManipulationMaskGenerator()

    @staticmethod
    def create_face_region_analyzer():
        return FaceRegionAnalyzer()