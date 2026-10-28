import logging
from pathlib import Path

from app.core.config import settings
from app.services.detection.detectors.image.trufor_detector import (
    TruForDetector
)
from app.services.detection.models.trufor_runner import (
    TruForRunner
)
from app.services.detection.detectors.image.univfd_detector import UnivFDDetector
from app.services.detection.detectors.image.bfree_detector import BFreeDetector
from app.services.detection.models.bfree_runner import BFreeRunner
from app.services.detection.models.univfd_runner import UnivFDRunner
from app.services.detection.detectors.face_detector import BaselineFaceDetector
from app.services.detection.detectors.object_detector import (
    BaselineObjectDetector,
    YOLOv8ObjectDetector,
)
from app.services.detection.detectors.audio_detector import BaselineAudioAnalyzer
from app.services.detection.detectors.video_detector import BaselineVideoDetector
from app.services.detection.detectors.image.forensic_image_detector import ForensicImageDetector
from app.services.detection.detectors.image.copy_move_detector import CopyMoveDetector
from app.services.detection.detectors.image.manipulation_mask_generator import ManipulationMaskGenerator
from app.services.detection.detectors.image.face_region_analyzer import FaceRegionAnalyzer
from app.services.detection.runtime import memory_preflight

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

        if not Path(settings.trufor_model_path).exists():
            # Before the memory check: a missing install must be reported as
            # missing, not as whatever memory happened to be free at the time.
            return TruForDetector(runner=None, initialization_error="TruFor initialization failed: FileNotFoundError")

        refused = memory_preflight(settings.model_load_min_free_mb)
        if refused:
            logger.warning("TruFor not loaded: %s", refused)
            return TruForDetector(runner=None, initialization_error=f"TruFor not loaded: {refused}")

        try:
            runner = TruForRunner(
                settings.trufor_model_path,
                device=settings.device,
                tile_size=settings.trufor_tile_size,
                tile_overlap=settings.trufor_tile_overlap,
            )
            runner.load()
            return TruForDetector(runner)
        except Exception as exc:
            logger.warning("TruFor initialization unavailable: %s", type(exc).__name__)
            return TruForDetector(
                runner=None,
                initialization_error=f"TruFor initialization failed: {type(exc).__name__}",
            )

    @staticmethod
    def create_univfd():
        if not settings.univfd_enabled:
            return None

        if not settings.univfd_model_path:
            return UnivFDDetector(
                runner=None,
                initialization_error="UnivFD model path is not configured",
            )

        if not Path(settings.univfd_model_path).exists():
            # Before the memory check: a missing install must be reported as
            # missing, not as whatever memory happened to be free at the time.
            return UnivFDDetector(runner=None, initialization_error="UnivFD initialization failed: FileNotFoundError")

        refused = memory_preflight(settings.model_load_min_free_mb)
        if refused:
            logger.warning("UnivFD not loaded: %s", refused)
            return UnivFDDetector(runner=None, initialization_error=f"UnivFD not loaded: {refused}")

        try:
            runner = UnivFDRunner(settings.univfd_model_path, device=settings.device)
            runner.load()
            return UnivFDDetector(runner, threshold=settings.univfd_threshold)
        except Exception as exc:
            logger.warning("UnivFD initialization unavailable: %s", type(exc).__name__)
            return UnivFDDetector(
                runner=None,
                initialization_error=f"UnivFD initialization failed: {type(exc).__name__}",
            )

    @staticmethod
    def create_bfree():
        if not settings.bfree_enabled:
            return None

        if not settings.bfree_model_path:
            return BFreeDetector(runner=None, initialization_error="B-Free model path is not configured")

        if not Path(settings.bfree_model_path).exists():
            # Before the memory check: a missing install must be reported as
            # missing, not as whatever memory happened to be free at the time.
            return BFreeDetector(runner=None, initialization_error="B-Free initialization failed: FileNotFoundError")

        refused = memory_preflight(settings.model_load_min_free_mb)
        if refused:
            logger.warning("B-Free not loaded: %s", refused)
            return BFreeDetector(runner=None, initialization_error=f"B-Free not loaded: {refused}")

        try:
            runner = BFreeRunner(
                settings.bfree_model_path, device=settings.device,
                small_input_mode=settings.bfree_small_input_mode,
                small_input_logit_offset=settings.bfree_small_input_logit_offset,
            )
            runner.load()
            return BFreeDetector(runner, threshold=settings.bfree_threshold)
        except Exception as exc:
            logger.warning("B-Free initialization unavailable: %s", type(exc).__name__)
            return BFreeDetector(
                runner=None,
                initialization_error=f"B-Free initialization failed: {type(exc).__name__}",
            )

    @staticmethod
    def create_forensic_image_detector():
        return ForensicImageDetector()

    @staticmethod
    def create_face_detector():
        return BaselineFaceDetector()

    @staticmethod
    def create_object_detector():
        if not settings.object_detection_enabled:
            return BaselineObjectDetector()
        device = None if settings.device in ("", "auto") else settings.device
        return YOLOv8ObjectDetector(str(settings.object_detection_model_path), device=device)

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