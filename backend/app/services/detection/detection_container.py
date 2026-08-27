"""
Detection container — wires detector factory into detection engines.

The container exposes a .router property for use by the pipeline.
"""
from app.services.detection.detector_factory import DetectorFactory
from app.services.detection.engine.image_detection_engine import ImageDetectionEngine
from app.services.detection.engine.video_detection_engine import VideoDetectionEngine
from app.services.detection.router.media_detection_router import MediaDetectionRouter


def create_detection_router() -> MediaDetectionRouter:
    """Create a fully-wired MediaDetectionRouter."""
    # Image detectors
    trufor_detector = DetectorFactory.create_trufor()
    forensic_image_detector = DetectorFactory.create_forensic_image_detector()
    face_detector = DetectorFactory.create_face_detector()
    object_detector = DetectorFactory.create_object_detector()

    image_detectors = [
        d for d in [trufor_detector, forensic_image_detector, face_detector, object_detector]
        if d is not None
    ]

    image_engine = ImageDetectionEngine(detectors=image_detectors)

    # Video engine handles its own detectors internally
    video_engine = VideoDetectionEngine()

    return MediaDetectionRouter(
        image_engine=image_engine,
        video_engine=video_engine,
    )


class DetectionContainer:
    """Singleton-style container for the detection router."""

    def __init__(self):
        self._router: MediaDetectionRouter | None = None

    @property
    def router(self) -> MediaDetectionRouter:
        if self._router is None:
            self._router = create_detection_router()
        return self._router