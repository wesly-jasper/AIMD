"""
Detection container -- wires detector factory into detection engines.

The container exposes a .router property for use by the pipeline.
"""
import threading

from app.services.detection.detector_factory import DetectorFactory
from app.services.detection.engine.image_detection_engine import ImageDetectionEngine
from app.services.detection.engine.video_detection_engine import VideoDetectionEngine
from app.services.detection.engine.video_keyframes import VideoKeyframeAnalyzer
from app.services.detection.router.media_detection_router import MediaDetectionRouter
from app.services.detection.runtime import release_cached_memory


def create_detection_router() -> MediaDetectionRouter:
    """Create a fully-wired MediaDetectionRouter."""
    # Image detectors
    trufor_detector = DetectorFactory.create_trufor()
    univfd_detector = DetectorFactory.create_univfd()
    bfree_detector = DetectorFactory.create_bfree()
    forensic_image_detector = DetectorFactory.create_forensic_image_detector()
    face_detector = DetectorFactory.create_face_detector()
    object_detector = DetectorFactory.create_object_detector()
    copy_move_detector = DetectorFactory.create_copy_move_detector()
    manipulation_mask_generator = DetectorFactory.create_manipulation_mask_generator()
    face_region_analyzer = DetectorFactory.create_face_region_analyzer()

    image_detectors = [
        d for d in [
            trufor_detector,
            univfd_detector,
            bfree_detector,
            forensic_image_detector,
            face_detector,
            object_detector,
            copy_move_detector,
            manipulation_mask_generator,
            face_region_analyzer,
        ]
        if d is not None
    ]

    image_engine = ImageDetectionEngine(detectors=image_detectors)

    # Video engine handles its own heuristics internally; the learned models run
    # on its keyframes as the SAME instances the image engine holds, so no model
    # is loaded twice.
    video_engine = VideoDetectionEngine(
        keyframe_analyzer=VideoKeyframeAnalyzer({
            "VideoKeyframeTruFor": trufor_detector,
            "VideoKeyframeUnivFD": univfd_detector,
        }),
    )

    return MediaDetectionRouter(
        image_engine=image_engine,
        video_engine=video_engine,
    )


class DetectionContainer:
    """Lazily built detection router. Use ``get_detection_container()`` for the
    process-wide one: each container loads its own copy of every model."""

    def __init__(self):
        self._router: MediaDetectionRouter | None = None
        self._lock = threading.Lock()

    @property
    def router(self) -> MediaDetectionRouter:
        if self._router is None:
            with self._lock:
                if self._router is None:
                    self._router = create_detection_router()
        return self._router

    def reload(self) -> MediaDetectionRouter:
        """
        Build the detectors again, e.g. once memory has been freed for a model
        that was refused at startup. The old router is released first so its
        models are not held twice.
        """
        with self._lock:
            self._router = None
            release_cached_memory()
            self._router = create_detection_router()
            return self._router


_shared: DetectionContainer | None = None
_shared_lock = threading.Lock()


def get_detection_container() -> DetectionContainer:
    """
    The one container per process.

    ``/health`` used to build a second router of its own beside the pipeline's,
    which loaded TruFor a second time into the same process -- on a 16 GB host
    that was enough to push the loads into the memory pressure that crashed it.
    """
    global _shared
    if _shared is None:
        with _shared_lock:
            if _shared is None:
                _shared = DetectionContainer()
    return _shared