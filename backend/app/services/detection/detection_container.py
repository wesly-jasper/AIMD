from app.services.detection.detector_factory import DetectorFactory
from app.services.detection.engine.image_detection_engine import (
    ImageDetectionEngine
)
from app.services.detection.engine.video_detection_engine import (
    VideoDetectionEngine
)
from app.services.detection.router.media_detection_router import (
    MediaDetectionRouter
)


def create_detection_router():

    # Image Detectors
    trufor_detector=DetectorFactory.create_trufor()
    forensic_image_detector = DetectorFactory.create_forensic_image_detector()
    face_detector = DetectorFactory.create_face_detector()
    object_detector = DetectorFactory.create_object_detector()
    
    image_detectors = [d for d in [trufor_detector, forensic_image_detector, face_detector, object_detector] if d is not None]

    image_engine=ImageDetectionEngine(
        detectors=image_detectors
    )

    # Video Detectors
    video_detector = DetectorFactory.create_video_detector()
    audio_detector = DetectorFactory.create_audio_detector()

    video_detectors = [d for d in [video_detector, audio_detector] if d is not None]

    video_engine = VideoDetectionEngine(
        detectors=video_detectors
    )

    return MediaDetectionRouter(
        image_engine=image_engine,
        video_engine=video_engine
    )