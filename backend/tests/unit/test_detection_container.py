from app.services.detection.detection_container import (
    create_detection_router
)


def test_detection_container_creates_router():

    router=create_detection_router()

    assert router is not None
    assert router.image_engine is not None
    # Depending on config, trufor might be None, but other baseline detectors will be added.
    # We just check that detectors is a list.
    assert isinstance(router.image_engine.detectors, list)
    
    assert router.video_engine is not None
    assert isinstance(router.video_engine.detectors, list)