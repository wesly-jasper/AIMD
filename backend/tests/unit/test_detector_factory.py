from app.services.detection.detector_factory import (
    DetectorFactory
)


def test_trufor_disabled():

    result=DetectorFactory.create_trufor()

    assert result is None