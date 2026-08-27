import pytest

from app.schemas.detection import DetectionResult
from app.services.detection.detectors.image.trufor_detector import (
    TruForDetector
)


class MockRunner:

    def predict(self,file_path):
        return {
            "manipulation_detected":True,
            "confidence":0.92,
            "regions":[],
            "metadata":{
                "score":0.92
            }
        }


class InvalidRunner:

    def predict(self,file_path):
        return {
            "invalid":"result"
        }


def test_trufor_detector_file_not_found():

    detector=TruForDetector(
        MockRunner()
    )

    with pytest.raises(FileNotFoundError):
        detector.detect(
            "missing.jpg"
        )


def test_trufor_detector(tmp_path):

    image=tmp_path/"test.jpg"
    image.write_bytes(b"test")

    detector=TruForDetector(
        MockRunner()
    )

    result=detector.detect(image)

    assert isinstance(
        result,
        DetectionResult
    )

    assert result.detector=="TruFor"
    assert result.media_type=="image"
    assert result.manipulation_detected is True
    assert result.confidence==0.92
    assert result.manipulation_type=="image"
    assert result.regions==[]
    assert result.metadata["score"]==0.92


def test_trufor_detector_invalid_runner_result(tmp_path):

    image=tmp_path/"test.jpg"
    image.write_bytes(b"test")

    detector=TruForDetector(
        InvalidRunner()
    )

    with pytest.raises(KeyError):
        detector.detect(image)