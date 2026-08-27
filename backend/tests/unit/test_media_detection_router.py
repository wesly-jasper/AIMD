import pytest

from app.services.detection.router.media_detection_router import (
    MediaDetectionRouter
)


class MockImageEngine:

    def detect(self,file_path):
        return {
            "engine":"image",
            "file":str(file_path)
        }


class MockVideoEngine:

    def detect(self,file_path):
        return {
            "engine":"video",
            "file":str(file_path)
        }


def test_router_image():

    router=MediaDetectionRouter(
        image_engine=MockImageEngine()
    )

    result=router.detect(
        "test.jpg",
        "image"
    )

    assert result["engine"]=="image"
    assert result["file"]=="test.jpg"


def test_router_video():

    router=MediaDetectionRouter(
        video_engine=MockVideoEngine()
    )

    result=router.detect(
        "test.mp4",
        "video"
    )

    assert result["engine"]=="video"
    assert result["file"]=="test.mp4"


def test_router_image_engine_missing():

    router=MediaDetectionRouter()

    with pytest.raises(RuntimeError):
        router.detect(
            "test.jpg",
            "image"
        )


def test_router_video_engine_missing():

    router=MediaDetectionRouter()

    with pytest.raises(RuntimeError):
        router.detect(
            "test.mp4",
            "video"
        )


def test_router_unsupported_media_type():

    router=MediaDetectionRouter()

    with pytest.raises(ValueError):
        router.detect(
            "test.txt",
            "audio"
        )