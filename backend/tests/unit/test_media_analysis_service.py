import pytest

from app.services.analysis.media_analysis_service import (
    MediaAnalysisService
)


class MockRouter:

    def detect(
        self,
        file_path,
        media_type
    ):
        return {
            "media_type":media_type,
            "file":str(file_path)
        }


def test_analysis_service_image(tmp_path):

    image=tmp_path/"test.jpg"
    image.write_bytes(b"test")

    service=MediaAnalysisService(
        router=MockRouter()
    )

    result=service.analyze(
        image,
        "image"
    )

    assert result["media_type"]=="image"
    assert result["file"]==str(image)


def test_analysis_service_video(tmp_path):

    video=tmp_path/"test.mp4"
    video.write_bytes(b"test")

    service=MediaAnalysisService(
        router=MockRouter()
    )

    result=service.analyze(
        video,
        "video"
    )

    assert result["media_type"]=="video"


def test_analysis_service_file_not_found():

    service=MediaAnalysisService(
        router=MockRouter()
    )

    with pytest.raises(FileNotFoundError):

        service.analyze(
            "missing.jpg",
            "image"
        )


def test_analysis_service_missing_media_type(tmp_path):

    image=tmp_path/"test.jpg"
    image.write_bytes(b"test")

    service=MediaAnalysisService(
        router=MockRouter()
    )

    with pytest.raises(ValueError):

        service.analyze(
            image,
            ""
        )