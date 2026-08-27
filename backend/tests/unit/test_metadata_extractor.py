import cv2
import numpy as np

from app.services.preprocessing.metadata_extractor import MetadataExtractor


def test_extract_image_metadata(tmp_path):

    image_path=tmp_path/"test.jpg"

    image=np.zeros(
        (100,200,3),
        dtype=np.uint8
    )

    cv2.imwrite(
        str(image_path),
        image
    )

    extractor=MetadataExtractor()

    result=extractor.extract(
        image_path
    )

    assert result["media_type"]=="image"
    assert result["format"]=="jpg"
    assert result["width"]==200
    assert result["height"]==100
    assert result["channels"]==3


def test_extract_video_metadata(tmp_path):

    video_path=tmp_path/"test.mp4"

    width=320
    height=240
    fps=10
    frame_count=20

    writer=cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width,height)
    )

    frame=np.zeros(
        (height,width,3),
        dtype=np.uint8
    )

    for _ in range(frame_count):
        writer.write(frame)

    writer.release()

    extractor=MetadataExtractor()

    result=extractor.extract(
        video_path
    )

    assert result["media_type"]=="video"
    assert result["format"]=="mp4"
    assert result["width"]==width
    assert result["height"]==height
    assert result["frame_count"]==frame_count
    assert result["fps"]>0
    assert result["duration_seconds"]>0


def test_file_not_found():

    extractor=MetadataExtractor()

    try:
        extractor.extract(
            "does_not_exist.mp4"
        )
        assert False
    except FileNotFoundError:
        assert True


def test_unsupported_format(tmp_path):

    file_path=tmp_path/"test.txt"

    file_path.write_text(
        "test"
    )

    extractor=MetadataExtractor()

    try:
        extractor.extract(
            file_path
        )
        assert False
    except ValueError:
        assert True