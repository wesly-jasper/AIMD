from pathlib import Path
import cv2
import numpy as np

from app.services.preprocessing.frame_extractor import FrameExtractor


def create_test_video(path,frame_count=5):
    fourcc=cv2.VideoWriter_fourcc(*"mp4v")
    writer=cv2.VideoWriter(
        str(path),
        fourcc,
        10,
        (200,100)
    )

    for i in range(frame_count):
        frame=np.zeros((100,200,3),dtype=np.uint8)
        frame[:]=i*30
        writer.write(frame)

    writer.release()


def test_extract_frames(tmp_path):
    video_path=tmp_path/"test.mp4"

    create_test_video(video_path,5)

    extractor=FrameExtractor(tmp_path/"frames")

    result=extractor.extract(
        video_path,
        "test-media",
        frame_interval=1
    )

    assert result["frame_count"]==5
    assert result["saved_count"]==5
    assert len(result["frames"])==5

    for frame_path in result["frames"]:
        assert Path(frame_path).exists()


def test_extract_every_second_frame(tmp_path):
    video_path=tmp_path/"test.mp4"

    create_test_video(video_path,6)

    extractor=FrameExtractor(tmp_path/"frames")

    result=extractor.extract(
        video_path,
        "test-media",
        frame_interval=2
    )

    assert result["frame_count"]==6
    assert result["saved_count"]==3


def test_video_not_found(tmp_path):
    extractor=FrameExtractor(tmp_path/"frames")

    try:
        extractor.extract(
            tmp_path/"missing.mp4",
            "test-media"
        )
        assert False
    except FileNotFoundError:
        assert True


def test_invalid_frame_interval(tmp_path):
    video_path=tmp_path/"test.mp4"

    create_test_video(video_path,3)

    extractor=FrameExtractor(tmp_path/"frames")

    try:
        extractor.extract(
            video_path,
            "test-media",
            frame_interval=0
        )
        assert False
    except ValueError:
        assert True