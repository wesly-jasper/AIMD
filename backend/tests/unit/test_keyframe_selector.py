from pathlib import Path

from app.services.preprocessing.keyframe_selector import KeyframeSelector


def create_test_frames(frame_dir,count=10):
    frame_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for i in range(count):
        frame_path=frame_dir/f"frame_{i+1:06d}.jpg"
        frame_path.write_bytes(b"test frame")


def test_select_keyframes(tmp_path):
    frame_dir=tmp_path/"frames"
    output_dir=tmp_path/"keyframes"

    create_test_frames(
        frame_dir,
        count=10
    )

    selector=KeyframeSelector(output_dir)

    result=selector.select(
        frame_dir,
        "test-media",
        frame_interval=2
    )

    assert result["total_frames"]==10
    assert result["keyframe_count"]==5
    assert len(result["keyframes"])==5

    for keyframe in result["keyframes"]:
        assert Path(keyframe).exists()


def test_select_all_frames(tmp_path):
    frame_dir=tmp_path/"frames"
    output_dir=tmp_path/"keyframes"

    create_test_frames(
        frame_dir,
        count=5
    )

    selector=KeyframeSelector(output_dir)

    result=selector.select(
        frame_dir,
        "test-media",
        frame_interval=1
    )

    assert result["total_frames"]==5
    assert result["keyframe_count"]==5


def test_frame_directory_not_found(tmp_path):
    selector=KeyframeSelector(tmp_path/"keyframes")

    try:
        selector.select(
            tmp_path/"missing",
            "test-media"
        )
        assert False
    except FileNotFoundError:
        assert True


def test_invalid_frame_interval(tmp_path):
    frame_dir=tmp_path/"frames"

    create_test_frames(
        frame_dir,
        count=5
    )

    selector=KeyframeSelector(tmp_path/"keyframes")

    try:
        selector.select(
            frame_dir,
            "test-media",
            frame_interval=0
        )
        assert False
    except ValueError:
        assert True