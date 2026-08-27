import pytest
from app.services.detection.detectors.video_detector import BaselineVideoDetector, BaseVideoDetector
from app.schemas.detection import DetectionResult

def test_video_detector_missing_file():
    detector = BaselineVideoDetector()
    with pytest.raises(FileNotFoundError):
        detector.detect("non_existent_file.mp4")

def test_video_detector_success(tmp_path):
    detector = BaselineVideoDetector()
    assert isinstance(detector, BaseVideoDetector)
    
    vid_file = tmp_path / "sample.mp4"
    vid_file.write_bytes(b"dummy video bytes")
    
    result = detector.detect(str(vid_file))
    assert isinstance(result, DetectionResult)
    assert result.detector == "BaselineVideoDetector"
    assert result.media_type == "video"
    assert result.manipulation_type == "temporal"
    assert result.metadata.get("status") == "baseline"
