import pytest
from app.services.detection.detectors.audio_detector import BaselineAudioAnalyzer, BaseAudioAnalyzer
from app.schemas.detection import DetectionResult

def test_audio_detector_missing_file():
    detector = BaselineAudioAnalyzer()
    with pytest.raises(FileNotFoundError):
        detector.detect("non_existent_file.mp3")

def test_audio_detector_success(tmp_path):
    detector = BaselineAudioAnalyzer()
    assert isinstance(detector, BaseAudioAnalyzer)
    
    audio_file = tmp_path / "sample.mp3"
    audio_file.write_bytes(b"dummy audio bytes")
    
    result = detector.detect(str(audio_file))
    assert isinstance(result, DetectionResult)
    assert result.detector == "BaselineAudioAnalyzer"
    assert result.media_type == "audio"
    assert result.manipulation_type == "audio"
    assert result.metadata.get("status") == "baseline"
