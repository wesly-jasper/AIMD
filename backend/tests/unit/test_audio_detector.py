import pytest

from app.schemas.detection import (
    DetectionResult,
    DetectorAvailability,
    DetectorExecutionStatus,
)
from app.services.detection.detectors.audio_detector import (
    BaseAudioAnalyzer,
    BaselineAudioAnalyzer,
    SpectralSpeechHeuristicDetector,
)


def test_audio_detector_missing_file():
    detector = BaselineAudioAnalyzer()
    with pytest.raises(FileNotFoundError):
        detector.detect("non_existent_file.mp3")


def test_undecodable_audio_is_failed_not_unavailable(tmp_path):
    """
    A corrupt audio file is a failed run, not a missing dependency.

    librosa is installed, so the capability is present; the problem is the file.
    These were the same UNAVAILABLE label, which made a crashing detector
    indistinguishable from one that was never installed -- a deployment fix and a
    code fix reported identically.
    """
    detector = BaselineAudioAnalyzer()
    assert isinstance(detector, BaseAudioAnalyzer)

    audio_file = tmp_path / "sample.mp3"
    audio_file.write_bytes(b"dummy audio bytes")

    result = detector.detect(str(audio_file))
    assert isinstance(result, DetectionResult)
    assert result.detector == "BaselineAudioAnalyzer"
    assert result.media_type == "audio"
    assert result.manipulation_type == "audio"
    assert result.availability == DetectorAvailability.AVAILABLE
    assert result.execution_status == DetectorExecutionStatus.FAILED
    assert result.status_reason
    assert result.produced_observations is False


def test_non_audio_input_is_skipped_not_unavailable(tmp_path):
    """
    Pointing an audio detector at a video file is our routing choice.

    Reporting it as UNAVAILABLE described that choice as a missing dependency.
    """
    video_file = tmp_path / "clip.mp4"
    video_file.write_bytes(b"not audio")

    result = BaselineAudioAnalyzer().detect(str(video_file))

    assert result.availability == DetectorAvailability.AVAILABLE
    assert result.execution_status == DetectorExecutionStatus.SKIPPED
    assert "not an audio file" in result.status_reason.lower()
    assert result.produced_observations is False


def test_spectral_speech_detector_missing_file():
    detector = SpectralSpeechHeuristicDetector()
    with pytest.raises(FileNotFoundError):
        detector.detect("missing_audio.wav")


def test_spectral_speech_detector_does_not_return_a_baseline_result(tmp_path):
    """
    A detector must never report under another detector's identity.

    Every non-success path used to ``return
    self.baseline_analyzer.detect(file_path)``, so an undecodable file produced a
    result named ``BaselineAudioAnalyzer`` with nothing recording that this
    detector had been requested, or why it was not used.
    """
    detector = SpectralSpeechHeuristicDetector()
    assert isinstance(detector, BaseAudioAnalyzer)

    audio_file = tmp_path / "test_synth.wav"
    audio_file.write_bytes(b"test wave audio")

    result = detector.detect(str(audio_file))
    assert isinstance(result, DetectionResult)
    assert result.detector == "SpectralSpeechHeuristicDetector"
    assert result.media_type == "audio"
    assert result.manipulation_type == "audio"
    assert result.produced_observations is False
    assert result.status_reason


def test_spectral_speech_detector_does_not_claim_neural_inference():
    """
    The identity an examiner reads must not claim a capability that is absent.

    This class was named ``NeuralSpeechDetector``, documented as a "Deep Neural
    Acoustic & Synthetic Speech (AI Voice Clone) Detector" and versioned
    "2.0-neural-acoustic", while performing no neural inference at all: it
    imported torch, never used it, and hand-thresholded two librosa statistics.
    """
    detector = SpectralSpeechHeuristicDetector()

    assert "neural" not in detector.detector_id.lower()
    assert "neural" not in detector.version.lower()

    source = type(detector).__module__
    import importlib
    import inspect

    module_source = inspect.getsource(importlib.import_module(source))
    # The unused torch import was itself a capability claim in the code.
    assert "import torch" not in module_source
