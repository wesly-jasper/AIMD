import pytest
from PIL import Image
import numpy as np
from app.services.detection.detectors.face_detector import (
    BaselineFaceDetector,
    DeepLearningFaceDetector,
    BaseFaceDetector,
)
from app.schemas.detection import (
    DetectionResult,
    DetectorAvailability,
    DetectorExecutionStatus,
    ScoreSemantics,
)

def test_face_detector_missing_file():
    detector = BaselineFaceDetector()
    with pytest.raises(FileNotFoundError):
        detector.detect("non_existent_file.jpg")

def test_face_detector_success(tmp_path):
    detector = BaselineFaceDetector()
    assert isinstance(detector, BaseFaceDetector)

    img_file = tmp_path / "sample.jpg"
    img = Image.new("RGB", (64, 64), color="red")
    img.save(str(img_file))

    result = detector.detect(str(img_file))
    assert isinstance(result, DetectionResult)
    assert result.detector == "BaselineFaceDetector"
    assert result.media_type == "image"
    assert result.manipulation_type == "face"
    assert "status" in result.metadata


def test_invalid_face_input_is_failed_not_clean(tmp_path):
    """
    Undecodable evidence is a FAILED run by an AVAILABLE detector.

    This asserted UNAVAILABLE, which blamed the capability for a problem with the
    input file. The Haar cascade loaded fine; it was handed bytes it could not
    decode. The two have different remedies, and only one of them is honestly
    described as "we do not have this capability".
    """
    image = tmp_path / "invalid.jpg"
    image.write_bytes(b"not an image")

    result = BaselineFaceDetector().detect(str(image))

    assert result.availability == DetectorAvailability.AVAILABLE
    assert result.execution_status == DetectorExecutionStatus.FAILED
    assert result.status_reason
    assert result.manipulation_detected is False
    assert result.confidence == 0.0
    assert result.score_semantics == ScoreSemantics.NOT_APPLICABLE
    assert result.regions == []
    assert result.produced_observations is False


def test_no_face_found_is_inconclusive_not_clean(tmp_path):
    """
    A face detector that located no face has examined no face.

    It must not report a negative result: "no face present" is not evidence that
    a face is unmanipulated, and a flat colour field is exactly the input where
    the old CLEAN label was most misleading.
    """
    image = tmp_path / "no-face.png"
    Image.new("RGB", (120, 120), color=(40, 90, 160)).save(str(image))

    result = BaselineFaceDetector().detect(str(image))

    assert result.availability == DetectorAvailability.AVAILABLE
    assert result.execution_status == DetectorExecutionStatus.INCONCLUSIVE
    assert result.metadata["faces_detected"] == 0
    assert result.manipulation_detected is False
    assert result.score_semantics == ScoreSemantics.NOT_APPLICABLE
    assert result.regions == []
    # It ran, so it is not an absence of evidence -- but it examined no face, so
    # it cannot carry a conclusion in either direction.
    assert result.supports_determination is False


def test_unloadable_cascade_is_unavailable_not_silently_faceless(monkeypatch, tmp_path):
    """
    An empty classifier matches nothing, which reads as "no faces in the image".

    Availability is therefore checked against the loaded model rather than
    assumed from OpenCV being importable -- otherwise a missing cascade file is
    indistinguishable from a photograph of a landscape.
    """
    class UnloadedCascade:
        """``cv2.CascadeClassifier.empty`` is read-only, so stand in for one."""

        def empty(self):
            return True

    detector = BaselineFaceDetector()
    detector.face_cascade = UnloadedCascade()

    availability, reason = detector.availability()

    assert availability == DetectorAvailability.UNAVAILABLE
    assert "cascade" in reason.lower()


def test_deep_learning_face_detector_missing_file():
    detector = DeepLearningFaceDetector()
    with pytest.raises(FileNotFoundError):
        detector.detect("missing_deepfake_file.jpg")

def test_deep_learning_face_detector_execution(tmp_path):
    detector = DeepLearningFaceDetector()
    assert isinstance(detector, BaseFaceDetector)

    img_file = tmp_path / "face_sample.jpg"
    img = Image.new("RGB", (100, 100), color="blue")
    img.save(str(img_file))

    result = detector.detect(str(img_file))
    assert isinstance(result, DetectionResult)
    assert result.media_type == "image"
    assert result.manipulation_type == "face"
    assert result.confidence >= 0.0
    # Whatever happened, the record names the code that actually ran.
    assert result.detector == "DeepLearningFaceDetector"


def test_mtcnn_unavailable_does_not_return_a_cascade_result(tmp_path, monkeypatch):
    """
    Neural inference must never be silently substituted by a classical detector.

    ``DeepLearningFaceDetector.detect`` used to ``return
    self.cascade_fallback.detect(file_path)`` when MTCNN had not loaded, so a
    Haar cascade result came back with nothing in the record saying that MTCNN
    had been requested or why it was not used. A reader comparing detectors would
    have counted a classical heuristic as neural corroboration.
    """
    image = tmp_path / "subject.png"
    Image.new("RGB", (100, 100), color="blue").save(str(image))

    detector = DeepLearningFaceDetector()
    # Simulate torch / facenet-pytorch absent.
    monkeypatch.setattr(detector, "_load", lambda: None)
    detector._load_attempted = True
    detector._loaded = False
    detector._mtcnn = None
    detector._load_error = "ModuleNotFoundError: No module named 'facenet_pytorch'"

    result = detector.detect(str(image))

    assert result.detector == "DeepLearningFaceDetector"
    assert result.availability == DetectorAvailability.UNAVAILABLE
    assert result.execution_status == DetectorExecutionStatus.SKIPPED
    assert "facenet_pytorch" in result.status_reason
    assert result.score_semantics == ScoreSemantics.NOT_APPLICABLE
    assert result.produced_observations is False
    assert result.regions == []


def test_mtcnn_runtime_crash_is_failed_not_a_completed_analysis(tmp_path, monkeypatch):
    """
    A crash inside MTCNN inference also used to fall through to the cascade, so a
    broken neural detector was recorded as a completed analysis.
    """
    image = tmp_path / "subject.png"
    Image.new("RGB", (100, 100), color="blue").save(str(image))

    class ExplodingMTCNN:
        def detect(self, *_args, **_kwargs):
            raise RuntimeError("forced inference failure")

    detector = DeepLearningFaceDetector()
    monkeypatch.setattr(detector, "_load", lambda: None)
    detector._load_attempted = True
    detector._loaded = True
    detector._mtcnn = ExplodingMTCNN()

    result = detector.detect(str(image))

    assert result.detector == "DeepLearningFaceDetector"
    assert result.availability == DetectorAvailability.AVAILABLE
    assert result.execution_status == DetectorExecutionStatus.FAILED
    assert "RuntimeError" in result.status_reason
    assert result.produced_observations is False
