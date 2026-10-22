import cv2
import numpy as np

from app.schemas.detection import (
    DetectorAvailability,
    DetectorExecutionStatus,
    ScoreSemantics,
)
from app.services.detection.detectors.image.copy_move_detector import CopyMoveDetector


def test_copy_move_runtime_failure_is_failed_not_unavailable(tmp_path, monkeypatch):
    image = np.zeros((96, 96, 3), dtype=np.uint8)
    image_path = tmp_path / "image.png"
    assert cv2.imwrite(str(image_path), image)
    detector = CopyMoveDetector()

    class TenKeypoints:
        def detectAndCompute(self, _gray, _mask):
            points = [cv2.KeyPoint(float(i), float(i), 1.0) for i in range(10)]
            descriptors = np.ones((10, 128), dtype=np.float32)
            return points, descriptors

    detector.detector = TenKeypoints()

    def fail_matching(*_args):
        raise RuntimeError("forced matcher failure")

    monkeypatch.setattr(detector, "_match_descriptors", fail_matching)
    result = detector.detect(str(image_path))

    # SIFT/ORB are installed and the matcher crashed: a defect, not an absent
    # capability. This asserted UNAVAILABLE, which hid the crash.
    assert result.availability == DetectorAvailability.AVAILABLE
    assert result.execution_status == DetectorExecutionStatus.FAILED
    assert "RuntimeError" in result.status_reason
    assert result.manipulation_detected is False
    assert result.confidence == 0.0
    assert result.score_semantics == ScoreSemantics.NOT_APPLICABLE
    assert result.produced_observations is False
    assert result.regions == []
