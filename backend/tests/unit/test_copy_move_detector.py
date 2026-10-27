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

    class ManyKeypoints:
        def detectAndCompute(self, _gray, _mask):
            points = [cv2.KeyPoint(float(i), float(i), 1.0) for i in range(200)]
            descriptors = np.ones((200, 128), dtype=np.float32)
            return points, descriptors

    detector.detector = ManyKeypoints()

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


def _textured(seed=0, size=(480, 640)):
    rng = np.random.default_rng(seed)
    base = cv2.resize(rng.integers(0, 255, (size[0] // 8, size[1] // 8, 3), dtype=np.uint8),
                      (size[1], size[0]), interpolation=cv2.INTER_CUBIC)
    noise = rng.normal(0, 18, base.shape)
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def test_a_pasted_region_is_found_with_both_boxes(tmp_path):
    image = _textured()
    image[300:420, 420:560] = image[40:160, 60:200]  # 120 x 140 block, shifted
    path = tmp_path / "clone.jpg"
    assert cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 90])

    result = CopyMoveDetector().detect(str(path))
    assert result.execution_status == DetectorExecutionStatus.COMPLETED
    assert result.manipulation_detected is True
    assert result.metadata["inliers"] >= 50
    boxes = {r.type: r.bbox for r in result.regions}
    sx, sy, sw, sh = boxes["copy_move_source"]
    cx, cy, cw, ch = boxes["copy_move_copy"]
    assert sx < 200 and sy < 160 and sx + sw > 60 and sy + sh > 40
    assert cx < 560 and cy < 420 and cx + cw > 420 and cy + ch > 300
    assert result.metadata["transforms"][0]["shift"] > 300


def test_an_unedited_textured_image_is_a_completed_negative(tmp_path):
    path = tmp_path / "plain.jpg"
    assert cv2.imwrite(str(path), _textured(seed=1), [cv2.IMWRITE_JPEG_QUALITY, 90])
    result = CopyMoveDetector().detect(str(path))
    assert result.execution_status == DetectorExecutionStatus.COMPLETED
    assert result.manipulation_detected is False
    assert result.regions == []
    assert result.confidence < 0.5


def test_a_flat_image_is_inconclusive(tmp_path):
    path = tmp_path / "flat.png"
    assert cv2.imwrite(str(path), np.full((200, 200, 3), 128, dtype=np.uint8))
    result = CopyMoveDetector().detect(str(path))
    assert result.execution_status == DetectorExecutionStatus.INCONCLUSIVE
