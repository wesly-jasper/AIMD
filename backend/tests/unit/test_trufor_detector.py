import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.schemas.detection import (
    DetectionResult,
    DetectorAvailability,
    DetectorExecutionStatus,
    ScoreSemantics,
)
from app.services.detection.detectors.image import trufor_detector as trufor_detector_module
from app.services.detection.detectors.image.trufor_detector import (
    TruForDetector
)
from app.services.detection.engine.image_detection_engine import ImageDetectionEngine


class MockRunner:

    def predict(self,file_path):
        return {
            "manipulation_detected":True,
            "confidence":0.92,
            "regions":[],
            "metadata":{
                "score":0.92
            }
        }


class InvalidRunner:

    def predict(self,file_path):
        return {
            "invalid":"result"
        }


class ExplodingRunner:

    def predict(self, file_path):
        raise RuntimeError("forced inference failure")


def test_trufor_detector_file_not_found():

    detector=TruForDetector(
        MockRunner()
    )

    with pytest.raises(FileNotFoundError):
        detector.detect(
            "missing.jpg"
        )


def test_trufor_detector(tmp_path):

    image=tmp_path/"test.jpg"
    image.write_bytes(b"test")

    detector=TruForDetector(
        MockRunner()
    )

    result=detector.detect(image)

    assert isinstance(
        result,
        DetectionResult
    )

    assert result.detector=="TruForDetector"
    assert result.media_type=="image"
    assert result.manipulation_detected is True
    assert result.confidence==0.92
    assert result.manipulation_type=="image_manipulation"
    assert result.regions==[]
    assert result.metadata["score"]==0.92
    assert result.availability == DetectorAvailability.AVAILABLE
    assert result.execution_status == DetectorExecutionStatus.COMPLETED
    # The only learned detector in the suite, and the only one entitled to call
    # its score a model output -- raw, because no calibration curve was fitted.
    assert result.score_semantics == ScoreSemantics.RAW_MODEL_SCORE


def test_trufor_identity_matches_the_engine_signal_family_map(tmp_path):
    """
    The reported identity must be the one the fusion layer keys on.

    Results went out as ``"TruFor"`` while the engine's ``_SIGNAL_FAMILY`` map,
    the factory and the class were all named ``"TruForDetector"``. The "learned"
    family therefore never matched a result, so the one learned detector in the
    suite was silently filed as "unclassified" and could not contribute to
    cross-family corroboration as designed.
    """
    image = tmp_path / "test.jpg"
    image.write_bytes(b"test")

    result = TruForDetector(MockRunner()).detect(image)

    assert ImageDetectionEngine._family(result.detector) == "learned"


def test_trufor_unavailable_runner_is_unavailable_not_clean(tmp_path):
    """
    The TruFor checkpoint is not redistributable, so absence is the normal state.

    It must be recorded as an absent capability and never as a completed run:
    an assessment reached without the learned family has narrower coverage, and
    the record has to show that.
    """
    image = tmp_path / "test.jpg"
    image.write_bytes(b"test")

    result = TruForDetector(None, "TruFor checkpoint is not installed").detect(image)

    assert result.detector == "TruForDetector"
    assert result.availability == DetectorAvailability.UNAVAILABLE
    assert result.execution_status == DetectorExecutionStatus.SKIPPED
    assert "checkpoint" in result.status_reason
    assert result.confidence == 0.0
    assert result.score_semantics == ScoreSemantics.NOT_APPLICABLE
    assert result.produced_observations is False


def test_trufor_inference_crash_is_failed_not_unavailable(tmp_path):
    """
    A loaded checkpoint that crashes is a defect, not a missing capability.

    Collapsing the two would tell an operator to install a model they already
    have.
    """
    image = tmp_path / "test.jpg"
    image.write_bytes(b"test")

    result = TruForDetector(ExplodingRunner()).detect(image)

    assert result.availability == DetectorAvailability.AVAILABLE
    assert result.execution_status == DetectorExecutionStatus.FAILED
    assert "RuntimeError" in result.status_reason
    assert result.produced_observations is False


def test_trufor_detector_malformed_runner_result_is_failed(tmp_path):
    """
    A malformed runner response is recorded, not raised.

    Raising surfaced as an engine-level error with no detector record at all, so
    the run left no trace of having been attempted. A FAILED result keeps the
    reason in the evidence where an examiner will see it.
    """
    image=tmp_path/"test.jpg"
    image.write_bytes(b"test")

    result = TruForDetector(InvalidRunner()).detect(image)

    assert result.availability == DetectorAvailability.AVAILABLE
    assert result.execution_status == DetectorExecutionStatus.FAILED
    assert "manipulation_detected" in result.status_reason
    assert result.produced_observations is False


class MapRunner:
    """A runner that returns heatmaps, as the real TruForRunner does."""

    version_tag = "trufor-ae54475.w-0123456789ab"

    def predict(self, file_path):
        loc = np.zeros((32, 48), dtype=np.float32)
        loc[8:24, 12:36] = 0.9
        return {
            "manipulation_detected": True,
            "confidence": 0.81,
            "regions": [{"type": "forgery_localization", "confidence": 0.9,
                         "bbox": [12.0, 8.0, 24.0, 16.0], "extra": {"localization_max": 0.9}}],
            "localization_map": loc,
            "reliability_map": np.full((32, 48), 0.6, dtype=np.float32),
            "metadata": {"checkpoint_sha256": "0123456789abcdef" * 4},
        }


def _png(path):
    Image.fromarray(np.full((32, 48, 3), 128, dtype=np.uint8)).save(path)
    return path


def test_trufor_heatmaps_become_artefacts_not_metadata(tmp_path):
    image = _png(tmp_path / "input.png")
    artefacts_dir = tmp_path / "artefacts"

    detector = TruForDetector(MapRunner(), artefacts_dir=artefacts_dir)
    result = detector.detect(image)

    assert result.execution_status == DetectorExecutionStatus.COMPLETED
    assert result.detector_version == "2.0.0+trufor-ae54475.w-0123456789ab"
    assert "execution_time_ms" in result.metadata
    # Arrays never reach the stored metadata.
    json.dumps(result.metadata)
    assert "localization_map" not in result.metadata

    artefacts = result.metadata["artefacts"]
    assert set(artefacts) == {"localization_map", "mask", "reliability_map", "overlay"}
    for entry in artefacts.values():
        relative = Path(entry["path"])
        assert not relative.is_absolute()
        assert str(tmp_path) not in entry["path"]
        written = artefacts_dir / relative
        assert written.is_file()
        assert entry["sha256"] == hashlib.sha256(written.read_bytes()).hexdigest()
        assert entry["media_type"] == "image/png"
        with Image.open(written) as png:
            assert png.size == (48, 32)

    assert "0123456789ab" in artefacts["localization_map"]["path"]
    region = result.regions[0]
    assert region.mask_path == artefacts["mask"]["path"]
    assert region.extra == {"localization_max": 0.9}


def test_trufor_mask_covers_the_localised_area_only(tmp_path):
    """The mask is the localisation map cut at the recorded threshold: the
    0.9 block is masked, the rest of the image is not."""
    image = _png(tmp_path / "input.png")
    artefacts_dir = tmp_path / "artefacts"

    result = TruForDetector(MapRunner(), artefacts_dir=artefacts_dir).detect(image)

    mask = result.metadata["artefacts"]["mask"]
    assert mask["threshold"] == 0.5
    assert mask["areas"] == 1
    assert "_mask_t050.png" in mask["path"]
    with Image.open(artefacts_dir / mask["path"]) as png:
        pixels = np.asarray(png)
    assert set(np.unique(pixels)) <= {0, 255}
    assert (pixels[8:24, 12:36] == 255).all()
    assert pixels.sum() == 255 * 16 * 24
    assert mask["masked_fraction"] == pytest.approx(16 * 24 / (32 * 48), abs=1e-6)


def test_trufor_mask_uses_the_runners_threshold(tmp_path):
    class HighCut(MapRunner):
        def predict(self, file_path):
            out = super().predict(file_path)
            out["metadata"]["threshold"] = 0.95
            return out

    image = _png(tmp_path / "input.png")
    result = TruForDetector(HighCut(), artefacts_dir=tmp_path / "a").detect(image)

    mask = result.metadata["artefacts"]["mask"]
    assert mask["threshold"] == 0.95
    assert mask["areas"] == 0 and mask["masked_fraction"] == 0.0
    assert "_mask_t095.png" in mask["path"]


def test_trufor_heatmap_write_failure_keeps_the_score(tmp_path, monkeypatch):
    """A lost heatmap is recorded; the model's score is still a valid observation."""
    image = _png(tmp_path / "input.png")

    def broken(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(trufor_detector_module, "write_heatmap_artefacts", broken)
    result = TruForDetector(MapRunner(), artefacts_dir=tmp_path).detect(image)

    assert result.execution_status == DetectorExecutionStatus.COMPLETED
    assert result.confidence == 0.81
    assert result.metadata["artefacts_error"] == "OSError"
    assert "artefacts" not in result.metadata
    assert result.regions[0].mask_path is None
