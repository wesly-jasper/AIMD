from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from app.schemas.detection import (
    Assessment,
    DetectionResult,
    DetectorAvailability,
    DetectorExecutionStatus,
    ScoreSemantics,
)
from app.services.detection.detectors.image.manipulation_mask_generator import (
    ManipulationMaskGenerator,
)
from app.services.detection.engine.image_detection_engine import ImageDetectionEngine
from app.services.orchestration import analysis_pipeline as pipeline_module
from app.services.orchestration.analysis_pipeline import AnalysisPipeline


def _write_test_image(path: Path, width: int = 160, height: int = 128) -> None:
    rng = np.random.default_rng(2025)
    image = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def test_valid_image_generates_heatmap_with_original_dimensions_and_schema(tmp_path):
    image_path = tmp_path / "deterministic.png"
    _write_test_image(image_path)

    result = ManipulationMaskGenerator().detect(str(image_path))

    assert isinstance(result, DetectionResult)
    assert result.detector == "ManipulationMaskGenerator"
    assert result.media_type == "image"
    assert result.metadata["status"] == "active_mask_generation"
    assert result.metadata["heatmap_shape"] == [128, 160]
    assert result.metadata["regions_found"] == len(result.regions)
    assert result.metadata["heatmap_units"] == "robust_z"
    # An uncalibrated heuristic: it ran and keeps its regions, but it never
    # reports a determination in either direction.
    assert result.execution_status == DetectorExecutionStatus.INCONCLUSIVE
    assert result.produced_observations is True
    assert result.manipulation_detected is False
    assert result.assessment == Assessment.INCONCLUSIVE
    for region in result.regions:
        assert 0.0 <= region.confidence <= 1.0
        assert region.type == "manipulation_mask"
        assert region.bbox is not None
        x, y, width, height = region.bbox
        assert 0 <= x <= 160 and 0 <= y <= 128
        assert 0 < width <= 160 and 0 < height <= 128


def test_every_cue_is_weighted_in_the_fusion(monkeypatch):
    """Regression: 7 cue maps were zipped with 6 weights, dropping colour."""
    generator = ManipulationMaskGenerator()
    seen = []
    original = generator._robust_z

    def spy(cue):
        seen.append(cue)
        return original(cue)

    monkeypatch.setattr(generator, "_robust_z", spy)
    image = np.random.default_rng(1).integers(0, 256, (96, 96, 3), dtype=np.uint8)
    generator._generate_single_scale_heatmap(image)

    # 3 ELA quality factors + FFT + noise + gradient + colour.
    assert len(seen) == len(generator.quality_factors) + 4


def test_uniform_texture_produces_no_outlier_regions(tmp_path):
    """
    Regression: per-image min-max normalisation guaranteed "suspicious" regions.

    Stretching every cue to [0, 1] forces some pixels to 1.0 in any image, so a
    fixed threshold always found regions. Robust z-scores leave a homogeneous
    image with nothing that stands out.
    """
    image_path = tmp_path / "uniform.png"
    noise = np.random.default_rng(3).normal(128, 8, (128, 160, 3))
    assert cv2.imwrite(str(image_path), noise.clip(0, 255).astype(np.uint8))

    result = ManipulationMaskGenerator().detect(str(image_path))

    assert result.regions == []
    assert result.metadata["regions_found"] == 0


def test_robust_z_does_not_stretch_a_flat_map():
    flat = np.full((32, 32), 0.4, dtype=np.float32)
    assert np.all(ManipulationMaskGenerator._robust_z(flat) == 0.0)


def test_invalid_and_empty_images_are_failed_not_clean(tmp_path):
    """
    Unreadable input is a FAILED run by an AVAILABLE detector.

    This previously asserted UNAVAILABLE, which blamed the capability for a
    problem with the evidence file. The detector is installed and working; it was
    handed something it could not decode.
    """
    invalid_path = tmp_path / "invalid.png"
    invalid_path.write_bytes(b"not an image")
    empty_path = tmp_path / "empty.png"
    empty_path.write_bytes(b"")

    for path in (invalid_path, empty_path):
        result = ManipulationMaskGenerator().detect(str(path))
        assert result.availability == DetectorAvailability.AVAILABLE
        assert result.execution_status == DetectorExecutionStatus.FAILED
        assert result.manipulation_detected is False
        assert result.confidence == 0.0
        assert result.score_semantics == ScoreSemantics.NOT_APPLICABLE
        assert result.regions == []
        assert result.status_reason
        # The result must not be usable as evidence in either direction.
        assert result.produced_observations is False


def test_missing_image_input_is_failed_with_a_recorded_reason(tmp_path):
    result = ManipulationMaskGenerator().detect(str(tmp_path / "missing.png"))

    assert result.availability == DetectorAvailability.AVAILABLE
    assert result.execution_status == DetectorExecutionStatus.FAILED
    assert "missing" in result.status_reason.lower()
    assert result.manipulation_detected is False
    assert result.confidence == 0.0
    assert result.regions == []
    assert result.produced_observations is False


def test_missing_optional_unet_keeps_classical_mask_generation_explicit(tmp_path):
    image_path = tmp_path / "optional-unet.png"
    _write_test_image(image_path)

    result = ManipulationMaskGenerator(
        enable_unet=True,
        unet_model_path=str(tmp_path / "missing-model.onnx"),
    ).detect(str(image_path))

    assert result.metadata["status"] == "active_mask_generation"
    assert result.metadata["unet_refinement"] == "unavailable"
    assert result.metadata["heatmap_shape"] == [128, 160]


def test_runtime_failure_is_failed_not_unavailable(tmp_path, monkeypatch):
    """
    A crash inside mask computation must stay visible as a defect.

    It was reported as UNAVAILABLE, which is indistinguishable from "this
    detector was never installed". The two have different remedies -- a code fix
    versus a deployment fix -- so recording a crash as an absent capability hides
    a bug behind what looks like a configuration gap.
    """
    image_path = tmp_path / "runtime-failure.png"
    _write_test_image(image_path)
    generator = ManipulationMaskGenerator()

    def fail_heatmap(_image):
        raise RuntimeError("forced mask computation failure")

    monkeypatch.setattr(generator, "_generate_multiscale_heatmap", fail_heatmap)
    result = generator.detect(str(image_path))

    assert result.availability == DetectorAvailability.AVAILABLE
    assert result.execution_status == DetectorExecutionStatus.FAILED
    assert "RuntimeError" in result.status_reason
    assert result.manipulation_detected is False
    assert result.confidence == 0.0
    assert result.score_semantics == ScoreSemantics.NOT_APPLICABLE
    assert result.regions == []
    assert result.produced_observations is False


def test_mask_failure_is_partial_and_other_detector_result_is_preserved(
    tmp_path, monkeypatch
):
    image_path = tmp_path / "pipeline-failure.png"
    _write_test_image(image_path)
    mask_generator = ManipulationMaskGenerator()

    def fail_heatmap(_image):
        raise RuntimeError("forced mask computation failure")

    monkeypatch.setattr(mask_generator, "_generate_multiscale_heatmap", fail_heatmap)
    other_result = DetectionResult(
        detector="ForensicImageDetector",
        media_type="image",
        manipulation_detected=False,
        confidence=0.1,
        manipulation_type="image_manipulation",
        assessment=Assessment.CLEAN,
        metadata={"assessment": "CLEAN"},
    )
    engine = ImageDetectionEngine(detectors=[mask_generator, _StaticDetector(other_result)])
    router = SimpleNamespace(route=lambda *_args: engine.detect(image_path))
    pipeline = AnalysisPipeline.__new__(AnalysisPipeline)
    pipeline._detection_container = SimpleNamespace(router=router)
    stored = []
    monkeypatch.setattr(
        pipeline_module.repo,
        "create_detection",
        lambda _db, data: stored.append(data),
    )
    failures = []

    pipeline._run_detection(
        image_path, "media-id", "analysis-id", "image", [], object(), failures
    )

    assert len(stored) == 2
    assert stored[0]["detector"] == "ManipulationMaskGenerator"
    # Persisted as a failure of an available detector, so an examiner reading the
    # record can tell a broken run from an uninstalled one.
    assert stored[0]["availability"] == "AVAILABLE"
    assert stored[0]["execution_status"] == "FAILED"
    assert "RuntimeError" in stored[0]["status_reason"]
    assert stored[0]["score_semantics"] == "NOT_APPLICABLE"
    assert stored[1]["detector"] == "ForensicImageDetector"
    assert any("ManipulationMaskGenerator incomplete" in failure for failure in failures)


class _StaticDetector:
    def __init__(self, result: DetectionResult):
        self.result = result

    def detect(self, _file_path):
        return self.result
