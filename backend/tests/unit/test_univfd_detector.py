"""
UnivFDDetector and UnivFDRunner with stand-in networks.

The real CLIP checkpoint is 890 MB and is not in CI; the runner is exercised
with a stub backbone and a real (tiny) linear head whose pinned hash is patched
to the file the test wrote. The real-weights path is in test_univfd_real_model.
"""
import math

import pytest
import torch
from PIL import Image

from app.schemas.detection import (
    Assessment,
    DetectorAvailability,
    DetectorExecutionStatus,
    ScoreSemantics,
)
from app.services.detection.detectors.image.univfd_detector import UnivFDDetector
from app.services.detection.models import univfd_release as release
from app.services.detection.models.trufor_runner import sha256_file
from app.services.detection.models.univfd_runner import LIMITATIONS, UnivFDRunner


class StubRunner:
    version_tag = "univfd-stub"

    def __init__(self, score=0.9, logit=2.2, error=None):
        self.score, self.logit, self.error = score, logit, error

    def predict(self, _path):
        if self.error:
            raise self.error
        return {"score": self.score, "logit": self.logit,
                "metadata": {"clip_precision": "fp32", "padded": False}}


@pytest.fixture
def image(tmp_path):
    path = tmp_path / "x.png"
    Image.new("RGB", (256, 256), (120, 60, 30)).save(path)
    return path


# ── Detector ─────────────────────────────────────────────────────────────────

def test_score_above_threshold_is_likely_ai_generated(image):
    result = UnivFDDetector(StubRunner(score=0.9, logit=2.2), threshold=0.5).detect(image)
    assert result.assessment == Assessment.LIKELY_AI_GENERATED
    assert result.manipulation_detected is True
    assert result.manipulation_type == "ai_generation"
    assert result.score_semantics == ScoreSemantics.RAW_MODEL_SCORE
    assert result.confidence == pytest.approx(0.9)
    assert result.detector_version == "1.0.0+univfd-stub"


def test_score_below_threshold_is_not_called_real(image):
    result = UnivFDDetector(StubRunner(score=0.1, logit=-2.2), threshold=0.5).detect(image)
    # "Nothing detected", never AUTHENTIC/REAL.
    assert result.assessment == Assessment.NO_SIGNIFICANT_ANOMALIES_DETECTED
    assert result.manipulation_detected is False


def test_threshold_is_the_configured_one(image):
    result = UnivFDDetector(StubRunner(score=0.6), threshold=0.7).detect(image)
    assert result.assessment == Assessment.NO_SIGNIFICANT_ANOMALIES_DETECTED
    assert result.metadata["threshold"] == 0.7


def test_result_carries_its_branch_semantics_and_limitations(image):
    meta = UnivFDDetector(StubRunner(), threshold=0.5).detect(image).metadata
    assert meta["branch"] == "ai_generation"
    assert meta["logit"] == pytest.approx(2.2)
    assert "uncalibrated" in meta["semantics"]
    assert meta["limitations"] == LIMITATIONS
    assert meta["clip_precision"] == "fp32"  # the runner's own metadata is kept


def test_no_runner_is_unavailable(image):
    result = UnivFDDetector(None).detect(image)
    assert result.availability == DetectorAvailability.UNAVAILABLE
    assert result.execution_status == DetectorExecutionStatus.SKIPPED


def test_initialisation_error_is_reported(image):
    detector = UnivFDDetector(None, initialization_error="UnivFD initialization failed: X")
    assert detector.availability() == (
        DetectorAvailability.UNAVAILABLE, "UnivFD initialization failed: X"
    )


def test_inference_error_is_a_failure_not_a_score(image):
    result = UnivFDDetector(StubRunner(error=RuntimeError("cuda oom"))).detect(image)
    assert result.execution_status == DetectorExecutionStatus.FAILED
    assert result.assessment != Assessment.NO_SIGNIFICANT_ANOMALIES_DETECTED


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        UnivFDDetector(StubRunner()).detect(tmp_path / "missing.png")


def test_factory_returns_none_when_disabled(monkeypatch):
    from app.core.config import settings
    from app.services.detection.detector_factory import DetectorFactory

    monkeypatch.setattr(settings, "univfd_enabled", False)
    assert DetectorFactory.create_univfd() is None


def test_factory_reports_missing_weights_as_unavailable(monkeypatch, tmp_path):
    from app.core.config import settings
    from app.services.detection.detector_factory import DetectorFactory

    monkeypatch.setattr(settings, "univfd_enabled", True)
    monkeypatch.setattr(settings, "univfd_model_path", tmp_path)
    detector = DetectorFactory.create_univfd()
    availability, reason = detector.availability()
    assert availability == DetectorAvailability.UNAVAILABLE
    assert reason == "UnivFD initialization failed: FileNotFoundError"


# ── Runner ───────────────────────────────────────────────────────────────────

class StubBackbone:
    checkpoint_sha256 = "c" * 64
    precision = "fp32"
    device = torch.device("cpu")

    def __init__(self):
        self.modes = []

    def features(self, images, mode):
        self.modes.append(mode)
        return torch.ones(len(images), release.FEATURE_DIM)


def _head(root, monkeypatch, bias=0.5):
    head = torch.nn.Linear(release.FEATURE_DIM, 1)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.fill_(bias)
    path = root / release.HEAD_FILE
    torch.save(head.state_dict(), path)
    monkeypatch.setattr(release, "HEAD_SHA256", sha256_file(path))
    return path


def test_runner_scores_with_univfd_preprocessing(tmp_path, image, monkeypatch):
    _head(tmp_path, monkeypatch, bias=1.0)
    backbone = StubBackbone()
    runner = UnivFDRunner(tmp_path, device="cpu", backbone_provider=lambda: backbone)
    runner.load()

    out = runner.predict(image)
    assert backbone.modes == ["univfd"]
    assert out["logit"] == pytest.approx(1.0)
    assert out["score"] == pytest.approx(1 / (1 + math.exp(-1.0)))
    assert out["metadata"]["image_size"] == [256, 256]
    assert out["metadata"]["padded"] is False
    assert out["metadata"]["head_sha256"] == release.HEAD_SHA256
    assert runner.version_tag.startswith(f"univfd-{release.UPSTREAM_COMMIT[:7]}.clip-cccc")


def test_runner_flags_small_images_as_padded(tmp_path, monkeypatch):
    _head(tmp_path, monkeypatch)
    small = tmp_path / "small.png"
    Image.new("RGB", (100, 300)).save(small)
    runner = UnivFDRunner(tmp_path, device="cpu", backbone_provider=StubBackbone)
    runner.load()
    assert runner.predict(small)["metadata"]["padded"] is True


def test_runner_refuses_a_head_that_fails_its_hash(tmp_path, monkeypatch):
    _head(tmp_path, monkeypatch)
    monkeypatch.setattr(release, "HEAD_SHA256", "0" * 64)
    runner = UnivFDRunner(tmp_path, device="cpu", backbone_provider=StubBackbone)
    with pytest.raises(RuntimeError, match="pinned SHA-256"):
        runner.load()
    assert not runner.loaded


def test_runner_without_a_head_is_not_installed(tmp_path):
    runner = UnivFDRunner(tmp_path, device="cpu", backbone_provider=StubBackbone)
    with pytest.raises(FileNotFoundError):
        runner.load()


def test_runner_without_a_path_is_not_configured():
    with pytest.raises(ValueError):
        UnivFDRunner(None).load()


def test_runner_predict_before_load_raises(tmp_path, image):
    with pytest.raises(RuntimeError, match="not loaded"):
        UnivFDRunner(tmp_path).predict(image)
