"""
TruForRunner: loading, verification and inference.

Most tests inject a tiny stand-in network through ``model_builder`` because
TruFor's code is research-only and cannot ship with the repository. Tests that
need the real architecture, the real weights or CocoGlide are skipped unless
``scripts/fetch_models.py`` has installed them into ``backend/models/trufor``.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from app.services.detection.models import trufor_release as release
from app.services.detection.models.trufor_runner import (
    TruForRunner,
    _import_upstream_builder,
    sha256_file,
    trufor_config,
)

TRUFOR_ROOT = Path(__file__).resolve().parents[2] / "models" / "trufor"
HAS_CODE = all((TRUFOR_ROOT / rel).is_file() for rel in release.CODE_FILES)
HAS_WEIGHTS = HAS_CODE and (TRUFOR_ROOT / release.WEIGHTS_RELPATH).is_file()

needs_code = pytest.mark.skipif(not HAS_CODE, reason="TruFor code not fetched (scripts/fetch_models.py)")
needs_weights = pytest.mark.skipif(not HAS_WEIGHTS, reason="TruFor weights not installed")


class FakeTruFor(torch.nn.Module):
    """TruFor's interface: ``forward(rgb) -> (pred, conf, det, npp)``."""

    def __init__(self, cfg=None):
        super().__init__()
        self.loc = torch.nn.Conv2d(3, 2, kernel_size=1)
        self.conf = torch.nn.Conv2d(3, 1, kernel_size=1)
        self.det = torch.nn.Linear(3, 1)
        self.seen_input_range = None

    def forward(self, rgb):
        self.seen_input_range = (float(rgb.min()), float(rgb.max()))
        pred = self.loc(rgb)
        conf = self.conf(rgb)
        det = self.det(rgb.mean(dim=(2, 3)))
        return pred, conf, det, None


class NotATensor:
    """Anything a ``weights_only`` unpickle refuses."""


def _image(path: Path, size=(48, 32)) -> Path:
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)).save(path)
    return path


def _install(root: Path, state, manifest_sha=None, verification="official_archive_md5") -> Path:
    checkpoint = root / release.WEIGHTS_RELPATH
    checkpoint.parent.mkdir(parents=True)
    torch.save(state, checkpoint)
    if manifest_sha is not None:
        sha = sha256_file(checkpoint) if manifest_sha == "actual" else manifest_sha
        (root / release.MANIFEST_NAME).write_text(
            json.dumps({"weights": {"sha256": sha, "verification": verification}})
        )
    return checkpoint


def _loaded_fake_runner(root: Path, **install) -> TruForRunner:
    _install(root, {"state_dict": FakeTruFor().state_dict(), "epoch": 1}, **install)
    runner = TruForRunner(root, device="cpu", model_builder=FakeTruFor)
    runner.load()
    return runner


# ── Loading ──────────────────────────────────────────────────────────────────

def test_runner_without_model_path():
    with pytest.raises(ValueError):
        TruForRunner(None).load()


def test_runner_model_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        TruForRunner(tmp_path / "missing").load()


def test_runner_root_without_weights_is_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="weights"):
        TruForRunner(tmp_path, model_builder=FakeTruFor).load()


def test_runner_not_loaded(tmp_path):
    with pytest.raises(RuntimeError):
        TruForRunner(tmp_path).predict(_image(tmp_path / "x.png"))


def test_runner_refuses_modified_upstream_code(tmp_path):
    """A code file that differs from the pinned release is never imported."""
    _install(tmp_path, {"state_dict": {}})
    for relpath in release.CODE_FILES:
        target = tmp_path / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("raise SystemExit('this must not run')\n")

    with pytest.raises(RuntimeError, match="does not match the pinned release"):
        TruForRunner(tmp_path, device="cpu").load()


def test_runner_refuses_checkpoint_that_differs_from_its_manifest(tmp_path):
    _install(tmp_path, {"state_dict": FakeTruFor().state_dict()}, manifest_sha="0" * 64)

    with pytest.raises(RuntimeError, match="manifest"):
        TruForRunner(tmp_path, device="cpu", model_builder=FakeTruFor).load()


def test_runner_will_not_fully_unpickle_an_unverified_checkpoint(tmp_path):
    """Only weights traced to the authors' archive may run arbitrary unpickling."""
    _install(tmp_path, {"state_dict": FakeTruFor().state_dict(), "extra": NotATensor()},
             manifest_sha="actual", verification="unverified")

    with pytest.raises(RuntimeError, match="provenance"):
        TruForRunner(tmp_path, device="cpu", model_builder=FakeTruFor).load()


def test_runner_rejects_weights_for_a_different_architecture(tmp_path):
    _install(tmp_path, {"state_dict": {"unrelated.weight": torch.zeros(1)}})

    with pytest.raises(RuntimeError):  # load_state_dict(strict=True)
        TruForRunner(tmp_path, device="cpu", model_builder=FakeTruFor).load()


def test_runner_accepts_the_checkpoint_file_itself(tmp_path):
    checkpoint = _install(tmp_path, {"state_dict": FakeTruFor().state_dict()})
    runner = TruForRunner(checkpoint, device="cpu", model_builder=FakeTruFor)
    runner.load()
    assert runner.root == tmp_path


def test_runner_records_weights_identity(tmp_path):
    runner = _loaded_fake_runner(tmp_path, manifest_sha="actual")
    checkpoint_sha = sha256_file(tmp_path / release.WEIGHTS_RELPATH)

    assert runner.checkpoint_sha256 == checkpoint_sha
    assert runner.weights_verification == "official_archive_md5"
    assert release.UPSTREAM_COMMIT[:7] in runner.version_tag
    assert checkpoint_sha[:12] in runner.version_tag


def test_runner_without_manifest_says_so(tmp_path):
    assert _loaded_fake_runner(tmp_path).weights_verification == "no_manifest"


# ── Inference ────────────────────────────────────────────────────────────────

def test_runner_file_not_found(tmp_path):
    runner = _loaded_fake_runner(tmp_path)
    with pytest.raises(FileNotFoundError):
        runner.predict(tmp_path / "missing.png")


def test_runner_inference_outputs(tmp_path):
    runner = _loaded_fake_runner(tmp_path)
    result = runner.predict(_image(tmp_path / "x.png", size=(48, 32)), threshold=0.4)

    assert set(result) >= {
        "manipulation_detected", "confidence", "regions",
        "localization_map", "reliability_map", "metadata",
    }
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["manipulation_detected"] == (result["confidence"] >= 0.4)
    for key in ("localization_map", "reliability_map"):
        heatmap = result[key]
        assert heatmap.shape == (32, 48)
        assert heatmap.dtype == np.float32
        assert 0.0 <= heatmap.min() and heatmap.max() <= 1.0

    meta = result["metadata"]
    assert meta["threshold"] == 0.4
    assert meta["image_size"] == [48, 32]
    assert meta["integrity_score"] == pytest.approx(result["confidence"], abs=1e-4)
    assert meta["upstream_commit"] == release.UPSTREAM_COMMIT
    assert meta["inference_time_ms"] >= 0.0


def test_runner_uses_upstream_preprocessing(tmp_path):
    """Pixels / 256 with no mean/std: the network normalises internally."""
    runner = _loaded_fake_runner(tmp_path)
    image = tmp_path / "white.png"
    Image.fromarray(np.full((16, 16, 3), 255, dtype=np.uint8)).save(image)

    runner.predict(image)

    low, high = runner.model.seen_input_range
    assert low == high == pytest.approx(255 / 256)


def test_runner_reports_no_label_and_no_host_path(tmp_path):
    runner = _loaded_fake_runner(tmp_path)
    result = runner.predict(_image(tmp_path / "x.png"))

    # A model wrapper reports a score; it does not declare an image clean.
    assert "assessment" not in result
    assert "status" not in result["metadata"]
    serialised = json.dumps(result["metadata"])
    assert str(tmp_path) not in serialised
    assert str(tmp_path).replace("\\", "/") not in serialised
    assert result["metadata"]["model_file"] == release.CHECKPOINT_NAME


def test_runner_regions_come_from_the_localization_map(tmp_path):
    runner = _loaded_fake_runner(tmp_path)
    loc = np.zeros((100, 100), dtype=np.float32)
    loc[20:60, 30:80] = 0.9
    rel = np.full((100, 100), 0.7, dtype=np.float32)

    regions = runner._regions(loc, rel, threshold=0.5)

    assert len(regions) == 1
    assert regions[0]["bbox"] == [30.0, 20.0, 50.0, 40.0]
    assert regions[0]["confidence"] == pytest.approx(0.9)
    assert regions[0]["extra"]["reliability_mean"] == pytest.approx(0.7)


# ── The real architecture (code fetched, weights optional) ──────────────────

@needs_code
def test_real_architecture_runs_with_random_weights(tmp_path):
    """
    The pinned upstream code imports, builds from our config and produces maps
    at the input resolution. Random weights: this checks plumbing, not accuracy.
    """
    builder = _import_upstream_builder(TRUFOR_ROOT / release.SRC_RELPATH)
    torch.manual_seed(0)
    model = builder(trufor_config()).eval()

    runner = TruForRunner(TRUFOR_ROOT, device="cpu")
    runner.model, runner.loaded = model, True
    result = runner.predict(_image(tmp_path / "x.png", size=(96, 64)))

    assert result["localization_map"].shape == (64, 96)
    assert result["reliability_map"].shape == (64, 96)
    assert 0.0 <= result["confidence"] <= 1.0


@needs_weights
def test_real_weights_load_and_score_the_upstream_samples():
    """The released checkpoint loads strictly and separates the demo pair."""
    samples = TRUFOR_ROOT / release.SAMPLES_RELPATH
    tampered, pristine = samples / "tampered2.png", samples / "pristine1.jpg"
    if not (tampered.is_file() and pristine.is_file()):
        pytest.skip("samples not fetched (scripts/fetch_models.py --samples)")

    runner = TruForRunner(TRUFOR_ROOT)
    runner.load()
    fake = runner.predict(tampered)
    real = runner.predict(pristine)

    assert fake["confidence"] > real["confidence"]
    assert fake["manipulation_detected"] is True
    assert fake["regions"], "the tampered sample should produce a localized region"


def _cocoglide_pairs(limit: int):
    root = TRUFOR_ROOT / release.COCOGLIDE_RELPATH
    if not root.is_dir():
        return []
    images = sorted(
        p for p in root.rglob("*")
        if p.suffix.lower() in {".png", ".jpg"} and "mask" not in p.as_posix().lower()
    )
    masks = {
        p.stem.lower().removesuffix("_mask"): p
        for p in root.rglob("*")
        if p.suffix.lower() in {".png", ".jpg"} and "mask" in p.as_posix().lower()
    }
    pairs = [(img, masks[img.stem.lower()]) for img in images if img.stem.lower() in masks]
    return pairs[:limit]


@needs_weights
def test_cocoglide_heatmaps_overlap_their_ground_truth():
    """
    Day 2 acceptance: on CocoGlide (generative inpainting, unseen in training),
    the localization map is higher inside the ground-truth mask than outside it,
    and at least one sample's thresholded map overlaps its mask substantially.
    """
    pairs = _cocoglide_pairs(limit=8)
    if not pairs:
        pytest.skip("CocoGlide not installed (scripts/fetch_models.py --cocoglide-parquet)")

    runner = TruForRunner(TRUFOR_ROOT)
    runner.load()
    inside_higher, ious = 0, []
    for image, mask_file in pairs:
        heatmap = runner.predict(image)["localization_map"]
        with Image.open(mask_file) as m:
            values = np.asarray(m.convert("L").resize(heatmap.shape[::-1], Image.NEAREST))
        # Copies differ in whether a binary mask is stored as 0/1 or 0/255.
        mask = values > values.max() / 2
        if mask.all() or not mask.any():
            continue
        inside_higher += heatmap[mask].mean() > heatmap[~mask].mean()
        predicted = heatmap >= 0.5
        ious.append((predicted & mask).sum() / max((predicted | mask).sum(), 1))

    assert ious, "no usable CocoGlide masks found"
    assert inside_higher >= 0.75 * len(ious)
    assert max(ious) >= 0.3
