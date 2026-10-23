"""
TruForRunner: loading, verification and inference.

Most tests inject a tiny stand-in network through ``model_builder`` because
TruFor's code is research-only and cannot ship with the repository. Tests that
need the real architecture, the real weights or CocoGlide are skipped unless
``scripts/fetch_models.py`` has installed them into ``backend/models/trufor``.
"""
import gc
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from app.services.detection.models import trufor_release as release
from app.services.detection.models.trufor_runner import (
    TILE_GRID,
    TruForRunner,
    _import_upstream_builder,
    sha256_file,
    tile_spans,
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


class HeadedFakeTruFor(FakeTruFor):
    """FakeTruFor with a ``detection`` MLP over pooled maps, as upstream's ``confpool`` has."""

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.detection = torch.nn.Linear(4, 1)


class RecordingPooling:
    """Stands in for upstream's pooling: a weighted mean and the max, per channel."""

    def __init__(self):
        self.input_shapes = []

    def __call__(self, x, log_w=None):
        self.input_shapes.append(tuple(x.shape))
        flat = x.flatten(2)
        weights = (
            torch.full_like(flat[:, :1], 1.0 / flat.shape[-1]) if log_w is None
            else torch.softmax(log_w.flatten(2), dim=-1)
        )
        return torch.cat(((flat * weights).sum(-1), flat.amax(-1)), dim=1)


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


# ── Tiled inference ──────────────────────────────────────────────────────────

def _tiled_fake_runner(root: Path, builder=HeadedFakeTruFor, pooling=None) -> TruForRunner:
    torch.manual_seed(0)
    _install(root, {"state_dict": builder().state_dict()})
    runner = TruForRunner(
        root, device="cpu", model_builder=builder,
        pooling=pooling or RecordingPooling(), tile_size=32, tile_overlap=8,
    )
    runner.load()
    return runner


@pytest.mark.parametrize("length", [33, 40, 56, 61, 80, 83, 100, 1000])
def test_tile_spans_cover_the_axis_on_the_jpeg_grid(length):
    spans = tile_spans(length, 32, 8)

    assert spans[0][0] == 0 and spans[-1][1] == length
    for (start, end), (next_start, _) in zip(spans, spans[1:]):
        assert end - next_start >= 8  # at least the nominal overlap
    for start, end in spans:
        assert start % TILE_GRID == 0
        assert 32 <= end - start <= 32 + TILE_GRID - 1


def test_tile_spans_leave_a_short_axis_whole():
    assert tile_spans(20, 32, 8) == [(0, 20)]
    assert tile_spans(32, 32, 8) == [(0, 32)]


@pytest.mark.parametrize("size, overlap", [(32, -1), (32, 16), (32, 20), (32, 5), (1024, 100)])
def test_invalid_tile_settings_are_refused(size, overlap):
    with pytest.raises(ValueError):
        TruForRunner(None, tile_size=size, tile_overlap=overlap)


def test_small_images_run_whole_and_say_so(tmp_path):
    runner = _tiled_fake_runner(tmp_path)
    meta = runner.predict(_image(tmp_path / "x.png", size=(32, 32)))["metadata"]

    assert meta["tiling"] == {"tiled": False, "tiles": 1, "tile_size": 32, "tile_overlap": 8}
    assert "limitations" not in meta
    assert runner.pooling.input_shapes == []  # the model's own score was used


def test_tiled_maps_match_a_whole_image_pass(tmp_path):
    """The stand-in network is pointwise, so blending overlapping tiles must be lossless."""
    image = _image(tmp_path / "x.png", size=(83, 61))
    tiled = _tiled_fake_runner(tmp_path / "tiled").predict(image)
    whole = _tiled_fake_runner(tmp_path / "whole")
    whole.tile_size = 10_000
    reference = whole.predict(image)

    for key in ("localization_map", "reliability_map"):
        assert tiled[key].shape == (61, 83)
        np.testing.assert_allclose(tiled[key], reference[key], atol=1e-5)
    meta = tiled["metadata"]
    assert meta["tiling"] == {"tiled": True, "tiles": 6, "tile_size": 32, "tile_overlap": 8}
    assert meta["limitations"] and "32 px tiles" in meta["limitations"][0]


def test_tiled_score_is_the_detection_head_on_the_whole_maps(tmp_path):
    """The head runs once, on full-size maps, and scores what a whole-image head would."""
    image = _image(tmp_path / "x.png", size=(80, 56))
    runner = _tiled_fake_runner(tmp_path)
    tensor, _ = runner._preprocess(image)
    pred, conf, _det = runner._forward(tensor, torch.device("cpu"))
    expected = float(torch.sigmoid(runner._detection_head(pred, conf, torch.device("cpu"))))
    runner.pooling.input_shapes.clear()

    result = runner.predict(image)

    assert runner.pooling.input_shapes == [(1, 1, 56, 80), (1, 1, 56, 80)]
    assert result["metadata"]["integrity_score"] == pytest.approx(expected, abs=1e-5)


def test_tiled_tiles_are_on_the_grid_and_cover_the_image(tmp_path):
    runner = _tiled_fake_runner(tmp_path)
    seen = []
    forward = runner._forward

    def recording_forward(tensor, device):
        seen.append(tuple(tensor.shape[-2:]))
        return forward(tensor, device)

    runner._forward = recording_forward
    runner.predict(_image(tmp_path / "x.png", size=(80, 56)))

    assert len(seen) == 6
    assert all(32 <= h <= 39 and 32 <= w <= 39 for h, w in seen)


def test_tiling_without_a_detection_head_is_refused(tmp_path):
    """No silent fallback to an average of per-tile scores."""
    runner = _tiled_fake_runner(tmp_path, builder=FakeTruFor)
    with pytest.raises(RuntimeError, match="detection head"):
        runner.predict(_image(tmp_path / "x.png", size=(80, 56)))


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


@needs_weights
def test_real_detection_head_reproduces_the_models_own_score(tmp_path):
    """
    The tiled path pools the stitched maps with upstream's head; on a whole image
    that head must give exactly the score the model computes itself.
    """
    runner = TruForRunner(TRUFOR_ROOT)
    runner.load()
    tensor, _ = runner._preprocess(_image(tmp_path / "x.png", size=(256, 192)))
    with torch.inference_mode():
        pred, conf, det = runner._forward(tensor, runner.device)
        head = runner._detection_head(pred, conf, runner.device)

    assert torch.allclose(head, det, atol=1e-4)


@needs_weights
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_real_1080p_image_runs_tiled_within_a_4gb_card(tmp_path):
    """
    A whole 1080p pass peaks near 4 GB; tiled, TruFor's weights and activations
    stay near 2.2 GB, leaving room for the CLIP backbone (~0.85 GB) beside it.
    Measured above whatever earlier tests left on the device.
    """
    sample = TRUFOR_ROOT / release.SAMPLES_RELPATH / "pristine1.jpg"
    if not sample.is_file():
        pytest.skip("samples not fetched (scripts/fetch_models.py --samples)")
    image = tmp_path / "1080p.png"
    with Image.open(sample) as img:
        img.convert("RGB").resize((1920, 1080), Image.BICUBIC).save(image)

    gc.collect()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    runner = TruForRunner(TRUFOR_ROOT, device="cuda")
    runner.load()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    result = runner.predict(image)
    peak = torch.cuda.max_memory_allocated() - baseline

    assert result["metadata"]["tiling"]["tiled"] is True
    assert result["metadata"]["tiling"]["tiles"] == 4
    assert result["localization_map"].shape == (1080, 1920)
    assert peak < 2.5 * 1024 ** 3, f"TruFor peak {peak / 1024 ** 2:.0f} MB"


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
