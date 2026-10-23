"""
ClipBackbone: preprocessing, hash refusal, and the shared process-wide instance.

The OpenAI checkpoint is replaced by a dummy file whose hash the test pins, and
the tower by a tiny module, through the ``tower_loader`` hook.
"""
import numpy as np
import pytest
import torch
from PIL import Image

from app.services.detection.models import univfd_release as release
from app.services.detection.models.clip_backbone import (
    ClipBackbone,
    get_clip_backbone,
    preprocess,
    reset_clip_backbones,
)
from app.services.detection.models.trufor_runner import sha256_file


class TinyTower(torch.nn.Module):
    """[N, 3, 224, 224] -> [N, 768], and remembers what it was given."""

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(3, release.FEATURE_DIM)
        self.seen_shape = None

    def forward(self, x):
        self.seen_shape = tuple(x.shape)
        return self.proj(x.mean(dim=(2, 3)))


class NarrowTower(torch.nn.Module):
    """A tower of the wrong model: 10-d features."""

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(3, 10)

    def forward(self, x):
        return self.proj(x.mean(dim=(2, 3)))


@pytest.fixture(autouse=True)
def fresh_instances():
    reset_clip_backbones()
    yield
    reset_clip_backbones()


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    path = tmp_path / release.CLIP_FILE
    path.write_bytes(b"stand-in for the OpenAI TorchScript archive")
    monkeypatch.setattr(release, "CLIP_SHA256", sha256_file(path))
    return path


def _loader(calls=None):
    def load(path, precision):
        if calls is not None:
            calls.append((path, precision))
        return TinyTower()
    return load


# ── Preprocessing ────────────────────────────────────────────────────────────

def test_univfd_mode_crops_without_resizing():
    # A 448 px image: an unresized centre crop sees the middle quarter only.
    arr = np.zeros((448, 448, 3), dtype=np.uint8)
    arr[112:336, 112:336] = 255
    out = preprocess(Image.fromarray(arr), "univfd")
    assert out.shape == (3, 224, 224)
    white = (1.0 - release.CLIP_MEAN[0]) / release.CLIP_STD[0]
    assert torch.allclose(out[0], torch.full((224, 224), white), atol=1e-4)


def test_univfd_mode_zero_pads_small_images():
    out = preprocess(Image.new("RGB", (100, 100), (255, 255, 255)), "univfd")
    assert out.shape == (3, 224, 224)
    black = (0.0 - release.CLIP_MEAN[0]) / release.CLIP_STD[0]
    assert out[0, 0, 0].item() == pytest.approx(black, abs=1e-4)


def test_retrieval_mode_resizes_so_a_half_size_copy_looks_the_same():
    arr = np.zeros((448, 448, 3), dtype=np.uint8)
    arr[:, :224] = 255
    full = preprocess(Image.fromarray(arr), "retrieval")
    half = preprocess(Image.fromarray(arr).resize((224, 224)), "retrieval")
    assert full.shape == half.shape == (3, 224, 224)
    assert torch.mean(torch.abs(full - half)).item() < 0.05


def test_unknown_mode_is_refused():
    with pytest.raises(ValueError):
        preprocess(Image.new("RGB", (10, 10)), "imagenet")


# ── Loading ──────────────────────────────────────────────────────────────────

def test_cpu_load_is_fp32_and_records_its_version(checkpoint):
    calls = []
    backbone = ClipBackbone(checkpoint.parent, device="cpu", tower_loader=_loader(calls))
    backbone.load()
    assert calls == [(checkpoint, "fp32")]
    assert backbone.precision == "fp32"
    assert backbone.version_tag == f"openai-vit-l14-quickgelu/{release.CLIP_SHA256[:12]}"


def test_a_path_to_the_checkpoint_itself_is_accepted(checkpoint):
    backbone = ClipBackbone(checkpoint, device="cpu", tower_loader=_loader())
    assert backbone.checkpoint_path == checkpoint


def test_checkpoint_failing_its_hash_is_refused(checkpoint, monkeypatch):
    monkeypatch.setattr(release, "CLIP_SHA256", "0" * 64)
    calls = []
    backbone = ClipBackbone(checkpoint.parent, device="cpu", tower_loader=_loader(calls))
    with pytest.raises(RuntimeError, match="pinned SHA-256"):
        backbone.load()
    assert calls == [], "an unverified checkpoint was opened"
    assert not backbone.loaded


def test_missing_checkpoint_is_not_installed(tmp_path):
    with pytest.raises(FileNotFoundError):
        ClipBackbone(tmp_path, device="cpu", tower_loader=_loader()).load()


# ── Features ─────────────────────────────────────────────────────────────────

def test_retrieval_embedding_is_unit_length(checkpoint, tmp_path):
    path = tmp_path / "x.png"
    Image.new("RGB", (300, 200), (10, 200, 90)).save(path)
    backbone = ClipBackbone(checkpoint.parent, device="cpu", tower_loader=_loader())
    backbone.load()
    vec = backbone.embed(path)
    assert vec.dtype == np.float32 and vec.shape == (768,)
    assert float(np.linalg.norm(vec)) == pytest.approx(1.0, abs=1e-5)
    assert backbone.tower.seen_shape == (1, 3, 224, 224)


def test_univfd_features_are_not_normalised(checkpoint):
    backbone = ClipBackbone(checkpoint.parent, device="cpu", tower_loader=_loader())
    backbone.load()
    raw = backbone.features([Image.new("RGB", (224, 224), (255, 255, 255))], "univfd")
    assert raw.shape == (1, 768)
    assert float(raw.norm()) != pytest.approx(1.0, abs=1e-3)


def test_features_before_load_raise(checkpoint):
    with pytest.raises(RuntimeError, match="not loaded"):
        ClipBackbone(checkpoint.parent, device="cpu").features([], "univfd")


def test_wrong_feature_width_is_refused(checkpoint):
    backbone = ClipBackbone(
        checkpoint.parent, device="cpu",
        tower_loader=lambda _p, _q: NarrowTower(),
    )
    backbone.load()
    with pytest.raises(ValueError, match="shape"):
        backbone.features([Image.new("RGB", (224, 224))], "univfd")


# ── Shared instance ──────────────────────────────────────────────────────────

def test_backbone_is_loaded_once_per_root_and_device(checkpoint):
    calls = []
    first = get_clip_backbone(checkpoint.parent, device="cpu", tower_loader=_loader(calls))
    second = get_clip_backbone(checkpoint.parent, device="cpu", tower_loader=_loader(calls))
    assert first is second
    assert len(calls) == 1


def test_a_failed_load_is_remembered_until_reset(checkpoint, monkeypatch):
    good = release.CLIP_SHA256
    monkeypatch.setattr(release, "CLIP_SHA256", "0" * 64)
    calls = []
    for _ in range(2):
        with pytest.raises(RuntimeError):
            get_clip_backbone(checkpoint.parent, device="cpu", tower_loader=_loader(calls))

    monkeypatch.setattr(release, "CLIP_SHA256", good)
    with pytest.raises(RuntimeError):  # still the remembered failure
        get_clip_backbone(checkpoint.parent, device="cpu", tower_loader=_loader(calls))

    reset_clip_backbones()
    assert get_clip_backbone(
        checkpoint.parent, device="cpu", tower_loader=_loader(calls)
    ).loaded
    assert len(calls) == 1
