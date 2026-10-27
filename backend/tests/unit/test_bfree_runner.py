"""
B-Free: the five-crop wrapper, the pinned-checkpoint loader, and the detector.

The architecture tests use a tiny timm ViT so they run without the weights. The
reference test reproduces the authors' demo logits and runs only when the
weights and the upstream demo images are installed.
"""
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from app.schemas.detection import DetectorAvailability, DetectorExecutionStatus
from app.services.detection.detectors.image.bfree_detector import BFreeDetector
from app.services.detection.models import bfree_release as release
from app.services.detection.models.bfree_runner import BFreeRunner, FiveCropViT

BACKEND = Path(__file__).resolve().parents[2]


def _tiny(crop=28):
    import timm

    vit = timm.create_model(
        "vit_small_patch14_reg4_dinov2", pretrained=False, num_classes=1,
        img_size=crop, embed_dim=32, depth=1, num_heads=2,
    )
    return FiveCropViT(vit, crop)


def test_five_crops_cover_centre_and_corners_without_resizing():
    model = _tiny().eval()
    with torch.inference_mode():
        large = model(torch.rand(1, 3, 70, 98))
        small = model(torch.rand(1, 3, 14, 20))  # smaller than one crop: wrapped
    assert large.shape == (1, 1) and small.shape == (1, 1)


def test_runner_refuses_a_checkpoint_that_does_not_match_the_pin(tmp_path):
    target = tmp_path / release.MODEL_DIR / release.WEIGHTS_FILE
    target.parent.mkdir(parents=True)
    torch.save({"model": _tiny().state_dict()}, target)
    runner = BFreeRunner(tmp_path, device="cpu", model_builder=_tiny)
    with pytest.raises(RuntimeError, match="pinned SHA-256"):
        runner.load()


def test_runner_scores_an_image(tmp_path):
    model = _tiny()
    state = {**{f"patch_embed.{k}": v for k, v in model.patch_embed.state_dict().items()},
             **model.vit.state_dict()}
    target = tmp_path / release.MODEL_DIR / release.WEIGHTS_FILE
    target.parent.mkdir(parents=True)
    torch.save({"model": state}, target)
    runner = BFreeRunner(tmp_path, device="cpu", model_builder=_tiny, verify_hash=False)
    runner.load()
    image = tmp_path / "x.png"
    Image.fromarray((np.random.default_rng(0).random((40, 60, 3)) * 255).astype(np.uint8)).save(image)
    out = runner.predict(image)
    assert 0.0 < out["score"] < 1.0
    assert out["metadata"]["replicated"] is True  # under 504 px


def test_detector_reports_on_the_ai_generation_branch(tmp_path):
    class Runner:
        version_tag = "bfree-test"

        def predict(self, path):
            return {"logit": 2.0, "score": 0.88, "metadata": {}}

    image = tmp_path / "x.png"
    Image.new("RGB", (8, 8)).save(image)
    result = BFreeDetector(Runner()).detect(image)
    assert result.detector == "BFreeDetector"
    assert result.detector_version.startswith("1.0.0+bfree-test")
    assert result.manipulation_detected is True
    assert result.metadata["branch"] == "ai_generation"
    assert "B-Free" in result.metadata["semantics"]

    missing = BFreeDetector(None, initialization_error="weights not installed").detect(image)
    assert missing.availability is DetectorAvailability.UNAVAILABLE


@pytest.mark.skipif(
    not (BACKEND / "models" / "bfree" / release.MODEL_DIR / release.WEIGHTS_FILE).is_file()
    or not (BACKEND / "models" / "_candidates" / "bfree" / "code" / "demo_images").is_dir(),
    reason="B-Free weights or upstream demo images not installed",
)
def test_matches_the_authors_reference_logits():
    """code/demo_images/results.csv in grip-unina/B-Free at the pinned commit."""
    runner = BFreeRunner(BACKEND / "models" / "bfree")
    runner.load()
    demo = BACKEND / "models" / "_candidates" / "bfree" / "code" / "demo_images"
    reference = {"img0000.png": -5.9374785, "img0001.png": -4.441922,
                 "img0002.png": 4.430519, "img0003.png": 3.8499813}
    for name, logit in reference.items():
        assert runner.predict(demo / name)["logit"] == pytest.approx(logit, abs=1e-3)
