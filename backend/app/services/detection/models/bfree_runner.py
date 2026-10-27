"""
B-Free inference: one logit per image from five 504 px crops.

Reproduces the authors' ``Wrapper5crops`` (grip-unina/B-Free, code/networks):
the whole image goes through the patch embedding unresized; from the embedding
grid it takes the centre and the four corner 36x36-token crops (504 px at
patch 14), replicate-wrapping images smaller than that, runs the ViT on all
five and averages the logits. No resizing anywhere -- resampling erases the
traces the model reads. Checked against the authors' reference logits on their
demo images (tests/unit/test_bfree_runner.py).

What the score is NOT:

  * not calibrated -- the logit orders images; ``sigmoid(logit)`` is not a
    probability of generation;
  * not localised -- one number for the image; an inpainted region in a real
    photograph is a manipulation question, answered by TruFor;
  * not universal -- trained on Stable Diffusion-family fakes; generators far
    from that family, and heavy recompression, can still be missed.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image

from app.services.detection.models import bfree_release as release
from app.services.detection.models.trufor_runner import resolve_device, sha256_file

logger = logging.getLogger(__name__)

SEMANTICS = (
    "sigmoid of the B-Free logit (DINOv2-initialised ViT-B/14, five 504 px crops "
    "averaged); trained bias-free on Stable Diffusion-derived fakes; uncalibrated"
)

LIMITATIONS: List[str] = [
    "Uncalibrated model score: it orders images, it is not a probability.",
    "Trained on fakes made with Stable Diffusion-family models; generators far "
    "from that family can be missed.",
    "Whole-image score with no localisation: an edited region in a captured "
    "photograph is a manipulation question, not a generation one.",
    "It also responds to regions filled in locally (it was trained on real photos "
    "with inpainted patches): it flagged 4 of 20 camera photos with an object removed "
    "by classical inpainting, and 0 of 60 unedited ones. On an edited photo an "
    "indication may mean a filled-in region, not a generated image.",
    "Five 504 px crops cover a large image only partly; the rest is not examined.",
    "'Not indicated' is not evidence that an image was captured by a camera.",
]


def _replicate_wrap(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
    reps_h = max(int(math.ceil(height / x.shape[-2])), 1)
    reps_w = max(int(math.ceil(width / x.shape[-1])), 1)
    return x.repeat(1, 1, reps_h, reps_w)[..., :height, :width]


class FiveCropViT(torch.nn.Module):
    """A timm ViT whose patch embedding sees the whole image, then five crops of tokens."""

    def __init__(self, vit: torch.nn.Module, crop_size: int):
        super().__init__()
        vit.set_input_size(img_size=crop_size)
        self.patch_embed = vit.patch_embed
        vit.patch_embed = torch.nn.Identity()
        self.vit = vit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gh, gw = self.patch_embed.grid_size
        emb = self.patch_embed.proj(x)
        if emb.shape[-2] < gh or emb.shape[-1] < gw:
            emb = _replicate_wrap(emb, max(gh, emb.shape[-2]), max(gw, emb.shape[-1]))
        hs = max((emb.shape[-2] - gh) // 2, 0)
        ws = max((emb.shape[-1] - gw) // 2, 0)
        crops = torch.cat((
            emb[:, :, hs:hs + gh, ws:ws + gw],   # centre
            emb[:, :, :gh, :gw],                 # top-left
            emb[:, :, -gh:, :gw],                # bottom-left
            emb[:, :, -gh:, -gw:],               # bottom-right
            emb[:, :, :gh, -gw:],                # top-right
        ), 0)
        if self.patch_embed.flatten:
            crops = crops.flatten(2).transpose(1, 2)
        crops = self.patch_embed.norm(crops)
        logits = self.vit(crops)
        return torch.stack(torch.split(logits, logits.shape[0] // 5, 0), 0).mean(0)

    def load_upstream_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self.patch_embed.load_state_dict(
            {k[len("patch_embed."):]: v for k, v in state.items() if k.startswith("patch_embed.")},
            strict=True,
        )
        self.vit.load_state_dict(
            {k: v for k, v in state.items() if not k.startswith("patch_embed.")}, strict=True
        )


def build_model() -> FiveCropViT:
    import timm

    vit = timm.create_model(release.TIMM_ARCH, pretrained=False, num_classes=1)
    return FiveCropViT(vit, release.CROP_SIZE)


class BFreeRunner:
    """Loads the pinned B-Free checkpoint and scores one image at a time."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]],
        device: Optional[str] = None,
        model_builder: Optional[Callable[[], torch.nn.Module]] = None,
        verify_hash: bool = True,
    ):
        self.root = Path(model_path) if model_path else None
        self.device = resolve_device(device)
        self._builder = model_builder or build_model
        self._verify_hash = verify_hash
        self.model: Optional[torch.nn.Module] = None
        self.weights_sha256: Optional[str] = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self.model is not None

    @property
    def version_tag(self) -> str:
        return f"bfree-{release.UPSTREAM_COMMIT}.w-{(self.weights_sha256 or 'unknown')[:12]}"

    def weights_path(self) -> Path:
        if self.root is None:
            raise ValueError("B-Free model path is not configured")
        return self.root / release.MODEL_DIR / release.WEIGHTS_FILE

    def load(self) -> None:
        path = self.weights_path()
        if not path.is_file():
            raise FileNotFoundError("B-Free weights are not installed (run scripts/fetch_bfree.py)")
        actual = sha256_file(path)
        if self._verify_hash and actual != release.WEIGHTS_SHA256:
            raise RuntimeError("B-Free checkpoint does not match the pinned SHA-256")
        # A plain state dict: the safe unpickler is enough, so nothing in the
        # file can execute code.
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state = checkpoint.get("model") if isinstance(checkpoint, dict) else None
        if not isinstance(state, dict):
            raise TypeError("B-Free checkpoint does not contain a 'model' state dict")
        model = self._builder()
        if hasattr(model, "load_upstream_state_dict"):
            model.load_upstream_state_dict(state)
        else:
            model.load_state_dict(state, strict=True)
        model.eval()
        self.model = model.to(self.device)
        self.weights_sha256 = actual
        logger.info("Loaded B-Free (%s) on %s", self.version_tag, self.device)

    def identity(self) -> Dict[str, Any]:
        return {
            "upstream_repo": release.UPSTREAM_REPO,
            "upstream_commit": release.UPSTREAM_COMMIT,
            "weights_sha256": self.weights_sha256,
            "device": str(self.device),
        }

    @staticmethod
    def preprocess(file_path: Union[str, Path]) -> tuple[torch.Tensor, List[int]]:
        with Image.open(file_path) as img:
            rgb = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy())
        mean = torch.tensor(release.IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(release.IMAGENET_STD).view(3, 1, 1)
        return ((tensor - mean) / std).unsqueeze(0), [rgb.shape[1], rgb.shape[0]]

    def predict(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("B-Free is not loaded")
        started = time.perf_counter()
        tensor, size = self.preprocess(file_path)
        with self._lock, torch.inference_mode():
            try:
                logit = float(self.model(tensor.to(self.device)).reshape(-1)[0].cpu())
            finally:
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
        if not math.isfinite(logit):
            raise ValueError("B-Free produced a non-finite logit")
        return {
            "logit": logit,
            "score": 1.0 / (1.0 + math.exp(-logit)),
            "metadata": {
                **self.identity(),
                "image_size": size,
                # Under 504 px a side, the crop is filled by repeating the image.
                "replicated": min(size) < release.CROP_SIZE,
                "inference_time_ms": round((time.perf_counter() - started) * 1000.0, 1),
            },
        }
