"""
UnivFD inference: the released linear head on CLIP ViT-L/14 features.

UnivFD (Ojha, Li, Lee, CVPR 2023) classifies an image as real or generated with a
single ``Linear(768, 1)`` trained on ProGAN images over frozen CLIP features. Its
output is one logit; ``sigmoid(logit) > 0.5`` is the paper's "fake" decision.

What that score is NOT, and what every result says about itself:

  * not calibrated -- no reliability curve has been fitted, so 0.8 is not "80%";
  * not localised  -- one number for the whole image, so it says nothing about
                      which part might be generated, and an inpainted region in a
                      real photograph usually scores as real;
  * strongest on GANs -- the paper reports roughly 70-80% average precision on
                      diffusion models (Guided, Glide, LDM) against near-perfect
                      on GAN families;
  * sensitive to processing -- resizing and recompression move the score, which
                      is why the input is cropped, never resized.
"""
from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from PIL import Image

from app.services.detection.models import univfd_release as release
from app.services.detection.models.clip_backbone import ClipBackbone, get_clip_backbone
from app.services.detection.models.trufor_runner import sha256_file

logger = logging.getLogger(__name__)

SEMANTICS = (
    "sigmoid of a linear-probe logit on CLIP ViT-L/14 image features (UnivFD); "
    "trained on ProGAN real/fake images; uncalibrated"
)

LIMITATIONS: List[str] = [
    "Uncalibrated model score: it orders images, it is not a probability.",
    "Trained on ProGAN only; the authors report markedly lower accuracy on "
    "diffusion models (roughly 70-80% AP on Guided, Glide and LDM) than on GANs.",
    "Whole-image score with no localisation: a locally inpainted or spliced region "
    "in a captured photograph will usually not be detected.",
    "Resizing, recompression and screenshots shift the score; a social-media copy "
    "may score differently from its original.",
    "'Not indicated' is not evidence that an image was captured by a camera.",
]


class UnivFDRunner:
    """The UnivFD head plus the shared CLIP backbone, for one image at a time."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]],
        device: Optional[str] = None,
        backbone_provider: Optional[Callable[[], ClipBackbone]] = None,
    ):
        self.root = Path(model_path) if model_path else None
        self.device = device
        self._backbone_provider = backbone_provider
        self.backbone: Optional[ClipBackbone] = None
        self.head: Optional[torch.nn.Linear] = None
        self.head_sha256: Optional[str] = None

    @property
    def loaded(self) -> bool:
        return self.head is not None and self.backbone is not None

    @property
    def version_tag(self) -> str:
        clip = (self.backbone.checkpoint_sha256 if self.backbone else None) or "unknown"
        return f"univfd-{release.UPSTREAM_COMMIT[:7]}.clip-{clip[:12]}.fc-{(self.head_sha256 or 'unknown')[:12]}"

    def load(self) -> None:
        if self.root is None:
            raise ValueError("UnivFD model path is not configured")
        head_path = self.root / release.HEAD_FILE
        if not head_path.is_file():
            raise FileNotFoundError("UnivFD head is not installed (run scripts/fetch_univfd.py)")
        actual = sha256_file(head_path)
        if actual != release.HEAD_SHA256:
            raise RuntimeError("UnivFD head does not match the pinned SHA-256")

        state = torch.load(head_path, map_location="cpu", weights_only=True)
        head = torch.nn.Linear(release.FEATURE_DIM, 1)
        head.load_state_dict(state, strict=True)
        head.eval()

        provider = self._backbone_provider or (
            lambda: get_clip_backbone(self.root, device=self.device)
        )
        self.backbone = provider()
        # fp32 on the CPU: 769 parameters, and fp16 features are upcast first.
        self.head = head
        self.head_sha256 = actual

    def identity(self) -> Dict[str, Any]:
        return {
            "upstream_repo": release.UPSTREAM_REPO,
            "upstream_commit": release.UPSTREAM_COMMIT,
            "clip_sha256": self.backbone.checkpoint_sha256 if self.backbone else None,
            "head_sha256": self.head_sha256,
            "clip_precision": self.backbone.precision if self.backbone else None,
            "device": str(self.backbone.device) if self.backbone else None,
        }

    def predict(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        if not self.loaded:
            raise RuntimeError("UnivFD is not loaded")
        started = time.perf_counter()
        with Image.open(file_path) as img:
            size = list(img.size)
            features = self.backbone.features([img], "univfd")
        with torch.inference_mode():
            logit = float(self.head(features).reshape(-1)[0])
        if not math.isfinite(logit):
            raise ValueError("UnivFD produced a non-finite logit")
        return {
            "logit": logit,
            "score": 1.0 / (1.0 + math.exp(-logit)),
            "metadata": {
                **self.identity(),
                "image_size": size,
                # Images under 224 px on a side are zero-padded, as upstream does;
                # the score is less reliable for them.
                "padded": min(size) < release.INPUT_SIZE,
                "inference_time_ms": round((time.perf_counter() - started) * 1000.0, 1),
            },
        }
