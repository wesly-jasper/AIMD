"""
The OpenAI CLIP ViT-L/14 image tower, loaded once per process and shared.

Two consumers read the same network:

  * the UnivFD detector, which puts a linear head on its features, and
  * the similarity search, which stores its features as an image embedding.

Each needs different preprocessing, and mixing them up would be silent:

  ``univfd``     CenterCrop(224) with no resize -- the UnivFD test protocol. An
                 image smaller than 224 is zero-padded. The head was trained on
                 unresized crops, so resizing first would hand it pixels whose
                 high-frequency statistics it has never seen.
  ``retrieval``  Resize(224, bicubic) on the short side, then CenterCrop(224), then
                 L2-normalised. A resized copy of an image must embed like the
                 original, and an unresized crop of a half-size copy would show
                 four times the scene.

The checkpoint is OpenAI's TorchScript archive, verified against the SHA-256 in
``univfd_release`` before it is opened. Only the image tower is kept: the text
tower is never used and would cost another 250 MB of GPU memory.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

from app.services.detection.models import univfd_release as release
from app.services.detection.models.trufor_runner import resolve_device, sha256_file

logger = logging.getLogger(__name__)

#: ``(checkpoint_path, precision) -> image tower``: a module mapping a normalised
#: [N, 3, 224, 224] batch to [N, 768] projected features. Injected by unit tests;
#: ``None`` means open_clip's loader for the OpenAI archive.
TowerLoader = Callable[[Path, str], torch.nn.Module]

MODES = ("univfd", "retrieval")


def _open_clip_tower(checkpoint: Path, precision: str) -> torch.nn.Module:
    import open_clip  # deferred: optional at import time, required to load

    model = open_clip.load_openai_model(str(checkpoint), precision=precision, device="cpu")
    return model.visual


def preprocess(img: Image.Image, mode: str) -> torch.Tensor:
    """One image as a normalised [3, 224, 224] float tensor."""
    from torchvision import transforms as T

    if mode == "univfd":
        steps = [T.CenterCrop(release.INPUT_SIZE)]
    elif mode == "retrieval":
        steps = [
            T.Resize(release.INPUT_SIZE, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(release.INPUT_SIZE),
        ]
    else:
        raise ValueError(f"unknown preprocessing mode {mode!r}")
    transform = T.Compose(
        steps + [T.ToTensor(), T.Normalize(release.CLIP_MEAN, release.CLIP_STD)]
    )
    return transform(img.convert("RGB"))


class ClipBackbone:
    """
    The verified CLIP image tower on one device.

    ``root`` is the directory ``scripts/fetch_univfd.py`` installed into; a path
    to ``ViT-L-14.pt`` itself is also accepted. fp16 on CUDA, fp32 on the CPU.
    """

    def __init__(
        self,
        root: Union[str, Path],
        device: Optional[str] = None,
        tower_loader: Optional[TowerLoader] = None,
    ):
        root = Path(root)
        self.checkpoint_path = root if root.suffix == ".pt" else root / release.CLIP_FILE
        self.device = resolve_device(device)
        self.precision = "fp16" if self.device.type == "cuda" else "fp32"
        self.tower_loader = tower_loader or _open_clip_tower
        self.tower: Optional[torch.nn.Module] = None
        self.checkpoint_sha256: Optional[str] = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self.tower is not None

    @property
    def version_tag(self) -> str:
        return f"openai-vit-l14-quickgelu/{(self.checkpoint_sha256 or 'unknown')[:12]}"

    def load(self) -> None:
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                "CLIP ViT-L/14 is not installed (run scripts/fetch_univfd.py)"
            )
        actual = sha256_file(self.checkpoint_path)
        if actual != release.CLIP_SHA256:
            raise RuntimeError("CLIP checkpoint does not match the pinned SHA-256")
        tower = self.tower_loader(self.checkpoint_path, self.precision)
        tower.eval()
        self.tower = tower.to(self.device)
        self.checkpoint_sha256 = actual
        logger.info("Loaded CLIP ViT-L/14 image tower (%s) on %s", self.precision, self.device)

    def features(self, images: list[Image.Image], mode: str) -> torch.Tensor:
        """
        Projected 768-d features as float32 on the CPU, one row per image.

        ``retrieval`` rows are L2-normalised; ``univfd`` rows are not, because the
        head was trained on the raw projection.
        """
        if self.tower is None:
            raise RuntimeError("CLIP backbone is not loaded")
        batch = torch.stack([preprocess(img, mode) for img in images])
        # The input meets the patch-embedding convolution first. In fp16 open_clip
        # leaves some parameters (class embedding, layer norms) in fp32, so the
        # first parameter's dtype is not the one the input has to match.
        conv = getattr(self.tower, "conv1", None)
        dtype = conv.weight.dtype if conv is not None else next(self.tower.parameters()).dtype
        with self._lock, torch.inference_mode():
            out = self.tower(batch.to(self.device, dtype=dtype)).float().cpu()
        if out.ndim != 2 or out.shape[1] != release.FEATURE_DIM:
            raise ValueError(f"CLIP tower returned features of shape {tuple(out.shape)}")
        if mode == "retrieval":
            out = torch.nn.functional.normalize(out, dim=1)
        return out

    def embed(self, file_path: Union[str, Path]) -> np.ndarray:
        """The retrieval embedding of one image file: float32 [768], unit length."""
        with Image.open(file_path) as img:
            return self.features([img], "retrieval")[0].numpy()


# ── Process-wide instance ─────────────────────────────────────────────────────

_instances: Dict[Tuple[str, str], Union[ClipBackbone, Exception]] = {}
_instances_lock = threading.Lock()


def get_clip_backbone(
    root: Union[str, Path],
    device: Optional[str] = None,
    tower_loader: Optional[TowerLoader] = None,
) -> ClipBackbone:
    """
    The loaded backbone for ``root`` on ``device``, loading it on first use.

    A failed load is remembered and re-raised: hashing 890 MB on every upload to
    rediscover that the file is missing would stall ingestion.
    """
    key = (str(Path(root).resolve()), str(resolve_device(device)))
    with _instances_lock:
        cached = _instances.get(key)
        if cached is None:
            backbone = ClipBackbone(root, device=device, tower_loader=tower_loader)
            try:
                backbone.load()
                cached = backbone
            except Exception as exc:
                cached = exc
            _instances[key] = cached
    if isinstance(cached, Exception):
        raise cached
    return cached


def reset_clip_backbones() -> None:
    """Forget every loaded or failed backbone (tests, or after installing weights)."""
    with _instances_lock:
        _instances.clear()
