"""
TruFor inference, run through the authors' own architecture code.

TruFor (Guillaro et al., "TruFor: Leveraging all-round clues for trustworthy
image forgery detection and localization", CVPR 2023) produces three outputs for
an image:

  * a localization map  -- per-pixel score that the pixel was manipulated
  * a reliability map   -- per-pixel confidence in the localization map; low where
                           the model knows it is guessing (flat or saturated areas)
  * an integrity score  -- one image-level detection score

The code and weights are research / non-profit only and are not in this
repository: ``scripts/fetch_models.py`` installs them below a TruFor root
(``backend/models/trufor`` by default) and records their hashes in a manifest.
This runner checks every code file against the hashes pinned in
``trufor_release`` before importing any of it, and checks the checkpoint against
the manifest before unpickling it, so a modified file is refused rather than run.

Memory grows with the pixel count: a whole 1080p pass needs ~4 GB of GPU memory,
more than a 4 GB laptop card has (and Windows spills into system RAM rather than
raising out-of-memory, so the CPU fallback never fires). Larger images therefore
run in overlapping tiles. Downscaling instead was measured and rejected: it
erases the noise and compression traces the model reads (image-level AUC on
CocoGlide fell from 0.81 to 0.62 at 0.67x). Tiling keeps the image-level score
global: the tile logits are blended into full-resolution maps and upstream's own
detection head (confidence-weighted statistics pooling, then an MLP) runs once
over them, exactly as it would on a whole-image pass.

The previous runner loaded "a .onnx or a pickled nn.Module" and applied ImageNet
normalisation itself. The released checkpoint is neither -- it is a state dict
for an architecture that normalises internally -- so it could never have loaded
the real model, and its preprocessing would have been wrong if it had.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import logging
import sys
import threading
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from app.services.detection.models import trufor_release as release

logger = logging.getLogger(__name__)

#: Builds an ``nn.Module`` with TruFor's interface from a config. Injected by unit
#: tests; ``None`` means the upstream ``myEncoderDecoder``.
ModelBuilder = Callable[[Any], torch.nn.Module]

#: ``(x, log_w=None) -> pooled statistics``: upstream's
#: ``weighted_statistics_pooling``, which the detection head is trained on.
#: Injected by unit tests together with ``model_builder``.
Pooling = Callable[..., torch.Tensor]

_CHUNK = 1024 * 1024

#: Tile origins stay on the 8x8 JPEG block grid, so every tile sees the
#: compression blocks aligned as they are in the whole image.
TILE_GRID = 8


def tiled_limitation(tile_size: int) -> str:
    return (
        f"Localisation was computed in overlapping {tile_size} px tiles: a "
        "manipulated area that fills most of a tile has little untouched context "
        "to be compared against, and the integrity score can differ slightly (up "
        "to ~0.07 in testing) from a single whole-image pass."
    )


def tile_spans(length: int, size: int, overlap: int) -> List[tuple[int, int]]:
    """
    ``(start, end)`` spans covering ``[0, length)`` with tiles of about ``size``.

    Consecutive tiles overlap by at least ``overlap``. Every start is a multiple
    of ``TILE_GRID``; the last tile therefore ends at ``length`` and may be up to
    ``TILE_GRID - 1`` pixels longer than ``size``.
    """
    if length <= size:
        return [(0, length)]
    step = size - overlap
    starts = list(range(0, length - size, step))
    last = (length - size) // TILE_GRID * TILE_GRID
    if last > starts[-1]:
        starts.append(last)
    return [(s, s + size) for s in starts[:-1]] + [(starts[-1], length)]


def _blend_ramp(start: int, end: int, length: int, overlap: int) -> torch.Tensor:
    """Per-pixel blend weight along one axis: rising over an edge shared with a neighbour."""
    ramp = torch.ones(end - start)
    if overlap == 0:
        return ramp
    rise = torch.arange(1, overlap + 1, dtype=torch.float32) / (overlap + 1)
    if start > 0:
        ramp[:overlap] = rise
    if end < length:
        ramp[-overlap:] = torch.minimum(ramp[-overlap:], rise.flip(0))
    return ramp


class _Node(dict):
    """The attribute-and-``in`` access upstream expects from its yacs config."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def trufor_config() -> _Node:
    """
    The inference config of the released model (upstream ``trufor.yaml`` plus the
    yacs defaults it relies on). Built here so yacs is not a dependency and
    nothing upstream that is not pinned gets imported.
    """
    return _Node(
        MODEL=_Node(
            NAME="detconfcmx",
            MODS=("RGB", "NP++"),
            PRETRAINED="",  # weights come from the checkpoint, not ImageNet init
            EXTRA=_Node(
                BACKBONE="mit_b2",
                DECODER="MLPDecoder",
                DECODER_EMBED_DIM=512,
                PREPRC="imagenet",
                BN_EPS=0.001,
                BN_MOMENTUM=0.1,
                DETECTION="confpool",
                CONF=True,
            ),
        ),
        DATASET=_Node(NUM_CLASSES=2),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _import_upstream_builder(src_dir: Path) -> ModelBuilder:
    """
    Import ``myEncoderDecoder`` from the verified upstream code.

    Upstream code imports itself as the top-level package ``models``
    (``from models.DnCNN import make_net``), so it is registered under that name.
    A ``models`` module from anywhere else is refused rather than shadowed: one
    of them would silently be the wrong code.
    """
    package_dir = src_dir / "models"
    init_file = (package_dir / "__init__.py").resolve()
    existing = sys.modules.get("models")
    if existing is not None:
        existing_file = getattr(existing, "__file__", None)
        if not existing_file or Path(existing_file).resolve() != init_file:
            raise RuntimeError("a different top-level 'models' package is already imported")

    with warnings.catch_warnings():
        # Upstream imports ``timm.models.layers``, deprecated (not removed) in timm 1.x.
        warnings.simplefilter("ignore", FutureWarning)
        if existing is None:
            spec = importlib.util.spec_from_file_location(
                "models", init_file, submodule_search_locations=[str(package_dir)]
            )
            if spec is None or spec.loader is None:
                raise ImportError("cannot create an import spec for the TruFor code")
            module = importlib.util.module_from_spec(spec)
            sys.modules["models"] = module
            try:
                spec.loader.exec_module(module)
            except Exception:
                del sys.modules["models"]
                raise
        builder_module = importlib.import_module("models.cmx.builder_np_conf")

    def build(cfg):
        # The backbone is imported when the model is constructed, not above.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            return builder_module.myEncoderDecoder(cfg=cfg)

    return build


def resolve_device(requested: Optional[str]) -> torch.device:
    if requested in (None, "", "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


class TruForRunner:
    """
    Loads the released TruFor model and runs it on one image at a time.

    ``model_path`` is the TruFor root that ``fetch_models.py`` installed into. A
    path to the checkpoint file itself is also accepted, for which the root is
    two levels up (``<root>/weights/trufor.pth.tar``).

    ``model_builder`` replaces the upstream architecture; it exists for unit
    tests, which cannot ship TruFor's code. With a builder injected the upstream
    code is neither verified nor imported, and ``pooling`` stands in for
    upstream's pooling in the tiled detection head.

    Images of more than ``tile_size ** 2`` pixels run in tiles of about
    ``tile_size`` overlapping by ``tile_overlap``; smaller ones in one pass.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        model_builder: Optional[ModelBuilder] = None,
        pooling: Optional[Pooling] = None,
        tile_size: int = 1024,
        tile_overlap: int = 128,
    ):
        if tile_overlap < 0 or 2 * tile_overlap >= tile_size:
            raise ValueError("TruFor tile overlap must be at least 0 and under half the tile size")
        if (tile_size - tile_overlap) % TILE_GRID:
            raise ValueError(
                f"TruFor tile size minus overlap must be a multiple of {TILE_GRID}"
            )
        self.model_path = Path(model_path) if model_path else None
        self.device = resolve_device(device)
        self.model_builder = model_builder
        self.pooling = pooling
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.model: Optional[torch.nn.Module] = None
        self.loaded = False
        self.root: Optional[Path] = None
        self.checkpoint_path: Optional[Path] = None
        self.checkpoint_sha256: Optional[str] = None
        self.weights_verification: Optional[str] = None
        # One forward pass at a time: the OOM fallback moves the model between
        # devices, and a shared runner may be called from several workers.
        self._lock = threading.Lock()

    # ── Identity ─────────────────────────────────────────────────────────────

    @property
    def version_tag(self) -> str:
        """Which code and which weights produced a result, short enough for a version string."""
        weights = (self.checkpoint_sha256 or "unknown")[:12]
        return f"trufor-{release.UPSTREAM_COMMIT[:7]}.w-{weights}"

    def identity(self) -> Dict[str, Any]:
        return {
            "upstream_repo": release.UPSTREAM_REPO,
            "upstream_commit": release.UPSTREAM_COMMIT,
            "model_file": self.checkpoint_path.name if self.checkpoint_path else None,
            "checkpoint_sha256": self.checkpoint_sha256,
            "weights_verification": self.weights_verification,
        }

    # ── Loading ──────────────────────────────────────────────────────────────

    def _resolve_layout(self) -> None:
        if not self.model_path:
            raise ValueError("TruFor model path is not configured")
        path = self.model_path
        if path.is_file():
            self.checkpoint_path = path
            self.root = path.parent.parent
        elif path.is_dir():
            self.root = path
            self.checkpoint_path = path / release.WEIGHTS_RELPATH
        else:
            raise FileNotFoundError("TruFor model path does not exist")
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                "TruFor weights are not installed (run scripts/fetch_models.py)"
            )

    def _verify_code(self) -> Path:
        assert self.root is not None
        for relpath, expected in release.CODE_FILES.items():
            target = self.root / relpath
            if not target.is_file():
                raise FileNotFoundError(f"TruFor code file missing: {relpath}")
            if sha256_file(target) != expected:
                raise RuntimeError(f"TruFor code file does not match the pinned release: {relpath}")
        return self.root / release.SRC_RELPATH

    def _read_manifest(self) -> Dict[str, Any]:
        assert self.root is not None
        manifest = self.root / release.MANIFEST_NAME
        if not manifest.is_file():
            return {}
        return json.loads(manifest.read_text(encoding="utf-8"))

    def _load_state_dict(self) -> Dict[str, torch.Tensor]:
        """
        Verify the checkpoint, then unpickle it as safely as its provenance allows.

        ``weights_only=True`` is tried first. Falling back to a full unpickle,
        which can execute code, is allowed only for a checkpoint whose hash the
        manifest recorded from the authors' MD5-verified archive.
        """
        assert self.checkpoint_path is not None
        actual = sha256_file(self.checkpoint_path)
        weights = self._read_manifest().get("weights") or {}
        recorded = weights.get("sha256")
        if recorded and recorded != actual:
            raise RuntimeError("TruFor checkpoint does not match the hash in its manifest")
        self.checkpoint_sha256 = actual
        self.weights_verification = (
            weights.get("verification", "unverified") if recorded else "no_manifest"
        )

        try:
            checkpoint = torch.load(
                self.checkpoint_path, map_location="cpu", weights_only=True
            )
        except Exception as exc:
            if self.weights_verification != "official_archive_md5":
                raise RuntimeError(
                    "TruFor checkpoint needs a full unpickle and its provenance is not verified"
                ) from exc
            checkpoint = torch.load(
                self.checkpoint_path, map_location="cpu", weights_only=False
            )

        if isinstance(checkpoint, dict) and isinstance(checkpoint.get("state_dict"), dict):
            checkpoint = checkpoint["state_dict"]
        if not isinstance(checkpoint, dict):
            raise TypeError("TruFor checkpoint does not contain a state dict")
        return checkpoint

    def load(self) -> None:
        self._resolve_layout()
        if self.model_builder is not None:
            builder = self.model_builder
        else:
            builder = _import_upstream_builder(self._verify_code())
            # Pinned in CODE_FILES, so verified above; importable once the
            # builder has registered the upstream ``models`` package.
            self.pooling = importlib.import_module(
                "models.cmx.layer_utils"
            ).weighted_statistics_pooling

        state_dict = self._load_state_dict()
        model = builder(trufor_config())
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        self.model = model.to(self.device)
        self.loaded = True
        logger.info(
            "Loaded TruFor (%s, weights %s) on %s",
            release.UPSTREAM_COMMIT[:7], self.weights_verification, self.device,
        )

    # ── Inference ────────────────────────────────────────────────────────────

    @staticmethod
    def _preprocess(file_path: Path) -> tuple[torch.Tensor, tuple[int, int]]:
        # Upstream's own preprocessing (data_core.myDataset): RGB, CHW, / 256.
        # No mean/std here -- the network applies ImageNet normalisation itself,
        # and Noiseprint++ must see the unnormalised pixels.
        with Image.open(file_path) as img:
            rgb = np.asarray(img.convert("RGB"))
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float() / 256.0
        return tensor.unsqueeze(0), (rgb.shape[0], rgb.shape[1])

    def _forward(self, tensor: torch.Tensor, device: torch.device):
        """One model pass; the localization and confidence logits and the detection logit, on the CPU."""
        assert self.model is not None
        with torch.inference_mode():
            pred, conf, det, _npp = self.model(tensor.to(device))
            if det is None or conf is None:
                raise ValueError("TruFor model returned no detection score or reliability map")
            return pred.float().cpu(), conf.float().cpu(), det.float().cpu()

    def _detection_head(
        self, pred: torch.Tensor, conf: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """
        Upstream's image-level head (``builder_np_conf``, ``confpool``) on full-size
        logit maps: the confidence-weighted statistics of the confidence map and
        of the localization logit margin, then the trained MLP.
        """
        assert self.model is not None
        detection = getattr(self.model, "detection", None)
        if self.pooling is None or detection is None:
            raise RuntimeError("TruFor detection head is not available for tiled inference")
        with torch.inference_mode():
            f1 = self.pooling(conf).view(1, -1)
            f2 = self.pooling(pred[:, 1:2] - pred[:, 0:1], F.logsigmoid(conf)).view(1, -1)
            features = torch.cat((f1, f2), -1).to(device)
            return detection(features).float().cpu()

    def _forward_tiled(self, tensor: torch.Tensor, device: torch.device):
        """
        The logit maps blended from overlapping tiles, then the detection head
        once over the whole stitched maps. Logits, not probabilities, are
        blended: they are what the head pools.
        """
        _, _, h, w = tensor.shape
        rows = tile_spans(h, self.tile_size, self.tile_overlap)
        cols = tile_spans(w, self.tile_size, self.tile_overlap)
        pred_sum = conf_sum = None
        weight_sum = torch.zeros(1, 1, h, w)
        for y0, y1 in rows:
            for x0, x1 in cols:
                pred, conf, _det = self._forward(tensor[:, :, y0:y1, x0:x1], device)
                if pred.shape[-2:] != (y1 - y0, x1 - x0):
                    pred = F.interpolate(pred, size=(y1 - y0, x1 - x0), mode="bilinear", align_corners=False)
                    conf = F.interpolate(conf, size=(y1 - y0, x1 - x0), mode="bilinear", align_corners=False)
                if pred_sum is None:
                    pred_sum = torch.zeros(1, pred.shape[1], h, w)
                    conf_sum = torch.zeros(1, conf.shape[1], h, w)
                weight = torch.outer(
                    _blend_ramp(y0, y1, h, self.tile_overlap),
                    _blend_ramp(x0, x1, w, self.tile_overlap),
                )
                pred_sum[:, :, y0:y1, x0:x1] += pred * weight
                conf_sum[:, :, y0:y1, x0:x1] += conf * weight
                weight_sum[:, :, y0:y1, x0:x1] += weight
        pred = pred_sum / weight_sum
        conf = conf_sum / weight_sum
        return pred, conf, self._detection_head(pred, conf, device), len(rows) * len(cols)

    def _run(self, tensor: torch.Tensor, device: torch.device):
        _, _, h, w = tensor.shape
        with torch.inference_mode():
            if h * w <= self.tile_size ** 2:
                pred, conf, det = self._forward(tensor, device)
                return pred, conf, det, 1
            return self._forward_tiled(tensor, device)

    def _infer(
        self, tensor: torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray, float, str, bool, int]:
        """Run on the configured device; on CUDA out-of-memory, once more on the CPU."""
        assert self.model is not None
        with self._lock:
            try:
                pred, conf, det, tiles = self._run(tensor, self.device)
                device_used, oom_fallback = str(self.device), False
            except torch.cuda.OutOfMemoryError:
                if self.device.type != "cuda":
                    raise
                logger.warning("TruFor ran out of GPU memory; retrying on the CPU")
                torch.cuda.empty_cache()
                cpu = torch.device("cpu")
                self.model.to(cpu)
                try:
                    pred, conf, det, tiles = self._run(tensor, cpu)
                finally:
                    self.model.to(self.device)
                device_used, oom_fallback = "cpu", True
            finally:
                # Blocks cached for one image's shapes fragment the next one's:
                # a 1024x768 pass followed by a tiled 1080p one reserved 3.7 GB
                # instead of 3.1 GB, past what a 4 GB card has free.
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
        localization = torch.softmax(pred.squeeze(0), dim=0)[1].numpy()
        reliability = torch.sigmoid(conf.squeeze(0))[0].numpy()
        score = float(torch.sigmoid(det).reshape(-1)[0].item())
        return localization, reliability, score, device_used, oom_fallback, tiles

    def predict(self, file_path: Union[str, Path], threshold: float = 0.50) -> Dict[str, Any]:
        """
        Run TruFor on one image.

        Returns the raw integrity score as ``confidence`` and the two maps as
        float32 arrays at the input resolution. The maps are for the caller to
        turn into evidence artefacts; they are not metadata and must not be
        serialised into a result.
        """
        if not self.loaded or self.model is None:
            raise RuntimeError("TruFor model is not loaded")

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path.name}")

        started = time.perf_counter()
        tensor, (h, w) = self._preprocess(file_path)
        localization, reliability, score, device_used, oom_fallback, tiles = self._infer(tensor)
        inference_ms = (time.perf_counter() - started) * 1000.0
        tiled = tiles > 1

        if localization.shape != (h, w):
            localization = cv2.resize(localization, (w, h), interpolation=cv2.INTER_LINEAR)
        if reliability.shape != (h, w):
            reliability = cv2.resize(reliability, (w, h), interpolation=cv2.INTER_LINEAR)
        localization = np.clip(localization, 0.0, 1.0).astype(np.float32)
        reliability = np.clip(reliability, 0.0, 1.0).astype(np.float32)
        score = float(np.clip(score, 0.0, 1.0))

        regions = self._regions(localization, reliability, threshold)

        return {
            # The raw integrity score and the threshold it was compared against.
            # No label: whether this constitutes a finding is decided downstream
            # from the status fields of the whole analysis, not inside a model
            # wrapper that cannot see which other detectors ran.
            "manipulation_detected": bool(score >= threshold),
            "confidence": round(score, 4),
            "regions": regions,
            "localization_map": localization,
            "reliability_map": reliability,
            "metadata": {
                **self.identity(),
                "device": device_used,
                "cuda_oom_cpu_fallback": oom_fallback,
                "integrity_score": round(score, 6),
                "threshold": threshold,
                "image_size": [w, h],
                "localization_max": round(float(localization.max()), 4),
                "localization_mean": round(float(localization.mean()), 4),
                "reliability_mean": round(float(reliability.mean()), 4),
                "regions_detected": len(regions),
                "inference_time_ms": round(inference_ms, 1),
                "tiling": {
                    "tiled": tiled,
                    "tiles": tiles,
                    "tile_size": self.tile_size,
                    "tile_overlap": self.tile_overlap,
                },
                **({"limitations": [tiled_limitation(self.tile_size)]} if tiled else {}),
            },
        }

    @staticmethod
    def _regions(
        localization: np.ndarray, reliability: np.ndarray, threshold: float
    ) -> List[Dict[str, Any]]:
        """Connected areas of the localization map above ``threshold``, cleaned of speckle."""
        h, w = localization.shape
        binary = (localization >= threshold).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = h * w * 0.001
        regions: List[Dict[str, Any]] = []
        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(contour)
            loc_crop = localization[y:y + bh, x:x + bw]
            rel_crop = reliability[y:y + bh, x:x + bw]
            regions.append({
                "type": "forgery_localization",
                "confidence": round(float(loc_crop.mean()), 4),
                "bbox": [float(x), float(y), float(bw), float(bh)],
                "extra": {
                    "localization_max": round(float(loc_crop.max()), 4),
                    "reliability_mean": round(float(rel_crop.mean()), 4),
                },
            })
        return regions
