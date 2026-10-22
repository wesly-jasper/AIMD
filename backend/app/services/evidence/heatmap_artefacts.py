"""
Heatmaps written as evidence artefacts.

A localization map is the part of a TruFor result an examiner actually looks
at, so it is kept as a file rather than reduced to a bounding box. Files are
content-addressed -- named by the SHA-256 of the analysed bytes and the weights
that produced them -- so the same image and model always map to the same file,
and a file's name says what it is evidence of.

Results record paths relative to ``settings.artefacts_dir`` and never an
absolute path: detector metadata reaches API responses and reports, and an
absolute path would disclose the host's directory layout.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
from PIL import Image

from app.core.config import settings

OVERLAY_ALPHA = 0.45


def sha256_bytes_of(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _to_u8(values: np.ndarray) -> np.ndarray:
    return np.round(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8)


def _write_png(root: Path, relpath: str, pixels: np.ndarray) -> Dict[str, Any]:
    # imencode + a byte write, not imwrite: OpenCV cannot open non-ASCII paths on Windows.
    ok, encoded = cv2.imencode(".png", pixels)
    if not ok:
        raise ValueError(f"PNG encoding failed for {relpath}")
    data = encoded.tobytes()
    target = root / relpath
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    partial.write_bytes(data)
    os.replace(partial, target)
    return {
        "path": relpath,
        "sha256": hashlib.sha256(data).hexdigest(),
        "media_type": "image/png",
    }


def write_heatmap_artefacts(
    image_path: Path,
    localization_map: np.ndarray,
    reliability_map: Optional[np.ndarray],
    *,
    producer: str,
    model_tag: str,
    input_sha256: Optional[str] = None,
    root: Optional[Path] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Write a localization map (and its reliability map) as PNG evidence.

    Produces, below ``<root>/<producer>/``:

      * ``*_localization.png`` -- the map itself, grayscale, value * 255
      * ``*_reliability.png``  -- the reliability map, same encoding
      * ``*_overlay.png``      -- the localization map colour-mapped over the image,
                                  for reading, not measurement

    Returns ``{kind: {path, sha256, media_type}}`` with paths relative to ``root``.
    """
    root = Path(root or settings.artefacts_dir)
    input_sha256 = input_sha256 or sha256_bytes_of(image_path)
    stem = f"{producer}/{input_sha256}_{model_tag}"

    loc_u8 = _to_u8(localization_map)
    artefacts = {"localization_map": _write_png(root, f"{stem}_localization.png", loc_u8)}

    if reliability_map is not None:
        artefacts["reliability_map"] = _write_png(
            root, f"{stem}_reliability.png", _to_u8(reliability_map)
        )

    with Image.open(image_path) as img:
        bgr = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    heat = cv2.applyColorMap(loc_u8, cv2.COLORMAP_JET)
    if heat.shape[:2] != bgr.shape[:2]:
        heat = cv2.resize(heat, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    overlay = cv2.addWeighted(bgr, 1.0 - OVERLAY_ALPHA, heat, OVERLAY_ALPHA, 0.0)
    artefacts["overlay"] = _write_png(root, f"{stem}_overlay.png", overlay)
    return artefacts
