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
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from app.core.config import settings

OVERLAY_ALPHA = 0.45
#: Localisation value at or above which a pixel is masked. TruFor's maps are a
#: per-pixel softmax, and 0.5 is the cut its authors binarise at.
MASK_THRESHOLD = 0.5
#: Connected areas smaller than this share of the image are dropped as speckle.
MASK_MIN_AREA_FRACTION = 0.001
#: How the mask is made, recorded beside it so a reader can reproduce it.
MASK_METHOD = (
    "localisation >= threshold, opened then closed with a 5x5 rectangle, "
    "areas under 0.1% of the image dropped"
)


def region_mask(
    localization: np.ndarray, threshold: float = MASK_THRESHOLD
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    The binary mask of the localised areas and the outer contour of each.

    The mask (uint8, 0 or 255) and the regions a runner reports come from this one
    function, so every masked area is a reported region and every reported
    region is masked. Holes inside an area stay unmasked.
    """
    h, w = localization.shape[:2]
    binary = (localization >= threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = h * w * MASK_MIN_AREA_FRACTION
    kept = [c for c in contours if cv2.contourArea(c) >= min_area]
    keep = np.zeros_like(binary)
    cv2.drawContours(keep, kept, -1, 255, thickness=cv2.FILLED)
    return cv2.bitwise_and(binary, keep), kept


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
    mask_threshold: float = MASK_THRESHOLD,
) -> Dict[str, Dict[str, Any]]:
    """
    Write a localization map (and its reliability map) as PNG evidence.

    Produces, below ``<root>/<producer>/``:

      * ``*_localization.png`` -- the map itself, grayscale, value * 255
      * ``*_mask.png``         -- the localised areas (``region_mask``), 0 or 255;
                                  the threshold and method are recorded with it
      * ``*_reliability.png``  -- the reliability map, same encoding
      * ``*_overlay.png``      -- the localization map colour-mapped over the image,
                                  for reading, not measurement

    Returns ``{kind: {path, sha256, media_type, ...}}`` with paths relative to ``root``.
    """
    root = Path(root or settings.artefacts_dir)
    input_sha256 = input_sha256 or sha256_bytes_of(image_path)
    stem = f"{producer}/{input_sha256}_{model_tag}"

    loc_u8 = _to_u8(localization_map)
    artefacts = {"localization_map": _write_png(root, f"{stem}_localization.png", loc_u8)}

    mask, areas = region_mask(np.asarray(localization_map, dtype=np.float32), mask_threshold)
    # The threshold is part of the file's name: a different cut is different evidence.
    artefacts["mask"] = {
        **_write_png(root, f"{stem}_mask_t{round(mask_threshold * 100):03d}.png", mask),
        "threshold": mask_threshold,
        "method": MASK_METHOD,
        "areas": len(areas),
        "masked_fraction": round(float(np.count_nonzero(mask)) / mask.size, 6),
    }

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
