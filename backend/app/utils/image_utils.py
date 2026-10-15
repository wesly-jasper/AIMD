"""Image utilities for forensic preprocessing."""
import io
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image


def load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Unable to read image: {path}")
    return img


def resize_for_analysis(img: np.ndarray, max_dim: int = 1024) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img


def compute_ela_map(image_path: Path, quality: int = 90, scale: int = 15) -> np.ndarray:
    """Return ELA difference map as numpy array."""
    from PIL import ImageChops

    original = Image.open(image_path).convert("RGB")
    buf = io.BytesIO()
    original.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    recompressed = Image.open(buf)
    diff = ImageChops.difference(original, recompressed)
    arr = np.array(diff, dtype=np.float32)
    if arr.ndim == 3:
        arr = np.mean(arr, axis=2)
    # Enhance
    arr = np.clip(arr * scale, 0, 255).astype(np.uint8)
    return arr


def compute_fft_magnitude(gray: np.ndarray) -> np.ndarray:
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = 20 * np.log(np.abs(fshift) + 1e-8)
    return magnitude


def save_visualization(arr: np.ndarray, out_path: Path, colormap=cv2.COLORMAP_JET) -> Path:
    """Normalize array to 0-255, apply colormap, save."""
    norm = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    colored = cv2.applyColorMap(norm, colormap)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), colored)
    return out_path
