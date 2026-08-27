"""
Hash generator — cryptographic and perceptual hashes.

Uses the imagehash library for pHash and dHash so the values are
compatible with standard perceptual hash libraries and can be compared
across tools.

SHA-256: exact file identity (cryptographic).
pHash:   perceptual similarity for images (DCT-based).
dHash:   perceptual similarity for images (gradient-based).

These are distinct: do not mix their meanings.
"""
import hashlib
import logging
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)

# imagehash is the standard library for pHash/dHash
try:
    import imagehash
    from PIL import Image as PILImage
    _IMAGEHASH_AVAILABLE = True
except ImportError:
    _IMAGEHASH_AVAILABLE = False
    logger.warning(
        "imagehash not available — perceptual hashing will fall back to custom DCT implementation"
    )


PHASH_ALGORITHM_VERSION = "imagehash-4.x-64bit"
DHASH_ALGORITHM_VERSION = "imagehash-4.x-64bit"


class HashGenerator:

    PHASH_VERSION = PHASH_ALGORITHM_VERSION
    DHASH_VERSION = DHASH_ALGORITHM_VERSION

    def generate_sha256(self, file_path: Path | str) -> str:
        """Compute SHA-256 of file contents. Streams the file to avoid loading large files into memory."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    def generate_phash(self, file_path: Path | str) -> str:
        """
        Compute perceptual hash (pHash) using imagehash library.

        Returns:
            64-bit binary hash string
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if _IMAGEHASH_AVAILABLE:
            try:
                img = PILImage.open(file_path).convert("RGB")
                h = imagehash.phash(img, hash_size=8)
                # Convert to binary string for Hamming distance comparison
                return bin(int(str(h), 16))[2:].zfill(64)
            except Exception as exc:
                logger.warning("imagehash pHash failed (%s), falling back to DCT", exc)

        # Fallback: custom DCT pHash (compatible only with itself)
        return self._phash_dct_fallback(file_path)

    def generate_dhash(self, file_path: Path | str) -> str:
        """
        Compute difference hash (dHash) using imagehash library.

        Returns:
            64-bit binary hash string
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if _IMAGEHASH_AVAILABLE:
            try:
                img = PILImage.open(file_path).convert("RGB")
                h = imagehash.dhash(img, hash_size=8)
                return bin(int(str(h), 16))[2:].zfill(64)
            except Exception as exc:
                logger.warning("imagehash dHash failed (%s), skipping", exc)

        return ""

    def _phash_dct_fallback(self, file_path: Path) -> str:
        """Custom DCT pHash fallback when imagehash is unavailable."""
        import numpy as np
        image = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("Unable to read image for pHash")
        image = cv2.resize(image, (32, 32))
        dct = cv2.dct(image.astype("float32"))
        dct_low = dct[:8, :8]
        median = dct_low[1:, :].mean()
        hash_bits = dct_low > median
        return "".join("1" if bit else "0" for bit in hash_bits.flatten())