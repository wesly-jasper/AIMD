"""
Fingerprint service.

Generates all fingerprints for a media file and returns structured
results with algorithm provenance metadata so every fingerprint is
traceable. Every fingerprint carries:
  - algorithm: the hash type (sha256 | phash | dhash | embedding)
  - algorithm_version: specific version string for comparison compatibility
  - value: the hash value
  - media_id: the media this belongs to
  - scope: full | frame | keyframe | audio
"""
import base64
import logging
from pathlib import Path
from typing import Any

import numpy as np

from app.services.fingerprinting.hash_generator import HashGenerator

logger = logging.getLogger(__name__)

# Supported image extensions for perceptual hashing
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tiff", ".tif", ".bmp"}
_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
_AUDIO_EXTS = {".wav", ".mp3", ".aac", ".m4a", ".flac"}


#: The image embedding stored for similarity retrieval.
EMBEDDING_ALGORITHM = "clip-vit-l14"


def embedding_version(embedder: Any) -> str:
    """The ``algorithm_version`` an embedding from ``embedder`` is stored under."""
    return f"{embedder.version_tag}/retrieval224/l2"


def encode_embedding(vector: np.ndarray) -> str:
    """A unit-length float vector as base64 of its little-endian fp16 bytes."""
    return base64.b64encode(np.asarray(vector, dtype="<f2").tobytes()).decode("ascii")


def decode_embedding(value: str) -> np.ndarray:
    """The inverse of ``encode_embedding``, as float32."""
    return np.frombuffer(base64.b64decode(value), dtype="<f2").astype(np.float32)


def _default_embedder() -> Any | None:
    """
    The shared CLIP backbone, or ``None`` when embeddings are off or not installed.

    Called on the first image, never at construction: loading 890 MB of weights
    must not happen just because an ingestion service was built.
    """
    from app.core.config import settings

    if not settings.similarity_embedding_enabled:
        return None
    if not settings.univfd_model_path:
        logger.warning("Image embeddings enabled but UNIVFD_MODEL_PATH is not set")
        return None
    try:
        from app.services.detection.models.clip_backbone import get_clip_backbone

        return get_clip_backbone(settings.univfd_model_path, device=settings.device)
    except Exception as exc:
        logger.warning("Image embeddings unavailable: %s", exc)
        return None


class FingerprintService:

    def __init__(self, embedder: Any | None = None, default_embedder: bool = True):
        """
        ``embedder`` maps an image path to a unit-length vector and has a
        ``version_tag``. ``None`` means the shared CLIP backbone, resolved lazily
        and only when ``SIMILARITY_EMBEDDING_ENABLED`` is set -- or no embedding
        at all with ``default_embedder=False``.
        """
        self.hash_generator = HashGenerator()
        self._embedder = embedder
        self._embedder_resolved = embedder is not None or not default_embedder
        self._uses_shared_embedder = embedder is None and default_embedder

    def _get_embedder(self) -> Any | None:
        if self._uses_shared_embedder:
            from app.core.config import settings

            # Checked on every call, not only the first: turning embeddings off
            # must take effect even after the backbone was resolved once.
            if not settings.similarity_embedding_enabled:
                return None
        if not self._embedder_resolved:
            self._embedder = _default_embedder()
            self._embedder_resolved = True
        return self._embedder

    def generate(
        self,
        file_path: Path | str,
        media_id: str | None = None,
        analysis_id: str | None = None,
        scope: str = "full",
        frame_index: int | None = None,
        skip: frozenset[str] | set[str] = frozenset(),
    ) -> dict[str, Any]:
        """
        Generate all applicable fingerprints for a file.

        Returns a dict with keys per algorithm, each containing
        {algorithm, algorithm_version, value, media_id, scope}.
        Also returns a 'fingerprints' list suitable for bulk DB insert.

        Algorithms in ``skip`` (already stored for this media) get no record, and
        the embedding is not computed at all.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = file_path.suffix.lower()
        results: dict[str, Any] = {}
        fingerprint_records: list[dict] = []

        # ── SHA-256 (all file types) ──────────────────────────────────────────
        try:
            sha256 = self.hash_generator.generate_sha256(file_path)
            results["sha256"] = sha256
            fingerprint_records.append({
                "media_id": media_id,
                "analysis_id": analysis_id,
                "algorithm": "sha256",
                "algorithm_version": "sha256",
                "value": sha256,
                "scope": scope,
                "frame_index": frame_index,
            })
        except Exception as exc:
            logger.warning("SHA-256 generation failed for %s: %s", file_path, exc)

        # ── Perceptual hashes (images only) ──────────────────────────────────
        if ext in _IMAGE_EXTS:
            try:
                phash = self.hash_generator.generate_phash(file_path)
                phash_ver = self.hash_generator.PHASH_VERSION
                if phash:
                    results["phash"] = phash
                    results["phash_version"] = phash_ver
                    fingerprint_records.append({
                        "media_id": media_id,
                        "analysis_id": analysis_id,
                        "algorithm": "phash",
                        "algorithm_version": phash_ver,
                        "value": phash,
                        "scope": scope,
                        "frame_index": frame_index,
                    })
            except Exception as exc:
                logger.warning("pHash generation failed for %s: %s", file_path, exc)

            try:
                dhash = self.hash_generator.generate_dhash(file_path)
                dhash_ver = self.hash_generator.DHASH_VERSION
                if dhash:
                    results["dhash"] = dhash
                    results["dhash_version"] = dhash_ver
                    fingerprint_records.append({
                        "media_id": media_id,
                        "analysis_id": analysis_id,
                        "algorithm": "dhash",
                        "algorithm_version": dhash_ver,
                        "value": dhash,
                        "scope": scope,
                        "frame_index": frame_index,
                    })
            except Exception as exc:
                logger.warning("dHash generation failed for %s: %s", file_path, exc)

            # ── Image embedding (retrieval only, never an identity) ──────────
            embedder = None if EMBEDDING_ALGORITHM in skip else self._get_embedder()
            if embedder is not None:
                try:
                    vector = embedder.embed(file_path)
                    version = embedding_version(embedder)
                    results[EMBEDDING_ALGORITHM + "_version"] = version
                    fingerprint_records.append({
                        "media_id": media_id,
                        "analysis_id": analysis_id,
                        "algorithm": EMBEDDING_ALGORITHM,
                        "algorithm_version": version,
                        "value": encode_embedding(vector),
                        "scope": scope,
                        "frame_index": frame_index,
                    })
                except Exception as exc:
                    logger.warning("Image embedding failed for %s: %s", file_path, exc)

        results["_records"] = [r for r in fingerprint_records if r["algorithm"] not in skip]
        results["_scope"] = scope
        return results