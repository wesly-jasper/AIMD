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
import logging
from pathlib import Path
from typing import Any

from app.services.fingerprinting.hash_generator import HashGenerator

logger = logging.getLogger(__name__)

# Supported image extensions for perceptual hashing
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tiff", ".tif", ".bmp"}
_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
_AUDIO_EXTS = {".wav", ".mp3", ".aac", ".m4a", ".flac"}


class FingerprintService:

    def __init__(self):
        self.hash_generator = HashGenerator()

    def generate(
        self,
        file_path: Path | str,
        media_id: str | None = None,
        analysis_id: str | None = None,
        scope: str = "full",
        frame_index: int | None = None,
    ) -> dict[str, Any]:
        """
        Generate all applicable fingerprints for a file.

        Returns a dict with keys per algorithm, each containing
        {algorithm, algorithm_version, value, media_id, scope}.
        Also returns a 'fingerprints' list suitable for bulk DB insert.
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

        results["_records"] = fingerprint_records
        results["_scope"] = scope
        return results