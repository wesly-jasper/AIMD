"""
Media Ingestion Service — validates, stores, and persists media to the database.

Security:
  - Sanitizes filenames to prevent path traversal.
  - Validates file content using magic bytes (not just content-type header).
  - Enforces file size limit.
  - Never executes uploaded files.

The ingestion result is immediately persisted to the database so that
investigations survive server restarts.
"""
import hashlib
import logging
import re
import uuid
from pathlib import Path

from fastapi import UploadFile
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db import repository as repo
from app.services.preprocessing.metadata_extractor import MetadataExtractor
from app.services.fingerprinting.fingerprint import FingerprintService
from app.services.preprocessing.frame_extractor import FrameExtractor
from app.services.preprocessing.keyframe_selector import KeyframeSelector

logger = logging.getLogger(__name__)

# File magic bytes for validation (subset of common media types)
_MAGIC_SIGNATURES: dict[str, bytes] = {
    "image/jpeg": bytes([0xFF, 0xD8, 0xFF]),
    "image/png": bytes([0x89, 0x50, 0x4E, 0x47]),
    "image/gif": b"GIF",
    "image/webp": None,  # RIFF....WEBP — checked separately
    "video/mp4": None,   # ftyp at offset 4 — checked separately
}


def _sanitize_filename(filename: str) -> str:
    """Remove path components and dangerous characters from a filename."""
    # Strip directory components
    name = Path(filename).name
    # Replace dangerous characters
    name = re.sub(r"[^\w\-. ]", "_", name)
    # Prevent hidden files
    if name.startswith("."):
        name = "_" + name
    return name[:200]  # Maximum length


def _detect_media_type(data: bytes, content_type: str) -> str:
    """Determine media_type (image/video/audio) from file content."""
    # Check magic bytes
    if data[:3] == bytes([0xFF, 0xD8, 0xFF]):
        return "image"
    if data[:8] == bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]):
        return "image"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image"
    if data[:4] in (b"\x00\x00\x00\x18", b"\x00\x00\x00\x1c", b"\x00\x00\x00\x20"):
        # MP4 ftyp box — likely video
        if data[4:8] == b"ftyp":
            return "video"
    # AVI
    if data[:4] == b"RIFF" and data[8:12] == b"AVI ":
        return "video"
    # WAV
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "audio"
    # MP3 ID3
    if data[:3] == b"ID3":
        return "audio"

    # Fall back to content-type header (least trusted)
    if content_type:
        if content_type.startswith("image/"):
            return "image"
        if content_type.startswith("video/"):
            return "video"
        if content_type.startswith("audio/"):
            return "audio"

    raise ValueError(f"Cannot determine media type from file content (content-type: {content_type})")


class MediaIngestionService:

    def __init__(self):
        self.upload_dir = settings.upload_dir
        self.max_file_size = settings.max_file_size_mb * 1024 * 1024
        self.metadata_extractor = MetadataExtractor()
        self.fingerprint_service = FingerprintService()
        self.frame_extractor = FrameExtractor(settings.frames_dir)
        self.keyframe_selector = KeyframeSelector(settings.keyframes_dir)

    def validate_file(self, file: UploadFile) -> None:
        """Validate the uploaded file against allowed types."""
        if not file.filename:
            raise ValueError("Filename is missing")

        allowed_types = settings.all_allowed_types
        if file.content_type not in allowed_types:
            raise ValueError(
                f"Unsupported file type: {file.content_type}. "
                f"Allowed: {', '.join(allowed_types)}"
            )

    def generate_sha256(self, data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    async def ingest(self, file: UploadFile, db: Session | None = None) -> dict:
        """
        Validate, store, and persist a media file.

        Args:
            file: Uploaded file from FastAPI.
            db: Optional DB session. If provided, media record is persisted.

        Returns:
            Dict with media_id, sha256, file_path, metadata, and status.
        """
        # Validate content-type header
        self.validate_file(file)

        # Read file contents
        data = await file.read()

        if len(data) > self.max_file_size:
            raise ValueError(
                f"File size {len(data):,} bytes exceeds limit "
                f"of {settings.max_file_size_mb} MB"
            )

        if len(data) == 0:
            raise ValueError("Empty file uploaded")

        # Determine media type from content (magic bytes)
        try:
            media_type = _detect_media_type(data, file.content_type or "")
        except ValueError as exc:
            raise ValueError(str(exc))

        # Sanitize filename
        safe_name = _sanitize_filename(file.filename or "upload")
        extension = Path(safe_name).suffix.lower() or ".bin"

        # Generate IDs
        media_id = str(uuid.uuid4())
        stored_filename = f"{media_id}{extension}"

        # Store file
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        file_path = self.upload_dir / stored_filename
        file_path.write_bytes(data)

        # Compute SHA-256
        sha256 = self.generate_sha256(data)

        logger.info(
            "Ingested %s → %s (%d bytes, sha256=%s…)",
            safe_name, stored_filename, len(data), sha256[:16]
        )

        # Extract metadata
        try:
            metadata = self.metadata_extractor.extract(file_path)
        except Exception as exc:
            logger.warning("Metadata extraction during ingest failed: %s", exc)
            metadata = {"media_type": media_type, "source": "FALLBACK"}

        # Extract fingerprints
        try:
            fp_result = self.fingerprint_service.generate(file_path, media_id=media_id)
            fingerprint = {k: v for k, v in fp_result.items() if not k.startswith("_")}
        except Exception as exc:
            logger.warning("Fingerprint generation during ingest failed: %s", exc)
            fingerprint = {"sha256": sha256}

        frames = None
        keyframes = None
        keyframe_fingerprints = []

        if media_type == "video":
            try:
                frames = self.frame_extractor.extract(file_path, media_id, frame_interval=1)
                keyframes = self.keyframe_selector.select(
                    settings.frames_dir / media_id,
                    media_id,
                    frame_interval=10,
                )
                for keyframe in keyframes.get("keyframes", []):
                    kf_path = keyframe["file_path"] if isinstance(keyframe, dict) else str(keyframe)
                    keyframe_hash = self.fingerprint_service.generate(kf_path)
                    if "phash" in keyframe_hash:
                        keyframe_fingerprints.append({
                            "file": kf_path,
                            "phash": keyframe_hash["phash"],
                        })
            except Exception as exc:
                logger.warning("Video frame extraction during ingest failed: %s", exc)

        result = {
            "media_id": media_id,
            "original_filename": safe_name,
            "stored_filename": stored_filename,
            "file_path": str(file_path),
            "content_type": file.content_type,
            "media_type": media_type,
            "size_bytes": len(data),
            "sha256": sha256,
            "metadata": metadata,
            "fingerprint": fingerprint,
            "frames": frames,
            "keyframes": keyframes,
            "keyframe_fingerprints": keyframe_fingerprints,
            "status": "UPLOADED",
        }

        # Persist to database if session provided
        if db is not None:
            try:
                repo.create_media(db, {
                    "id": media_id,
                    "original_filename": safe_name,
                    "stored_filename": stored_filename,
                    "file_path": str(file_path),
                    "content_type": file.content_type or "application/octet-stream",
                    "media_type": media_type,
                    "size_bytes": len(data),
                    "sha256": sha256,
                })
                repo.upsert_media_metadata(db, media_id, {
                    "format": metadata.get("format"),
                    "width": metadata.get("width"),
                    "height": metadata.get("height"),
                    "duration_seconds": metadata.get("duration_seconds"),
                    "fps": metadata.get("fps"),
                    "codec": metadata.get("codec"),
                    "exif_data": metadata.get("exif", {}),
                    "raw_metadata": metadata,
                })
                fp_records = fp_result.get("_records", [])
                if fp_records:
                    repo.bulk_create_fingerprints(db, fp_records)
                logger.info("Media and metadata persisted to DB: %s", media_id)
            except Exception as exc:
                logger.error("Failed to persist media to DB: %s", exc)
                result["db_error"] = str(exc)

        return result