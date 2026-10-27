"""
Media Ingestion Service -- validates, stores, and persists media to the database.

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
from app.core.safe_errors import safe_reason
from app.db import repository as repo
from app.services.preprocessing.metadata_extractor import MetadataExtractor
from app.services.fingerprinting.fingerprint import FingerprintService
from app.services.preprocessing.frame_extractor import FrameExtractor
from app.services.preprocessing.keyframe_selector import KeyframeSelector

logger = logging.getLogger(__name__)

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


UNSUPPORTED_CONTENT = (
    "The file's content is not a supported image, video or audio format "
    "(JPEG, PNG, WebP, TIFF, MP4, MOV, AVI, MKV, WAV, MP3, AAC or M4A)."
)


def _sniff(data: bytes) -> tuple[str, str, str] | None:
    """
    ``(MIME type, media type, extension)`` from the file's own bytes, or None.

    The bytes are the only authority. This used to fall back to the declared
    Content-Type when no signature matched, so an HTML page or a text file sent
    as ``image/jpeg`` was stored and analysed as an image, while a real PNG sent
    as ``application/octet-stream`` was refused.
    """
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg", "image", ".jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png", "image", ".png"
    if data[:4] in (b"II*\x00", b"MM\x00*"):
        return "image/tiff", "image", ".tiff"
    if data[:4] == b"RIFF":
        kind = data[8:12]
        if kind == b"WEBP":
            return "image/webp", "image", ".webp"
        if kind == b"AVI ":
            return "video/x-msvideo", "video", ".avi"
        if kind == b"WAVE":
            return "audio/wav", "audio", ".wav"
    if data[4:8] == b"ftyp":  # ISO base media; the box size at 0..4 varies
        brand = data[8:12]
        if brand == b"qt  ":
            return "video/quicktime", "video", ".mov"
        if brand in (b"M4A ", b"M4B "):
            return "audio/mp4", "audio", ".m4a"
        return "video/mp4", "video", ".mp4"
    if data[:4] == b"\x1aE\xdf\xa3":
        return "video/x-matroska", "video", ".mkv"
    if data[:3] == b"ID3":
        return "audio/mpeg", "audio", ".mp3"
    if len(data) > 1 and data[0] == 0xFF and data[1] & 0xE0 == 0xE0:
        # MPEG audio frame sync: layer bits 00 are AAC ADTS, anything else MP3.
        if data[1] & 0x06 == 0:
            return "audio/aac", "audio", ".aac"
        return "audio/mpeg", "audio", ".mp3"
    return None


def _check_decodes(media_type: str, data: bytes, path: Path) -> None:
    """Refuse an image or video that has a valid signature but cannot be read."""
    if media_type == "image":
        import io

        from PIL import Image

        try:
            with Image.open(io.BytesIO(data)) as img:
                img.verify()
            with Image.open(io.BytesIO(data)) as img:
                img.load()  # verify() does not decode pixels; a truncated JPEG passes it
        except Exception as exc:
            raise ValueError(
                "The image could not be decoded: it may be truncated or corrupt."
            ) from exc
    elif media_type == "video":
        import cv2

        cap = cv2.VideoCapture(str(path))
        try:
            readable = cap.isOpened() and cap.read()[0]
        finally:
            cap.release()
        if not readable:
            raise ValueError("The video could not be decoded: it may be truncated or corrupt.")


class MediaIngestionService:

    def __init__(self):
        self.upload_dir = settings.upload_dir
        self.max_file_size = settings.max_file_size_mb * 1024 * 1024
        self.metadata_extractor = MetadataExtractor()
        self.fingerprint_service = FingerprintService()
        self.frame_extractor = FrameExtractor(settings.frames_dir)
        self.keyframe_selector = KeyframeSelector(settings.keyframes_dir)

    def validate_file(self, file: UploadFile) -> None:
        """Checks that need no bytes. The type is decided by the content in ``ingest``."""
        if not file.filename:
            raise ValueError("Filename is missing")

    def generate_sha256(self, data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    async def ingest(
        self,
        file: UploadFile,
        db: Session | None = None,
        case_id: str | None = None,
        uploaded_by: str | None = None,
    ) -> dict:
        """
        Validate, store, and persist a media file.

        Args:
            file: Uploaded file from FastAPI.
            db: Optional DB session. If provided, media record is persisted.
            case_id: Case the evidence belongs to; must already exist.
            uploaded_by: Username of the authenticated uploader.

        Returns:
            Dict with media_id, sha256, file_path, metadata, and status.
        """
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

        # The type comes from the bytes, never from the declared Content-Type.
        sniffed = _sniff(data)
        if sniffed is None or sniffed[0] not in settings.all_allowed_types:
            raise ValueError(UNSUPPORTED_CONTENT)
        content_type, media_type, extension = sniffed
        if file.content_type and file.content_type != content_type:
            logger.info(
                "Declared type %s differs from content %s; the content is used",
                file.content_type, content_type,
            )
        if media_type == "image":
            _check_decodes(media_type, data, Path())  # a video is checked once on disk

        safe_name = _sanitize_filename(file.filename or "upload")

        # Generate IDs
        media_id = str(uuid.uuid4())
        stored_filename = f"{media_id}{extension}"

        # Store file
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        file_path = self.upload_dir / stored_filename
        file_path.write_bytes(data)
        if media_type == "video":
            try:
                _check_decodes(media_type, data, file_path)
            except ValueError:
                file_path.unlink(missing_ok=True)
                raise

        # Compute SHA-256
        sha256 = self.generate_sha256(data)

        logger.info(
            "Ingested %s -> %s (%d bytes, sha256=%s)",
            safe_name, stored_filename, len(data), sha256[:16]
        )

        # Extract metadata
        try:
            metadata = self.metadata_extractor.extract(file_path)
        except Exception as exc:
            logger.warning("Metadata extraction during ingest failed: %s", exc)
            metadata = {"media_type": media_type, "source": "FALLBACK"}

        # Extract fingerprints. ``fp_result`` must exist on failure too: the
        # persistence step below reads it, and used to raise NameError there.
        fp_result: dict = {}
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
            "content_type": content_type,
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
                    "content_type": content_type,
                    "media_type": media_type,
                    "size_bytes": len(data),
                    "sha256": sha256,
                    "case_id": case_id,
                    "uploaded_by": uploaded_by,
                })
                result["persisted"] = True
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
                # Without this the session stays in a failed transaction and the
                # caller's next query (the audit event) raises instead.
                db.rollback()
                result["db_error"] = safe_reason(exc)

        return result