"""
Metadata extractor — extracts and classifies technical metadata.

Evidence categories:
  FACT:        Values read directly from file headers/EXIF/container.
  OBSERVATION: Values computed by AIMD (e.g. duration from FPS × frame count).

Supports: images (JPEG/PNG/WebP/TIFF), video (MP4/AVI/MOV/MKV),
          audio (WAV/MP3/AAC/M4A).
"""
import logging
from pathlib import Path
from typing import Any

import cv2

logger = logging.getLogger(__name__)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tiff", ".tif"}
_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
_AUDIO_EXTS = {".wav", ".mp3", ".aac", ".m4a", ".flac", ".ogg"}


class MetadataExtractor:

    def extract(self, file_path: Path | str) -> dict[str, Any]:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = file_path.suffix.lower()

        if ext in _IMAGE_EXTS:
            return self._extract_image_metadata(file_path)
        if ext in _VIDEO_EXTS:
            return self._extract_video_metadata(file_path)
        if ext in _AUDIO_EXTS:
            return self._extract_audio_metadata(file_path)

        raise ValueError(f"Unsupported media format: {ext}")

    # ── Images ────────────────────────────────────────────────────────────────

    def _extract_image_metadata(self, file_path: Path) -> dict[str, Any]:
        image = cv2.imread(str(file_path))
        if image is None:
            raise ValueError(f"Unable to read image: {file_path}")

        height, width, channels = image.shape

        meta: dict[str, Any] = {
            "media_type": "image",
            "format": file_path.suffix.lower().lstrip("."),
            # FACT: read directly from image data
            "width": width,
            "height": height,
            "channels": channels,
            "source": "FACT",
        }

        # ── EXIF ──────────────────────────────────────────────────────────────
        exif_data = self._extract_exif(file_path)
        meta["exif"] = exif_data
        if exif_data:
            meta["exif_source"] = "FACT"

        return meta

    def _extract_exif(self, file_path: Path) -> dict[str, Any]:
        exif: dict[str, Any] = {}
        try:
            from PIL import Image, ExifTags
            img = Image.open(file_path)

            # PNG / WebP text chunks (AI generator signatures live here)
            if hasattr(img, "info") and isinstance(img.info, dict):
                text_info = {
                    k: str(v)[:512]  # truncate large blobs
                    for k, v in img.info.items()
                    if isinstance(v, (str, bytes, int, float))
                }
                if text_info:
                    exif["image_info"] = text_info

            # JPEG EXIF
            raw_exif = img._getexif() if hasattr(img, "_getexif") else None
            if raw_exif:
                for tag_id, value in raw_exif.items():
                    tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                    # Skip binary blobs
                    if isinstance(value, (str, int, float, tuple)):
                        exif[tag_name] = value

        except Exception as exc:
            logger.debug("EXIF extraction failed for %s: %s", file_path, exc)

        return exif

    # ── Video ─────────────────────────────────────────────────────────────────

    def _extract_video_metadata(self, file_path: Path) -> dict[str, Any]:
        cap = cv2.VideoCapture(str(file_path))
        if not cap.isOpened():
            raise ValueError(f"Unable to open video: {file_path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        finally:
            cap.release()

        # OBSERVATION: duration derived from FPS and frame count
        duration = round(frame_count / fps, 3) if fps > 0 else None

        # Decode FourCC codec string
        codec = (
            chr(fourcc_int & 0xFF)
            + chr((fourcc_int >> 8) & 0xFF)
            + chr((fourcc_int >> 16) & 0xFF)
            + chr((fourcc_int >> 24) & 0xFF)
        ).strip("\x00") if fourcc_int else None

        return {
            "media_type": "video",
            "format": file_path.suffix.lower().lstrip("."),
            # FACT: read from video container
            "width": width,
            "height": height,
            "fps": round(fps, 3) if fps else None,
            "frame_count": frame_count,
            "codec": codec,
            # OBSERVATION: computed
            "duration_seconds": duration,
            "source": {
                "width": "FACT",
                "height": "FACT",
                "fps": "FACT",
                "frame_count": "FACT",
                "codec": "FACT",
                "duration_seconds": "OBSERVATION",
            },
        }

    # ── Audio ─────────────────────────────────────────────────────────────────

    def _extract_audio_metadata(self, file_path: Path) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "media_type": "audio",
            "format": file_path.suffix.lower().lstrip("."),
        }

        # Try librosa for duration, sample rate, channels
        try:
            import librosa
            import soundfile as sf

            # soundfile gives exact sample rate + channel count without decoding
            info = sf.info(str(file_path))
            meta["sample_rate"] = info.samplerate   # FACT
            meta["channels"] = info.channels          # FACT
            meta["duration_seconds"] = round(info.duration, 3)  # FACT
            meta["format_info"] = info.format         # FACT
            meta["subtype"] = info.subtype            # FACT
            meta["source"] = "FACT"
        except Exception as exc:
            logger.debug("soundfile audio metadata failed for %s: %s", file_path, exc)
            # Try librosa as fallback
            try:
                import librosa
                y, sr = librosa.load(str(file_path), sr=None, mono=False)
                meta["sample_rate"] = int(sr)
                meta["channels"] = 1 if y.ndim == 1 else y.shape[0]
                meta["duration_seconds"] = round(len(y) / sr if y.ndim == 1 else y.shape[1] / sr, 3)
                meta["source"] = "OBSERVATION"  # librosa may resample
            except Exception as exc2:
                logger.warning("Audio metadata extraction failed for %s: %s", file_path, exc2)
                meta["source"] = "UNAVAILABLE"

        return meta