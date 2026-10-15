"""
Central constants for AIMD forensic platform.

Single source of truth for verdicts, severities, media types,
allowed extensions, and forensic thresholds.
"""
from enum import Enum

# ── Verdicts / Assessments ────────────────────────────────────────────────────
class Verdict(str, Enum):
    AUTHENTIC = "AUTHENTIC"
    LIKELY_AUTHENTIC = "LIKELY_AUTHENTIC"
    UNCERTAIN = "UNCERTAIN"
    LIKELY_MANIPULATED = "LIKELY_MANIPULATED"
    MANIPULATED = "MANIPULATED"
    AI_GENERATED = "AI_GENERATED"
    # Legacy / API compatibility
    CLEAN = "CLEAN"
    SUSPICIOUS = "SUSPICIOUS"
    INCONCLUSIVE = "INCONCLUSIVE"
    UNAVAILABLE = "UNAVAILABLE"


# ── Evidence severity ─────────────────────────────────────────────────────────
class Severity(str, Enum):
    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# ── Evidence category (FACT / OBSERVATION / INFERENCE / UNCERTAINTY) ─────────
class EvidenceCategory(str, Enum):
    FACT = "FACT"
    OBSERVATION = "OBSERVATION"
    INFERENCE = "INFERENCE"
    UNCERTAINTY = "UNCERTAINTY"


# ── Job / Analysis status ─────────────────────────────────────────────────────
class JobStatus(str, Enum):
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    ANALYZING = "ANALYZING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PARTIAL = "PARTIAL"
    INCONCLUSIVE = "INCONCLUSIVE"


# ── Risk levels ───────────────────────────────────────────────────────────────
class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# ── Supported media extensions ────────────────────────────────────────────────
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tiff", ".tif", ".bmp"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg"}

SUPPORTED_EXTENSIONS = ALLOWED_IMAGE_EXTENSIONS | ALLOWED_VIDEO_EXTENSIONS | ALLOWED_AUDIO_EXTENSIONS

# MIME to media_type
MIME_TO_MEDIA_TYPE = {
    "image/jpeg": "image",
    "image/png": "image",
    "image/webp": "image",
    "image/tiff": "image",
    "image/bmp": "image",
    "video/mp4": "video",
    "video/quicktime": "video",
    "video/x-msvideo": "video",
    "video/x-matroska": "video",
    "video/webm": "video",
    "audio/wav": "audio",
    "audio/x-wav": "audio",
    "audio/mpeg": "audio",
    "audio/mp3": "audio",
    "audio/m4a": "audio",
    "audio/x-m4a": "audio",
    "audio/flac": "audio",
    "audio/aac": "audio",
    "audio/ogg": "audio",
}

# ── Forensic thresholds ───────────────────────────────────────────────────────
ELA_SUSPICIOUS_THRESHOLD = 0.50
FFT_SUSPICIOUS_THRESHOLD = 0.60
NOISE_SUSPICIOUS_THRESHOLD = 0.55
OVERALL_SUSPICIOUS_THRESHOLD = 0.50

# Performance limits (configurable via env)
DEFAULT_MAX_UPLOAD_MB = 500
DEFAULT_MAX_FRAMES = 50
DEFAULT_ANALYSIS_FPS = 1.0
MAX_IMAGE_DIMENSION = 8000  # pixels

# ── Model status ──────────────────────────────────────────────────────────────
MODEL_AVAILABLE = "available"
MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
INTERNET_SEARCH_UNAVAILABLE = "INTERNET_SEARCH_UNAVAILABLE"
