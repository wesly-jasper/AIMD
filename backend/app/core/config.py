from pathlib import Path
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    # ── Application ──────────────────────────────────────────────────────────
    app_name: str = "AIMD"
    app_version: str = "1.0.0"
    log_level: str = "INFO"

    # ── Storage ───────────────────────────────────────────────────────────────
    upload_dir: Path = BASE_DIR / "uploads"
    processed_dir: Path = BASE_DIR / "processed"
    frames_dir: Path = BASE_DIR / "processed" / "frames"
    keyframes_dir: Path = BASE_DIR / "processed" / "keyframes"

    max_file_size_mb: int = 500

    # ── Database ──────────────────────────────────────────────────────────────
    # SQLite for local dev; set DATABASE_URL=postgresql+psycopg2://... for prod
    database_url: str = f"sqlite:///{BASE_DIR}/aimd.db"

    # ── CORS ──────────────────────────────────────────────────────────────────
    cors_origins: List[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://localhost:8080",
    ]

    # ── Media Types ───────────────────────────────────────────────────────────
    allowed_image_types: List[str] = [
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/tiff",
    ]
    allowed_video_types: List[str] = [
        "video/mp4",
        "video/avi",
        "video/quicktime",  # .mov
        "video/x-msvideo",  # .avi
        "video/x-matroska",  # .mkv
        "video/mov",
        "video/mkv",
    ]
    allowed_audio_types: List[str] = [
        "audio/wav",
        "audio/x-wav",
        "audio/mpeg",    # .mp3
        "audio/mp3",
        "audio/aac",
        "audio/x-m4a",
        "audio/m4a",
        "audio/mp4",
    ]

    # ── Frame extraction ──────────────────────────────────────────────────────
    # How many frames per second to sample for analysis (not native FPS)
    analysis_fps: float = 1.0
    max_keyframes: int = 50

    # ── Optional ML models ────────────────────────────────────────────────────
    trufor_enabled: bool = False
    trufor_model_path: Optional[Path] = None
    ai_image_model_path: Optional[Path] = None
    deepfake_model_path: Optional[Path] = None
    audio_model_path: Optional[Path] = None
    device: str = "auto"  # auto | cpu | cuda

    # ── Advanced Localization Detectors ─────────────────────────────────────
    copymove_enabled: bool = True
    copymove_detector_type: str = "SIFT"  # SIFT | ORB
    manipulation_mask_enabled: bool = True
    face_region_analyzer_enabled: bool = True
    face_region_use_mediapipe: bool = True

    # ── Reports ───────────────────────────────────────────────────────────────
    report_dir: Path = BASE_DIR / "reports"
    logs_dir: Path = BASE_DIR / "logs"

    # ── Performance limits ────────────────────────────────────────────────────
    max_video_duration_seconds: int = 600
    max_audio_duration_seconds: int = 600
    max_image_dimension: int = 8000
    processing_timeout_seconds: int = 600

    # ── Security / App ────────────────────────────────────────────────────────
    # Controls whether a missing JWT signing key is fatal; see app.core.auth.
    app_env: str = "development"
    # There is deliberately no ``secret_key`` setting. It held a copy of the same
    # hardcoded signing key as app.core.auth and was imported by nothing, so a
    # deployment that dutifully overrode it changed no behaviour at all. The
    # signing key is read from AIMD_JWT_SECRET by app.core.auth, which has no
    # default and refuses the committed placeholder.

    # ── Rate limits ───────────────────────────────────────────────────────────
    # ``limits`` notation ("10/minute", "100/hour;10/minute"). Counted per client
    # address *per endpoint*; see app.core.rate_limit for what that does and does
    # not protect against. An unparseable value is fatal at import -- the library
    # would otherwise log it and install no limit at all.
    rate_limit_default: str = "120/minute"
    rate_limit_login: str = "10/minute"
    rate_limit_signup: str = "5/hour"
    rate_limit_upload: str = "30/minute"
    rate_limit_analysis: str = "10/minute"

    # ── Per-account quota ─────────────────────────────────────────────────────
    # Uploads per account in any rolling 24 hours. Counted from the database, so
    # it holds across workers and restarts, unlike the per-address rate limits.
    daily_upload_limit: int = 50

    # ── External search providers ─────────────────────────────────────────────
    google_vision_api_key: Optional[str] = None
    tineye_api_key: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    @property
    def all_allowed_types(self) -> List[str]:
        return (
            self.allowed_image_types
            + self.allowed_video_types
            + self.allowed_audio_types
        )


settings = Settings()