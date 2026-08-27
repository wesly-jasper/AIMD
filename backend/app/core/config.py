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