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
    # Detector outputs kept as evidence (heatmaps, masks). Inside processed_dir,
    # so inside the evidence roots; results store paths relative to this.
    artefacts_dir: Path = BASE_DIR / "processed" / "artefacts"

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
    # The TruFor root scripts/fetch_models.py installs into (backend/models/trufor).
    trufor_model_path: Optional[Path] = None
    # Images over trufor_tile_size² pixels run in overlapping tiles so peak GPU
    # memory stays near 2.2 GB whatever the image size (a whole 1080p pass needs
    # ~4 GB). The step (size - overlap) must be a multiple of 8, the JPEG grid.
    trufor_tile_size: int = 1024
    trufor_tile_overlap: int = 128
    # UnivFD AI-generation detector: a linear head on CLIP ViT-L/14 features. The
    # root scripts/fetch_univfd.py installs into (backend/models/univfd) holds the
    # CLIP backbone and the head; similarity embeddings use the same backbone.
    univfd_enabled: bool = False
    univfd_model_path: Optional[Path] = None
    # Applied to sigmoid(logit), an uncalibrated model score; 0.5 is the paper's.
    univfd_threshold: float = 0.5
    ai_image_model_path: Optional[Path] = None
    deepfake_model_path: Optional[Path] = None
    audio_model_path: Optional[Path] = None
    device: str = "auto"  # auto | cpu | cuda

    # ── Similarity search ─────────────────────────────────────────────────────
    # Store a CLIP embedding per image at ingestion (needs univfd_model_path).
    similarity_embedding_enabled: bool = False
    # A candidate is VERIFIED_MATCH only when all three agree: pHash Hamming
    # distance (of 64 bits), embedding cosine, and ORB keypoints consistent with
    # one homography. Retrieval casts a wider net than verification.
    # The cosine gate is a sanity floor, not the discriminator: on CocoGlide's
    # 256 px real photographs a half-size JPEG q70 copy embeds as low as 0.73,
    # while a different photograph of the same class reaches 0.88. pHash (copies
    # <= 2, lookalikes >= 22) and ORB inliers (copies >= 20, lookalikes <= 15)
    # separate the two; the cosine only rejects a geometric match whose content
    # CLIP reads as unrelated.
    similarity_phash_max_distance: int = 10
    similarity_cosine_min: float = 0.70
    similarity_min_inliers: int = 20
    similarity_candidate_phash_distance: int = 20
    similarity_candidate_cosine: float = 0.80
    # A crop fails the whole-frame pHash; it verifies instead when, after warping
    # the candidate into the query's frame, the query lies inside it and the
    # aligned content agrees. Measured on crops of 120 Commons images: true crops
    # aligned pHash <= 6 (90 % <= 2), NCC mostly >= 0.92; burst shots and
    # different speakers on the same stage NCC <= 0.91 or aligned pHash >= 6.
    similarity_aligned_phash_max_distance: int = 4
    similarity_aligned_ncc_min: float = 0.90
    similarity_aligned_min_coverage: float = 0.90

    # ── Reference corpus ──────────────────────────────────────────────────────
    # Public images indexed by scripts/build_reference_corpus.py. Outside the
    # evidence roots: these files are citations, not anyone's evidence.
    reference_dir: Path = BASE_DIR / "storage" / "reference"
    # Wikimedia asks every client to identify itself with a way to reach its
    # operator (an email address or URL); anonymous clients are rate-limited.
    wikimedia_contact: Optional[str] = None
    # Search the indexed corpus for each image upload (no network at analysis time).
    source_wikimedia_enabled: bool = True
    # Fetch and check the URL a user supplies with an analysis (SSRF-guarded).
    source_url_verification_enabled: bool = True
    # Look up verified sources in the Wayback Machine (network at analysis time).
    source_wayback_enabled: bool = True

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

    # ── NVIDIA Nemotron via Nebius Token Factory ──────────────────────────────
    # OpenAI-compatible endpoint. With no key the investigation agent reports
    # UNAVAILABLE; nothing else depends on it.
    nebius_api_key: Optional[str] = None
    nebius_base_url: str = "https://api.tokenfactory.nebius.com/v1/"
    nemotron_model: str = "nvidia/nemotron-3-super-120b-a12b"

    # ── Investigation agent ───────────────────────────────────────────────────
    # Bounds on one investigation: model calls, wall-clock seconds for the whole
    # run, client retries per call, completion tokens per call, and how many
    # pages verify_source_match may fetch.
    agent_enabled: bool = True
    agent_max_steps: int = 8
    agent_timeout_seconds: float = 60.0
    agent_max_retries: int = 2
    agent_max_tokens: int = 2048
    agent_max_url_checks: int = 2
    # Investigations per account in any rolling hour. Counted from the database,
    # like the upload quota, so public demo traffic cannot drain the credits.
    agent_hourly_limit: int = 10

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