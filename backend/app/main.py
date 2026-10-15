import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.core.config import settings
from app.core.logging import setup_logging
from app.db.database import init_db
from app.api.v1.endpoints.media import router as media_router
from app.api.v1.endpoints.detection import router as detection_router
from app.api.v1.endpoints.investigation import router as investigation_router
from app.api.v1.endpoints.analysis import router as analysis_router
from app.api.v1.endpoints.auth import router as auth_router
from app.api.v1.endpoints.jobs import router as jobs_router

setup_logging(settings.log_level)
logger = logging.getLogger("aimd")

# Rate Limiter
limiter = Limiter(key_func=get_remote_address, default_limits=["120/minute"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup."""
    logger.info("AIMD starting -- initialising database")
    init_db()
    logger.info("Database ready")
    yield
    logger.info("AIMD shutting down")


app = FastAPI(
    title="AIMD -- AI Media Detection & Digital Forensics Platform",
    version=settings.app_version,
    description=(
        "Forensic platform for analysing digital media and determining "
        "whether it is AI-generated or manipulated, with full evidence traceability."
    ),
    lifespan=lifespan,
)

# SlowAPI Rate Limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(auth_router, prefix="/api/v1")
app.include_router(media_router, prefix="/api/v1")
app.include_router(detection_router, prefix="/api/v1")
app.include_router(investigation_router, prefix="/api/v1")
app.include_router(analysis_router, prefix="/api/v1")
app.include_router(jobs_router, prefix="/api/v1")


@app.get("/health", tags=["System"])
@app.get("/api/v1/health", tags=["System"])
def health():
    from sqlalchemy import text
    from app.db.database import SessionLocal

    db_status = "ok"
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
    except Exception as e:
        db_status = f"error: {e}"

    trufor_status = "unavailable"
    if settings.trufor_enabled and settings.trufor_model_path and settings.trufor_model_path.exists():
        trufor_status = "available"
    elif settings.trufor_enabled:
        trufor_status = "model_missing"

    return {
        "status": "ok" if db_status == "ok" else "degraded",
        "version": settings.app_version,
        "database": db_status,
        "health": "healthy" if db_status == "ok" else "degraded",
        "models": {
            "trufor": trufor_status,
            "image_detector": "available",
            "video_detector": "available",
            "audio_detector": "available",
            "face_detector": "available",
            "embedding": "available",
        },
    }
