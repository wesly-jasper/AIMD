import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.db.database import init_db
from app.api.v1.endpoints.media import router as media_router
from app.api.v1.endpoints.detection import router as detection_router
from app.api.v1.endpoints.investigation import router as investigation_router
from app.api.v1.endpoints.analysis import router as analysis_router


# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("aimd")


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup."""
    logger.info("AIMD starting — initialising database")
    init_db()
    logger.info("Database ready")
    yield
    logger.info("AIMD shutting down")


# ── Application ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="AIMD — AI Media Detection & Digital Forensics Platform",
    version=settings.app_version,
    description=(
        "Forensic platform for analysing digital media and determining "
        "whether it is AI-generated or manipulated, with full evidence traceability."
    ),
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(media_router, prefix="/api/v1")
app.include_router(detection_router, prefix="/api/v1")
app.include_router(investigation_router, prefix="/api/v1")
app.include_router(analysis_router, prefix="/api/v1")


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
@app.get("/api/v1/health", tags=["System"])
def health():
    return {
        "status": "ok",
        "version": settings.app_version,
        "database": settings.database_url.split("://")[0],
    }