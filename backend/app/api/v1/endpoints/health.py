"""
Health endpoint -- spec compliant with model status.
"""
from fastapi import APIRouter
from sqlalchemy import text

from app.core.config import settings
from app.db.database import SessionLocal

router = APIRouter(tags=["System"])


@router.get("/health", summary="System health")
@router.get("/api/v1/health", summary="System health")
def health():
    # DB check
    db_status = "ok"
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
    except Exception as e:
        db_status = f"error: {e}"

    # Model status
    trufor_status = "unavailable"
    if settings.trufor_enabled and settings.trufor_model_path and settings.trufor_model_path.exists():
        trufor_status = "available"
    elif settings.trufor_enabled:
        trufor_status = "model_missing"

    return {
        "status": "healthy" if db_status == "ok" else "degraded",
        "version": settings.app_version,
        "database": db_status,
        "models": {
            "trufor": trufor_status,
            "image_detector": "available",
            "video_detector": "available",
            "audio_detector": "available",
            "face_detector": "available",
            "embedding": "available",
        },
    }
