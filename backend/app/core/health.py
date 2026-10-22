"""
Single source of truth for system health reporting.

Previously two independent copies of this logic existed -- one inline in
``app.main`` and one in the (unmounted) ``app.api.v1.endpoints.health`` router --
and both returned the same hardcoded detector map:

    "image_detector": "available", "video_detector": "available", ...

Those strings were never consulted from the detector registry, so /health
reported detectors as available regardless of whether they could run. Reporting an
unavailable detector as available is precisely the misrepresentation this platform
must not make, so the map is now derived or omitted -- never asserted.
"""
import logging
from functools import lru_cache
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)


def check_database() -> str:
    """
    Return "ok" or "error". Never raises.

    The exception text is logged, not returned. This route is unauthenticated,
    and a database driver's error routinely carries the database file path or a
    connection string -- it previously went verbatim to any anonymous caller.
    """
    from sqlalchemy import text

    from app.db import database

    try:
        db = database.SessionLocal()
        try:
            db.execute(text("SELECT 1"))
        finally:
            db.close()
        return "ok"
    except Exception as exc:
        logger.error("Health check: database unreachable (%s: %s)", type(exc).__name__, exc)
        return "error"


def trufor_configuration_status() -> str:
    """
    Report TruFor *configuration*, not capability.

    "configured" means a model path is set and the file exists. It does NOT mean
    the checkpoint was validated, that the architecture is present, or that
    inference would succeed. TruFor availability is only known when the runner is
    constructed.
    """
    if not settings.trufor_enabled:
        return "disabled"
    if settings.trufor_model_path and settings.trufor_model_path.exists():
        return "configured"
    return "model_missing"


@lru_cache(maxsize=1)
def _registered_image_detectors() -> tuple[str, ...]:
    """
    Class names of the image detectors the container constructs.

    Computed once per process. Building the container loads MediaPipe and the
    other detectors -- about 0.7 s of CPU -- and this sits behind the one
    unauthenticated route, so doing it per request let any anonymous caller buy
    most of a core per probe. The set only changes with configuration, which
    requires a restart anyway. A construction failure raises and is therefore
    not cached: the next probe retries.
    """
    from app.services.detection.detection_container import create_detection_router

    engine = create_detection_router().image_engine
    return tuple(sorted(type(d).__name__ for d in getattr(engine, "detectors", [])))


def detector_report() -> dict[str, Any]:
    """
    Describe the detector set that is actually constructed.

    Registration is not availability: a registered detector may still return
    UNAVAILABLE at call time. That distinction is reported explicitly rather than
    being collapsed into a single misleading status.
    """
    registered: list[str] = []
    registry_status = "ok"
    try:
        registered = list(_registered_image_detectors())
    except Exception as exc:
        logger.error("Health check: detector registry failed (%s)", type(exc).__name__)
        registry_status = f"error: {type(exc).__name__}"

    return {
        "registry_status": registry_status,
        "trufor": trufor_configuration_status(),
        "registered_image_detectors": registered,
        # Per-detector availability reporting requires the detector availability
        # contract (roadmap Phase 2). Until then this is stated as absent rather
        # than guessed.
        "availability_introspection": "NOT_IMPLEMENTED",
    }


def build_health_report() -> dict[str, Any]:
    db_status = check_database()
    healthy = db_status == "ok"

    return {
        "status": "ok" if healthy else "degraded",
        "version": settings.app_version,
        "database": db_status,
        "health": "healthy" if healthy else "degraded",
        "detectors": detector_report(),
    }
