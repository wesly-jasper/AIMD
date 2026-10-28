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
from pathlib import Path
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

    "configured" means a model path is set and a checkpoint is there: the path is
    the checkpoint file, or a TruFor root holding one. It does NOT mean the
    checkpoint was validated, that the architecture is present, or that inference
    would succeed. TruFor availability is only known when the runner is
    constructed.
    """
    if not settings.trufor_enabled:
        return "disabled"
    path = settings.trufor_model_path
    if path and (path.is_file() or (path / "weights" / "trufor.pth.tar").is_file()):
        return "configured"
    return "model_missing"


def univfd_configuration_status() -> str:
    """UnivFD *configuration*, in the same sense as ``trufor_configuration_status``."""
    from app.services.detection.models import univfd_release as release

    if not settings.univfd_enabled:
        return "disabled"
    path = settings.univfd_model_path
    if path and (path / release.CLIP_FILE).is_file() and (path / release.HEAD_FILE).is_file():
        return "configured"
    return "model_missing"


def _image_detectors() -> list:
    """
    The image detectors the pipeline actually uses -- the process-wide container,
    not a second copy. Building a separate router here loaded every model twice.
    A construction failure raises and is not cached: the next probe retries.
    """
    from app.services.detection.detection_container import get_detection_container

    engine = get_detection_container().router.image_engine
    return list(getattr(engine, "detectors", []))


def detector_operational_status(detectors: list) -> dict[str, Any]:
    """
    Per detector: whether it can run (``availability``) and what its runs since
    startup actually did (``state``: OPERATIONAL, DEGRADED, FAILING, UNTESTED,
    UNAVAILABLE). Reasons are scrubbed: this route is unauthenticated.
    """
    from app.core.safe_errors import scrub
    from app.schemas.detection import DetectorAvailability
    from app.services.detection.runtime import detector_health

    report: dict[str, Any] = {}
    for detector in detectors:
        name = type(detector).__name__
        try:
            availability, reason = detector.availability()
        except Exception as exc:
            availability, reason = DetectorAvailability.UNAVAILABLE, type(exc).__name__
        runs = detector_health.snapshot(name)
        report[name] = {
            "state": detector_health.state(
                name, availability, failing_after=settings.detector_failing_after
            ),
            "availability": availability.value,
            "reason": scrub(reason),
            "runs": runs.get("runs", 0),
            "failed": runs.get("failed", 0),
            "timeouts": runs.get("timeouts", 0),
            "consecutive_failures": runs.get("consecutive_failures", 0),
            "last_status": runs.get("last_status"),
            "last_reason": scrub(runs.get("last_reason")),
            "last_run_at": runs.get("last_run_at"),
            "last_success_at": runs.get("last_success_at"),
            "p95_duration_ms": runs.get("p95_duration_ms"),
        }
    return report


def calibration_report(detectors: list) -> dict[str, Any]:
    """
    Each configured calibration file against the detectors actually running:
    present, readable, checksum, and whether each detector's version (which for
    B-Free names its small-input mode and offset) is the one the file was fitted
    for. A mismatch is not an error inside an analysis -- that detector's
    evidence is reported as direction only -- but it means the deployment is not
    the configuration that was validated, so it is surfaced here.
    """
    from app.services.detection.calibration import consistency

    live = {type(d).__name__: getattr(d, "version", None) for d in detectors}
    report: dict[str, Any] = {}
    for key in ("calibration_path", "fusion_calibration_path"):
        status = consistency(getattr(settings, key, None), live)
        # Unauthenticated route: no directory layout and no hashes. ``checksum``
        # says whether the file is the recorded one; the hash itself is in each
        # analysis's run provenance.
        if status.get("path"):
            status["path"] = Path(status["path"]).name
        status.pop("sha256", None)
        report[key] = status
    report["consistent"] = all(r["consistent"] for r in report.values() if isinstance(r, dict))
    return report


def detector_report() -> dict[str, Any]:
    """
    Describe the detector set that is actually constructed, and how it is doing.

    Registration is not availability, and availability is not operation: a
    detector whose checkpoint is present can still fail on every request.
    ``detectors`` reports all three, so a detector that is configured but broken
    shows as FAILING instead of "configured".
    """
    detectors: list = []
    registry_status = "ok"
    try:
        detectors = _image_detectors()
    except Exception as exc:
        logger.error("Health check: detector registry failed (%s)", type(exc).__name__)
        registry_status = f"error: {type(exc).__name__}"

    operational = detector_operational_status(detectors)
    failing = sorted(n for n, d in operational.items() if d["state"] == "FAILING")
    return {
        "registry_status": registry_status,
        "trufor": trufor_configuration_status(),
        "univfd": univfd_configuration_status(),
        "registered_image_detectors": sorted(operational),
        "detectors": operational,
        "failing_detectors": failing,
        "calibration": calibration_report(detectors),
    }


def build_health_report() -> dict[str, Any]:
    db_status = check_database()
    detectors = detector_report()
    # A detector failing on every request degrades the service even though the
    # process is up: analyses complete, but without that detector's evidence.
    # So does a calibration that is configured but cannot apply: evidence goes
    # unquantified without any analysis failing.
    healthy = (db_status == "ok" and not detectors["failing_detectors"]
               and detectors["calibration"]["consistent"])
    if not detectors["calibration"]["consistent"]:
        logger.warning("Health check: calibration does not match the running detectors: %s",
                       {k: v.get("state") for k, v in detectors["calibration"].items() if isinstance(v, dict)})

    return {
        "status": "ok" if healthy else "degraded",
        "version": settings.app_version,
        "database": db_status,
        "health": "healthy" if healthy else "degraded",
        "detectors": detectors,
    }
