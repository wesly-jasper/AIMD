"""
Structured logging for AIMD.

Every analysis logs:
- request_id
- analysis_id
- media_id
- model / model_version
- stage
- duration
- status / error

Sensitive media contents are never logged.
"""
import logging
import sys
import uuid
from contextvars import ContextVar

# Context variables for request-scoped logging
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
analysis_id_var: ContextVar[str | None] = ContextVar("analysis_id", default=None)

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s [req=%(request_id)s ana=%(analysis_id)s] -- %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get() or "-"
        record.analysis_id = analysis_id_var.get() or "-"
        return True


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging with context filter."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    handler.addFilter(ContextFilter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Reduce noisy libs
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def set_request_context(request_id: str | None = None, analysis_id: str | None = None) -> str:
    """Set context for current request. Returns request_id."""
    rid = request_id or str(uuid.uuid4())[:8]
    request_id_var.set(rid)
    if analysis_id:
        analysis_id_var.set(analysis_id)
    return rid


def clear_context() -> None:
    request_id_var.set(None)
    analysis_id_var.set(None)
