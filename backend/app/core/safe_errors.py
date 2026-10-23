"""
User-facing reasons for internal failures.

Stage failures and a failed analysis's ``error_message`` are returned to the
client and shown in the UI. They used to be ``str(exc)``, which put a database
failure's SQL statement and bound parameters, and the absolute storage path of
the evidence file, into API responses. The full exception is still logged where
it is caught; what is stored and returned goes through ``safe_reason``.
"""
from __future__ import annotations

import re
from typing import Optional

from sqlalchemy.exc import SQLAlchemyError

_MAX_CHARS = 240

#: A Windows drive or UNC path, or an absolute POSIX path of two or more
#: segments. Directory names may contain spaces (``C:\Users\Jane Doe\...``) as
#: long as a separator follows; the last segment may not, so the match stops at
#: the file name. The look-behinds keep URLs (``https://host/a/b``) intact: the
#: ``s:/`` of ``https:/`` is not a drive, and a URL's path follows ``/``.
_END = r"[^\s\\/'\"\],;)]*"
_PATH = re.compile(
    r"(?:(?<!\w)[A-Za-z]:[\\/]|\\\\)(?:[^\\/\n'\"\],;)]*[\\/])*" + _END
    + r"|(?<![\w.:/])/(?:[^/\n'\"\],;)]+/)+" + _END
)

DATABASE_ERROR = "a database error occurred while saving this step"


def scrub(text: Optional[str]) -> Optional[str]:
    """``text`` without file paths or SQL, on one line, at most 240 characters."""
    if not text:
        return text
    text = text.split("[SQL:")[0]  # SQLAlchemy appends the statement and parameters
    if "Session's transaction has been rolled back" in text or "(sqlite3." in text:
        return DATABASE_ERROR
    text = " ".join(_PATH.sub("<file>", text).split())
    return text if len(text) <= _MAX_CHARS else text[: _MAX_CHARS - 1] + "…"


def safe_reason(exc: BaseException) -> str:
    """A reason for ``exc`` that is safe to store and return to the client."""
    if isinstance(exc, SQLAlchemyError):
        return DATABASE_ERROR
    return scrub(str(exc)) or type(exc).__name__
