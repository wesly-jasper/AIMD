"""
Security utilities for AIMD.

Covers:
- Filename sanitization
- Path traversal protection
- Evidence path confinement
- File size / MIME validation
- Safe subprocess handling
- Secrets handling
"""
import re
import secrets
from pathlib import Path
from typing import Iterable, Optional

from fastapi import HTTPException

from app.core.config import settings


def sanitize_filename(filename: str, max_length: int = 200) -> str:
    """Sanitize filename to prevent path traversal and injection."""
    name = Path(filename).name  # strip directory
    # Replace dangerous chars
    name = re.sub(r"[^\w\-. ]", "_", name)
    if name.startswith("."):
        name = "_" + name
    # Prevent empty
    if not name or name in {".", ".."}:
        name = f"upload_{secrets.token_hex(4)}"
    return name[:max_length]


def ensure_safe_path(base_dir: Path, filename: str) -> Path:
    """
    Ensure resolved path stays within base_dir (path traversal protection).
    Raises HTTPException if traversal detected.
    """
    base = base_dir.resolve()
    target = (base / filename).resolve()
    try:
        target.relative_to(base)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid file path: traversal detected")
    return target


def evidence_roots() -> list[Path]:
    """
    Directories AIMD is permitted to read media bytes from.

    Anything outside these roots is not AIMD-managed evidence and must never be
    opened on behalf of a client request.
    """
    roots = [
        settings.upload_dir,
        settings.processed_dir,
        settings.frames_dir,
        settings.keyframes_dir,
    ]
    resolved: list[Path] = []
    for root in roots:
        try:
            resolved.append(Path(root).resolve())
        except OSError:
            continue
    return resolved


def resolve_evidence_path(
    stored_path: str | Path,
    allowed_roots: Optional[Iterable[Path]] = None,
) -> Path:
    """
    Resolve a stored evidence path and confine it to the AIMD evidence roots.

    This is the only sanctioned way to turn a persisted ``file_path`` into a path
    that may be opened. It defends against three distinct problems:

    1. Path traversal (``../``) in a stored value.
    2. Absolute paths pointing anywhere on the host filesystem -- the condition
       that made unauthenticated arbitrary file read possible when a client-
       supplied path was persisted verbatim and later served for download.
    3. Symlinks that escape the evidence roots (``resolve()`` follows them before
       the containment check, so an escaping link is rejected).

    Raises HTTPException(400) when the path escapes confinement. Existence is
    deliberately NOT checked here -- callers decide whether a missing file is a
    404 (never served) or a hard forensic error (integrity failure).
    """
    roots = list(allowed_roots) if allowed_roots is not None else evidence_roots()

    try:
        target = Path(stored_path).resolve()
    except (OSError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid evidence path")

    for root in roots:
        try:
            target.relative_to(root)
            return target
        except ValueError:
            continue

    # Do not echo the rejected path back to the client: it would confirm host
    # filesystem layout to an attacker probing for readable files.
    raise HTTPException(
        status_code=400,
        detail="Evidence path is outside the managed evidence storage area",
    )


def validate_file_size(size_bytes: int, max_mb: int) -> None:
    limit = max_mb * 1024 * 1024
    if size_bytes > limit:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {size_bytes:,} bytes exceeds {max_mb} MB limit",
        )
    if size_bytes == 0:
        raise HTTPException(status_code=400, detail="Empty file not allowed")


def sanitize_ffmpeg_arg(arg: str) -> str:
    """Basic sanitization for ffmpeg arguments -- disallow shell metachars."""
    if re.search(r"[;&|`$()]", arg):
        raise ValueError(f"Unsafe ffmpeg argument: {arg}")
    return arg


def get_secret(key: str, default: str | None = None) -> str | None:
    """Fetch secret from environment -- never log the value."""
    import os

    return os.getenv(key, default)
