"""
Security utilities for AIMD.

Covers:
- Filename sanitization
- Path traversal protection
- File size / MIME validation
- Safe subprocess handling
- Secrets handling
"""
import re
import secrets
from pathlib import Path

from fastapi import HTTPException


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
