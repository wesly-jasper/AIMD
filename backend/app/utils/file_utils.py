"""File utilities -- safe operations for forensic file handling."""
import hashlib
import re
from pathlib import Path


def sanitize_filename(filename: str) -> str:
    name = Path(filename).name
    name = re.sub(r"[^\w\-. ]", "_", name)
    if name.startswith("."):
        name = "_" + name
    return name[:200]


def compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_join(base: Path, *parts: str) -> Path:
    """Join and ensure result stays within base (path traversal protection)."""
    target = (base.joinpath(*parts)).resolve()
    base_resolved = base.resolve()
    try:
        target.relative_to(base_resolved)
    except ValueError:
        raise ValueError(f"Path traversal detected: {target} not in {base_resolved}")
    return target
