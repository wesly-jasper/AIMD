"""
The frozen holdout: a test set nothing is tuned against.

Every threshold, calibration and detector choice so far was checked on the same
test split, so its figures are no longer an unbiased estimate. A holdout is
frozen once, by ``scripts/evaluate.py freeze``, into a lock file committed with
the code (``backend/evaluation/holdout.lock.json``):

  * the lock lists each holdout file by content hash, so renaming a file or
    relabelling its split does not move it out of the holdout;
  * ``calibrate`` refuses to fit on any locked file;
  * ``split`` never moves a locked file;
  * ``run`` verifies every locked file it evaluates and refuses a file whose
    bytes changed, and the run records the lock it was checked against.

The lock holds hashes and ids only, never the images.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from app.services.evaluation.dataset import EvalItem

DEFAULT_LOCK = Path(__file__).resolve().parents[3] / "evaluation" / "holdout.lock.json"
SCHEMA = 1


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_lock(items: Iterable[EvalItem], *, note: str = "") -> dict[str, Any]:
    """The lock for every item whose split is ``holdout``."""
    files: dict[str, dict[str, Any]] = {}
    for item in items:
        if item.split != "holdout":
            continue
        sha = file_sha256(item.path)
        if sha in files:
            raise ValueError(f"{item.id} has the same bytes as {files[sha]['id']}")
        files[sha] = {"id": item.id, "label": item.label, "generator": item.generator,
                      "tags": list(item.tags)}
    if not files:
        raise ValueError("no item has split 'holdout'")
    labels: dict[str, int] = {}
    for f in files.values():
        labels[f["label"]] = labels.get(f["label"], 0) + 1
    return {
        "schema": SCHEMA,
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": note,
        "count": len(files),
        "labels": labels,
        "holdout_sha256": hashlib.sha256("\n".join(sorted(files)).encode()).hexdigest(),
        "files": dict(sorted(files.items())),
    }


def load_lock(path: Optional[Path] = None) -> Optional[dict[str, Any]]:
    path = Path(path or DEFAULT_LOCK)
    if not path.is_file():
        return None
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("schema") != SCHEMA:
        raise ValueError(f"{path}: unknown holdout lock schema {lock.get('schema')}")
    return lock


def locked_ids(lock: Optional[dict[str, Any]]) -> set[str]:
    return {f["id"] for f in (lock or {}).get("files", {}).values()}


def check(items: Iterable[EvalItem], lock: Optional[dict[str, Any]]) -> list[str]:
    """
    Problems with ``items`` against the lock: a holdout item whose bytes are not
    in the lock (changed or added after freezing), or a locked file listed under
    another split. Empty when there is no lock.
    """
    if not lock:
        return []
    files = lock["files"]
    problems = []
    for item in items:
        sha = file_sha256(item.path)
        if item.split == "holdout" and sha not in files:
            problems.append(f"{item.id}: holdout file not in the lock (changed or added after freezing)")
        elif sha in files and item.split != "holdout":
            problems.append(f"{item.id}: a locked holdout file listed under split {item.split!r}")
    return problems


def is_locked(item: EvalItem, lock: Optional[dict[str, Any]]) -> bool:
    return bool(lock) and (item.id in locked_ids(lock) or file_sha256(item.path) in lock["files"])
