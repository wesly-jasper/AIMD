"""
Reference corpus: public images indexed so an upload can be traced to them.

``build_corpus`` walks Commons categories, downloads a scaled rendition of each
acceptable file, and stores it with its citation and the same fingerprints an
upload gets (sha256, pHash, dHash and, when the CLIP backbone is available, the
retrieval embedding). Rerunning it is cheap: a file whose Commons SHA-1 is
unchanged and still on disk is not downloaded again; only fingerprints it is
missing, or whose embedding came from a different model, are recomputed.
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

import httpx
from PIL import Image
from sqlalchemy.orm import Session

from app.db import repository as repo
from app.services.fingerprinting.fingerprint import EMBEDDING_ALGORITHM, FingerprintService
from app.services.sources.commons_client import CORPUS, CommonsClient, CommonsError, CommonsFile

logger = logging.getLogger(__name__)

#: Originals Commons can render to a JPEG/PNG/WebP that PIL reads. Others
#: (SVG, TIFF, GIF, PDF pages) are left out rather than half-supported.
ACCEPTED_MIMES = {"image/jpeg", "image/png", "image/webp"}
_EXTENSIONS = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}
MAX_DOWNLOAD_BYTES = 25 * 1024 * 1024


@dataclass
class BuildStats:
    indexed: int = 0      # downloaded and fingerprinted this run
    unchanged: int = 0    # already indexed, nothing to do
    refreshed: int = 0    # already indexed, fingerprints added or replaced
    failed: int = 0
    skipped: Counter = field(default_factory=Counter)  # reason -> count

    @property
    def in_corpus(self) -> int:
        return self.indexed + self.unchanged + self.refreshed


def skip_reason(file: CommonsFile, min_side: int) -> Optional[str]:
    """Why ``file`` is not indexed, or ``None`` to index it."""
    if file.mime not in ACCEPTED_MIMES:
        return f"type {file.mime}"
    if not file.width or not file.height or min(file.width, file.height) < min_side:
        return f"smaller than {min_side} px"
    if not file.sha1:
        return "no SHA-1 to detect replacement"
    return None


def _missing_fingerprints(
    db: Session, record, embedding_version: Optional[str]
) -> set[str]:
    """Algorithms already stored, minus an embedding from another model (deleted here)."""
    stored = {fp.algorithm: fp.algorithm_version for fp in record.fingerprints}
    if EMBEDDING_ALGORITHM in stored and embedding_version and stored[EMBEDDING_ALGORITHM] != embedding_version:
        repo.delete_reference_fingerprints(db, record.id, [EMBEDDING_ALGORITHM])
        del stored[EMBEDDING_ALGORITHM]
    return set(stored)


def _fingerprint(db: Session, fingerprinter: FingerprintService, record, skip: set[str]) -> int:
    results = fingerprinter.generate(record.file_path, skip=frozenset(skip))
    rows = [
        {
            "reference_id": record.id,
            "algorithm": r["algorithm"],
            "algorithm_version": r["algorithm_version"],
            "value": r["value"],
        }
        for r in results["_records"]
    ]
    if rows:
        repo.bulk_create_reference_fingerprints(db, rows)
    return len(rows)


def _download(client: CommonsClient, file: CommonsFile, root: Path) -> dict:
    """Fetch the rendition and describe it; the file is named by its SHA-256."""
    staging = root / CORPUS / "incoming"
    tmp = staging / f"{file.sha1}.download"
    sha256, size, content_type = client.download(file.download_url, tmp, max_bytes=MAX_DOWNLOAD_BYTES)
    ext = _EXTENSIONS.get(content_type)
    if ext is None:
        tmp.unlink(missing_ok=True)
        raise CommonsError(f"unsupported rendition type {content_type}")
    try:
        with Image.open(tmp) as img:
            img.verify()
        with Image.open(tmp) as img:
            width, height = img.size
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise CommonsError(f"downloaded file is not a readable image: {exc}") from exc
    final = root / CORPUS / sha256[:2] / f"{sha256}{ext}"
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp.replace(final)
    return {"file_path": str(final), "sha256": sha256, "size_bytes": size,
            "width": width, "height": height}


def _record_fields(file: CommonsFile) -> dict:
    return {
        "corpus": CORPUS,
        "external_id": file.title,
        "page_url": file.page_url,
        "file_url": file.file_url,
        "download_url": file.download_url,
        "uploader": file.uploader,
        "uploaded_at": file.uploaded_at,
        "licence": file.licence,
        "licence_url": file.licence_url,
        "artist": file.artist,
        "original_sha1": file.sha1,
        "original_width": file.width,
        "original_height": file.height,
        "original_mime": file.mime,
        "source_metadata": file.metadata or None,
    }


def index_file(
    db: Session,
    client: CommonsClient,
    file: CommonsFile,
    category: str,
    *,
    root: Path,
    fingerprinter: FingerprintService,
    embedding_version: Optional[str],
) -> str:
    """Index one accepted file; returns ``indexed``, ``unchanged`` or ``refreshed``."""
    record = repo.get_reference_image(db, CORPUS, file.title)
    current = (
        record is not None
        and record.original_sha1 == file.sha1
        and Path(record.file_path).is_file()
    )
    if current:
        if category not in record.categories:
            repo.upsert_reference_image(db, {
                "corpus": CORPUS, "external_id": file.title,
                "categories": [*record.categories, category],
            })
        skip = _missing_fingerprints(db, record, embedding_version)
        added = _fingerprint(db, fingerprinter, record, skip)
        return "refreshed" if added else "unchanged"

    stale_path = Path(record.file_path) if record is not None else None
    stored = _download(client, file, root)
    categories = record.categories if record is not None else []
    record = repo.upsert_reference_image(db, {
        **_record_fields(file),
        **stored,
        "categories": categories if category in categories else [*categories, category],
    })
    repo.delete_reference_fingerprints(db, record.id)
    if stale_path is not None and stale_path != Path(stored["file_path"]):
        stale_path.unlink(missing_ok=True)  # the file was replaced on Commons
    db.refresh(record)
    _fingerprint(db, fingerprinter, record, set())
    return "indexed"


def build_corpus(
    db: Session,
    client: CommonsClient,
    categories: Iterable[str],
    *,
    root: Path,
    fingerprinter: FingerprintService,
    embedding_version: Optional[str],
    per_category: int,
    limit: int,
    thumb_width: int,
    min_side: int,
    progress: Callable[[str], None] = lambda _msg: None,
) -> BuildStats:
    """
    Up to ``per_category`` images from each category and ``limit`` in total.

    One file failing (a download refused, an unreadable image) is counted and
    skipped; a failed category listing stops the build, since everything after
    it would be a guess about what the category holds.
    """
    stats = BuildStats()
    for category in categories:
        taken = 0
        for file in client.category_files(category, thumb_width=thumb_width):
            if taken >= per_category or stats.in_corpus >= limit:
                break
            reason = skip_reason(file, min_side)
            if reason:
                stats.skipped[reason] += 1
                continue
            try:
                outcome = index_file(
                    db, client, file, category, root=root,
                    fingerprinter=fingerprinter, embedding_version=embedding_version,
                )
            except (CommonsError, httpx.HTTPError, OSError) as exc:
                stats.failed += 1
                progress(f"  failed {file.title}: {exc}")
                continue
            setattr(stats, outcome, getattr(stats, outcome) + 1)
            taken += 1
            progress(f"  [{stats.in_corpus}] {outcome} {file.title}")
        if stats.in_corpus >= limit:
            break
    return stats
