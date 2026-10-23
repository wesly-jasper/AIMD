"""
WikimediaCorpusSource: the locally indexed Commons reference corpus.

Retrieval and verification are the ones local similarity uses -- the same
fingerprints, the same ranking, the same three gates -- run against
``reference_images`` instead of an owner's media. Nothing here reaches the
network: the corpus was fetched ahead of time by
``scripts/build_reference_corpus.py``.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from app.db import repository as repo
from app.services.similarity.similarity_engine import (
    NO_MATCH,
    rank_candidates,
    verify_candidate,
)
from app.services.sources.commons_client import CORPUS
from app.services.sources.connectors import SourceConnector, SourceHit, SourceQuery


def _utc(value: Optional[datetime]) -> Optional[datetime]:
    """Stored as UTC; SQLite returns it without the timezone."""
    if value is None or value.tzinfo is not None:
        return value
    return value.replace(tzinfo=timezone.utc)


class WikimediaCorpusSource(SourceConnector):
    name = "wikimedia_commons"

    def __init__(self, enabled: Optional[bool] = None, top_k: int = 10):
        self._enabled = enabled
        self.top_k = top_k

    def _is_enabled(self) -> bool:
        if self._enabled is not None:
            return self._enabled
        from app.core.config import settings

        return settings.source_wikimedia_enabled

    def unavailable_reason(self, db) -> Optional[str]:
        if not self._is_enabled():
            return "disabled (SOURCE_WIKIMEDIA_ENABLED=false)"
        # An empty corpus is a setup gap, not a search that found nothing.
        if repo.count_reference_images(db, CORPUS) == 0:
            return "reference corpus is empty (run scripts/build_reference_corpus.py)"
        return None

    def search(self, query: SourceQuery, db) -> list[SourceHit]:
        def stored(algorithm: str):
            return [
                (fp.reference_id, fp.value, fp.algorithm_version)
                for fp in repo.get_reference_fingerprints(db, algorithm, CORPUS)
            ]

        hits = []
        for cand in rank_candidates(query.fingerprints, stored, self.top_k, query.embedding_version):
            ref = repo.get_reference_image_by_id(db, cand["id"])
            if ref is None:
                continue
            status, verification = verify_candidate(query.file_path, cand, ref.file_path)
            if status == NO_MATCH:
                continue
            hits.append(SourceHit(
                url=ref.page_url,
                title=ref.external_id,
                provider=self.name,
                verification_status=status,
                verification=verification,
                similarity=cand["cosine"],
                published_at=_utc(ref.uploaded_at),
                citation={
                    "corpus": ref.corpus,
                    "reference_id": ref.id,
                    "file_url": ref.file_url,
                    "original_sha1": ref.original_sha1,
                    "uploader": ref.uploader,
                    "uploaded_at": _utc(ref.uploaded_at).isoformat() if ref.uploaded_at else None,
                    "uploaded_at_meaning": "upload of the current file version on Wikimedia Commons",
                    "licence": ref.licence,
                    "licence_url": ref.licence_url,
                    "artist": ref.artist,
                    "date_time_original": (ref.source_metadata or {}).get("DateTimeOriginal"),
                    "retrieved_at": _utc(ref.retrieved_at).isoformat() if ref.retrieved_at else None,
                },
            ))
        return hits
