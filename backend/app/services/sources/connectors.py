"""
Source connectors: where else an uploaded image has been published.

Each connector answers one question -- "does this source hold this image?" --
and its result says which of three things happened, never blurring them:

  UNAVAILABLE  the connector is disabled or not set up; nothing was searched.
  FAILED       it was asked and broke; nothing can be concluded.
  COMPLETED    it searched; ``hits`` may be empty, which then means "not in
               this source", not "not published anywhere".

A hit is only as strong as its ``verification_status``: VERIFIED_MATCH means
the source holds the same image (possibly resized, recompressed or cropped);
SIMILAR_UNVERIFIED is a lead to check, not a source.
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class SourceStatus(str, Enum):
    UNAVAILABLE = "UNAVAILABLE"
    FAILED = "FAILED"
    COMPLETED = "COMPLETED"


@dataclass
class SourceQuery:
    """What is being looked for: the stored upload and its fingerprints."""
    file_path: str
    fingerprints: dict[str, str]
    embedding_version: Optional[str] = None
    # The URL the user says the image came from, if any.
    claimed_url: Optional[str] = None
    # Hits from the connectors that ran before this one (Wayback looks them up).
    prior_hits: list["SourceHit"] = field(default_factory=list)


@dataclass
class SourceHit:
    url: str                          # the page that publishes the image
    title: Optional[str]
    provider: str
    verification_status: str
    verification: dict[str, Any]
    similarity: Optional[float] = None
    # When the source says the image was published there (e.g. a Commons upload).
    published_at: Optional[datetime] = None
    # Who published it, the licence, the direct file URL, ...: the source's own
    # statements, cited, not facts established by AIMD.
    citation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "provider": self.provider,
            "verification_status": self.verification_status,
            "verification": self.verification,
            "similarity": self.similarity,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "citation": self.citation,
        }


@dataclass
class SourceResult:
    connector: str
    status: SourceStatus
    reason: Optional[str] = None
    hits: list[SourceHit] = field(default_factory=list)
    duration_ms: Optional[int] = None
    # What was searched, when it is not implied by the connector (e.g. a URL).
    detail: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """What is stored on the analysis: the outcome, not the hits themselves."""
        return {
            "status": self.status.value,
            "reason": self.reason,
            "hits": len(self.hits),
            "verified": sum(h.verification_status == "VERIFIED_MATCH" for h in self.hits),
            "duration_ms": self.duration_ms,
            **self.detail,
        }


class SourceConnector(ABC):
    name: str

    @abstractmethod
    def unavailable_reason(self, db) -> Optional[str]:
        """Why this connector cannot search right now, or ``None`` if it can."""

    @abstractmethod
    def search(self, query: SourceQuery, db) -> list[SourceHit]:
        """The hits for ``query``. May raise; ``run`` turns that into FAILED."""

    def run(self, query: SourceQuery, db) -> SourceResult:
        """Never raises: every outcome is a ``SourceResult``."""
        reason = self.unavailable_reason(db)
        if reason:
            return SourceResult(self.name, SourceStatus.UNAVAILABLE, reason)
        started = time.perf_counter()
        try:
            hits = self.search(query, db)
        except Exception as exc:
            logger.warning("Source connector %s failed: %s", self.name, exc)
            return SourceResult(
                self.name, SourceStatus.FAILED, f"{type(exc).__name__}: {exc}",
                duration_ms=int((time.perf_counter() - started) * 1000),
            )
        return SourceResult(
            self.name, SourceStatus.COMPLETED, hits=hits,
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
