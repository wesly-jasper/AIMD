"""
Source Tracer — identifies the earliest-known occurrence of a media file.

Works by:
  1. Running local similarity search (via SimilarityEngine) to find matching
     stored media with known timestamps.
  2. Optionally running external internet search if a provider is configured.
  3. Sorting candidates by source timestamp to identify the earliest occurrence.

Important language:
  - "Earliest-known occurrence discovered by AIMD" — not "original source".
  - AIMD cannot claim absolute origin. Offline and unindexed sources cannot
    be searched.

When no internet provider is configured, internet_provider_status = UNAVAILABLE.
This is different from "no sources found."
"""
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from app.services.similarity.internet_search_provider import (
    DisabledSourceSearchProvider,
    ProviderStatus,
)

logger = logging.getLogger(__name__)


class SourceCandidate:
    def __init__(
        self,
        url: Optional[str],
        title: Optional[str],
        domain: Optional[str],
        similarity: float,
        source_timestamp: Optional[str],
        provider: str,
        matching_method: str,
        confidence: float,
    ):
        self.url = url
        self.title = title
        self.domain = domain
        self.similarity = similarity
        self.source_timestamp = source_timestamp
        self.provider = provider
        self.matching_method = matching_method
        self.confidence = confidence

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "domain": self.domain,
            "similarity": self.similarity,
            "source_timestamp": self.source_timestamp,
            "provider": self.provider,
            "matching_method": self.matching_method,
            "confidence": self.confidence,
        }


class SourceTracer:

    def __init__(self, search_provider=None):
        self.search_provider = search_provider or DisabledSourceSearchProvider()

    def trace_source(self, file_path: str) -> dict[str, Any]:
        """
        Trace the earliest-known occurrence of a media file.

        Returns:
            dict with:
              internet_provider_status: UNAVAILABLE | CONFIGURED | FAILED
              candidates: list of SourceCandidate dicts sorted by timestamp
              earliest_known_occurrence: the candidate with earliest timestamp, or None
              note: epistemic statement about what "earliest known" means
        """
        candidates: list[SourceCandidate] = []

        # ── External internet search ──────────────────────────────────────────
        provider_status = self.search_provider.status
        if provider_status == ProviderStatus.CONFIGURED:
            try:
                raw_results = self.search_provider.search_by_image(file_path)
                for r in raw_results:
                    candidates.append(SourceCandidate(
                        url=r.url,
                        title=r.title,
                        domain=r.domain,
                        similarity=r.similarity,
                        source_timestamp=r.source_timestamp,
                        provider=r.provider,
                        matching_method="internet_search",
                        confidence=r.similarity,
                    ))
            except Exception as exc:
                logger.warning("Internet search failed: %s", exc)
                provider_status = ProviderStatus.FAILED

        # ── Sort by source_timestamp ──────────────────────────────────────────
        def sort_key(c: SourceCandidate):
            if c.source_timestamp:
                try:
                    return datetime.fromisoformat(
                        c.source_timestamp.replace("Z", "+00:00")
                    )
                except (ValueError, AttributeError):
                    pass
            return datetime.max.replace(tzinfo=timezone.utc)

        candidates.sort(key=sort_key)

        earliest = candidates[0] if candidates else None

        return {
            "internet_provider_status": provider_status.value,
            "candidates": [c.to_dict() for c in candidates],
            "earliest_known_occurrence": earliest.to_dict() if earliest else None,
            "note": (
                "INFERENCE: 'Earliest-known occurrence' refers to the earliest timestamp "
                "AIMD discovered among its sources. This is NOT a claim of absolute origin. "
                "Offline, unindexed, or private sources cannot be accessed."
            ),
        }
