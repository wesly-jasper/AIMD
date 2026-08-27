"""
Similarity engine.

Implements local similarity search first (comparing stored fingerprints
from the database), then optionally queries external search providers.

Local search methods (in priority order):
  1. SHA-256 exact match
  2. pHash perceptual distance
  3. dHash perceptual distance
  4. Embedding cosine similarity (if available)

External search is performed only when a configured provider is present.
"""
import logging
import math
from typing import Any

from sqlalchemy.orm import Session

from app.services.similarity.internet_search_provider import (
    BaseSourceSearchProvider,
    DisabledSourceSearchProvider,
    ProviderStatus,
    SourceSearchResult,
)

logger = logging.getLogger(__name__)


def _hamming_distance(hash_a: str, hash_b: str) -> int:
    """Compute Hamming distance between two binary hash strings."""
    if len(hash_a) != len(hash_b):
        return len(hash_a)  # Treat incompatible lengths as maximum distance
    return sum(a != b for a, b in zip(hash_a, hash_b))


def _hamming_similarity(hash_a: str, hash_b: str) -> float:
    """Convert Hamming distance to [0,1] similarity score."""
    if not hash_a or not hash_b:
        return 0.0
    dist = _hamming_distance(hash_a, hash_b)
    max_bits = max(len(hash_a), len(hash_b))
    return 1.0 - (dist / max_bits)


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two float vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class SimilarityMatch:
    """A ranked similarity match result."""

    def __init__(
        self,
        match_type: str,          # "local" | "external"
        source_media_id: str | None,
        external_url: str | None,
        similarity: float,
        method: str,              # "sha256" | "phash" | "dhash" | "embedding" | "internet"
        algorithm: str | None = None,
        rank: int = 1,
    ):
        self.match_type = match_type
        self.source_media_id = source_media_id
        self.external_url = external_url
        self.similarity = round(float(similarity), 4)
        self.method = method
        self.algorithm = algorithm
        self.rank = rank

    def to_dict(self) -> dict[str, Any]:
        return {
            "match_type": self.match_type,
            "source_media_id": self.source_media_id,
            "external_url": self.external_url,
            "similarity": self.similarity,
            "method": self.method,
            "algorithm": self.algorithm,
            "rank": self.rank,
        }


class SimilarityEngine:
    """
    Coordinates multi-method similarity search.

    Local search is always attempted first using stored fingerprints.
    External search is attempted only if a configured provider is present.
    """

    # Thresholds for declaring a match
    PHASH_MATCH_THRESHOLD = 0.85   # 85% similarity → perceptual match
    DHASH_MATCH_THRESHOLD = 0.85
    EMBEDDING_MATCH_THRESHOLD = 0.80

    def __init__(
        self,
        search_provider: BaseSourceSearchProvider | None = None,
    ):
        self.search_provider = search_provider or DisabledSourceSearchProvider()

    def search_local(
        self,
        query_fingerprints: dict[str, str],  # {algorithm: value}
        db: Session,
        exclude_media_id: str | None = None,
    ) -> list[SimilarityMatch]:
        """
        Search stored fingerprints for similar media.

        Args:
            query_fingerprints: Dict of {algorithm: hash_value} for the query media.
            db: Database session.
            exclude_media_id: Media ID to exclude (e.g. the query itself).

        Returns:
            Ranked list of SimilarityMatch objects.
        """
        from app.db.repository import get_all_fingerprints

        matches: list[SimilarityMatch] = []

        # ── SHA-256 exact match ───────────────────────────────────────────────
        query_sha256 = query_fingerprints.get("sha256")
        if query_sha256:
            stored = get_all_fingerprints(db, "sha256")
            for fp in stored:
                if exclude_media_id and fp.media_id == exclude_media_id:
                    continue
                if fp.value == query_sha256:
                    matches.append(SimilarityMatch(
                        match_type="local",
                        source_media_id=fp.media_id,
                        external_url=None,
                        similarity=1.0,
                        method="exact",
                        algorithm="sha256",
                    ))

        # ── pHash perceptual match ────────────────────────────────────────────
        query_phash = query_fingerprints.get("phash")
        if query_phash:
            stored = get_all_fingerprints(db, "phash")
            for fp in stored:
                if exclude_media_id and fp.media_id == exclude_media_id:
                    continue
                sim = _hamming_similarity(query_phash, fp.value)
                if sim >= self.PHASH_MATCH_THRESHOLD:
                    matches.append(SimilarityMatch(
                        match_type="local",
                        source_media_id=fp.media_id,
                        external_url=None,
                        similarity=sim,
                        method="perceptual",
                        algorithm="phash",
                    ))

        # ── dHash perceptual match ────────────────────────────────────────────
        query_dhash = query_fingerprints.get("dhash")
        if query_dhash:
            stored = get_all_fingerprints(db, "dhash")
            for fp in stored:
                if exclude_media_id and fp.media_id == exclude_media_id:
                    continue
                sim = _hamming_similarity(query_dhash, fp.value)
                if sim >= self.DHASH_MATCH_THRESHOLD:
                    matches.append(SimilarityMatch(
                        match_type="local",
                        source_media_id=fp.media_id,
                        external_url=None,
                        similarity=sim,
                        method="perceptual",
                        algorithm="dhash",
                    ))

        # Deduplicate by media_id, keep highest similarity
        seen: dict[str, SimilarityMatch] = {}
        for m in matches:
            key = m.source_media_id or ""
            if key not in seen or m.similarity > seen[key].similarity:
                seen[key] = m

        ranked = sorted(seen.values(), key=lambda m: m.similarity, reverse=True)
        for i, m in enumerate(ranked):
            m.rank = i + 1

        return ranked

    def search_external(self, file_path: str) -> tuple[list[SimilarityMatch], str]:
        """
        Search external sources using the configured provider.

        Returns:
            (matches, provider_status)
        """
        if self.search_provider.status == ProviderStatus.UNAVAILABLE:
            return [], ProviderStatus.UNAVAILABLE.value

        try:
            raw_results: list[SourceSearchResult] = self.search_provider.search_by_image(file_path)
        except Exception as exc:
            logger.warning("External search provider failed: %s", exc)
            return [], ProviderStatus.FAILED.value

        matches = []
        for i, r in enumerate(raw_results):
            matches.append(SimilarityMatch(
                match_type="external",
                source_media_id=None,
                external_url=r.url,
                similarity=float(r.similarity),
                method="internet",
                algorithm=r.provider,
                rank=i + 1,
            ))

        return matches, ProviderStatus.CONFIGURED.value

    def search(
        self,
        file_path: str,
        query_fingerprints: dict[str, str] | None = None,
        db: Session | None = None,
        exclude_media_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Run full similarity search: local first, then external.

        Returns a dict with:
          local_matches: list of local SimilarityMatch dicts
          external_matches: list of external SimilarityMatch dicts
          external_provider_status: UNAVAILABLE | CONFIGURED | FAILED
          total_matches: int
        """
        local_matches: list[SimilarityMatch] = []
        if db is not None and query_fingerprints:
            local_matches = self.search_local(query_fingerprints, db, exclude_media_id)

        external_matches, provider_status = self.search_external(file_path)

        all_matches = local_matches + external_matches
        return {
            "local_matches": [m.to_dict() for m in local_matches],
            "external_matches": [m.to_dict() for m in external_matches],
            "external_provider_status": provider_status,
            "total_matches": len(all_matches),
        }
