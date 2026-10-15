"""Similarity schemas."""
from pydantic import BaseModel, Field


class SimilarityMatch(BaseModel):
    similarity: float = Field(ge=0.0, le=1.0)
    match_type: str  # local | external
    method: str  # sha256 | phash | dhash | embedding | internet
    algorithm: str | None = None
    source_media_id: str | None = None
    external_url: str | None = None
    rank: int = 1
    confidence: float | None = None


class SimilaritySearchResult(BaseModel):
    media_id: str
    query_fingerprints: dict[str, str] | None = None
    local_matches: list[SimilarityMatch] = Field(default_factory=list)
    external_matches: list[SimilarityMatch] = Field(default_factory=list)
    total_matches: int = 0
    provider_status: str = "UNAVAILABLE"
