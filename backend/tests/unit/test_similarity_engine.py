import pytest
from app.services.similarity.similarity_engine import SimilarityEngine
from app.services.similarity.internet_search_provider import (
    DisabledSourceSearchProvider,
    BaseSourceSearchProvider,
    ProviderStatus,
    SourceSearchResult,
)


class MockExternalProvider(BaseSourceSearchProvider):
    @property
    def status(self) -> ProviderStatus:
        return ProviderStatus.CONFIGURED

    def search_by_image(self, file_path: str):
        return [
            SourceSearchResult(
                url="https://verified-source.org/image.jpg",
                title="Verified Source Image",
                domain="verified-source.org",
                similarity=0.92,
                source_timestamp="2023-01-01T00:00:00Z",
                provider="MockExternalProvider",
            )
        ]


def test_similarity_engine_disabled_provider():
    engine = SimilarityEngine(search_provider=DisabledSourceSearchProvider())
    result = engine.search("sample.jpg")
    assert isinstance(result, dict)
    assert result["external_provider_status"] == "UNAVAILABLE"
    assert result["external_matches"] == []
    assert result["total_matches"] == 0


def test_similarity_engine_configured_provider():
    provider = MockExternalProvider()
    engine = SimilarityEngine(search_provider=provider)
    
    result = engine.search("sample.jpg")
    assert isinstance(result, dict)
    assert result["external_provider_status"] == "CONFIGURED"
    assert len(result["external_matches"]) == 1
    assert result["external_matches"][0]["external_url"] == "https://verified-source.org/image.jpg"
    assert result["external_matches"][0]["similarity"] == 0.92
