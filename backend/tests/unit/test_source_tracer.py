import pytest
from app.services.provenance.source_tracer import SourceTracer
from app.services.similarity.internet_search_provider import (
    DisabledSourceSearchProvider,
    BaseSourceSearchProvider,
    ProviderStatus,
    SourceSearchResult,
)


class MockExternalSourceProvider(BaseSourceSearchProvider):
    @property
    def status(self) -> ProviderStatus:
        return ProviderStatus.CONFIGURED

    def search_by_image(self, file_path: str):
        return [
            SourceSearchResult(
                url="https://source-early.org/image.jpg",
                title="Early Source",
                domain="source-early.org",
                similarity=0.95,
                source_timestamp="2021-06-15T12:00:00Z",
                provider="MockSearch",
            ),
            SourceSearchResult(
                url="https://source-late.org/image.jpg",
                title="Late Source",
                domain="source-late.org",
                similarity=0.90,
                source_timestamp="2023-08-20T12:00:00Z",
                provider="MockSearch",
            ),
        ]


def test_source_tracer_disabled():
    tracer = SourceTracer(search_provider=DisabledSourceSearchProvider())
    result = tracer.trace_source("sample.jpg")
    assert "earliest_known_occurrence" in result
    assert result["earliest_known_occurrence"] is None
    assert result["candidates"] == []
    assert result["internet_provider_status"] == "UNAVAILABLE"
    assert "INFERENCE" in result["note"]


def test_source_tracer_finds_earliest():
    provider = MockExternalSourceProvider()
    tracer = SourceTracer(search_provider=provider)
    result = tracer.trace_source("sample.jpg")

    assert result["internet_provider_status"] == "CONFIGURED"
    assert len(result["candidates"]) == 2
    assert result["earliest_known_occurrence"] is not None
    assert result["earliest_known_occurrence"]["url"] == "https://source-early.org/image.jpg"
    assert result["earliest_known_occurrence"]["source_timestamp"] == "2021-06-15T12:00:00Z"
