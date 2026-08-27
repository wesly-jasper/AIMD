from app.services.provenance.source_tracer import SourceTracer
from app.services.similarity.similarity_engine import SimilarityEngine
from app.services.similarity.internet_search_provider import BaselineInternetSearchProvider

def test_source_tracer():
    provider = BaselineInternetSearchProvider()
    similarity_engine = SimilarityEngine(search_provider=provider)
    tracer = SourceTracer(similarity_engine=similarity_engine)
    
    result = tracer.trace_source("sample.jpg")
    assert "earliest_known_occurrence" in result
    assert "candidates" in result
    assert len(result["candidates"]) > 0
    assert result["earliest_known_occurrence"] is not None
    assert "media_id" in result["earliest_known_occurrence"]
    assert "similarity" in result["earliest_known_occurrence"]
