import pytest
from app.services.similarity.similarity_engine import SimilarityEngine
from app.services.similarity.internet_search_provider import BaselineInternetSearchProvider

def test_similarity_engine_no_provider():
    engine = SimilarityEngine()
    with pytest.raises(RuntimeError):
        engine.search("sample.jpg")

def test_similarity_engine_search():
    provider = BaselineInternetSearchProvider()
    engine = SimilarityEngine(search_provider=provider)
    
    results = engine.search("sample.jpg")
    assert isinstance(results, list)
    assert len(results) > 0
    assert "url" in results[0]
    assert "similarity" in results[0]
