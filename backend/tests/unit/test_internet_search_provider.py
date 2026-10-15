import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from app.services.similarity.internet_search_provider import (
    DisabledSourceSearchProvider,
    GoogleVisionSearchProvider,
    TinEyeSearchProvider,
    ProviderStatus,
    SourceSearchResult,
)


def test_disabled_source_search_provider():
    provider = DisabledSourceSearchProvider()
    assert provider.status == ProviderStatus.UNAVAILABLE
    results = provider.search_by_image("any_file.png")
    assert results == []


def test_google_vision_provider_status_unavailable():
    provider = GoogleVisionSearchProvider(api_key=None)
    assert provider.status == ProviderStatus.UNAVAILABLE
    assert provider.search_by_image("test.png") == []


def test_google_vision_provider_mock_response(tmp_path):
    img_path = tmp_path / "img.png"
    img_path.write_bytes(b"mock image bytes")

    provider = GoogleVisionSearchProvider(api_key="fake_key_123")
    assert provider.status == ProviderStatus.CONFIGURED

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "responses": [
            {
                "webDetection": {
                    "fullMatchingImages": [
                        {"url": "https://example.com/origin.jpg"}
                    ],
                    "partialMatchingImages": [
                        {"url": "https://mirror.com/cropped.jpg"}
                    ],
                    "pagesWithMatchingImages": [
                        {"url": "https://news.com/article", "pageTitle": "Original News Story"}
                    ],
                }
            }
        ]
    }

    with patch("requests.post", return_value=mock_resp):
        results = provider.search_by_image(str(img_path))
        assert len(results) == 3
        assert results[0].url == "https://example.com/origin.jpg"
        assert results[0].similarity == 1.0
        assert results[0].domain == "example.com"
        assert results[1].similarity == 0.85
        assert results[2].similarity == 0.75
        assert results[2].title == "Original News Story"


def test_tineye_provider_mock_response(tmp_path):
    img_path = tmp_path / "img2.png"
    img_path.write_bytes(b"mock bytes")

    provider = TinEyeSearchProvider(api_key="tineye_test_key")
    assert provider.status == ProviderStatus.CONFIGURED

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "results": {
            "matches": [
                {
                    "domain": "archive.org",
                    "score": 95.0,
                    "backlinks": [
                        {
                            "url": "https://archive.org/photo123",
                            "crawl_date": "2021-04-12T10:00:00Z",
                        }
                    ],
                }
            ]
        }
    }

    with patch("requests.post", return_value=mock_resp):
        results = provider.search_by_image(str(img_path))
        assert len(results) == 1
        assert results[0].url == "https://archive.org/photo123"
        assert results[0].domain == "archive.org"
        assert results[0].similarity == 0.95
        assert results[0].source_timestamp == "2021-04-12T10:00:00Z"
