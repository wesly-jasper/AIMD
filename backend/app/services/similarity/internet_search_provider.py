"""
Source search provider abstraction.

Providers search external sources (reverse image search, etc.) for
matching media. When no provider is configured, the system returns
status=UNAVAILABLE -- which is distinct from "no sources found".

AIMD never fabricates internet search results.
AIMD never claims "absolute original source" -- only
"earliest-known occurrence discovered by AIMD".
"""
from abc import ABC, abstractmethod
from enum import Enum
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import urllib.parse

logger = logging.getLogger(__name__)


class ProviderStatus(str, Enum):
    UNAVAILABLE = "UNAVAILABLE"   # Provider not configured
    CONFIGURED = "CONFIGURED"     # Provider ready
    FAILED = "FAILED"             # Provider errored during request


class SourceSearchResult:
    """A single result from an external source search."""

    def __init__(
        self,
        url: str,
        title: Optional[str] = None,
        domain: Optional[str] = None,
        similarity: float = 0.0,
        source_timestamp: Optional[str] = None,
        provider: str = "unknown",
        raw: Optional[dict] = None,
    ):
        self.url = url
        self.title = title
        self.domain = domain or (urllib.parse.urlparse(url).netloc if url else None)
        self.similarity = similarity
        self.source_timestamp = source_timestamp
        self.provider = provider
        self.raw = raw or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "domain": self.domain,
            "similarity": self.similarity,
            "source_timestamp": self.source_timestamp,
            "provider": self.provider,
        }


class BaseSourceSearchProvider(ABC):

    @property
    @abstractmethod
    def status(self) -> ProviderStatus:
        ...

    @abstractmethod
    def search_by_image(self, file_path: str) -> List[SourceSearchResult]:
        """Search for the image in external sources.

        Returns an empty list when no matches are found.
        Raises RuntimeError if the provider is in a failed state.
        """
        ...


class DisabledSourceSearchProvider(BaseSourceSearchProvider):
    """
    Default provider used when no external search API is configured.

    Returns UNAVAILABLE status and an empty result list.
    This is the correct forensic response: we cannot claim
    "no sources exist on the internet" without actually searching.
    """

    @property
    def status(self) -> ProviderStatus:
        return ProviderStatus.UNAVAILABLE

    def search_by_image(self, file_path: str) -> List[SourceSearchResult]:
        return []


class GoogleVisionSearchProvider(BaseSourceSearchProvider):
    """
    Google Cloud Vision Web Detection provider for reverse image search.
    Queries WebDetection API for full/partial image matches and web occurrences.
    """

    def __init__(self, api_key: Optional[str] = None, credentials_path: Optional[str] = None):
        self.api_key = api_key
        self.credentials_path = credentials_path
        self._status = ProviderStatus.CONFIGURED if (api_key or credentials_path) else ProviderStatus.UNAVAILABLE

    @property
    def status(self) -> ProviderStatus:
        return self._status

    def search_by_image(self, file_path: str) -> List[SourceSearchResult]:
        if self._status == ProviderStatus.UNAVAILABLE:
            return []

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {file_path}")

        results: List[SourceSearchResult] = []

        try:
            import requests

            # If REST API key configured
            if self.api_key:
                import base64
                with open(path, "rb") as image_file:
                    content = base64.b64encode(image_file.read()).decode("utf-8")

                url = f"https://vision.googleapis.com/v1/images:annotate?key={self.api_key}"
                payload = {
                    "requests": [
                        {
                            "image": {"content": content},
                            "features": [{"type": "WEB_DETECTION", "maxResults": 20}],
                        }
                    ]
                }
                resp = requests.post(url, json=payload, timeout=15)
                if resp.status_code != 200:
                    self._status = ProviderStatus.FAILED
                    raise RuntimeError(f"Google Vision API error {resp.status_code}: {resp.text}")

                data = resp.json()
                web_detection = data.get("responses", [{}])[0].get("webDetection", {})
                results = self._parse_web_detection(web_detection)
            return results

        except Exception as exc:
            self._status = ProviderStatus.FAILED
            logger.error(f"Google Vision search failed: {exc}")
            raise RuntimeError(f"Google Vision search error: {exc}") from exc

    def _parse_web_detection(self, web_data: dict) -> List[SourceSearchResult]:
        results: List[SourceSearchResult] = []

        # 1. Full matching images (1.0 similarity)
        for item in web_data.get("fullMatchingImages", []):
            u = item.get("url")
            if u:
                results.append(
                    SourceSearchResult(
                        url=u,
                        title="Full Visual Match",
                        similarity=1.0,
                        provider="GoogleVision",
                        raw=item,
                    )
                )

        # 2. Partial matching images (0.85 similarity)
        for item in web_data.get("partialMatchingImages", []):
            u = item.get("url")
            if u:
                results.append(
                    SourceSearchResult(
                        url=u,
                        title="Partial Visual Match",
                        similarity=0.85,
                        provider="GoogleVision",
                        raw=item,
                    )
                )

        # 3. Pages with matching images (0.75 similarity)
        for page in web_data.get("pagesWithMatchingImages", []):
            u = page.get("url")
            t = page.get("pageTitle")
            if u:
                results.append(
                    SourceSearchResult(
                        url=u,
                        title=t or "Web Occurrence",
                        similarity=0.75,
                        provider="GoogleVision",
                        raw=page,
                    )
                )

        return results


class TinEyeSearchProvider(BaseSourceSearchProvider):
    """
    TinEye Commercial Reverse Image Search Provider.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self._status = ProviderStatus.CONFIGURED if api_key else ProviderStatus.UNAVAILABLE

    @property
    def status(self) -> ProviderStatus:
        return self._status

    def search_by_image(self, file_path: str) -> List[SourceSearchResult]:
        if self._status == ProviderStatus.UNAVAILABLE:
            return []

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {file_path}")

        try:
            import requests

            url = "https://api.tineye.com/rest/search/"
            headers = {"x-api-key": self.api_key}
            with open(path, "rb") as f:
                files = {"image": f}
                resp = requests.post(url, headers=headers, files=files, timeout=15)

            if resp.status_code != 200:
                self._status = ProviderStatus.FAILED
                raise RuntimeError(f"TinEye API returned error {resp.status_code}")

            data = resp.json()
            matches = data.get("results", {}).get("matches", [])
            results = []
            for m in matches:
                backlinks = m.get("backlinks", [])
                primary_url = backlinks[0].get("url") if backlinks else m.get("image_url")
                crawl_date = backlinks[0].get("crawl_date") if backlinks else None
                score = float(m.get("score", 85.0)) / 100.0
                if primary_url:
                    results.append(
                        SourceSearchResult(
                            url=primary_url,
                            title=f"TinEye Match ({m.get('domain', '')})",
                            domain=m.get("domain"),
                            similarity=round(score, 4),
                            source_timestamp=crawl_date,
                            provider="TinEye",
                            raw=m,
                        )
                    )
            return results
        except Exception as exc:
            self._status = ProviderStatus.FAILED
            logger.error(f"TinEye search failed: {exc}")
            raise RuntimeError(f"TinEye search error: {exc}") from exc


# Aliases for backwards compatibility
BaselineInternetSearchProvider = DisabledSourceSearchProvider
BaseInternetSearchProvider = BaseSourceSearchProvider
