"""
Source search provider abstraction.

Providers search external sources (reverse image search, etc.) for
matching media. When no provider is configured, the system returns
status=UNAVAILABLE — which is distinct from "no sources found".

AIMD never fabricates internet search results.
AIMD never claims "absolute original source" — only
"earliest-known occurrence discovered by AIMD".
"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any


class ProviderStatus(str, Enum):
    UNAVAILABLE = "UNAVAILABLE"   # Provider not configured
    CONFIGURED = "CONFIGURED"     # Provider ready
    FAILED = "FAILED"             # Provider errored during request


class SourceSearchResult:
    """A single result from an external source search."""

    def __init__(
        self,
        url: str,
        title: str | None,
        domain: str | None,
        similarity: float,
        source_timestamp: str | None,
        provider: str,
        raw: dict | None = None,
    ):
        self.url = url
        self.title = title
        self.domain = domain
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
    def search_by_image(self, file_path: str) -> list[SourceSearchResult]:
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

    To enable real internet search, configure a provider such as
    GoogleVisionProvider or TinEyeProvider via environment variables.
    """

    @property
    def status(self) -> ProviderStatus:
        return ProviderStatus.UNAVAILABLE

    def search_by_image(self, file_path: str) -> list[SourceSearchResult]:
        # Return empty — do NOT fabricate results
        return []


# Keep the old name as an alias for backwards compatibility
BaselineInternetSearchProvider = DisabledSourceSearchProvider
BaseInternetSearchProvider = BaseSourceSearchProvider
