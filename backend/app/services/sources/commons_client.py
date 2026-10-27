"""
Wikimedia Commons client for building the reference corpus.

Lists the files in a category with the metadata a citation needs (page URL,
uploader, upload time, licence) and downloads a scaled rendition of each.

Commons rate-limits hard, and anonymous clients first: every request carries a
User-Agent with the operator's contact (Wikimedia's User-Agent policy), requests
are paced, and a 429/503 is retried after the ``Retry-After`` the server sends.
"""
from __future__ import annotations

import hashlib
import html
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Callable, Iterator, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx

logger = logging.getLogger(__name__)

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
CORPUS = "wikimedia_commons"
CLIENT_NAME = "AIMD-reference-corpus/1.0"

#: The descriptive fields kept from ``extmetadata``. All are the uploader's
#: statements about the file, not facts verified here.
EXTMETADATA_FIELDS = (
    "DateTimeOriginal", "LicenseShortName", "LicenseUrl", "Artist", "Credit",
    "AttributionRequired", "ImageDescription", "ObjectName",
)
#: Files per API request. extmetadata makes each page of results expensive to
#: build, so Commons may return fewer and continue.
BATCH_SIZE = 50

_RETRY_STATUSES = {429, 503}
_TAG = re.compile(r"<[^>]+>")
_SPACE = re.compile(r"\s+")


class CommonsError(RuntimeError):
    """Commons refused or failed a request in a way a retry will not fix."""


@dataclass
class CommonsFile:
    """One file as Commons describes it. ``uploaded_at`` is the current version's upload."""
    title: str
    page_url: str
    file_url: str
    download_url: str
    uploader: Optional[str]
    uploaded_at: Optional[datetime]
    sha1: Optional[str]
    width: Optional[int]
    height: Optional[int]
    mime: Optional[str]
    size: Optional[int]
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def licence(self) -> Optional[str]:
        return self.metadata.get("LicenseShortName")

    @property
    def licence_url(self) -> Optional[str]:
        return self.metadata.get("LicenseUrl")

    @property
    def artist(self) -> Optional[str]:
        return self.metadata.get("Artist")


def user_agent(contact: str) -> str:
    return f"{CLIENT_NAME} ({contact}) httpx/{httpx.__version__}"


def plain_text(value: object) -> str:
    """extmetadata values are HTML fragments; keep the text."""
    text = html.unescape(_TAG.sub(" ", str(value)))
    return _SPACE.sub(" ", text).strip()


def category_title(name: str) -> str:
    """``"Quality images of buildings"`` -> ``"Category:Quality images of buildings"``."""
    name = name.strip().replace("_", " ")
    return name if name.lower().startswith("category:") else f"Category:{name}"


def clean_url(url: str) -> str:
    """Drop the ``utm_*`` tracking parameters Commons appends: a citation is the bare URL."""
    parts = urlsplit(url)
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
             if not k.lower().startswith("utm_")]
    return urlunsplit(parts._replace(query=urlencode(query)))


def _timestamp(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def parse_file(page: dict) -> Optional[CommonsFile]:
    """A ``CommonsFile`` from one page of an imageinfo query, or ``None`` without imageinfo."""
    infos = page.get("imageinfo") or []
    if page.get("missing") or not infos:
        return None
    info = infos[0]
    if not info.get("url") or not info.get("descriptionurl"):
        return None
    metadata = {}
    for key, entry in (info.get("extmetadata") or {}).items():
        if key in EXTMETADATA_FIELDS and isinstance(entry, dict) and entry.get("value") is not None:
            text = plain_text(entry["value"])
            if text:
                metadata[key] = text
    return CommonsFile(
        title=page["title"],
        page_url=clean_url(info["descriptionurl"]),
        file_url=clean_url(info["url"]),
        download_url=clean_url(info.get("thumburl") or info["url"]),
        uploader=info.get("user"),
        uploaded_at=_timestamp(info.get("timestamp")),
        sha1=info.get("sha1"),
        width=info.get("width"),
        height=info.get("height"),
        mime=info.get("mime"),
        size=info.get("size"),
        metadata=metadata,
    )


def _retry_after(response: httpx.Response, attempt: int) -> float:
    value = response.headers.get("retry-after")
    if value:
        try:
            return max(0.0, float(value))
        except ValueError:
            try:
                when = parsedate_to_datetime(value)
                return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())
            except (TypeError, ValueError):
                pass
    return float(2 ** attempt)


class CommonsClient:
    """
    ``contact`` is required: an email address or URL where Wikimedia can reach
    whoever runs this. ``transport``, ``sleep`` and ``clock`` exist for tests.
    """

    def __init__(
        self,
        contact: str,
        *,
        api_url: str = COMMONS_API,
        min_interval: float = 1.0,
        max_retries: int = 5,
        max_retry_after: float = 120.0,
        timeout: float = 30.0,
        transport: Optional[httpx.BaseTransport] = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        if not contact or not contact.strip():
            raise ValueError("a contact (email address or URL) is required by Wikimedia's policy")
        self.api_url = api_url
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.max_retry_after = max_retry_after
        self._sleep = sleep
        self._clock = clock
        self._last_request: Optional[float] = None
        self._client = httpx.Client(
            headers={"User-Agent": user_agent(contact.strip())},
            timeout=timeout,
            follow_redirects=True,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "CommonsClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _pace(self) -> None:
        if self._last_request is not None:
            wait = self.min_interval - (self._clock() - self._last_request)
            if wait > 0:
                self._sleep(wait)
        self._last_request = self._clock()

    def _send(self, method: str, url: str, **kwargs) -> httpx.Response:
        """A streamed response with a non-retryable status; the caller closes it."""
        for attempt in range(self.max_retries + 1):
            self._pace()
            request = self._client.build_request(method, url, **kwargs)
            response = self._client.send(request, stream=True)
            if response.status_code not in _RETRY_STATUSES:
                if response.status_code >= 400:
                    response.close()
                    raise CommonsError(f"HTTP {response.status_code} from {request.url.host}")
                return response
            response.close()
            wait = _retry_after(response, attempt)
            if attempt == self.max_retries or wait > self.max_retry_after:
                raise CommonsError(
                    f"HTTP {response.status_code} from {request.url.host}: "
                    f"rate-limited (retry after {wait:.0f} s)"
                )
            logger.info("HTTP %s, retrying in %.0f s", response.status_code, wait)
            self._sleep(wait)
        raise AssertionError("unreachable")

    def _api(self, params: dict) -> dict:
        for attempt in range(self.max_retries + 1):
            response = self._send("GET", self.api_url, params=params)
            try:
                response.read()
                break
            except httpx.TimeoutException:
                # extmetadata makes some pages slow to build; a timed-out read
                # is retried like a 503, then reported.
                if attempt == self.max_retries:
                    raise CommonsError("Commons API timed out") from None
                self._sleep(min(self.max_retry_after, 5.0 * (attempt + 1)))
            finally:
                response.close()
        try:
            data = response.json()
        except ValueError as exc:
            raise CommonsError("Commons API returned a non-JSON response") from exc
        if "error" in data:
            error = data["error"]
            raise CommonsError(f"Commons API error {error.get('code')}: {error.get('info')}")
        return data

    def subcategories(self, category: str) -> list[str]:
        """The names (without "Category:") of ``category``'s direct subcategories."""
        params = {
            "action": "query", "format": "json", "formatversion": "2",
            "list": "categorymembers", "cmtitle": category_title(category),
            "cmtype": "subcat", "cmlimit": "500",
        }
        names: list[str] = []
        cont: dict = {}
        while True:
            data = self._api({**params, **cont})
            for member in (data.get("query") or {}).get("categorymembers") or []:
                names.append(member["title"].split(":", 1)[1])
            cont = data.get("continue") or {}
            if not cont:
                return names

    def search_files(self, query: str, *, thumb_width: int, limit: int = 500) -> Iterator[CommonsFile]:
        """Files matching a Commons full-text ``query`` (CirrusSearch syntax), with imageinfo."""
        params = {
            "action": "query", "format": "json", "formatversion": "2",
            "generator": "search", "gsrsearch": query, "gsrnamespace": "6",
            "gsrlimit": str(BATCH_SIZE), "prop": "imageinfo",
            "iiprop": "url|timestamp|user|sha1|size|mime|extmetadata",
            "iiurlwidth": str(thumb_width),
            "iiextmetadatafilter": "|".join(EXTMETADATA_FIELDS),
            "iiextmetadatalanguage": "en",
        }
        seen: set[str] = set()
        cont: dict = {}
        while len(seen) < limit:
            data = self._api({**params, **cont})
            for page in (data.get("query") or {}).get("pages") or []:
                parsed = parse_file(page)
                if parsed is None or parsed.title in seen:
                    continue
                seen.add(parsed.title)
                yield parsed
            cont = data.get("continue") or {}
            if not cont:
                return

    def category_files(
        self, category: str, *, thumb_width: int, start_sortkey_prefix: Optional[str] = None,
    ) -> Iterator[CommonsFile]:
        """
        Every file directly in ``category`` (not its subcategories), lazily, page
        by page; from ``start_sortkey_prefix`` on when given, which lets a caller
        sample a large category at several points rather than only its start.
        """
        params = {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "generator": "categorymembers",
            "gcmtitle": category_title(category),
            "gcmtype": "file",
            "gcmlimit": str(BATCH_SIZE),
            "prop": "imageinfo",
            "iiprop": "url|timestamp|user|sha1|size|mime|extmetadata",
            "iiurlwidth": str(thumb_width),
            "iiextmetadatafilter": "|".join(EXTMETADATA_FIELDS),
            "iiextmetadatalanguage": "en",
        }
        if start_sortkey_prefix:
            params["gcmstartsortkeyprefix"] = start_sortkey_prefix
        seen: set[str] = set()
        cont: dict = {}
        while True:
            data = self._api({**params, **cont})
            for page in (data.get("query") or {}).get("pages") or []:
                parsed = parse_file(page)
                # A page can come back without imageinfo when the batch is split
                # across continuations; it is returned whole in a later one.
                if parsed is None or parsed.title in seen:
                    continue
                seen.add(parsed.title)
                yield parsed
            cont = data.get("continue") or {}
            if not cont:
                return

    def download(self, url: str, dest: Path, *, max_bytes: int) -> tuple[str, int, str]:
        """
        Stream ``url`` to ``dest``; returns (sha256, size, content type).

        Written to a ``.part`` file and renamed, so an interrupted download never
        leaves a truncated file under the final name.
        """
        dest.parent.mkdir(parents=True, exist_ok=True)
        partial = dest.with_name(dest.name + ".part")
        response = self._send("GET", url)
        digest, size = hashlib.sha256(), 0
        try:
            content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
            if not content_type.startswith("image/"):
                raise CommonsError(f"expected an image, got {content_type or 'no content type'}")
            with open(partial, "wb") as fh:
                for chunk in response.iter_bytes():
                    size += len(chunk)
                    if size > max_bytes:
                        raise CommonsError(f"download exceeds {max_bytes} bytes")
                    digest.update(chunk)
                    fh.write(chunk)
            os.replace(partial, dest)
        finally:
            response.close()
            partial.unlink(missing_ok=True)
        return digest.hexdigest(), size, content_type
