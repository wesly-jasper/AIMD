"""
WaybackSource: when did the Internet Archive first and last capture a source?

Runs after the other connectors and looks up the URLs of their VERIFIED hits in
the Wayback Machine's CDX index (earliest and latest capture with HTTP 200).

What a capture proves depends on what was captured:

  * the image file itself (a Commons file URL): the CDX ``digest`` is the SHA-1
    of the archived bytes, and Commons publishes the original's SHA-1. When they
    agree the archive held this exact file then -- a VERIFIED_MATCH, and an
    ARCHIVED event on the timeline. When they differ, the URL was archived with
    other bytes (the file was since replaced), and that is said.
  * a web page: it existed then, but whether it showed this image is not
    checked. Reported as SIMILAR_UNVERIFIED, a lead -- never the earliest
    appearance.
"""
from __future__ import annotations

import base64
import logging
from datetime import datetime, timezone
from typing import Optional

import httpx

from app.services.sources.connectors import SourceConnector, SourceHit, SourceQuery

logger = logging.getLogger(__name__)

CDX_API = "https://web.archive.org/cdx/search/cdx"
USER_AGENT = "AIMD-provenance/1.0 (+https://github.com/wesly-jasper/AIMD)"


def sha1_to_cdx_digest(sha1_hex: str) -> str:
    """CDX digests are the base32 of the raw SHA-1."""
    return base64.b32encode(bytes.fromhex(sha1_hex)).decode("ascii")


def _capture_time(ts: str) -> datetime:
    return datetime.strptime(ts, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)


class WaybackSource(SourceConnector):
    name = "wayback"

    def __init__(self, enabled: Optional[bool] = None, *, timeout: float = 15.0,
                 transport: Optional[httpx.BaseTransport] = None, max_urls: int = 4):
        self._enabled = enabled
        self.timeout = timeout
        self._transport = transport
        self.max_urls = max_urls

    def unavailable_reason(self, db) -> Optional[str]:
        enabled = self._enabled
        if enabled is None:
            from app.core.config import settings

            enabled = settings.source_wayback_enabled
        return None if enabled else "disabled (SOURCE_WAYBACK_ENABLED=false)"

    def _captures(self, client: httpx.Client, url: str) -> list[list[str]]:
        """Earliest and latest 200 captures: rows of [timestamp, original, digest]."""
        rows = []
        for limit in ("1", "-1"):
            response = client.get(CDX_API, params={
                "url": url, "output": "json", "fl": "timestamp,original,digest",
                "filter": "statuscode:200", "limit": limit,
            })
            response.raise_for_status()
            data = response.json() if response.content.strip() else []
            rows.extend(data[1:])  # the first row is the header
        unique = {row[0]: row for row in rows}
        return [unique[k] for k in sorted(unique)]

    def _file_captures(self, client: httpx.Client, url: str, sha1: str) -> list[list[str]]:
        """Every capture of ``url`` whose bytes are the file with ``sha1`` (oldest first)."""
        response = client.get(CDX_API, params={
            "url": url, "output": "json", "fl": "timestamp,original,digest",
            "filter": ["statuscode:200", f"digest:{sha1_to_cdx_digest(sha1)}"], "limit": "5",
        })
        response.raise_for_status()
        data = response.json() if response.content.strip() else []
        return data[1:]

    def search(self, query: SourceQuery, db) -> list[SourceHit]:
        targets = []  # (url, sha1 or None, the hit it came from)
        for hit in query.prior_hits:
            if hit.verification_status != "VERIFIED_MATCH" or hit.provider == self.name:
                continue
            if hit.citation.get("file_url") and hit.citation.get("original_sha1"):
                targets.append((hit.citation["file_url"], hit.citation["original_sha1"], hit))
            targets.append((hit.url, None, hit))
        # Two connectors often verify the same page: look each URL up once.
        seen: set[str] = set()
        targets = [t for t in targets if not (t[0] in seen or seen.add(t[0]))]
        hits: list[SourceHit] = []
        if not targets:
            return hits
        with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=self.timeout,
                          transport=self._transport, trust_env=False) as client:
            for url, sha1, origin in targets[: self.max_urls]:
                if sha1:
                    matching = self._file_captures(client, url, sha1)
                    if matching:
                        ts, original, digest = matching[0]
                        hits.append(self._hit(url, ts, original, origin, "VERIFIED_MATCH", {
                            "method": "cdx-sha1-digest",
                            "gates": {"digest": True},
                            "reasons": [],
                            "digest": digest,
                            "expected_sha1": sha1,
                        }))
                        continue
                    captures = self._captures(client, url)
                    if captures:
                        ts, original, digest = captures[0]
                        hits.append(self._hit(url, ts, original, origin, "SIMILAR_UNVERIFIED", {
                            "method": "cdx-sha1-digest",
                            "gates": {"digest": False},
                            "reasons": ["the archived bytes differ from the indexed file "
                                        "(the file at this URL was replaced)"],
                            "digest": digest,
                            "expected_sha1": sha1,
                        }))
                    continue
                captures = self._captures(client, url)
                if captures:
                    ts, original, digest = captures[0]
                    hits.append(self._hit(url, ts, original, origin, "SIMILAR_UNVERIFIED", {
                        "method": "page_capture",
                        "gates": {"image_on_archived_page": None},
                        "reasons": ["the archived page was not checked for the image"],
                        "digest": digest,
                        "latest_capture": captures[-1][0],
                    }))
        return hits

    def _hit(self, url, ts, original, origin: SourceHit, status, verification) -> SourceHit:
        return SourceHit(
            url=f"https://web.archive.org/web/{ts}/{original}",
            title=f"Wayback Machine capture of {original}",
            provider=self.name,
            verification_status=status,
            verification=verification,
            published_at=_capture_time(ts),
            citation={
                "archived_url": url,
                "capture_timestamp": ts,
                "cdx_api": CDX_API,
                "found_via": origin.provider,
                "found_via_url": origin.url,
            },
        )
