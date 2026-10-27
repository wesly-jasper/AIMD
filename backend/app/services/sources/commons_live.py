"""
CommonsSha1Source: is this exact file on Wikimedia Commons?

The ``wikimedia_commons`` connector searches a local corpus of a few hundred
pre-fetched Commons images, so a file taken straight from Commons was "not
found" unless it happened to be in that corpus. This asks Commons itself, live,
for any file with the upload's SHA-1 (``list=allimages&aisha1=``).

Only byte-identical copies match: a resized or re-saved copy has another hash.
A match is therefore VERIFIED_MATCH by construction, and its citation carries
the file URL and SHA-1 that the Wayback connector looks up next.
"""
from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Optional

import httpx

from app.services.sources.commons_client import COMMONS_API, user_agent
from app.services.sources.connectors import SourceConnector, SourceHit, SourceQuery


def _sha1(path: str) -> str:
    digest = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _when(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


class CommonsSha1Source(SourceConnector):
    name = "wikimedia_commons_live"

    def __init__(self, enabled: Optional[bool] = None, contact: Optional[str] = None, *,
                 timeout: float = 15.0, transport: Optional[httpx.BaseTransport] = None):
        self._enabled = enabled
        self._contact = contact
        self.timeout = timeout
        self._transport = transport

    def _settings(self) -> tuple[bool, Optional[str]]:
        from app.core.config import settings

        enabled = settings.source_commons_live_enabled if self._enabled is None else self._enabled
        contact = self._contact if self._contact is not None else settings.wikimedia_contact
        return enabled, (contact or "").strip() or None

    def unavailable_reason(self, db) -> Optional[str]:
        enabled, contact = self._settings()
        if not enabled:
            return "disabled (SOURCE_COMMONS_LIVE_ENABLED=false)"
        if not contact:
            # Wikimedia's User-Agent policy: identify the operator, or be blocked.
            return "set WIKIMEDIA_CONTACT (an email or URL) to query Commons live"
        return None

    def search(self, query: SourceQuery, db) -> list[SourceHit]:
        _, contact = self._settings()
        sha1 = _sha1(query.file_path)
        params = {
            "action": "query", "format": "json", "formatversion": "2",
            "list": "allimages", "aisha1": sha1, "aiprop": "url|timestamp|user|sha1|size|mime",
            "ailimit": "10",
        }
        with httpx.Client(headers={"User-Agent": user_agent(contact)}, timeout=self.timeout,
                          transport=self._transport) as client:
            response = client.get(COMMONS_API, params=params)
            response.raise_for_status()
            images = response.json().get("query", {}).get("allimages", [])

        hits = []
        for image in images:
            if image.get("sha1") != sha1:  # the API filters by it; keep the check
                continue
            hits.append(SourceHit(
                url=image.get("descriptionurl") or image.get("url"),
                title=image.get("title") or image.get("name"),
                provider=self.name,
                verification_status="VERIFIED_MATCH",
                verification={"method": "sha1-exact", "sha1": sha1},
                similarity=1.0,
                published_at=_when(image.get("timestamp")),
                citation={
                    "file_url": image.get("url"),
                    "original_sha1": sha1,
                    "uploader": image.get("user"),
                    "size_bytes": image.get("size"),
                    "mime": image.get("mime"),
                },
            ))
        return hits
