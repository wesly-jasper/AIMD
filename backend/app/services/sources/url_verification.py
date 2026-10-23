"""
UrlVerificationSource: does the page the user names actually hold this image?

The user supplies the URL where they say they found the image. It is fetched
through ``GuardedFetcher`` (SSRF guard, size and time caps). A direct image URL
is compared as is; an HTML page yields its declared images (og:image,
twitter:image, image_src, then <img> tags), of which at most ``max_images`` are
fetched. Each is verified exactly as a local match is, and kept as an artefact
so the comparison can be re-run.

A page that does not hold the image is a COMPLETED search with no verified hit,
not a failure: the claim was checked and not supported.
"""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlsplit

import numpy as np

from app.services.fingerprinting.fingerprint import (
    EMBEDDING_ALGORITHM,
    FingerprintService,
    decode_embedding,
)
from app.services.similarity.similarity_engine import NO_MATCH, _hamming_distance, verify_candidate
from app.services.sources.commons_client import clean_url
from app.services.sources.connectors import (
    SourceConnector,
    SourceHit,
    SourceQuery,
    SourceResult,
    SourceStatus,
)
from app.services.sources.url_guard import BlockedURL, FetchError, GuardedFetcher

logger = logging.getLogger(__name__)

_EXTENSIONS = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp",
               "image/gif": ".gif", "image/bmp": ".bmp", "image/tiff": ".tiff"}
_META_IMAGE_KEYS = {"og:image", "og:image:url", "og:image:secure_url", "twitter:image",
                    "twitter:image:src"}


class _ImageLinks(HTMLParser):
    """Declared images first (meta / link), then <img> tags, in page order."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.declared: list[str] = []
        self.inline: list[str] = []

    def handle_starttag(self, tag, attrs):
        a = {k.lower(): (v or "") for k, v in attrs}
        if tag == "meta" and (a.get("property") or a.get("name", "")).lower() in _META_IMAGE_KEYS:
            if a.get("content"):
                self.declared.append(a["content"])
        elif tag == "link" and a.get("rel", "").lower() == "image_src" and a.get("href"):
            self.declared.append(a["href"])
        elif tag == "img":
            src = a.get("src") or a.get("data-src")
            if src and not src.startswith("data:"):
                self.inline.append(src)


def image_links(html: str, base_url: str) -> list[str]:
    parser = _ImageLinks()
    parser.feed(html)
    seen, out = set(), []
    for link in parser.declared + parser.inline:
        url = clean_url(urljoin(base_url, link.strip()))
        if urlsplit(url).scheme in ("http", "https") and url not in seen:
            seen.add(url)
            out.append(url)
    return out


def _cosine(a: Optional[str], b: Optional[str]) -> Optional[float]:
    if not a or not b:
        return None
    va, vb = decode_embedding(a), decode_embedding(b)
    if va.shape != vb.shape:
        return None
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    return round(float(va @ vb) / denom, 6) if denom else None


class UrlVerificationSource(SourceConnector):
    name = "url_verification"

    def __init__(
        self,
        enabled: Optional[bool] = None,
        *,
        fetcher_factory=GuardedFetcher,
        fingerprinter: Optional[FingerprintService] = None,
        artefact_dir: Optional[Path] = None,
        max_images: int = 8,
    ):
        self._enabled = enabled
        self._fetcher_factory = fetcher_factory
        self._fingerprinter = fingerprinter
        self._artefact_dir = artefact_dir
        self.max_images = max_images

    def _settings(self):
        from app.core.config import settings

        return settings

    def unavailable_reason(self, db) -> Optional[str]:
        enabled = self._enabled if self._enabled is not None else self._settings().source_url_verification_enabled
        if not enabled:
            return "disabled (SOURCE_URL_VERIFICATION_ENABLED=false)"
        return None

    def run(self, query: SourceQuery, db) -> SourceResult:
        if not query.claimed_url:
            return SourceResult(self.name, SourceStatus.UNAVAILABLE, "no source URL was supplied")
        result = super().run(query, db)
        result.detail["claimed_url"] = query.claimed_url
        return result

    def _store(self, body: bytes, content_type: str) -> Path:
        root = self._artefact_dir or (self._settings().artefacts_dir / "sources")
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{hashlib.sha256(body).hexdigest()}{_EXTENSIONS.get(content_type, '.img')}"
        if not path.exists():
            path.write_bytes(body)
        return path

    def search(self, query: SourceQuery, db) -> list[SourceHit]:
        fingerprinter = self._fingerprinter or FingerprintService()
        page_url = query.claimed_url
        with self._fetcher_factory() as fetcher:
            page = fetcher.fetch(page_url)
            retrieved_at = datetime.now(timezone.utc)
            if page.content_type.startswith("image/"):
                fetched = [(page, retrieved_at)]
                links: list[str] = []
            else:
                links = image_links(page.body.decode("utf-8", errors="replace"), page.url)
                fetched = []
                for link in links[: self.max_images]:
                    try:
                        fetched.append((fetcher.fetch(link, accept=("image/",)), datetime.now(timezone.utc)))
                    except (BlockedURL, FetchError) as exc:
                        logger.info("Skipped image %s: %s", link, exc)

        best: Optional[SourceHit] = None
        rank = {"VERIFIED_MATCH": 2, "SIMILAR_UNVERIFIED": 1}
        for image, when in fetched:
            path = self._store(image.body, image.content_type)
            fps = {r["algorithm"]: r for r in fingerprinter.generate(path)["_records"]}
            q_phash = query.fingerprints.get("phash")
            c_phash = fps.get("phash", {}).get("value")
            same_model = fps.get(EMBEDDING_ALGORITHM, {}).get("algorithm_version") == query.embedding_version
            candidate = {
                "sha256_exact": query.fingerprints.get("sha256") == fps.get("sha256", {}).get("value"),
                "phash_distance": (_hamming_distance(q_phash, c_phash)
                                   if q_phash and c_phash and len(q_phash) == len(c_phash) else None),
                "cosine": (_cosine(query.fingerprints.get(EMBEDDING_ALGORITHM),
                                   fps.get(EMBEDDING_ALGORITHM, {}).get("value"))
                           if same_model else None),
                "retrieved_by": ["url"],
            }
            status, verification = verify_candidate(query.file_path, candidate, path)
            if status == NO_MATCH:
                continue
            hit = SourceHit(
                url=page_url,
                title=None,
                provider=self.name,
                verification_status=status,
                verification=verification,
                similarity=candidate["cosine"],
                published_at=None,  # a page does not reliably say when it was published
                citation={
                    "claimed_url": page_url,
                    "final_url": page.url,
                    "file_url": image.url,
                    "retrieved_at": when.isoformat(),
                    "peer_ip": image.peer_ip,
                    "redirects": page.redirects,
                    "images_on_page": len(links),
                    "images_compared": len(fetched),
                    "stored_copy_sha256": path.stem,
                },
            )
            if best is None or rank[status] > rank[best.verification_status]:
                best = hit
        return [best] if best else []
