"""WaybackSource against a fake CDX API (response shapes captured 2026-10-23)."""
from datetime import datetime, timezone
from types import SimpleNamespace

import httpx
import pytest

from app.services.provenance.timeline import build_timeline
from app.services.sources.connectors import SourceHit, SourceQuery, SourceStatus
from app.services.sources.wayback import WaybackSource, sha1_to_cdx_digest

SHA1 = "83c52b418e206ff6a29f5ba6c9d4bba019a1107c"
FILE = "https://upload.wikimedia.org/wikipedia/commons/3/3e/USA_09562_Boston.jpg"
PAGE = "https://commons.wikimedia.org/wiki/File:USA_09562_Boston.jpg"
HEADER = ["timestamp", "original", "digest"]


def commons_hit(status="VERIFIED_MATCH"):
    return SourceHit(url=PAGE, title=None, provider="wikimedia_commons",
                     verification_status=status, verification={},
                     citation={"file_url": FILE, "original_sha1": SHA1})


def cdx(captures: dict[str, list[list[str]]]):
    """``captures[url]`` rows; honours the digest filter and limit=1 / -1."""
    calls = []

    def handler(request):
        params = request.url.params
        calls.append(dict(params))
        rows = captures.get(params["url"], [])
        for f in params.get_list("filter"):
            if f.startswith("digest:"):
                rows = [r for r in rows if r[2] == f.removeprefix("digest:")]
        limit = int(params["limit"])
        rows = rows[:limit] if limit > 0 else rows[limit:]
        return httpx.Response(200, json=[HEADER, *rows] if rows else [])

    return handler, calls


def run(handler, hits):
    source = WaybackSource(enabled=True, transport=httpx.MockTransport(handler))
    return source.run(SourceQuery("x.jpg", {}, prior_hits=hits), db=None)


def test_digest_is_the_base32_sha1():
    assert sha1_to_cdx_digest(SHA1) == "QPCSWQMOEBX7NIU7LOTMTVF3UAM2CED4"


def test_an_archived_copy_with_the_same_bytes_is_verified():
    digest = sha1_to_cdx_digest(SHA1)
    handler, _ = cdx({
        FILE: [["20120626225734", "http://upload.wikimedia.org/x.jpg", digest]],
        PAGE: [["20130902235928", "http://commons.wikimedia.org/wiki/File:x", "AAAA"],
               ["20240101000000", "http://commons.wikimedia.org/wiki/File:x", "BBBB"]],
    })
    result = run(handler, [commons_hit()])
    assert result.status == SourceStatus.COMPLETED
    file_hit, page_hit = result.hits
    assert file_hit.verification_status == "VERIFIED_MATCH"
    assert file_hit.published_at == datetime(2012, 6, 26, 22, 57, 34, tzinfo=timezone.utc)
    assert file_hit.url == "https://web.archive.org/web/20120626225734/http://upload.wikimedia.org/x.jpg"
    assert file_hit.citation["found_via"] == "wikimedia_commons"
    assert page_hit.verification_status == "SIMILAR_UNVERIFIED"
    assert page_hit.verification["method"] == "page_capture"
    assert page_hit.verification["latest_capture"] == "20240101000000"


def test_a_capture_with_other_bytes_is_not_verified_and_says_so():
    handler, _ = cdx({FILE: [["20100101000000", "http://upload.wikimedia.org/x.jpg", "ZZZZ"]]})
    [hit] = [h for h in run(handler, [commons_hit()]).hits if "upload" in h.url]
    assert hit.verification_status == "SIMILAR_UNVERIFIED"
    assert "was replaced" in hit.verification["reasons"][0]


def test_a_page_verified_by_two_connectors_is_looked_up_once():
    handler, calls = cdx({})
    url_hit = SourceHit(url=PAGE, title=None, provider="url_verification",
                        verification_status="VERIFIED_MATCH", verification={}, citation={})
    run(handler, [commons_hit(), url_hit])
    assert [c["url"] for c in calls].count(PAGE) == 2  # earliest + latest, once


def test_only_verified_hits_are_looked_up():
    handler, calls = cdx({})
    result = run(handler, [commons_hit(status="SIMILAR_UNVERIFIED")])
    assert result.status == SourceStatus.COMPLETED and result.hits == [] and calls == []


def test_disabled_is_unavailable():
    source = WaybackSource(enabled=False)
    assert source.run(SourceQuery("x.jpg", {}), db=None).status == SourceStatus.UNAVAILABLE


def test_an_archive_outage_is_a_failed_search():
    handler = lambda request: httpx.Response(503)
    assert run(handler, [commons_hit()]).status == SourceStatus.FAILED


def test_a_verified_capture_becomes_the_earliest_known_appearance():
    digest = sha1_to_cdx_digest(SHA1)
    handler, _ = cdx({FILE: [["20120626225734", "http://upload.wikimedia.org/x.jpg", digest]]})
    [archived] = [h for h in run(handler, [commons_hit()]).hits if h.verification_status == "VERIFIED_MATCH"]
    as_record = SimpleNamespace(
        id="s2", url=archived.url, provider="wayback", domain="web.archive.org",
        verification_status=archived.verification_status,
        source_timestamp=archived.published_at, citation=archived.citation,
    )
    commons = SimpleNamespace(
        id="s1", url=PAGE, provider="wikimedia_commons", domain="commons.wikimedia.org",
        verification_status="VERIFIED_MATCH", source_timestamp=datetime(2013, 1, 1),
        citation={"uploader": "Lucag"},
    )
    earliest = build_timeline([commons, as_record])["earliest_known_appearance"]
    assert earliest["type"] == "ARCHIVED"
    assert earliest["citation"]["url"].startswith("https://web.archive.org/web/2012")
