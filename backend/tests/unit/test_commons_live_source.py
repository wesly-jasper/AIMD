"""The live Commons lookup: an exact copy by SHA-1, cited for Wayback to follow."""
import hashlib

import httpx

from app.services.sources.commons_live import CommonsSha1Source
from app.services.sources.connectors import SourceQuery, SourceStatus

DATA = b"\xff\xd8\xff\xe0 a photograph's bytes"
SHA1 = hashlib.sha1(DATA).hexdigest()


def _query(tmp_path) -> SourceQuery:
    path = tmp_path / "upload.jpg"
    path.write_bytes(DATA)
    return SourceQuery(file_path=str(path), fingerprints={})


def _commons(images, seen):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"query": {"allimages": images}})
    return httpx.MockTransport(handler)


def test_an_exact_copy_on_commons_is_a_verified_match(tmp_path):
    seen = []
    image = {
        "name": "Fishing_boat.jpg", "title": "File:Fishing boat.jpg", "sha1": SHA1,
        "timestamp": "2019-05-04T10:20:30Z", "user": "Photographer",
        "url": "https://upload.wikimedia.org/wikipedia/commons/a/ab/Fishing_boat.jpg",
        "descriptionurl": "https://commons.wikimedia.org/wiki/File:Fishing_boat.jpg",
        "size": len(DATA), "mime": "image/jpeg",
    }
    source = CommonsSha1Source(enabled=True, contact="ops@example.org", transport=_commons([image], seen))

    result = source.run(_query(tmp_path), db=None)

    assert result.status == SourceStatus.COMPLETED
    [hit] = result.hits
    assert hit.verification_status == "VERIFIED_MATCH" and hit.similarity == 1.0
    assert hit.url == image["descriptionurl"]
    assert hit.published_at.year == 2019
    assert hit.citation["file_url"] == image["url"] and hit.citation["original_sha1"] == SHA1
    request = seen[0]
    assert request.url.params["aisha1"] == SHA1
    assert "ops@example.org" in request.headers["user-agent"]


def test_a_file_not_on_commons_is_completed_with_no_hits(tmp_path):
    source = CommonsSha1Source(enabled=True, contact="ops@example.org", transport=_commons([], []))
    result = source.run(_query(tmp_path), db=None)
    assert result.status == SourceStatus.COMPLETED and result.hits == []


def test_a_different_hash_in_the_answer_is_not_a_match(tmp_path):
    other = {"name": "x.jpg", "sha1": "0" * 40, "url": "https://upload.wikimedia.org/x.jpg"}
    source = CommonsSha1Source(enabled=True, contact="ops@example.org", transport=_commons([other], []))
    assert source.run(_query(tmp_path), db=None).hits == []


def test_without_a_contact_or_when_disabled_it_does_not_search(tmp_path):
    seen = []
    no_contact = CommonsSha1Source(enabled=True, contact="", transport=_commons([], seen))
    result = no_contact.run(_query(tmp_path), db=None)
    assert result.status == SourceStatus.UNAVAILABLE and "WIKIMEDIA_CONTACT" in result.reason

    disabled = CommonsSha1Source(enabled=False, contact="ops@example.org", transport=_commons([], seen))
    assert disabled.run(_query(tmp_path), db=None).status == SourceStatus.UNAVAILABLE
    assert seen == []


def test_an_error_from_commons_is_failed_not_empty(tmp_path):
    broken = httpx.MockTransport(lambda request: httpx.Response(503))
    source = CommonsSha1Source(enabled=True, contact="ops@example.org", transport=broken)
    assert source.run(_query(tmp_path), db=None).status == SourceStatus.FAILED
