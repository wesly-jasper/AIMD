"""
The SSRF guard and the URL verification source.

DNS is a fake resolver and HTTP a MockTransport, so nothing leaves the machine.
The real-network peer check (the address actually connected to) is exercised by
a real transport only; with an injected transport the resolved address is used.
"""
import io

import httpx
import pytest

from app.services.fingerprinting.fingerprint import EMBEDDING_ALGORITHM, FingerprintService
from app.services.similarity.similarity_engine import VERIFIED_MATCH
from app.services.sources.connectors import SourceQuery, SourceStatus
from app.services.sources.url_guard import (
    BlockedURL,
    FetchError,
    GuardedFetcher,
    check_url,
    is_public_address,
)
from app.services.sources.url_verification import UrlVerificationSource, image_links
from tests.unit.test_similarity_verification import ConstantEmbedder, photo

PUBLIC = "93.184.215.14"


def resolver(table):
    def resolve(host, _port):
        return table.get(host, [PUBLIC])
    return resolve


def fetcher(handler, table=None, **kwargs):
    return GuardedFetcher(resolver=resolver(table or {}),
                          transport=httpx.MockTransport(handler), **kwargs)


def jpeg(img, scale=1.0):
    if scale != 1.0:
        img = img.resize((int(img.width * scale), int(img.height * scale)))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=80)
    return buf.getvalue()


# ── The guard ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("url", [
    "http://127.0.0.1/", "http://10.1.2.3/x", "http://192.168.0.1/", "http://169.254.169.254/latest",
    "http://[::1]/", "http://[::ffff:10.0.0.1]/", "http://0.0.0.0/", "http://100.64.0.1/",
])
def test_private_and_special_addresses_are_blocked(url):
    with pytest.raises(BlockedURL):
        check_url(url, resolver({}))


@pytest.mark.parametrize("url, why", [
    ("ftp://example.org/a.jpg", "scheme"),
    ("file:///etc/passwd", "scheme"),
    ("https://user:pw@example.org/", "credentials"),
    ("https://example.org:8443/", "port"),
    ("https:///nohost", "no host"),
])
def test_disallowed_urls_say_why(url, why):
    with pytest.raises(BlockedURL, match=why):
        check_url(url, resolver({}))


def test_a_name_that_also_resolves_privately_is_blocked():
    with pytest.raises(BlockedURL, match="non-public"):
        check_url("https://mixed.example/", resolver({"mixed.example": [PUBLIC, "10.0.0.7"]}))


def test_public_addresses_pass():
    assert is_public_address(PUBLIC) and is_public_address("2606:4700::1111")
    assert check_url("https://example.org/a.jpg", resolver({}))[2] == [PUBLIC]


def test_a_redirect_into_the_private_network_is_blocked():
    def handler(request):
        if request.url.host == "example.org":
            return httpx.Response(302, headers={"location": "http://internal.example/admin"})
        return httpx.Response(200, content=b"secret", headers={"content-type": "text/html"})

    with fetcher(handler, {"internal.example": ["10.0.0.5"]}) as f:
        with pytest.raises(BlockedURL, match="non-public"):
            f.fetch("https://example.org/a")


def test_redirects_are_followed_and_capped():
    hops = iter(range(10))

    def handler(request):
        return httpx.Response(302, headers={"location": f"/hop{next(hops)}"})

    with fetcher(handler, max_redirects=3) as f:
        with pytest.raises(FetchError, match="redirects"):
            f.fetch("https://example.org/start")


def test_size_and_type_are_capped():
    big = lambda request: httpx.Response(200, content=b"x" * 5000, headers={"content-type": "image/jpeg"})
    declared = lambda request: httpx.Response(
        200, content=b"x", headers={"content-type": "image/jpeg", "content-length": "999999999"})
    wrong = lambda request: httpx.Response(200, content=b"{}", headers={"content-type": "application/json"})
    for handler, match in ((big, "exceeds"), (declared, "exceeds"), (wrong, "content type")):
        with fetcher(handler, max_bytes=1000) as f:
            with pytest.raises(FetchError, match=match):
                f.fetch("https://example.org/a.jpg")


def test_a_non_200_is_an_error():
    with fetcher(lambda request: httpx.Response(404)) as f:
        with pytest.raises(FetchError, match="404"):
            f.fetch("https://example.org/a.jpg")


# ── Page parsing ─────────────────────────────────────────────────────────────

def test_declared_images_come_first_and_links_are_resolved():
    html = """<html><head>
      <meta property="og:image" content="/media/hero.jpg?utm_source=feed&amp;w=1200">
      <meta name="twitter:image" content="https://cdn.example/t.jpg">
    </head><body>
      <img src="inline.png"><img src="data:image/png;base64,AAAA"><img data-src="/lazy.jpg">
      <img src="/media/hero.jpg">
    </body></html>"""
    assert image_links(html, "https://news.example/story/1") == [
        "https://news.example/media/hero.jpg?w=1200",  # tracking parameter dropped
        "https://cdn.example/t.jpg",
        "https://news.example/story/inline.png",
        "https://news.example/lazy.jpg",
        "https://news.example/media/hero.jpg",
    ]


# ── The connector ────────────────────────────────────────────────────────────

@pytest.fixture
def upload(tmp_path):
    original = photo(31)
    path = tmp_path / "upload.jpg"
    path.write_bytes(jpeg(original))
    embedder = ConstantEmbedder()
    fps = FingerprintService(embedder=embedder).generate(path)
    query = SourceQuery(
        file_path=str(path),
        fingerprints={r["algorithm"]: r["value"] for r in fps["_records"]},
        embedding_version=fps[EMBEDDING_ALGORITHM + "_version"],
        claimed_url="https://news.example/story/1",
    )
    return original, query


def connector(handler, tmp_path):
    return UrlVerificationSource(
        enabled=True,
        fetcher_factory=lambda: fetcher(handler),
        fingerprinter=FingerprintService(embedder=ConstantEmbedder()),
        artefact_dir=tmp_path / "sources",
    )


def test_a_page_whose_og_image_is_the_upload_is_verified(upload, tmp_path):
    original, query = upload
    page = b'<meta property="og:image" content="/img/hero.jpg"><img src="/img/other.jpg">'

    def handler(request):
        if request.url.path == "/story/1":
            return httpx.Response(200, content=page, headers={"content-type": "text/html"})
        if request.url.path == "/img/hero.jpg":
            return httpx.Response(200, content=jpeg(original, 0.6), headers={"content-type": "image/jpeg"})
        return httpx.Response(200, content=jpeg(photo(32)), headers={"content-type": "image/jpeg"})

    result = connector(handler, tmp_path).run(query, db=None)
    assert result.status == SourceStatus.COMPLETED, result.reason
    [hit] = result.hits
    assert hit.verification_status == VERIFIED_MATCH
    assert hit.url == "https://news.example/story/1"
    assert hit.citation["file_url"] == "https://news.example/img/hero.jpg"
    assert hit.citation["retrieved_at"] and hit.citation["peer_ip"] == PUBLIC
    assert hit.citation["images_compared"] == 2
    assert (tmp_path / "sources" / f"{hit.citation['stored_copy_sha256']}.jpg").is_file()
    assert result.summary()["claimed_url"] == "https://news.example/story/1"


def test_a_direct_image_url_is_compared_as_is(upload, tmp_path):
    original, query = upload
    handler = lambda request: httpx.Response(
        200, content=jpeg(original, 0.5), headers={"content-type": "image/jpeg"})
    result = connector(handler, tmp_path).run(query, db=None)
    assert [h.verification_status for h in result.hits] == [VERIFIED_MATCH]


def test_a_page_without_the_image_is_checked_and_not_supported(upload, tmp_path):
    _, query = upload

    def handler(request):
        if request.url.path == "/story/1":
            return httpx.Response(200, content=b'<img src="/a.jpg">', headers={"content-type": "text/html"})
        return httpx.Response(200, content=jpeg(photo(33)), headers={"content-type": "image/jpeg"})

    result = connector(handler, tmp_path).run(query, db=None)
    assert result.status == SourceStatus.COMPLETED
    assert all(h.verification_status != VERIFIED_MATCH for h in result.hits)


def test_no_url_or_disabled_is_unavailable(upload, tmp_path):
    _, query = upload
    query.claimed_url = None
    assert connector(None, tmp_path).run(query, db=None).status == SourceStatus.UNAVAILABLE
    query.claimed_url = "https://news.example/story/1"
    disabled = UrlVerificationSource(enabled=False)
    assert disabled.run(query, db=None).status == SourceStatus.UNAVAILABLE


def test_a_blocked_url_is_a_failed_search_that_says_why(upload, tmp_path):
    _, query = upload
    query.claimed_url = "http://169.254.169.254/latest/meta-data/"
    result = connector(lambda request: httpx.Response(200), tmp_path).run(query, db=None)
    assert result.status == SourceStatus.FAILED
    assert "BlockedURL" in result.reason
