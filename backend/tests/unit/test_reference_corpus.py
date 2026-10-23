"""
Reference corpus: the Commons client and the indexer, against a fake Commons.

No test reaches the network. ``FakeCommons`` answers the API with pages shaped
like a real imageinfo response (captured 2026-10-23, including the ``utm_*``
parameters Commons now appends to every URL) and serves generated JPEGs as the
renditions, so the real download, PIL check and fingerprinting all run.
"""
import io
import json
import uuid
from datetime import datetime, timezone

import httpx
import numpy as np
import pytest
from PIL import Image

from app.db import repository as repo
from app.db.database import SessionLocal
from app.services.fingerprinting.fingerprint import EMBEDDING_ALGORITHM, FingerprintService
from app.services.sources.commons_client import (
    CORPUS,
    CommonsClient,
    CommonsError,
    category_title,
    clean_url,
    parse_file,
    plain_text,
)
from app.services.sources.reference_corpus import build_corpus, skip_reason

UTM = "?utm_source=commons.wikimedia.org&utm_campaign=imageinfo&utm_content=original"


class StubEmbedder:
    def __init__(self, version_tag="stub/1"):
        self.version_tag = version_tag

    def embed(self, _path):
        v = np.ones(768, dtype=np.float32)
        return v / np.linalg.norm(v)


def jpeg_bytes(seed: int, size=(640, 480)) -> bytes:
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, (size[1] // 16, size[0] // 16, 3), dtype=np.uint8)
    img = Image.fromarray(pixels).resize(size, Image.Resampling.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def page(title: str, *, seed: int, sha1: str | None = None, mime="image/jpeg",
         width=2000, height=1333) -> dict:
    name = title.removeprefix("File:").replace(" ", "_")
    return {
        "pageid": seed,
        "ns": 6,
        "title": title,
        "imageinfo": [{
            "timestamp": "2007-12-10T00:02:15Z",
            "user": "Lucag",
            "size": 1776026,
            "width": width,
            "height": height,
            "thumburl": f"https://thumb.wikimedia.org/thumb/{name}/1280px-{name}{UTM}",
            "url": f"https://upload.wikimedia.org/wikipedia/commons/3/3e/{name}{UTM}",
            "descriptionurl": f"https://commons.wikimedia.org/wiki/File:{name}",
            "sha1": sha1 or f"{seed:040x}",
            "mime": mime,
            "extmetadata": {
                "DateTimeOriginal": {"value": "2007-07-21", "source": "commons-desc-page"},
                "Artist": {"value": 'Luca Galuzzi (<a href="//commons.wikimedia.org/wiki/User:Lucag">'
                                    "Lucag</a>) &amp; friends", "source": "commons-desc-page"},
                "LicenseShortName": {"value": "CC BY-SA 2.5", "source": "commons-desc-page"},
                "LicenseUrl": {"value": "https://creativecommons.org/licenses/by-sa/2.5",
                               "source": "commons-desc-page"},
                "Categories": {"value": "not kept", "source": "commons-categories"},
            },
        }],
    }


class FakeCommons:
    """The API (paged by ``per_batch``) and the rendition host."""

    def __init__(self, categories: dict[str, list[dict]], per_batch=2):
        self.categories = categories
        self.per_batch = per_batch
        self.api_calls: list[dict] = []
        self.downloads: list[str] = []
        self.fail_downloads: set[str] = set()
        self.content_type = "image/jpeg"

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.host == "commons.wikimedia.org":
            params = dict(request.url.params)
            self.api_calls.append(params)
            pages = self.categories[params["gcmtitle"]]
            start = int(params.get("gcmcontinue", 0))
            body = {"batchcomplete": True,
                    "query": {"pages": pages[start:start + self.per_batch]}}
            if start + self.per_batch < len(pages):
                body["continue"] = {"gcmcontinue": str(start + self.per_batch),
                                    "continue": "gcmcontinue||"}
            return httpx.Response(200, json=body)
        self.downloads.append(str(request.url))
        if any(f in str(request.url) for f in self.fail_downloads):
            return httpx.Response(404)
        seed = sum(str(request.url).encode())
        return httpx.Response(200, content=jpeg_bytes(seed),
                              headers={"content-type": self.content_type})

    def client(self, **kwargs) -> CommonsClient:
        kwargs.setdefault("min_interval", 0)
        return CommonsClient("tests@example.invalid",
                             transport=httpx.MockTransport(self.handler), **kwargs)


@pytest.fixture
def db():
    with SessionLocal() as session:
        yield session


def _titles(n: int, tag: str = "") -> list[str]:
    run = uuid.uuid4().hex[:8]
    return [f"File:{tag}{run} building {i}.jpg" for i in range(n)]


def _build(db, fake, categories, tmp_path, embedder=None, **kwargs):
    kwargs.setdefault("per_category", 10)
    kwargs.setdefault("limit", 100)
    embedder = embedder or StubEmbedder()
    with fake.client() as client:
        return build_corpus(
            db, client, categories, root=tmp_path,
            fingerprinter=FingerprintService(embedder=embedder),
            embedding_version=f"{embedder.version_tag}/retrieval224/l2",
            thumb_width=1280, min_side=600, **kwargs,
        )


# ── Parsing ──────────────────────────────────────────────────────────────────

def test_a_real_shaped_page_becomes_a_citation_without_tracking_parameters():
    f = parse_file(page("File:USA 09562 Boston.jpg", seed=7))
    assert f.page_url == "https://commons.wikimedia.org/wiki/File:USA_09562_Boston.jpg"
    assert f.file_url == "https://upload.wikimedia.org/wikipedia/commons/3/3e/USA_09562_Boston.jpg"
    assert "utm_" not in f.download_url and "1280px" in f.download_url
    assert f.uploader == "Lucag"
    assert f.uploaded_at == datetime(2007, 12, 10, 0, 2, 15, tzinfo=timezone.utc)
    assert f.artist == "Luca Galuzzi ( Lucag ) & friends"
    assert f.licence == "CC BY-SA 2.5"
    assert f.metadata["DateTimeOriginal"] == "2007-07-21"
    assert "Categories" not in f.metadata


def test_clean_url_keeps_other_query_parameters():
    assert clean_url("https://x.org/a.jpg?page=2&utm_source=y") == "https://x.org/a.jpg?page=2"


def test_a_page_without_imageinfo_is_not_a_file():
    assert parse_file({"title": "File:A.jpg", "missing": True}) is None
    assert parse_file({"title": "File:A.jpg"}) is None


def test_plain_text_and_category_title():
    assert plain_text("<b>A</b>&nbsp;&amp;\n B") == "A & B"
    assert category_title("Quality_images_of_buildings") == "Category:Quality images of buildings"
    assert category_title("Category:X") == "Category:X"


def test_only_large_raster_images_with_a_hash_are_indexed():
    assert skip_reason(parse_file(page("File:A.jpg", seed=1)), 600) is None
    assert skip_reason(parse_file(page("File:A.svg", seed=1, mime="image/svg+xml")), 600)
    assert skip_reason(parse_file(page("File:A.jpg", seed=1, width=800, height=500)), 600)


# ── Client ───────────────────────────────────────────────────────────────────

def test_a_contact_is_required():
    with pytest.raises(ValueError):
        CommonsClient("  ")


def test_listing_follows_continuations_and_identifies_the_client():
    titles = _titles(5)
    fake = FakeCommons({"Category:Bridges": [page(t, seed=i) for i, t in enumerate(titles)]})
    seen_agents = []
    inner = fake.handler

    def handler(request):
        seen_agents.append(request.headers["user-agent"])
        return inner(request)

    with CommonsClient("ops@example.invalid", min_interval=0,
                       transport=httpx.MockTransport(handler)) as client:
        got = [f.title for f in client.category_files("Bridges", thumb_width=1280)]
    assert got == titles
    assert len(fake.api_calls) == 3
    assert fake.api_calls[0]["gcmtype"] == "file"
    assert all("(ops@example.invalid)" in ua for ua in seen_agents)


def test_rate_limit_is_retried_after_the_servers_delay():
    calls, sleeps = [], []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(429, headers={"retry-after": "28"}, text="slow down")
        return httpx.Response(200, json={"query": {"pages": []}})

    with CommonsClient("t@example.invalid", min_interval=0, sleep=sleeps.append,
                       transport=httpx.MockTransport(handler)) as client:
        assert list(client.category_files("X", thumb_width=1280)) == []
    assert sleeps == [28.0]


def test_a_rate_limit_longer_than_allowed_stops_instead_of_waiting():
    handler = lambda request: httpx.Response(429, headers={"retry-after": "3600"})
    with CommonsClient("t@example.invalid", min_interval=0, sleep=lambda s: None,
                       transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(CommonsError, match="rate-limited"):
            list(client.category_files("X", thumb_width=1280))


def test_an_api_error_or_non_json_reply_is_an_error():
    error = lambda request: httpx.Response(200, json={"error": {"code": "badvalue", "info": "no"}})
    text = lambda request: httpx.Response(200, text="<html>maintenance</html>")
    for handler in (error, text):
        with CommonsClient("t@example.invalid", min_interval=0,
                           transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(CommonsError):
                list(client.category_files("X", thumb_width=1280))


def test_requests_are_paced():
    now, sleeps = [100.0], []
    handler = lambda request: httpx.Response(200, json={"query": {"pages": []}})
    with CommonsClient("t@example.invalid", min_interval=1.0, sleep=sleeps.append,
                       clock=lambda: now[0], transport=httpx.MockTransport(handler)) as client:
        list(client.category_files("A", thumb_width=1280))
        now[0] += 0.25
        list(client.category_files("B", thumb_width=1280))
    assert sleeps == [0.75]


def test_download_refuses_a_non_image_and_an_oversized_file(tmp_path):
    fake = FakeCommons({})
    dest = tmp_path / "x.jpg"
    with fake.client() as client:
        fake.content_type = "text/html"
        with pytest.raises(CommonsError, match="expected an image"):
            client.download("https://thumb.wikimedia.org/a.jpg", dest, max_bytes=10**7)
        fake.content_type = "image/jpeg"
        with pytest.raises(CommonsError, match="exceeds"):
            client.download("https://thumb.wikimedia.org/a.jpg", dest, max_bytes=100)
    assert list(tmp_path.iterdir()) == []


# ── Building the corpus ──────────────────────────────────────────────────────

def test_build_indexes_each_file_with_its_citation_and_fingerprints(db, tmp_path):
    titles = _titles(3)
    fake = FakeCommons({"Category:Buildings": [page(t, seed=i) for i, t in enumerate(titles)]})
    stats = _build(db, fake, ["Buildings"], tmp_path)
    assert (stats.indexed, stats.failed) == (3, 0)

    record = repo.get_reference_image(db, CORPUS, titles[0])
    assert record.page_url.startswith("https://commons.wikimedia.org/wiki/File:")
    assert record.uploader == "Lucag" and record.licence == "CC BY-SA 2.5"
    assert record.uploaded_at is not None
    assert record.categories == ["Buildings"]
    assert record.source_metadata["DateTimeOriginal"] == "2007-07-21"
    assert record.file_path.startswith(str(tmp_path)) and record.sha256 in record.file_path
    assert (record.width, record.height) == (640, 480)
    algorithms = {fp.algorithm: fp.algorithm_version for fp in record.fingerprints}
    assert {"sha256", "phash", "dhash", EMBEDDING_ALGORITHM} <= set(algorithms)
    assert algorithms[EMBEDDING_ALGORITHM] == "stub/1/retrieval224/l2"
    assert not (tmp_path / CORPUS / "incoming").exists() or not any((tmp_path / CORPUS / "incoming").iterdir())


def test_unsuitable_files_are_skipped_and_do_not_count(db, tmp_path):
    titles = _titles(4)
    pages = [page(titles[0], seed=1, mime="image/svg+xml"),
             page(titles[1], seed=2, width=500, height=400),
             page(titles[2], seed=3), page(titles[3], seed=4)]
    fake = FakeCommons({"Category:Mixed": pages})
    stats = _build(db, fake, ["Mixed"], tmp_path, per_category=2)
    assert stats.indexed == 2
    assert sum(stats.skipped.values()) == 2
    assert repo.get_reference_image(db, CORPUS, titles[0]) is None


def test_per_category_and_total_limits(db, tmp_path):
    a, b = _titles(5, "a"), _titles(5, "b")
    fake = FakeCommons({
        "Category:A": [page(t, seed=10 + i) for i, t in enumerate(a)],
        "Category:B": [page(t, seed=20 + i) for i, t in enumerate(b)],
    })
    stats = _build(db, fake, ["A", "B"], tmp_path, per_category=3, limit=4)
    assert stats.indexed == 4
    assert all(repo.get_reference_image(db, CORPUS, t) for t in a[:3])
    assert repo.get_reference_image(db, CORPUS, b[0]) is not None
    assert repo.get_reference_image(db, CORPUS, b[1]) is None


def test_a_rerun_downloads_nothing_already_indexed(db, tmp_path):
    titles = _titles(2)
    fake = FakeCommons({"Category:C": [page(t, seed=30 + i) for i, t in enumerate(titles)]})
    _build(db, fake, ["C"], tmp_path)
    fake.downloads.clear()
    stats = _build(db, fake, ["C"], tmp_path)
    assert (stats.unchanged, stats.indexed) == (2, 0)
    assert fake.downloads == []
    record = repo.get_reference_image(db, CORPUS, titles[0])
    assert len([fp for fp in record.fingerprints if fp.algorithm == "phash"]) == 1


def test_a_new_embedding_model_replaces_only_the_embedding(db, tmp_path):
    titles = _titles(1)
    fake = FakeCommons({"Category:D": [page(titles[0], seed=40)]})
    _build(db, fake, ["D"], tmp_path)
    stats = _build(db, fake, ["D"], tmp_path, embedder=StubEmbedder("stub/2"))
    assert stats.refreshed == 1 and fake.downloads.__len__() == 1
    db.expire_all()
    record = repo.get_reference_image(db, CORPUS, titles[0])
    embeddings = [fp for fp in record.fingerprints if fp.algorithm == EMBEDDING_ALGORITHM]
    assert [fp.algorithm_version for fp in embeddings] == ["stub/2/retrieval224/l2"]
    assert len([fp for fp in record.fingerprints if fp.algorithm == "sha256"]) == 1


def test_a_file_replaced_on_commons_is_reindexed_and_the_old_copy_removed(db, tmp_path):
    title = _titles(1)[0]
    fake = FakeCommons({"Category:E": [page(title, seed=50)]})
    _build(db, fake, ["E"], tmp_path)
    old = repo.get_reference_image(db, CORPUS, title)
    old_path, old_sha = old.file_path, old.sha256

    replaced = page(title, seed=50, sha1="f" * 40)
    replaced["imageinfo"][0]["thumburl"] += "&v=2"  # new pixels behind a new URL
    fake.categories["Category:E"] = [replaced]
    stats = _build(db, fake, ["E"], tmp_path)
    assert stats.indexed == 1
    db.expire_all()
    new = repo.get_reference_image(db, CORPUS, title)
    assert new.id == old.id and new.original_sha1 == "f" * 40
    assert new.sha256 != old_sha
    assert not __import__("pathlib").Path(old_path).exists()
    assert len([fp for fp in new.fingerprints if fp.algorithm == "sha256"]) == 1


def test_a_failed_download_is_counted_and_the_build_goes_on(db, tmp_path):
    titles = _titles(3)
    fake = FakeCommons({"Category:F": [page(t, seed=60 + i) for i, t in enumerate(titles)]})
    fake.fail_downloads.add(titles[1].removeprefix("File:").replace(" ", "_"))
    stats = _build(db, fake, ["F"], tmp_path)
    assert (stats.indexed, stats.failed) == (2, 1)
    assert repo.get_reference_image(db, CORPUS, titles[1]) is None


def test_a_file_in_two_categories_is_stored_once_with_both(db, tmp_path):
    title = _titles(1)[0]
    fake = FakeCommons({"Category:G": [page(title, seed=70)], "Category:H": [page(title, seed=70)]})
    stats = _build(db, fake, ["G", "H"], tmp_path)
    assert (stats.indexed, stats.unchanged) == (1, 1)
    db.expire_all()
    assert repo.get_reference_image(db, CORPUS, title).categories == ["G", "H"]
