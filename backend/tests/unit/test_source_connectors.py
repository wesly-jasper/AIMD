"""
Source connectors and the Wikimedia corpus source.

The corpus rows are written directly, as ``build_corpus`` would leave them, with
the real pHash and ORB check and a stub embedder (as in the Day 3 tests).
"""
import uuid
from datetime import datetime, timezone

import pytest

from app.db import repository as repo
from app.db.database import SessionLocal
from app.db.models import ReferenceImageRecord
from app.services.fingerprinting.fingerprint import (
    EMBEDDING_ALGORITHM,
    FingerprintService,
    embedding_version,
)
from app.services.similarity.similarity_engine import SIMILAR_UNVERIFIED, VERIFIED_MATCH
from app.services.sources.commons_client import CORPUS
from app.services.sources.connectors import (
    SourceConnector,
    SourceQuery,
    SourceStatus,
)
from app.services.sources.wikimedia_corpus import WikimediaCorpusSource
from tests.unit.test_similarity_verification import (
    ConstantEmbedder,
    SeededEmbedder,
    photo,
    resized_copy,
)


@pytest.fixture
def db():
    with SessionLocal() as session:
        yield session


@pytest.fixture
def isolated_corpus(db):
    """Each test sees only the reference rows it adds."""
    db.query(ReferenceImageRecord).delete()
    db.commit()
    yield
    db.query(ReferenceImageRecord).delete()
    db.commit()


def add_reference(db, path, embedder, title=None) -> ReferenceImageRecord:
    record = repo.upsert_reference_image(db, {
        "corpus": CORPUS,
        "external_id": title or f"File:{uuid.uuid4().hex}.jpg",
        "page_url": "https://commons.wikimedia.org/wiki/File:Example.jpg",
        "file_url": "https://upload.wikimedia.org/wikipedia/commons/a/ab/Example.jpg",
        "download_url": "https://thumb.wikimedia.org/thumb/Example.jpg/1280px-Example.jpg",
        "uploader": "Lucag",
        "uploaded_at": datetime(2007, 12, 10, 0, 2, 15, tzinfo=timezone.utc),
        "licence": "CC BY-SA 2.5",
        "artist": "Luca Galuzzi",
        "original_sha1": "a" * 40,
        "file_path": str(path),
        "sha256": "0" * 64,
        "size_bytes": path.stat().st_size,
        "categories": ["Quality images of buildings"],
        "source_metadata": {"DateTimeOriginal": "2007-07-21"},
    })
    fps = FingerprintService(embedder=embedder).generate(path)
    repo.bulk_create_reference_fingerprints(db, [
        {"reference_id": record.id, "algorithm": r["algorithm"],
         "algorithm_version": r["algorithm_version"], "value": r["value"]}
        for r in fps["_records"]
    ])
    return record


def query_for(path, embedder) -> SourceQuery:
    fps = FingerprintService(embedder=embedder).generate(path)
    return SourceQuery(
        file_path=str(path),
        fingerprints={r["algorithm"]: r["value"] for r in fps["_records"]},
        embedding_version=fps.get(EMBEDDING_ALGORITHM + "_version"),
    )


# ── The connector contract ───────────────────────────────────────────────────

class Broken(SourceConnector):
    name = "broken"

    def unavailable_reason(self, db):
        return None

    def search(self, query, db):
        raise TimeoutError("upstream took too long")


def test_a_connector_that_raises_is_failed_not_empty():
    result = Broken().run(SourceQuery("x.jpg", {}), db=None)
    assert result.status == SourceStatus.FAILED
    assert "TimeoutError" in result.reason and result.hits == []
    assert result.summary()["status"] == "FAILED"


def test_disabled_wikimedia_source_is_unavailable(db):
    result = WikimediaCorpusSource(enabled=False).run(SourceQuery("x.jpg", {}), db)
    assert result.status == SourceStatus.UNAVAILABLE
    assert "disabled" in result.reason


def test_an_empty_corpus_is_unavailable_not_a_negative(db, isolated_corpus):
    result = WikimediaCorpusSource(enabled=True).run(SourceQuery("x.jpg", {}), db)
    assert result.status == SourceStatus.UNAVAILABLE
    assert "empty" in result.reason


# ── Searching the corpus ─────────────────────────────────────────────────────

def test_a_resized_copy_of_a_commons_image_is_a_verified_cited_source(db, isolated_corpus, tmp_path):
    original, copy = tmp_path / "commons.png", tmp_path / "upload.jpg"
    img = photo(11)
    img.save(original)
    resized_copy(img, copy)
    embedder = ConstantEmbedder()
    add_reference(db, original, embedder, title="File:USA 09562 Boston.jpg")

    result = WikimediaCorpusSource(enabled=True).run(query_for(copy, embedder), db)

    assert result.status == SourceStatus.COMPLETED
    [hit] = result.hits
    assert hit.verification_status == VERIFIED_MATCH
    assert hit.url == "https://commons.wikimedia.org/wiki/File:Example.jpg"
    assert hit.title == "File:USA 09562 Boston.jpg"
    assert hit.published_at == datetime(2007, 12, 10, 0, 2, 15, tzinfo=timezone.utc)
    assert hit.citation["uploader"] == "Lucag"
    assert hit.citation["licence"] == "CC BY-SA 2.5"
    assert hit.citation["file_url"].startswith("https://upload.wikimedia.org/")
    assert hit.citation["date_time_original"] == "2007-07-21"
    assert "current file version" in hit.citation["uploaded_at_meaning"]
    assert result.summary()["verified"] == 1


def test_a_lookalike_is_at_most_an_unverified_lead(db, isolated_corpus, tmp_path):
    reference, upload = tmp_path / "commons.png", tmp_path / "upload.png"
    photo(12).save(reference)
    photo(13).save(upload)
    embedder = ConstantEmbedder()  # CLIP says they look alike
    add_reference(db, reference, embedder)

    result = WikimediaCorpusSource(enabled=True).run(query_for(upload, embedder), db)

    assert result.status == SourceStatus.COMPLETED
    assert all(h.verification_status == SIMILAR_UNVERIFIED for h in result.hits)
    assert result.summary()["verified"] == 0


def test_an_unrelated_image_finds_nothing_and_that_is_completed(db, isolated_corpus, tmp_path):
    reference, upload = tmp_path / "commons.png", tmp_path / "upload.png"
    photo(14).save(reference)
    photo(15).save(upload)
    embedder = SeededEmbedder()
    add_reference(db, reference, embedder)

    result = WikimediaCorpusSource(enabled=True).run(query_for(upload, embedder), db)
    assert result.status == SourceStatus.COMPLETED
    assert result.hits == []


def test_embeddings_from_another_model_are_not_compared(db, isolated_corpus, tmp_path):
    reference, upload = tmp_path / "commons.png", tmp_path / "upload.png"
    photo(16).save(reference)
    photo(17).save(upload)
    add_reference(db, reference, ConstantEmbedder())
    query = query_for(upload, ConstantEmbedder())
    query.embedding_version = "other-model/retrieval224/l2"

    result = WikimediaCorpusSource(enabled=True).run(query, db)
    assert all(h.verification["cosine"] is None for h in result.hits)


def test_embedding_version_matches_what_fingerprinting_stores(tmp_path):
    path = tmp_path / "a.png"
    photo(18).save(path)
    fps = FingerprintService(embedder=ConstantEmbedder()).generate(path)
    assert fps[EMBEDDING_ALGORITHM + "_version"] == embedding_version(ConstantEmbedder())
