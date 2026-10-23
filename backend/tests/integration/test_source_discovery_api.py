"""
Day 4 done-when, through the API: uploading a cropped copy of an indexed Commons
image yields a verified external source with a real URL and a provenance
timeline of at least two cited events.

The CLIP backbone is replaced by a stub that embeds everything alike, so the
outcome rests on what must decide it: the ORB geometry and the content after
alignment. The corpus holds the original and two lookalikes.
"""
import uuid
from datetime import datetime, timezone

import pytest

from app.api.v1.endpoints import analysis as analysis_endpoint
from app.db import repository as repo
from app.db.database import SessionLocal
from app.db.models import ReferenceImageRecord
from app.services.fingerprinting.fingerprint import FingerprintService
from app.services.similarity.similarity_engine import VERIFIED_MATCH
from app.services.sources.commons_client import CORPUS
from tests.integration.test_ai_generation_branch import _upload, client
from tests.unit.test_similarity_verification import ConstantEmbedder, photo

PAGE = "https://commons.wikimedia.org/wiki/File:Boston_Back_Bay_2007.jpg"


@pytest.fixture
def corpus(tmp_path):
    """The original and two lookalikes, indexed as build_corpus would."""
    embedder = ConstantEmbedder()
    seed = int(uuid.uuid4().int % 10_000) + 2_000
    original = photo(seed)
    with SessionLocal() as db:
        db.query(ReferenceImageRecord).delete()
        db.commit()
        for i, img in enumerate([original, photo(seed + 1), photo(seed + 2)]):
            path = tmp_path / f"ref{i}.png"
            img.save(path)
            record = repo.upsert_reference_image(db, {
                "corpus": CORPUS,
                "external_id": f"File:Ref {seed} {i}.jpg",
                "page_url": PAGE if i == 0 else f"https://commons.wikimedia.org/wiki/File:Other_{i}.jpg",
                "file_url": f"https://upload.wikimedia.org/wikipedia/commons/a/ab/Ref_{i}.jpg",
                "download_url": f"https://thumb.wikimedia.org/thumb/Ref_{i}.jpg/1280px-Ref_{i}.jpg",
                "uploader": "Lucag",
                "uploaded_at": datetime(2007, 12, 10, 0, 2, 15, tzinfo=timezone.utc),
                "licence": "CC BY-SA 2.5",
                "original_sha1": f"{i}" * 40,
                "file_path": str(path), "sha256": f"{i}" * 64,
                "size_bytes": path.stat().st_size, "categories": ["Test"],
                "source_metadata": {"DateTimeOriginal": "2007-07-21"},
            })
            fps = FingerprintService(embedder=embedder).generate(path)
            repo.bulk_create_reference_fingerprints(db, [
                {"reference_id": record.id, "algorithm": r["algorithm"],
                 "algorithm_version": r["algorithm_version"], "value": r["value"]}
                for r in fps["_records"]
            ])
    yield original
    with SessionLocal() as db:
        db.query(ReferenceImageRecord).delete()
        db.commit()


def test_a_cropped_commons_image_is_traced_to_its_page_with_a_cited_timeline(corpus, monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "similarity_embedding_enabled", False)  # upload: no real CLIP
    monkeypatch.setattr(settings, "source_wikimedia_enabled", True)
    monkeypatch.setattr(analysis_endpoint._pipeline, "_fingerprint_service",
                        FingerprintService(embedder=ConstantEmbedder()))

    w, h = corpus.size
    crop = corpus.crop((w // 5, h // 5, w * 4 // 5, h * 4 // 5))  # 60 % centre crop
    media_id = _upload(crop, "JPEG", "found_online.jpg")
    run = client.post("/api/v1/analysis/", json={"media_id": media_id})
    assert run.status_code == 200, run.text
    body = client.get(f"/api/v1/analysis/{run.json()['analysis_id']}").json()

    assert body["source_searches"]["wikimedia_commons"]["status"] == "COMPLETED"
    verified = [s for s in body["sources"] if s["verification_status"] == VERIFIED_MATCH]
    assert [s["url"] for s in verified] == [PAGE], body["sources"]
    [source] = verified
    assert source["citation"]["uploader"] == "Lucag"
    assert source["verification"]["method"] == "embedding+orb+aligned"
    assert source["is_earliest_known"] is True

    timeline = body["provenance_timeline"]
    cited = [e for e in timeline["events"] if e["citation"].get("url")]
    assert len(cited) >= 2
    assert {"PUBLISHED", "RETRIEVED"} <= {e["type"] for e in cited}
    earliest = timeline["earliest_known_appearance"]
    assert earliest["type"] == "PUBLISHED" and earliest["citation"]["url"] == PAGE
    assert earliest["timestamp"] == "2007-12-10T00:02:15+00:00"


def test_an_empty_corpus_says_the_source_was_not_searched(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "similarity_embedding_enabled", False)
    with SessionLocal() as db:
        db.query(ReferenceImageRecord).delete()
        db.commit()
    media_id = _upload(photo(int(uuid.uuid4().int % 10_000) + 5_000), "PNG", "x.png")
    run = client.post("/api/v1/analysis/", json={"media_id": media_id})
    body = client.get(f"/api/v1/analysis/{run.json()['analysis_id']}").json()

    search = body["source_searches"]["wikimedia_commons"]
    assert search["status"] == "UNAVAILABLE" and "empty" in search["reason"]
    assert body["sources"] == []
    assert body["provenance_timeline"]["earliest_known_appearance"] is None
