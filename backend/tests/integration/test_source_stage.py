"""
The pipeline's source stage: what it stores from the source connectors.

Connectors are stubs here (their own behaviour is covered in
test_source_connectors); what is under test is that every connector's outcome
is recorded, hits are stored with their verification and citation, and only a
dated VERIFIED_MATCH can become the earliest known occurrence.
"""
import uuid
from datetime import datetime, timezone

import pytest

from app.db import repository as repo
from app.db.database import SessionLocal
from app.services.evidence.evidence_service import EvidenceService
from app.services.orchestration.analysis_pipeline import AnalysisPipeline
from app.services.provenance.source_tracer import SourceTracer
from app.services.similarity.similarity_engine import SIMILAR_UNVERIFIED, VERIFIED_MATCH
from app.services.sources.connectors import SourceConnector, SourceHit


@pytest.fixture
def db():
    with SessionLocal() as session:
        yield session


def _analysis(db) -> tuple[str, str]:
    media_id, analysis_id = str(uuid.uuid4()), str(uuid.uuid4())
    repo.create_media(db, {
        "id": media_id, "original_filename": "x.jpg", "stored_filename": "x.jpg",
        "file_path": "x.jpg", "content_type": "image/jpeg",
        "media_type": "image", "size_bytes": 1, "sha256": "0" * 64,
    })
    repo.create_analysis(db, {"id": analysis_id, "media_id": media_id, "status": "PROCESSING"})
    return media_id, analysis_id


def hit(status, published_at=None, url="https://commons.wikimedia.org/wiki/File:A.jpg"):
    return SourceHit(
        url=url, title="File:A.jpg", provider="stub", verification_status=status,
        verification={"gates": {}}, similarity=0.9, published_at=published_at,
        citation={"uploader": "Lucag"},
    )


class Stub(SourceConnector):
    def __init__(self, name, hits=(), unavailable=None, error=None):
        self.name, self._hits, self._unavailable, self._error = name, list(hits), unavailable, error

    def unavailable_reason(self, db):
        return self._unavailable

    def search(self, query, db):
        if self._error:
            raise self._error
        return self._hits


def run_stage(db, connectors):
    media_id, analysis_id = _analysis(db)
    pipeline = AnalysisPipeline.__new__(AnalysisPipeline)
    pipeline._source_connectors = connectors
    pipeline._source_tracer = SourceTracer()
    failures = []
    pipeline._run_source_tracing("x.jpg", media_id, analysis_id, db, failures)
    assert failures == []
    db.expire_all()
    return analysis_id


def test_every_connector_outcome_is_recorded(db):
    analysis_id = run_stage(db, [
        Stub("off", unavailable="disabled"),
        Stub("broken", error=RuntimeError("boom")),
        Stub("ok", hits=[hit(VERIFIED_MATCH)]),
    ])
    searches = repo.get_analysis(db, analysis_id).source_searches
    assert searches["off"]["status"] == "UNAVAILABLE" and searches["off"]["reason"] == "disabled"
    assert searches["broken"]["status"] == "FAILED"
    assert searches["ok"] | {"duration_ms": None} == {
        "status": "COMPLETED", "reason": None, "hits": 1, "verified": 1, "duration_ms": None,
    }


def test_hits_are_stored_with_verification_and_citation(db):
    when = datetime(2007, 12, 10, 0, 2, 15, tzinfo=timezone.utc)
    analysis_id = run_stage(db, [Stub("ok", hits=[hit(VERIFIED_MATCH, when)])])
    [source] = repo.get_sources_for_analysis(db, analysis_id)
    assert source.verification_status == VERIFIED_MATCH
    assert source.citation == {"uploader": "Lucag"}
    assert source.domain == "commons.wikimedia.org"
    assert source.source_timestamp.replace(tzinfo=timezone.utc) == when
    assert source.is_earliest_known


def test_only_a_dated_verified_match_can_be_earliest_known(db):
    early = datetime(2001, 1, 1, tzinfo=timezone.utc)
    later = datetime(2010, 1, 1, tzinfo=timezone.utc)
    analysis_id = run_stage(db, [Stub("ok", hits=[
        hit(SIMILAR_UNVERIFIED, early, url="https://a.example/1"),  # earlier, but unverified
        hit(VERIFIED_MATCH, later, url="https://a.example/2"),
        hit(VERIFIED_MATCH, None, url="https://a.example/3"),      # verified, undated
    ])])
    earliest = [s.url for s in repo.get_sources_for_analysis(db, analysis_id) if s.is_earliest_known]
    assert earliest == ["https://a.example/2"]


def test_unverified_leads_alone_have_no_earliest_known_occurrence(db):
    analysis_id = run_stage(db, [Stub("ok", hits=[
        hit(SIMILAR_UNVERIFIED, datetime(2001, 1, 1, tzinfo=timezone.utc)),
    ])])
    assert not any(s.is_earliest_known for s in repo.get_sources_for_analysis(db, analysis_id))


def test_limitations_name_the_sources_that_were_not_searched(db):
    analysis_id = run_stage(db, [Stub("off", unavailable="reference corpus is empty")])
    analysis = repo.get_analysis(db, analysis_id)
    limitations = EvidenceService()._collect_limitations([], [], analysis)
    assert any("'off' was not searched" in lim and "corpus is empty" in lim for lim in limitations)
    assert any("No source was searched" in lim for lim in limitations)


def test_a_search_that_found_nothing_is_not_called_unsearched(db):
    analysis_id = run_stage(db, [Stub("ok")])
    analysis = repo.get_analysis(db, analysis_id)
    limitations = EvidenceService()._collect_limitations([], [], analysis)
    assert not any("No source was searched" in lim for lim in limitations)
    assert any("No verified source was found in the indexes searched" in lim for lim in limitations)
