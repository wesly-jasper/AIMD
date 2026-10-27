"""
Day 5 done-when: a Token Factory conversation calls at least two AIMD tools and
every citation in the stored summary resolves to a tool call in the trace.

The conversation is replayed from ``tests/fixtures/nemotron_investigation_recorded.json``
through a client that stands in for ``openai.OpenAI``; the one live smoke test
at the bottom makes the same request against Nebius Token Factory and is
skipped unless ``NEBIUS_API_KEY`` is exported in the shell running pytest.

Also guarded here:
  * the grounding guard strips and flags a URL, date, number and citation the
    tools never produced;
  * the summary is stored as INFERENCE, the trace with FACT / OBSERVATION labels;
  * no key, a model error or a timeout ends as UNAVAILABLE and the analysis is
    untouched;
  * the endpoints are owner-scoped, refuse a second concurrent run, and are
    rate-limited per user.
"""
import copy
import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from openai.types.chat import ChatCompletion

from app.api.v1.endpoints import analysis as analysis_endpoint
from app.core.config import settings
from app.db import repository as repo
from app.db.database import SessionLocal
from app.db.models import AuditEventRecord, InvestigationRecord
from app.services.agent import nemotron_agent
from app.services.agent.nemotron_agent import NemotronAgent
from tests.conftest import TEST_USERNAME
from tests.integration.test_ai_generation_branch import client

RECORDING = Path(__file__).resolve().parents[1] / "fixtures" / "nemotron_investigation_recorded.json"
SOURCE_URL = "https://example.org/news/photo.jpg"


class RecordedClient:
    """Replays recorded chat.completion responses; keeps every request it got."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))
        if not self._responses:
            raise AssertionError("the agent asked for more completions than were recorded")
        return ChatCompletion.model_validate(self._responses.pop(0))


def _recorded():
    return json.loads(RECORDING.read_text(encoding="utf-8"))["responses"]


@pytest.fixture(autouse=True)
def no_investigations():
    """The hourly quota counts the test user's investigations across tests."""
    with SessionLocal() as db:
        db.query(InvestigationRecord).delete()
        db.commit()
    yield


@pytest.fixture
def agent_slot(monkeypatch):
    """Install an agent on the endpoint; returns a setter."""
    def install(agent):
        monkeypatch.setattr(analysis_endpoint, "_agent", agent)
        return agent
    return install


def seed_analysis(owner: str = TEST_USERNAME, status: str = "COMPLETED") -> str:
    """A finished analysis with known values the recording refers to."""
    with SessionLocal() as db:
        media_id, analysis_id = str(uuid.uuid4()), str(uuid.uuid4())
        repo.create_media(db, {
            "id": media_id, "original_filename": "photo.jpg", "stored_filename": "photo.jpg",
            "file_path": "photo.jpg", "content_type": "image/jpeg", "media_type": "image",
            "size_bytes": 2048, "sha256": "a" * 64, "uploaded_by": owner,
        })
        repo.upsert_media_metadata(db, media_id, {
            "format": "JPEG", "width": 640, "height": 480,
            "exif_data": {"Make": "Canon", "GPSLatitude": "51.5", "DateTimeOriginal": "2019:05:03 18:20:11"},
        })
        repo.create_analysis(db, {
            "id": analysis_id, "media_id": media_id, "status": status,
            "assessment": "INCONCLUSIVE", "evidence_strength": "LIMITED",
        })
        repo.upsert_assessment(db, analysis_id, {
            "category": "INCONCLUSIVE", "evidence_strength": "LIMITED", "rule": "NO_SIGNAL",
            "rationale": "No detector produced a manipulation signal.", "coverage_gaps": [],
            "ai_generation": {"status": "NOT_INDICATED"},
        })
        for detector, availability, execution, confidence in (
            ("trufor", "AVAILABLE", "COMPLETED", 0.87),
            ("ela", "UNAVAILABLE", "SKIPPED", 0.0),
        ):
            repo.create_detection(db, {
                "analysis_id": analysis_id, "detector": detector, "media_type": "image",
                "manipulation_type": "splicing", "availability": availability,
                "execution_status": execution, "assessment": "INCONCLUSIVE",
                "manipulation_detected": False, "confidence": confidence,
                "score_semantics": "RAW_MODEL_SCORE", "detector_metadata": {"secret": "not for the model"},
            })
        repo.create_source(db, {
            "analysis_id": analysis_id, "url": SOURCE_URL, "title": "News photo",
            "domain": "example.org", "provider": "wikimedia_commons",
            "source_timestamp": datetime(2019, 5, 4, 10, 0, tzinfo=timezone.utc),
            "verification_status": "VERIFIED_MATCH", "is_earliest_known": True,
            "citation": {"page_url": SOURCE_URL, "uploader": "Lucag"},
        })
        return analysis_id


def _snapshot(analysis_id: str) -> dict:
    """Everything an investigation must leave alone."""
    body = client.get(f"/api/v1/analysis/{analysis_id}").json()
    return {k: v for k, v in body.items() if k != "investigation"}


# ── The recorded conversation ─────────────────────────────────────────────────

def test_recorded_investigation_calls_tools_and_every_citation_resolves(agent_slot):
    analysis_id = seed_analysis()
    before = _snapshot(analysis_id)
    fake = RecordedClient(_recorded())
    agent_slot(NemotronAgent(client_factory=lambda: fake))

    started = client.post(f"/api/v1/analysis/{analysis_id}/investigate")
    assert started.status_code == 202, started.text
    assert started.json()["status"] == "RUNNING"

    # TestClient runs the background task before returning.
    result = client.get(f"/api/v1/analysis/{analysis_id}/investigation").json()
    assert result["status"] == "COMPLETED", result
    assert result["investigation_id"] == started.json()["investigation_id"]

    names = [c["name"] for c in result["tool_calls"]]
    assert names == ["get_analysis_overview", "list_detections", "get_sources"]
    assert len(set(names)) >= 2
    assert [c["id"] for c in result["tool_calls"]] == ["T1", "T2", "T3"]
    assert [c["category"] for c in result["tool_calls"]] == ["OBSERVATION", "OBSERVATION", "FACT"]
    # The plan as executed: the model asked for the first two together.
    assert [c["step"] for c in result["tool_calls"]] == [1, 1, 2]
    assert all("result" in c and "error" not in c for c in result["tool_calls"])

    # Every citation resolves to a tool call in the trace.
    trace_ids = {c["id"] for c in result["tool_calls"]}
    assert result["citations"]
    for citation in result["citations"]:
        assert citation["tool_call_ids"] and set(citation["tool_call_ids"]) <= trace_ids

    summary = result["summary"]
    assert summary["category"] == "INFERENCE"
    assert "not independent forensic proof" in summary["note"]
    text = summary["text"]
    assert "<think>" not in text
    assert SOURCE_URL in text and "2019-05-04" in text and "0.87" in text
    # The fabricated parts are gone, and each is flagged.
    assert "fabricated.example.com" not in text and "2018-01-01" not in text and "99%" not in text
    assert "T9" not in text
    flagged = {(f["type"], f["value"]) for f in result["flags"]}
    assert ("unsupported_url", "https://fabricated.example.com/original") in flagged
    assert ("unsupported_timestamp", "2018-01-01") in flagged
    assert ("unsupported_number", "99%") in flagged
    assert ("unknown_citation", "T9") in flagged
    assert result["grounded"] is False
    # The closing caveat cites nothing and is flagged, not removed.
    assert any(f["type"] == "uncited" and "interpretation" in f["sentence"] for f in result["flags"])

    # Steps, latency and token usage are recorded.
    assert result["steps"] == 3
    assert result["usage"]["total_tokens"] == 1894 + 2741 + 3262
    assert [c["step"] for c in result["usage"]["calls"]] == [1, 2, 3]
    assert all("latency_ms" in c for c in result["usage"]["calls"])

    # What went over the wire: strict tool specs, our T-ids, the provider's call ids.
    first = fake.requests[0]
    assert first["model"] == settings.nemotron_model
    assert len(first["tools"]) == 8
    assert all(t["function"]["strict"] is True for t in first["tools"])
    assert all(t["function"]["parameters"]["additionalProperties"] is False for t in first["tools"])
    assert first["max_tokens"] == settings.agent_max_tokens
    assert 0 < first["timeout"] <= settings.agent_timeout_seconds
    tool_messages = [m for m in fake.requests[2]["messages"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_messages] == ["call_overview", "call_detections", "call_sources"]
    assert [json.loads(m["content"])["tool_call_id"] for m in tool_messages] == ["T1", "T2", "T3"]
    sent = " ".join(m["content"] for m in tool_messages)
    assert "not for the model" not in sent      # detector_metadata stays home
    assert settings.nebius_api_key in ("", None)  # and no key was ever in play

    assert _snapshot(analysis_id) == before

    with SessionLocal() as db:
        actions = [e.action for e in db.query(AuditEventRecord).filter_by(analysis_id=analysis_id)]
    assert actions[-2:] == ["INVESTIGATION_STARTED", "INVESTIGATION_FINISHED"]


# ── UNAVAILABLE never touches results ─────────────────────────────────────────

def test_without_a_key_the_investigation_is_unavailable_at_once(agent_slot):
    agent_slot(NemotronAgent())  # the real default: no NEBIUS_API_KEY in the suite
    analysis_id = seed_analysis()
    before = _snapshot(analysis_id)

    started = client.post(f"/api/v1/analysis/{analysis_id}/investigate")
    assert started.status_code == 202
    assert started.json()["status"] == "UNAVAILABLE"
    assert "NEBIUS_API_KEY" in started.json()["error_message"]

    result = client.get(f"/api/v1/analysis/{analysis_id}/investigation").json()
    assert result["status"] == "UNAVAILABLE"
    assert result["summary"] is None and result["tool_calls"] == []
    assert result["completed_at"] is not None
    assert _snapshot(analysis_id) == before


class _Broken:
    def __init__(self, exc):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._raise))
        self._exc = exc

    def _raise(self, **kwargs):
        raise self._exc


def test_a_model_error_is_unavailable_and_leaves_the_analysis_alone(agent_slot):
    agent_slot(NemotronAgent(client_factory=lambda: _Broken(ConnectionError("upstream 503"))))
    analysis_id = seed_analysis()
    before = _snapshot(analysis_id)

    client.post(f"/api/v1/analysis/{analysis_id}/investigate")
    result = client.get(f"/api/v1/analysis/{analysis_id}/investigation").json()
    assert result["status"] == "UNAVAILABLE"
    assert "ConnectionError" in result["error_message"]
    assert result["summary"] is None
    assert _snapshot(analysis_id) == before


# ── Endpoint guards ───────────────────────────────────────────────────────────

def test_investigation_endpoints_are_owner_scoped(agent_slot):
    agent_slot(NemotronAgent(client_factory=lambda: RecordedClient(_recorded())))
    theirs = seed_analysis(owner="someone-else@example.invalid")
    assert client.post(f"/api/v1/analysis/{theirs}/investigate").status_code == 404
    assert client.get(f"/api/v1/analysis/{theirs}/investigation").status_code == 404
    with SessionLocal() as db:
        assert db.query(InvestigationRecord).filter_by(analysis_id=theirs).count() == 0


def test_get_before_any_investigation_is_404():
    analysis_id = seed_analysis()
    assert client.get(f"/api/v1/analysis/{analysis_id}/investigation").status_code == 404


def test_an_unfinished_analysis_cannot_be_investigated(agent_slot):
    agent_slot(NemotronAgent(client_factory=lambda: RecordedClient(_recorded())))
    analysis_id = seed_analysis(status="PROCESSING")
    assert client.post(f"/api/v1/analysis/{analysis_id}/investigate").status_code == 409


def test_a_second_concurrent_investigation_is_refused_unless_the_first_is_stale(agent_slot):
    agent_slot(NemotronAgent(client_factory=lambda: RecordedClient(_recorded())))
    analysis_id = seed_analysis()
    with SessionLocal() as db:
        running = repo.create_investigation(db, {"analysis_id": analysis_id, "requested_by": TEST_USERNAME})
    assert client.post(f"/api/v1/analysis/{analysis_id}/investigate").status_code == 409

    with SessionLocal() as db:
        repo.update_investigation(
            db, running.id,
            created_at=datetime.now(timezone.utc) - timedelta(seconds=settings.agent_timeout_seconds * 4),
        )
    assert client.post(f"/api/v1/analysis/{analysis_id}/investigate").status_code == 202
    with SessionLocal() as db:
        lost = repo.get_investigation(db, running.id)
        assert lost.status == "FAILED" and "run lost" in lost.error_message


def test_investigations_are_rate_limited_per_user(agent_slot, monkeypatch):
    agent_slot(NemotronAgent())  # UNAVAILABLE at once, but each attempt still counts
    monkeypatch.setattr(settings, "agent_hourly_limit", 2)
    analysis_id = seed_analysis()
    assert client.post(f"/api/v1/analysis/{analysis_id}/investigate").status_code == 202
    assert client.post(f"/api/v1/analysis/{analysis_id}/investigate").status_code == 202
    refused = client.post(f"/api/v1/analysis/{analysis_id}/investigate")
    assert refused.status_code == 429
    assert "2 investigations per hour" in refused.text

    # Another user's quota is their own.
    with SessionLocal() as db:
        db.query(InvestigationRecord).update({"requested_by": "someone-else@example.invalid"})
        db.commit()
    assert client.post(f"/api/v1/analysis/{analysis_id}/investigate").status_code == 202


def test_a_source_url_in_the_body_must_be_http(agent_slot):
    agent_slot(NemotronAgent(client_factory=lambda: RecordedClient(_recorded())))
    analysis_id = seed_analysis()
    for bad in ("file:///etc/passwd", "javascript:alert(1)", "ftp://example.org/x.jpg"):
        r = client.post(f"/api/v1/analysis/{analysis_id}/investigate", json={"source_url": bad})
        assert r.status_code == 422, bad


# ── Live smoke test ───────────────────────────────────────────────────────────

LIVE_KEY = os.environ.get("AIMD_LIVE_NEBIUS_API_KEY", "")


@pytest.mark.live
@pytest.mark.skipif(not LIVE_KEY, reason="NEBIUS_API_KEY not exported: live Nemotron call not run")
def test_live_token_factory_investigation(monkeypatch):
    """One real request to Nebius Token Factory against the seeded analysis."""
    monkeypatch.setattr(settings, "nebius_api_key", LIVE_KEY)
    analysis_id = seed_analysis()
    with SessionLocal() as db:
        outcome = NemotronAgent().investigate(db, analysis_id, TEST_USERNAME)

    assert outcome.status == nemotron_agent.COMPLETED, outcome.error
    names = {c["name"] for c in outcome.tool_calls}
    assert len(names) >= 2, names
    trace_ids = {c["id"] for c in outcome.tool_calls}
    assert outcome.citations, outcome.summary
    for citation in outcome.citations:
        assert set(citation["tool_call_ids"]) <= trace_ids
    assert not any(f["type"] == "unknown_citation" for f in outcome.flags), outcome.flags
    assert outcome.usage["total_tokens"] > 0


# ── A URL the user supplies is always checked ─────────────────────────────────

def _completion(content: str) -> dict:
    return {
        "id": "chatcmpl-url", "object": "chat.completion", "created": 0, "model": settings.nemotron_model,
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def test_a_supplied_source_url_is_verified_before_the_model_runs(agent_slot):
    """
    The live run never called verify_source_match for the URL the user gave, so
    the investigation said nothing about it. AIMD now makes that call itself, as
    T1 in step 0, and the model sees the result before its first turn.
    """
    from tests.unit.test_investigation_agent_parts import _UrlSource

    url = "https://commons.wikimedia.org/wiki/File:Example.jpg"
    source = _UrlSource()
    fake = RecordedClient([_completion("AIMD fetched the supplied page and found a verified match [T1].")])
    agent_slot(NemotronAgent(client_factory=lambda: fake, url_source_factory=source))
    analysis_id = seed_analysis()

    assert client.post(f"/api/v1/analysis/{analysis_id}/investigate", json={"source_url": url}).status_code == 202
    result = client.get(f"/api/v1/analysis/{analysis_id}/investigation").json()

    assert result["status"] == "COMPLETED", result
    first = result["tool_calls"][0]
    assert (first["id"], first["name"], first["step"]) == ("T1", "verify_source_match", 0)
    assert first["result"]["hits"][0]["verification_status"] == "VERIFIED_MATCH"
    assert source.queries[0].claimed_url == url
    assert result["grounded"] is True and "[T1]" in result["summary"]["text"]

    sent = fake.requests[0]["messages"]
    assert sent[-2]["role"] == "assistant" and sent[-2]["tool_calls"][0]["function"]["name"] == "verify_source_match"
    assert sent[-1]["role"] == "tool" and json.loads(sent[-1]["content"])["tool_call_id"] == "T1"
    assert "tool call T1" in sent[1]["content"]
