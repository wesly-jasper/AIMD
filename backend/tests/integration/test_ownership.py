"""
One account can never see, use or infer another account's media.

Before the product became self-service every authenticated caller saw every
row. These tests drive two identities through every route that takes an id or
lists rows, and through similarity search -- which compared an upload against
every account's fingerprints, and so could hand one customer another's media id.
"""
import io
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.core.auth import User, get_current_user
from app.core.config import settings
from app.core.security import resolve_evidence_path
from app.db import repository as repo
from app.db.database import SessionLocal
from app.main import app
from app.services.audit.audit_log import events_for, verify_chain

client = TestClient(app)

ALICE = "alice@example.test"
BOB = "bob@example.test"


@contextmanager
def acting_as(username: str):
    previous = app.dependency_overrides.get(get_current_user)
    app.dependency_overrides[get_current_user] = lambda: User(
        username=username, email=username, role="user", is_active=True
    )
    try:
        yield
    finally:
        if previous is None:
            app.dependency_overrides.pop(get_current_user, None)
        else:
            app.dependency_overrides[get_current_user] = previous


def _png(colour="navy") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (48, 48), color=colour).save(buf, format="PNG")
    return buf.getvalue()


def _upload(data: bytes, case_id=None):
    form = {"case_id": case_id} if case_id else {}
    return client.post("/api/v1/media/upload", files={"file": ("x.png", data, "image/png")}, data=form)


@pytest.fixture
def alices_things():
    """A folder, an upload in it and an analysis of it, all Alice's."""
    with acting_as(ALICE):
        folder = client.post("/api/v1/cases/", json={"title": "Alice's folder"}).json()["case_id"]
        media = _upload(_png(), folder).json()["media_id"]
        analysis = client.post("/api/v1/analysis/", json={"media_id": media}).json()["analysis_id"]
    return {"folder": folder, "media": media, "analysis": analysis}


def test_the_owner_sees_their_own(alices_things):
    with acting_as(ALICE):
        assert client.get(f"/api/v1/media/{alices_things['media']}").status_code == 200
        assert client.get(f"/api/v1/analysis/{alices_things['analysis']}").status_code == 200
        assert client.get(f"/api/v1/cases/{alices_things['folder']}").status_code == 200
        ids = [m["media_id"] for m in client.get("/api/v1/media/?limit=200").json()["items"]]
        assert alices_things["media"] in ids


@pytest.mark.parametrize("method,path", [
    ("get", "/api/v1/media/{media}"),
    ("get", "/api/v1/media/{media}/download"),
    ("delete", "/api/v1/media/{media}"),
    ("get", "/api/v1/analysis/{analysis}"),
    ("get", "/api/v1/cases/{folder}"),
])
def test_another_account_gets_404_not_403(alices_things, method, path):
    """404, exactly as for an id that does not exist: a 403 would confirm it does."""
    with acting_as(BOB):
        response = getattr(client, method)(path.format(**alices_things))
    assert response.status_code == 404, response.text
    assert ALICE not in response.text


def test_another_account_cannot_analyse_or_file_into_it(alices_things):
    with acting_as(BOB):
        assert client.post("/api/v1/analysis/", json={"media_id": alices_things["media"]}).status_code == 404
        assert _upload(_png("olive"), alices_things["folder"]).status_code == 404


def test_lists_hold_only_the_callers_rows(alices_things):
    with acting_as(BOB):
        media = client.get("/api/v1/media/?limit=200").json()["items"]
        analyses = client.get("/api/v1/analysis/?limit=200").json()["items"]
        filtered = client.get(f"/api/v1/analysis/?media_id={alices_things['media']}").json()["items"]
        folders = client.get("/api/v1/cases/?limit=200").json()["items"]
        events = client.get("/api/v1/audit/events?limit=500").json()["items"]

    assert all(m["uploaded_by"] == BOB for m in media)
    assert alices_things["analysis"] not in [a["analysis_id"] for a in analyses]
    assert filtered == []
    assert alices_things["folder"] not in [c["case_id"] for c in folders]
    assert all(e["actor"] == BOB for e in events)


def test_similarity_never_matches_another_accounts_media(alices_things):
    """Byte-identical upload: an exact sha256 match exists, but it is Alice's."""
    with SessionLocal() as db:
        alice_bytes = resolve_evidence_path(repo.get_media(db, alices_things["media"]).file_path).read_bytes()

    with acting_as(BOB):
        bobs = _upload(alice_bytes).json()["media_id"]
        analysis = client.post("/api/v1/analysis/", json={"media_id": bobs}).json()["analysis_id"]

    with SessionLocal() as db:
        matches = repo.get_similarity_matches_for_analysis(db, analysis)
        assert alices_things["media"] not in {m.source_media_id for m in matches}
        nodes = repo.get_provenance_for_analysis(db, analysis)
        assert alices_things["media"] not in str(nodes)


def test_similarity_still_matches_within_one_account(alices_things):
    with SessionLocal() as db:
        alice_bytes = resolve_evidence_path(repo.get_media(db, alices_things["media"]).file_path).read_bytes()

    with acting_as(ALICE):
        again = _upload(alice_bytes).json()["media_id"]
        analysis = client.post("/api/v1/analysis/", json={"media_id": again}).json()["analysis_id"]

    with SessionLocal() as db:
        matches = repo.get_similarity_matches_for_analysis(db, analysis)
        assert alices_things["media"] in {m.source_media_id for m in matches}


def test_delete_removes_the_media_its_analyses_and_bytes_but_not_the_record(alices_things):
    with SessionLocal() as db:
        path = resolve_evidence_path(repo.get_media(db, alices_things["media"]).file_path)
    assert path.is_file()

    with acting_as(ALICE):
        assert client.delete(f"/api/v1/media/{alices_things['media']}").status_code == 204
        assert client.get(f"/api/v1/media/{alices_things['media']}").status_code == 404
        assert client.get(f"/api/v1/analysis/{alices_things['analysis']}").status_code == 404

    assert not path.exists()
    with SessionLocal() as db:
        assert repo.get_analysis(db, alices_things["analysis"]) is None
        actions = [e.action for e in events_for(db, media_id=alices_things["media"])]
        assert actions[0] == "EVIDENCE_INGESTED" and actions[-1] == "EVIDENCE_DELETED"
        assert verify_chain(db).intact


def test_the_daily_upload_limit_is_per_account(monkeypatch):
    monkeypatch.setattr(settings, "daily_upload_limit", 2)
    with acting_as("quota@example.test"):
        assert _upload(_png("red")).status_code == 200
        assert _upload(_png("green")).status_code == 200
        refused = _upload(_png("blue"))
    assert refused.status_code == 429
    assert "Daily limit" in refused.json()["error"]["message"]

    # Another account's allowance is untouched.
    with acting_as("other-quota@example.test"):
        assert _upload(_png("red")).status_code == 200


def test_upload_response_does_not_disclose_the_storage_path():
    response = _upload(_png("purple"))
    assert response.status_code == 200
    assert "file_path" not in response.json()
    assert str(settings.upload_dir) not in response.text


def test_usage_reports_the_callers_own_count(monkeypatch):
    monkeypatch.setattr(settings, "daily_upload_limit", 7)
    with acting_as("usage@example.test"):
        assert client.get("/api/v1/media/usage").json() == {"used": 0, "limit": 7, "window_hours": 24}
        _upload(_png("teal"))
        assert client.get("/api/v1/media/usage").json()["used"] == 1


def test_media_list_carries_the_latest_result_and_filters_by_folder(alices_things):
    with acting_as(ALICE):
        listed = client.get(f"/api/v1/media/?case_id={alices_things['folder']}").json()["items"]
    assert [m["media_id"] for m in listed] == [alices_things["media"]]
    assert listed[0]["latest_analysis"]["analysis_id"] == alices_things["analysis"]
