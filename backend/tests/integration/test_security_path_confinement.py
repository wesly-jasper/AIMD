"""
Regression tests for SEC-1 (unauthenticated arbitrary file read) and
SEC-2 (fabricated cryptographic hash written into the evidence record).

Both defects were confirmed by live exploitation against the running application
during the Phase 0 audit, not merely inferred from source. These tests re-run the
exploit shape so it cannot return.

Original exploit chain:
  1. POST /api/v1/source/trace  {"file_path": "<any host path>", ...}
       -> persisted the attacker-supplied path verbatim into the media table,
          together with a hardcoded sha256 that was the digest of the empty string
  2. GET  /api/v1/media/{id}/download
       -> served that path via FileResponse

Any file readable by the server process was exfiltratable, including aimd.db and
backend/.env.
"""
import hashlib
import io
import uuid

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image

from app.core.config import settings
from app.core.security import resolve_evidence_path
from app.db import repository as repo
from app.db.database import SessionLocal
from app.main import app
from tests.conftest import TEST_USERNAME

client = TestClient(app)

EMPTY_STRING_SHA256 = hashlib.sha256(b"").hexdigest()


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), color="teal").save(buf, format="PNG")
    return buf.getvalue()


# ── SEC-2: the fabricated hash constant ───────────────────────────────────────

def test_empty_string_digest_constant_is_what_the_audit_claimed():
    """Documents the finding: the hardcoded constant was sha256 of zero bytes."""
    assert EMPTY_STRING_SHA256 == (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )


def test_real_ingestion_records_a_real_digest():
    data = _png_bytes()
    response = client.post(
        "/api/v1/media/upload",
        files={"file": ("real.png", data, "image/png")},
    )
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["sha256"] == hashlib.sha256(data).hexdigest()
    assert body["sha256"] != EMPTY_STRING_SHA256


def test_no_endpoint_fabricates_a_hash_for_unknown_evidence():
    """
    The exploit's first step must now fail closed.

    Previously this created a media row carrying a fabricated digest.
    """
    media_id = f"sec1-{uuid.uuid4()}"

    # The route the exploit used no longer exists at all...
    response = client.post(
        "/api/v1/source/trace",
        json={"media_id": media_id, "file_path": str(settings.database_url)},
    )
    assert response.status_code in (404, 405)

    # ...and the route that does take a media id refuses an unknown one.
    response = client.post("/api/v1/analysis/", json={"media_id": media_id})
    assert response.status_code == 404

    # Nothing may have been persisted under that id.
    db = SessionLocal()
    try:
        assert repo.get_media(db, media_id) is None
    finally:
        db.close()


# ── SEC-1: the full exploit chain ─────────────────────────────────────────────

def test_full_exploit_chain_fails(tmp_path):
    """Re-runs the confirmed two-step exfiltration end to end."""
    canary = tmp_path / "canary.txt"
    canary.write_text("TOP-SECRET-CANARY-VALUE", encoding="utf-8")

    media_id = f"pentest-{uuid.uuid4()}"

    trace = client.post(
        "/api/v1/source/trace",
        json={
            "media_id": media_id,
            "analysis_id": f"pentest-analysis-{uuid.uuid4()}",
            "file_path": str(canary),
        },
    )
    # Step 1 must not succeed in registering the canary as evidence.
    assert trace.status_code in (404, 405)

    # Step 2 must find nothing to serve.
    download = client.get(f"/api/v1/media/{media_id}/download")
    assert download.status_code == 404
    assert "TOP-SECRET-CANARY-VALUE" not in download.text


def test_download_refuses_a_poisoned_stored_path(tmp_path):
    """
    Defence in depth for the download endpoint.

    Even when a media row already contains a path outside managed storage -- for
    example a row written before this fix -- the bytes must not be served.
    """
    canary = tmp_path / "legacy_canary.txt"
    canary.write_text("TOP-SECRET-CANARY-VALUE", encoding="utf-8")

    media_id = f"legacy-poisoned-{uuid.uuid4()}"
    db = SessionLocal()
    try:
        repo.create_media(db, {
            "id": media_id,
            "original_filename": "legacy_canary.txt",
            "stored_filename": "legacy_canary.txt",
            "file_path": str(canary),
            "content_type": "text/plain",
            "media_type": "image",
            "size_bytes": canary.stat().st_size,
            "sha256": hashlib.sha256(canary.read_bytes()).hexdigest(),
            # Owned by the test identity, so a 404 for "not yours" cannot mask
            # the path-confinement refusal this test is about.
            "uploaded_by": TEST_USERNAME,
        })

        response = client.get(f"/api/v1/media/{media_id}/download")
        assert response.status_code == 400
        assert "TOP-SECRET-CANARY-VALUE" not in response.text

        # Metadata must report the bytes as unreachable rather than "STORED".
        meta = client.get(f"/api/v1/media/{media_id}")
        assert meta.status_code == 200
        assert meta.json()["storage_status"] == "BYTES_MISSING"
    finally:
        # Remove only the row this test created.
        media = repo.get_media(db, media_id)
        if media:
            db.delete(media)
            db.commit()
        db.close()


# ── The confinement primitive itself ──────────────────────────────────────────

def test_resolve_evidence_path_accepts_managed_paths():
    managed = settings.upload_dir / "some-file.png"
    resolved = resolve_evidence_path(managed)
    assert resolved == managed.resolve()


@pytest.mark.parametrize(
    "hostile",
    [
        "C:/Windows/win.ini",
        "/etc/passwd",
        "../../../../etc/passwd",
        "..\\..\\..\\..\\Windows\\win.ini",
    ],
)
def test_resolve_evidence_path_rejects_paths_outside_managed_storage(hostile):
    with pytest.raises(HTTPException) as excinfo:
        resolve_evidence_path(hostile)
    assert excinfo.value.status_code == 400


def test_resolve_evidence_path_rejects_traversal_out_of_upload_dir():
    escape = settings.upload_dir / ".." / ".." / "secret.txt"
    with pytest.raises(HTTPException) as excinfo:
        resolve_evidence_path(escape)
    assert excinfo.value.status_code == 400


def test_rejection_message_does_not_leak_the_probed_path():
    """An error must not confirm host filesystem layout to a prober."""
    probe = "C:/Users/Administrator/.ssh/id_rsa"
    with pytest.raises(HTTPException) as excinfo:
        resolve_evidence_path(probe)
    assert "id_rsa" not in str(excinfo.value.detail)
    assert "Administrator" not in str(excinfo.value.detail)
