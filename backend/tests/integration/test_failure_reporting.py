"""
Camera photos upload and analyse; failures say what failed without leaking.

Three things the end-to-end run found broken (docs/E2E_TEST_REPORT.md, C1/C2):
  * a camera JPEG's EXIF rationals failed the metadata insert, and the session
    was left in a failed transaction, so the upload returned 500 and the
    analysis FAILED;
  * a stage failure's text reached the client with the absolute storage path;
  * a FAILED analysis stored, and returned, the raw database error with its SQL
    statement and bound parameters.
"""
import pytest
from sqlalchemy.exc import OperationalError

from app.api.v1.endpoints import analysis as analysis_endpoint
from app.core.safe_errors import DATABASE_ERROR
from app.db import repository as repo
from app.db.database import SessionLocal
from app.db.models import MediaMetadataRecord
from app.main import app  # noqa: F401  -- registers the routes the client uses
from tests.integration.test_ai_generation_branch import client
from tests.integration.test_investigation_agent import seed_analysis
from tests.unit.test_metadata_exif import camera_jpeg

RAW_DB_ERROR = (
    "This Session's transaction has been rolled back due to a previous exception during flush. "
    "Original exception was: (builtins.TypeError) Object of type IFDRational is not JSON serializable\n"
    "[SQL: INSERT INTO media_metadata (id, media_id, exif_data) VALUES (?, ?, ?)]\n"
    "[parameters: [{'raw_metadata': {'file': 'D:\\\\projects\\\\AIMD-2\\\\backend\\\\uploads\\\\x.jpg'}}]]"
)


def _upload(data: bytes, name: str = "camera.jpg") -> str:
    r = client.post("/api/v1/media/upload", files={"file": (name, data, "image/jpeg")})
    assert r.status_code == 200, r.text
    return r.json()["media_id"]


def _poisoning_upsert(db, media_id, data):
    """A metadata write whose flush fails the way IFDRational did: in the driver."""
    db.add(MediaMetadataRecord(media_id=media_id, exif_data={"ExposureTime": object()}))
    db.flush()


def test_a_camera_jpeg_uploads_and_analyses_with_its_exif():
    media_id = _upload(camera_jpeg((160, 120)))

    with SessionLocal() as db:
        exif = repo.get_media(db, media_id).metadata_record.exif_data
    assert exif["Make"] == "Canon" and exif["ExposureTime"] == pytest.approx(0.004)

    r = client.post("/api/v1/analysis/", json={"media_id": media_id})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("COMPLETED", "PARTIAL")
    assert "metadata:" not in (body["message"] or "")


def test_a_failed_metadata_write_does_not_fail_the_upload(monkeypatch):
    monkeypatch.setattr(repo, "upsert_media_metadata", _poisoning_upsert)
    r = client.post("/api/v1/media/upload", files={"file": ("camera.jpg", camera_jpeg(), "image/jpeg")})
    # The session is rolled back, so the audit event after it is written.
    assert r.status_code == 200, r.text


def test_a_failed_stage_does_not_poison_the_rest_of_the_run(monkeypatch):
    media_id = _upload(camera_jpeg((160, 120)))
    monkeypatch.setattr(repo, "upsert_media_metadata", _poisoning_upsert)

    body = client.post("/api/v1/analysis/", json={"media_id": media_id}).json()
    assert body["status"] == "PARTIAL"  # not FAILED: later stages still ran
    assert f"metadata: {DATABASE_ERROR}" in body["message"]
    assert "INSERT" not in body["message"]

    detail = client.get(f"/api/v1/analysis/{body['analysis_id']}").json()
    assert detail["detections"]


def test_stage_failures_name_no_storage_path(monkeypatch):
    media_id = _upload(camera_jpeg())

    def unreadable(path):
        raise ValueError(f"Unable to read image: {path}")

    monkeypatch.setattr(analysis_endpoint._pipeline._metadata_extractor, "extract", unreadable)
    message = client.post("/api/v1/analysis/", json={"media_id": media_id}).json()["message"]
    assert "metadata: Unable to read image: <file>" in message
    assert "uploads" not in message and media_id not in message


def test_a_failed_run_stores_a_safe_reason(monkeypatch):
    media_id = _upload(camera_jpeg())

    def broken(analysis_id, db):
        raise OperationalError("UPDATE analyses SET status=?", ("x",), Exception("disk I/O error"))

    monkeypatch.setattr(analysis_endpoint._pipeline, "_compute_final_assessment", broken)
    body = client.post("/api/v1/analysis/", json={"media_id": media_id}).json()
    assert body["status"] == "FAILED"

    detail = client.get(f"/api/v1/analysis/{body['analysis_id']}").json()
    assert detail["status"] == "FAILED"
    assert detail["error_message"] == DATABASE_ERROR


def test_raw_errors_stored_before_the_fix_are_scrubbed_on_read():
    analysis_id = seed_analysis(status="FAILED")
    with SessionLocal() as db:
        repo.update_analysis_status(db, analysis_id, status="FAILED", error_message=RAW_DB_ERROR)

    shown = client.get(f"/api/v1/analysis/{analysis_id}").json()["error_message"]
    assert shown == DATABASE_ERROR
    assert "INSERT" not in shown and "uploads" not in shown
