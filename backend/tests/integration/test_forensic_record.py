"""
Observations, findings and the assessment are persisted as linked entities.

Guards:
  * a detector that did not run is a coverage gap, never a negative observation
  * a finding of anomaly links the completed observations that found nothing
    as CONTRADICTS -- contradicting evidence is recorded, not discarded
  * an inconclusive analysis states no finding
  * a finding cannot cite another analysis's observations
  * cases, uploads and analyses leave a verifiable audit trail
"""
import io
import uuid

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.core.auth import User, get_current_user
from app.db import repository as repo
from app.db.database import SessionLocal
from app.main import app
from tests.conftest import TEST_USERNAME
from app.services.audit.audit_log import events_for, verify_chain
from app.services.orchestration.analysis_pipeline import AnalysisPipeline

client = TestClient(app)


@pytest.fixture
def db():
    with SessionLocal() as session:
        yield session


def _analysis(db) -> str:
    media_id, analysis_id = str(uuid.uuid4()), str(uuid.uuid4())
    repo.create_media(db, {
        "id": media_id, "original_filename": "x.jpg", "stored_filename": "x.jpg",
        "file_path": "x.jpg", "content_type": "image/jpeg", "media_type": "image",
        "size_bytes": 1, "sha256": "0" * 64,
    })
    repo.create_analysis(db, {"id": analysis_id, "media_id": media_id, "status": "PROCESSING"})
    return analysis_id


def _detection(db, analysis_id, detector, assessment, *, availability="AVAILABLE",
               execution="COMPLETED", regions=()):
    return repo.create_detection(db, {
        "analysis_id": analysis_id, "detector": detector, "media_type": "image",
        "manipulation_type": "test", "availability": availability,
        "execution_status": execution, "assessment": assessment, "confidence": 0.5,
        "score_semantics": "HEURISTIC_SEVERITY",
        "regions": [{"region_type": "r", "confidence": 0.5, "bbox": list(b)} for b in regions],
    }).id


def _record(db, analysis_id):
    failures = []
    AnalysisPipeline.__new__(AnalysisPipeline)._run_forensic_record(analysis_id, db, failures)
    assert failures == []


def test_anomaly_finding_links_support_and_contradiction(db):
    analysis_id = _analysis(db)
    compression = _detection(db, analysis_id, "ForensicImageDetector", "SUSPICIOUS",
                             regions=[(0, 0, 4, 4)])
    geometric = _detection(db, analysis_id, "CopyMoveDetector", "SUSPICIOUS")
    face = _detection(db, analysis_id, "FaceRegionAnalyzer", "NO_SIGNIFICANT_ANOMALIES_DETECTED")
    absent = _detection(db, analysis_id, "TruForDetector", "UNAVAILABLE",
                        availability="UNAVAILABLE", execution="SKIPPED")

    _record(db, analysis_id)

    observations = {o.detection_id: o for o in repo.get_observations_for_analysis(db, analysis_id)}
    assert set(observations) == {compression, geometric, face}, (
        "a detector that never ran produced an observation"
    )
    assert observations[face].polarity == "NO_ANOMALY_OBSERVED"
    assert observations[compression].localised is True

    [finding] = repo.get_findings_for_analysis(db, analysis_id)
    relations = {link.observation_id: link.relation for link in finding.links}
    assert relations == {
        observations[compression].id: "SUPPORTS",
        observations[geometric].id: "SUPPORTS",
        observations[face].id: "CONTRADICTS",
    }

    assessment = repo.get_assessment_for_analysis(db, analysis_id)
    assert assessment.category == "LIKELY_MANIPULATED"
    assert assessment.rule == "multi_family_localised"
    assert assessment.coverage_gaps == [absent]


def test_a_failed_detector_is_a_gap_not_a_negative(db):
    analysis_id = _analysis(db)
    _detection(db, analysis_id, "ForensicImageDetector", "NO_SIGNIFICANT_ANOMALIES_DETECTED")
    failed = _detection(db, analysis_id, "CopyMoveDetector", "INCONCLUSIVE", execution="FAILED")

    _record(db, analysis_id)

    polarities = [o.polarity for o in repo.get_observations_for_analysis(db, analysis_id)]
    assert polarities == ["NO_ANOMALY_OBSERVED"]
    assessment = repo.get_assessment_for_analysis(db, analysis_id)
    assert assessment.category == "INCONCLUSIVE"
    assert assessment.rule == "incomplete_coverage"
    assert assessment.coverage_gaps == [failed]
    # Inconclusive: nothing to state as a finding.
    assert repo.get_findings_for_analysis(db, analysis_id) == []


def test_full_coverage_finding_disclaims_authenticity(db):
    analysis_id = _analysis(db)
    _detection(db, analysis_id, "ForensicImageDetector", "NO_SIGNIFICANT_ANOMALIES_DETECTED")
    _detection(db, analysis_id, "CopyMoveDetector", "NO_SIGNIFICANT_ANOMALIES_DETECTED")

    _record(db, analysis_id)

    [finding] = repo.get_findings_for_analysis(db, analysis_id)
    assert "not a determination of authenticity" in finding.statement
    assert {link.relation for link in finding.links} == {"SUPPORTS"}
    assert len(finding.links) == 2


def test_object_detection_is_never_linked_to_a_finding(db):
    analysis_id = _analysis(db)
    _detection(db, analysis_id, "ForensicImageDetector", "SUSPICIOUS")
    yolo = _detection(db, analysis_id, "YOLOv8ObjectDetector", "NO_SIGNIFICANT_ANOMALIES_DETECTED")

    _record(db, analysis_id)

    yolo_obs = next(o for o in repo.get_observations_for_analysis(db, analysis_id)
                    if o.detection_id == yolo)
    [finding] = repo.get_findings_for_analysis(db, analysis_id)
    assert yolo_obs.id not in {link.observation_id for link in finding.links}


def test_a_finding_cannot_cite_another_analysis(db):
    first, second = _analysis(db), _analysis(db)
    _detection(db, first, "ForensicImageDetector", "SUSPICIOUS")
    _record(db, first)
    foreign = repo.get_observations_for_analysis(db, first)[0].id

    with pytest.raises(ValueError, match="do not belong"):
        repo.create_finding(db, second, "borrowed evidence", [(foreign, "SUPPORTS")])


def test_unknown_finding_relation_is_refused(db):
    analysis_id = _analysis(db)
    with pytest.raises(ValueError, match="relation"):
        repo.create_finding(db, analysis_id, "x", [("any", "PROVES")])


# ── API ───────────────────────────────────────────────────────────────────────

def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color="teal").save(buf, format="PNG")
    return buf.getvalue()


def test_case_upload_analysis_flow_is_recorded_and_audited():
    case = client.post("/api/v1/cases/", json={"title": "Exhibit review"})
    assert case.status_code == 201
    case_id = case.json()["case_id"]
    assert case.json()["created_by"] == TEST_USERNAME

    upload = client.post(
        "/api/v1/media/upload",
        files={"file": ("exhibit.png", _png(), "image/png")},
        data={"case_id": case_id},
    )
    assert upload.status_code == 200, upload.text
    media_id = upload.json()["media_id"]

    detail = client.get(f"/api/v1/cases/{case_id}").json()
    assert [m["media_id"] for m in detail["media"]] == [media_id]
    assert detail["media"][0]["uploaded_by"] == TEST_USERNAME

    run = client.post("/api/v1/analysis/", json={"media_id": media_id})
    assert run.status_code == 200, run.text
    analysis_id = run.json()["analysis_id"]

    body = client.get(f"/api/v1/analysis/{analysis_id}").json()
    assert body["run"]["requested_by"] == TEST_USERNAME
    assert body["run"]["software_version"]
    assert len(body["run"]["detector_set_version"]) == 64
    assert body["run"]["duration_ms"] >= 0
    assert "google_vision_api_key" not in body["run"]["config_snapshot"]
    assert body["assessment_detail"]["category"] == body["assessment"]
    for d in body["detections"]:
        assert d["availability"] in ("AVAILABLE", "UNAVAILABLE")
        assert d["execution_status"] in ("COMPLETED", "INCONCLUSIVE", "FAILED", "SKIPPED")
    observation_ids = {o["id"] for o in body["observations"]}
    for f in body["findings"]:
        assert set(f["supporting_observation_ids"]) <= observation_ids
        assert set(f["contradicting_observation_ids"]) <= observation_ids

    # What the frontend reads to place evidence and its latest result.
    assert client.get(f"/api/v1/media/{media_id}").json()["case_id"] == case_id
    listed = client.get("/api/v1/cases/").json()["items"]
    assert next(c for c in listed if c["case_id"] == case_id)["evidence_count"] == 1
    only = client.get(f"/api/v1/analysis/?media_id={media_id}").json()["items"]
    assert [a["analysis_id"] for a in only] == [analysis_id]

    with SessionLocal() as session:
        actions = [e.action for e in events_for(session, case_id=case_id)]
        assert actions == [
            "CASE_CREATED", "EVIDENCE_INGESTED", "ANALYSIS_STARTED", "ANALYSIS_FINISHED"
        ]
        ingested = events_for(session, media_id=media_id)[0]
        assert ingested.details["sha256"] == upload.json()["sha256"]
        assert verify_chain(session).intact


def test_upload_to_an_unknown_case_is_refused():
    response = client.post(
        "/api/v1/media/upload",
        files={"file": ("x.png", _png(), "image/png")},
        data={"case_id": "no-such-case"},
    )
    assert response.status_code == 404


def test_chain_verification_is_admin_only():
    assert client.get("/api/v1/audit/verify").status_code == 403

    app.dependency_overrides[get_current_user] = lambda: User(
        username="pytest_admin", role="admin", is_active=True
    )
    try:
        response = client.get("/api/v1/audit/verify")
    finally:
        app.dependency_overrides.pop(get_current_user, None)
    assert response.status_code == 200
    assert response.json()["intact"] is True
