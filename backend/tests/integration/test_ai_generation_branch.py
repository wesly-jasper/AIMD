"""
The AI-generation branch is reported beside the manipulation verdict, never in it.

Guards:
  * a UnivFD positive does not move the manipulation category or rule
  * a missing or failed UnivFD is not a coverage gap in the manipulation verdict
  * the AI-generation status is its own record: INDICATED, NOT_INDICATED or
    UNAVAILABLE, never "real"
  * GET /analysis/{id} exposes the branch and the verified similarity matches
"""
import io
import uuid
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.db import repository as repo
from app.db.database import SessionLocal
from app.main import app
from app.schemas.detection import DetectorAvailability
from app.services.orchestration.analysis_pipeline import AnalysisPipeline
from app.services.similarity.similarity_engine import SIMILAR_UNVERIFIED, VERIFIED_MATCH
from tests.integration.test_pipeline_truthfulness import _run_detection_with, _skipped
from tests.unit.test_similarity_verification import photo

client = TestClient(app)

LIMITATIONS = ["global score, no localisation", "uncalibrated"]


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


def _detection(db, analysis_id, detector, assessment, *, detected=False, confidence=0.5,
               availability="AVAILABLE", execution="COMPLETED", metadata=None):
    return repo.create_detection(db, {
        "analysis_id": analysis_id, "detector": detector, "media_type": "image",
        "manipulation_type": "test", "availability": availability,
        "execution_status": execution, "assessment": assessment,
        "manipulation_detected": detected, "confidence": confidence,
        "score_semantics": "RAW_MODEL_SCORE", "detector_metadata": metadata or {},
    }).id


def _univfd(db, analysis_id, score, **kwargs):
    indicated = score >= 0.5
    return _detection(
        db, analysis_id, "UnivFDDetector",
        "LIKELY_AI_GENERATED" if indicated else "NO_SIGNIFICANT_ANOMALIES_DETECTED",
        detected=indicated, confidence=score,
        metadata={"threshold": 0.5, "logit": 2.0 if indicated else -2.0,
                  "branch": "ai_generation", "semantics": "uncalibrated linear probe",
                  "limitations": LIMITATIONS},
        **kwargs,
    )


def _record(db, analysis_id):
    failures = []
    AnalysisPipeline.__new__(AnalysisPipeline)._run_forensic_record(analysis_id, db, failures)
    assert failures == []
    return repo.get_assessment_for_analysis(db, analysis_id)


def _verdict(db, with_univfd=None):
    """The manipulation assessment for one clean forensic detector, plus UnivFD."""
    analysis_id = _analysis(db)
    _detection(db, analysis_id, "ForensicImageDetector", "NO_SIGNIFICANT_ANOMALIES_DETECTED")
    if with_univfd is not None:
        _univfd(db, analysis_id, with_univfd)
    return _record(db, analysis_id)


# ── Independence from the manipulation verdict ───────────────────────────────

def test_ai_generation_positive_does_not_move_the_manipulation_verdict(db):
    without = _verdict(db)
    indicated = _verdict(db, with_univfd=0.93)
    assert (indicated.category, indicated.rule) == (without.category, without.rule)
    assert indicated.category == "NO_SIGNIFICANT_ANOMALIES_DETECTED"
    assert indicated.category != "LIKELY_AI_GENERATED"


def test_ai_generation_positive_alone_is_not_evidence_of_editing(db):
    analysis_id = _analysis(db)
    _univfd(db, analysis_id, 0.93)
    assessment = _record(db, analysis_id)
    assert assessment.category == "INCONCLUSIVE"
    assert assessment.ai_generation["status"] == "AI_GENERATION_INDICATED"


def test_ai_generation_positive_does_not_corroborate_a_manipulation_family(db):
    # One manipulation family plus UnivFD stays single-family: generation and
    # editing are different questions, not two families agreeing.
    analysis_id = _analysis(db)
    _detection(db, analysis_id, "ForensicImageDetector", "SUSPICIOUS", detected=True)
    _univfd(db, analysis_id, 0.93)
    assessment = _record(db, analysis_id)
    assert assessment.rule == "single_family"


def test_indicated_branch_carries_its_score_threshold_and_limitations(db):
    branch = _verdict(db, with_univfd=0.93).ai_generation
    assert branch["status"] == "AI_GENERATION_INDICATED"
    assert branch["detector"] == "UnivFDDetector"
    assert branch["score"] == pytest.approx(0.93)
    assert branch["threshold"] == 0.5
    assert branch["logit"] == 2.0
    assert branch["limitations"] == LIMITATIONS
    assert "uncalibrated" in branch["rationale"]


def test_a_low_score_is_not_indicated_and_never_called_real(db):
    branch = _verdict(db, with_univfd=0.08).ai_generation
    assert branch["status"] == "NOT_INDICATED"
    assert "not a finding that the image was captured by a camera" in branch["rationale"]


def test_no_ai_generation_detector_is_unavailable(db):
    branch = _verdict(db).ai_generation
    assert branch["status"] == "UNAVAILABLE"
    assert branch["detector"] is None


def test_a_video_says_the_check_is_image_only_not_unconfigured():
    """Regression: a video was told no AI-generation detector is configured."""
    video = [SimpleNamespace(detector="VideoTemporalAnalysisEngine", media_type="video")]
    branch = AnalysisPipeline._decide_ai_generation(video)
    assert branch["status"] == "UNAVAILABLE"
    assert branch["reason"] == "image_only"
    assert "not configured" not in branch["rationale"]
    assert "still images" in branch["rationale"]


# ── Coverage ─────────────────────────────────────────────────────────────────

def test_unavailable_univfd_is_not_a_gap_in_the_manipulation_verdict(db):
    analysis_id = _analysis(db)
    _detection(db, analysis_id, "ForensicImageDetector", "NO_SIGNIFICANT_ANOMALIES_DETECTED")
    _detection(db, analysis_id, "UnivFDDetector", "UNAVAILABLE",
               availability="UNAVAILABLE", execution="SKIPPED")
    assessment = _record(db, analysis_id)
    assert assessment.category == "NO_SIGNIFICANT_ANOMALIES_DETECTED"
    assert assessment.coverage_gaps == []
    assert assessment.ai_generation["status"] == "UNAVAILABLE"
    assert assessment.ai_generation["detector"] == "UnivFDDetector"


def test_failed_univfd_is_reported_on_its_own_branch(db):
    analysis_id = _analysis(db)
    _detection(db, analysis_id, "ForensicImageDetector", "NO_SIGNIFICANT_ANOMALIES_DETECTED")
    _detection(db, analysis_id, "UnivFDDetector", "INCONCLUSIVE", execution="FAILED")
    assessment = _record(db, analysis_id)
    assert assessment.coverage_gaps == []
    assert assessment.ai_generation["status"] == "UNAVAILABLE"


def test_unconfigured_univfd_does_not_make_the_run_partial(monkeypatch):
    failures = _run_detection_with(
        [_skipped("UnivFDDetector", DetectorAvailability.UNAVAILABLE)], monkeypatch
    )
    assert failures == []


# ── API ──────────────────────────────────────────────────────────────────────

def _upload(img, fmt, name) -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt, **({"quality": 70} if fmt == "JPEG" else {}))
    response = client.post(
        "/api/v1/media/upload",
        files={"file": (name, buf.getvalue(), f"image/{fmt.lower()}")},
    )
    assert response.status_code == 200, response.text
    return response.json()["media_id"]


def test_api_reports_both_branches_and_the_verified_copy(monkeypatch):
    from app.core.config import settings

    # No CLIP weights in CI: the copy is found by pHash and geometry, and the
    # missing embedding keeps it short of VERIFIED_MATCH, with the reason said.
    monkeypatch.setattr(settings, "similarity_embedding_enabled", False)
    original = photo(int(uuid.uuid4().int % 10_000) + 100)
    original_id = _upload(original, "PNG", "original.png")
    w, h = original.size
    copy_id = _upload(original.resize((w // 2, h // 2)), "JPEG", "copy.jpg")

    run = client.post("/api/v1/analysis/", json={"media_id": copy_id})
    assert run.status_code == 200, run.text
    body = client.get(f"/api/v1/analysis/{run.json()['analysis_id']}").json()

    branch = body["ai_generation"]
    assert branch["status"] in ("AI_GENERATION_INDICATED", "NOT_INDICATED", "UNAVAILABLE")
    assert body["assessment"] != "LIKELY_AI_GENERATED"
    assert body["assessment_detail"]["category"] == body["assessment"]

    [match] = [m for m in body["similarity_matches"]
               if m["source_media_id"] == original_id]
    assert match["verification_status"] in (SIMILAR_UNVERIFIED, VERIFIED_MATCH)
    gates = match["verification"]["gates"]
    assert gates["phash"] is True and gates["geometry"] is True
    assert gates["cosine"] is None
    assert match["verification_status"] == SIMILAR_UNVERIFIED
    assert "no comparable image embedding for one of the images" in (
        match["verification"]["reasons"]
    )


def test_ai_indication_on_an_edited_file_carries_the_overlap_note():
    ai = {"status": "AI_GENERATION_INDICATED", "rationale": "B-Free indicated generation."}
    noted = AnalysisPipeline._with_edit_note(ai, "LIKELY_MANIPULATED")
    assert "same local edit" in noted["edit_overlap_note"]
    assert noted["rationale"].startswith("B-Free indicated generation.")
    assert noted["status"] == "AI_GENERATION_INDICATED"  # the branch's finding is not changed
    assert "edit_overlap_note" not in AnalysisPipeline._with_edit_note(ai, "NO_SIGNIFICANT_ANOMALIES_DETECTED")
    cleared = {"status": "NOT_INDICATED", "rationale": "x"}
    assert AnalysisPipeline._with_edit_note(cleared, "SUSPICIOUS") is cleared
