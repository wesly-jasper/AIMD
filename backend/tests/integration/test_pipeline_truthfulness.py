"""
Regression tests for pipeline-level truthfulness bugs.

  * a skipped object detector is not a stage failure (every image was PARTIAL)
  * a FAILED evidential detector is a gap even when labelled INCONCLUSIVE
  * evidence items use the detection's own assessment, not stale metadata
    (every detector was described as INCONCLUSIVE)
  * the report is built after the analysis assessment is stored
    (every report said INCONCLUSIVE)
"""
import uuid
from types import SimpleNamespace

import pytest

from app.db import repository as repo
from app.db.database import SessionLocal
from app.schemas.detection import (
    DetectionResult,
    DetectorAvailability,
    DetectorExecutionStatus,
    ScoreSemantics,
)
from app.services.orchestration import analysis_pipeline as pipeline_module
from app.services.orchestration.analysis_pipeline import AnalysisPipeline


@pytest.fixture
def db():
    with SessionLocal() as session:
        yield session


def _media(db, file_path="x.jpg") -> str:
    media_id = str(uuid.uuid4())
    repo.create_media(db, {
        "id": media_id, "original_filename": "x.jpg", "stored_filename": "x.jpg",
        "file_path": str(file_path), "content_type": "image/jpeg",
        "media_type": "image", "size_bytes": 1, "sha256": "0" * 64,
    })
    return media_id


def _analysis(db) -> str:
    analysis_id = str(uuid.uuid4())
    repo.create_analysis(db, {
        "id": analysis_id, "media_id": _media(db), "status": "PROCESSING",
    })
    return analysis_id


def _skipped(detector: str, availability: DetectorAvailability) -> DetectionResult:
    return DetectionResult(
        detector=detector, media_type="image", manipulation_detected=False,
        confidence=0.0, manipulation_type="unknown", availability=availability,
        execution_status=DetectorExecutionStatus.SKIPPED,
        score_semantics=ScoreSemantics.NOT_APPLICABLE, status_reason="not run",
    )


def _run_detection_with(results, monkeypatch) -> list[str]:
    pipeline = AnalysisPipeline.__new__(AnalysisPipeline)
    router = SimpleNamespace(route=lambda *_a: {"media_type": "image", "detections": results})
    pipeline._detection_container = SimpleNamespace(router=router)
    monkeypatch.setattr(pipeline_module.repo, "create_detection", lambda _db, _data: None)
    failures: list[str] = []
    pipeline._run_detection("x.jpg", "m", "a", "image", [], object(), failures)
    return failures


def test_unavailable_object_detector_is_not_a_stage_failure(monkeypatch):
    failures = _run_detection_with(
        [_skipped("YOLOv8ObjectDetector", DetectorAvailability.UNAVAILABLE)], monkeypatch
    )
    assert failures == []


def test_deliberately_skipped_evidential_detector_is_not_a_stage_failure(monkeypatch):
    failures = _run_detection_with(
        [_skipped("CopyMoveDetector", DetectorAvailability.AVAILABLE)], monkeypatch
    )
    assert failures == []


def test_unavailable_evidential_detector_is_still_a_stage_failure(monkeypatch):
    failures = _run_detection_with(
        [_skipped("TruForDetector", DetectorAvailability.UNAVAILABLE)], monkeypatch
    )
    assert any("TruForDetector incomplete" in f for f in failures)


def _detection(db, analysis_id, detector, assessment, *, execution="COMPLETED",
               availability="AVAILABLE", status_reason=None, metadata=None):
    repo.create_detection(db, {
        "analysis_id": analysis_id, "detector": detector, "media_type": "image",
        "manipulation_type": "test", "availability": availability,
        "execution_status": execution, "assessment": assessment, "confidence": 0.4,
        "score_semantics": "HEURISTIC_SEVERITY", "status_reason": status_reason,
        # Deliberately no "assessment" key: detectors no longer write one here.
        "detector_metadata": metadata or {},
    })


def test_failed_detector_labelled_inconclusive_is_a_gap():
    rows = [
        SimpleNamespace(detector="ForensicImageDetector", assessment="NO_SIGNIFICANT_ANOMALIES_DETECTED",
                        availability="AVAILABLE", execution_status="COMPLETED", regions=[]),
        SimpleNamespace(detector="CopyMoveDetector", assessment="INCONCLUSIVE",
                        availability="AVAILABLE", execution_status="FAILED", regions=[]),
    ]
    decision = AnalysisPipeline._decide(rows)
    assert decision.category == "INCONCLUSIVE"
    assert decision.rule == "incomplete_coverage"


def test_evidence_items_use_the_detection_assessment(db):
    analysis_id = _analysis(db)
    _detection(db, analysis_id, "CopyMoveDetector", "SUSPICIOUS")
    _detection(db, analysis_id, "FaceRegionAnalyzer", "NO_SIGNIFICANT_ANOMALIES_DETECTED")
    _detection(db, analysis_id, "ForensicImageDetector", "INCONCLUSIVE",
               execution="INCONCLUSIVE", status_reason="Uncalibrated heuristic")
    _detection(db, analysis_id, "TruForDetector", "UNAVAILABLE",
               availability="UNAVAILABLE", execution="SKIPPED",
               status_reason="weights not installed")

    failures: list[str] = []
    AnalysisPipeline.__new__(AnalysisPipeline)._run_evidence(analysis_id, db, failures)
    assert failures == []

    by_detector = {
        e.detector: e for e in repo.get_evidence_for_analysis(db, analysis_id)
        if e.detector
    }
    assert by_detector["CopyMoveDetector"].category == "OBSERVATION"
    assert "detected forensic anomalies" in by_detector["CopyMoveDetector"].description
    assert by_detector["FaceRegionAnalyzer"].category == "OBSERVATION"
    assert "no anomalies" in by_detector["FaceRegionAnalyzer"].description
    assert by_detector["ForensicImageDetector"].category == "UNCERTAINTY"
    assert "Uncalibrated heuristic" in by_detector["ForensicImageDetector"].description
    assert by_detector["TruForDetector"].category == "UNCERTAINTY"
    assert "weights not installed" in by_detector["TruForDetector"].description


def test_report_is_built_after_the_assessment_is_stored(db, tmp_path, monkeypatch):
    image = tmp_path / "x.jpg"
    image.write_bytes(b"\xff\xd8\xff")
    media_id = _media(db, image)

    pipeline = AnalysisPipeline.__new__(AnalysisPipeline)
    for stage in ("_run_metadata", "_run_detection", "_run_fingerprinting",
                  "_run_similarity", "_run_source_tracing", "_run_provenance",
                  "_run_evidence", "_run_forensic_record"):
        monkeypatch.setattr(pipeline, stage, lambda *_a, **_k: None)
    monkeypatch.setattr(
        pipeline, "_compute_final_assessment", lambda *_a: ("SUSPICIOUS", "WEAK")
    )
    seen = {}

    def capture_report(analysis_id, session, _failures):
        seen["assessment"] = repo.get_analysis(session, analysis_id).assessment

    monkeypatch.setattr(pipeline, "_run_report", capture_report)

    pipeline.run(media_id, db)

    assert pipeline_module._plain_value(seen["assessment"]) == "SUSPICIOUS"


# ── Evidence package ──────────────────────────────────────────────────────────

from app.services.evidence.evidence_service import EvidenceService  # noqa: E402

_TRUFOR_WARNING = "No learned forgery-localisation model (TruFor)"


def test_evidence_package_has_no_fabricated_case_id(db):
    """Regression: an analysis with no case was given "AIMD-<id prefix>"."""
    package = EvidenceService().build_evidence_package(_analysis(db), db)
    assert package["case_information"]["case_id"] is None


def test_evidence_package_reports_the_real_case(db):
    case = repo.create_case(db, {"title": "Harbour fire footage"})
    analysis_id = _analysis(db)
    media = repo.get_media(db, repo.get_analysis(db, analysis_id).media_id)
    media.case_id = case.id
    db.commit()

    info = EvidenceService().build_evidence_package(analysis_id, db)["case_information"]
    assert info["case_id"] == case.id
    assert info["case_title"] == "Harbour fire footage"


def test_limitations_come_from_the_detection_columns(db):
    """Regression: limitations keyed on metadata["assessment"], which no detector writes."""
    analysis_id = _analysis(db)
    _detection(db, analysis_id, "ForensicImageDetector", "INCONCLUSIVE",
               execution="INCONCLUSIVE", status_reason="Uncalibrated heuristic")

    limitations = EvidenceService().build_evidence_package(analysis_id, db)["limitations"]
    assert any("Uncalibrated heuristic" in lim for lim in limitations)


def test_trufor_warning_only_when_trufor_did_not_run(db):
    """Regression: the TruFor-not-configured warning was appended unconditionally."""
    without = _analysis(db)
    _detection(db, without, "TruForDetector", "UNAVAILABLE",
               availability="UNAVAILABLE", execution="SKIPPED",
               status_reason="weights not installed")
    with_trufor = _analysis(db)
    _detection(db, with_trufor, "TruForDetector", "NO_SIGNIFICANT_ANOMALIES_DETECTED")

    service = EvidenceService()
    assert any(_TRUFOR_WARNING in lim
               for lim in service.build_evidence_package(without, db)["limitations"])
    assert not any(_TRUFOR_WARNING in lim
                   for lim in service.build_evidence_package(with_trufor, db)["limitations"])


def test_tiled_trufor_result_carries_its_limitation(db):
    """A completed TruFor result run in tiles says so, once, in the package."""
    analysis_id = _analysis(db)
    _detection(db, analysis_id, "TruForDetector", "LIKELY_MANIPULATED",
               metadata={"tiling": {"tiled": True}, "limitations": ["computed in tiles"]})

    limitations = EvidenceService().build_evidence_package(analysis_id, db)["limitations"]
    assert limitations.count("TruFor: computed in tiles") == 1
    assert not any(_TRUFOR_WARNING in lim for lim in limitations)
