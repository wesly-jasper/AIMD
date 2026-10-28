"""Verdicts stay ADVISORY until they pass independent evaluation, and every
place a verdict is read says which state it was produced under."""
from app.core.config import settings
from app.services.evidence.report_generator import ReportGenerator
from app.services.evidence.verdict_state import ADVISORY, STATEMENTS, VALIDATED, verdict_state
from app.services.orchestration.analysis_pipeline import AnalysisPipeline


def test_the_default_release_state_is_advisory():
    assert settings.verdict_release_state == ADVISORY
    assert AnalysisPipeline._verdict_policy()["release_state"] == ADVISORY


def test_release_state_is_outside_the_rules_hash(monkeypatch):
    """Validating the rules must not change their identity."""
    before = AnalysisPipeline._verdict_policy()["rules_sha256"]
    monkeypatch.setattr(settings, "verdict_release_state", VALIDATED)
    after = AnalysisPipeline._verdict_policy()
    assert after["rules_sha256"] == before and after["release_state"] == VALIDATED


def test_a_result_keeps_the_state_it_was_recorded_under():
    snapshot = {"verdict_policy": {"release_state": VALIDATED}}
    assert verdict_state(snapshot) == {"state": VALIDATED, "statement": STATEMENTS[VALIDATED]}


def test_results_without_a_recorded_state_are_advisory():
    for snapshot in (None, {}, {"verdict_policy": {}}, {"verdict_policy": {"release_state": "BOGUS"}}):
        assert verdict_state(snapshot)["state"] == ADVISORY
    assert "must not be relied on" in verdict_state(None)["statement"]


def _package(**extra):
    return {
        "case_information": {"analysis_id": "a1"},
        "overall_assessment": "SUSPICIOUS",
        "evidence_strength": "WEAK",
        "detection_evidence": [],
        "run": {"config_snapshot": {}},
        **extra,
    }


def test_the_report_states_the_advisory_state_beside_the_verdict():
    md = ReportGenerator().generate_markdown_report(_package())
    summary = md[md.index("## 1. Executive Summary"):]
    assert summary.index("ADVISORY [UNCERTAINTY]") < summary.index("Forensic Assessment")


def test_the_report_names_the_mask_file_and_its_hash():
    detection = {
        "detector": "TruForDetector", "assessment": "SUSPICIOUS", "confidence": 0.97,
        "regions": [{"type": "forgery_localization", "confidence": 0.9, "bbox": [1, 2, 3, 4]}],
        "metadata": {"artefacts": {"mask": {
            "path": "trufor/x_mask_t050.png", "sha256": "ab" * 32, "areas": 1,
            "masked_fraction": 0.06, "threshold": 0.5, "method": "cut at 0.5",
        }}},
    }
    md = ReportGenerator().generate_markdown_report(_package(detection_evidence=[detection]))
    line = next(l for l in md.splitlines() if "Manipulation mask" in l)
    assert "1 area(s), 6.0% of the image" in line
    assert "trufor/x_mask_t050.png" in line and "ab" * 32 in line
    assert "not proof of editing" in line
