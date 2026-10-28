"""
The evaluation harness and calibration: splits cannot leak, metrics count
abstentions instead of hiding them, verdicts come from the production decision
rules, and a calibration is applied only to the weights it was fitted for.
"""
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.core.config import settings
from app.schemas.detection import (
    DetectionRegion,
    DetectionResult,
    DetectorAvailability,
    DetectorExecutionStatus,
    ScoreSemantics,
)
from app.services.detection import calibration
from app.services.evaluation.dataset import (
    assign_splits,
    load_manifest,
    split_leakage,
    validate,
    write_manifest,
)
from app.services.evaluation.metrics import (
    binary_metrics,
    compare,
    compute_report,
    localisation_scores,
    wilson,
    MANIPULATION_CLEARED,
    MANIPULATION_FLAGGED,
)
from app.services.evaluation.runner import predict_item


# ── Dataset ───────────────────────────────────────────────────────────────────


def _dataset(tmp_path, n_groups=30):
    (tmp_path / "files").mkdir()
    (tmp_path / "masks").mkdir()
    entries = []
    for g in range(n_groups):
        for label in ("real", "manipulated"):
            name = f"g{g}_{label}.png"
            Image.new("RGB", (8, 8)).save(tmp_path / "files" / name)
            entry = {"file": name, "label": label, "group": f"g{g}"}
            if label == "manipulated":
                entry.update(kind="splice", mask=f"g{g}.png")
                Image.new("L", (8, 8)).save(tmp_path / "masks" / f"g{g}.png")
            entries.append(entry)
    entries.append({"file": "gen.png", "label": "ai_generated", "kind": "Midjourney"})
    Image.new("RGB", (8, 8)).save(tmp_path / "files" / "gen.png")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(entries))
    return manifest


def test_split_is_by_group_deterministic_and_leak_free(tmp_path):
    items = load_manifest(_dataset(tmp_path))
    assign_splits(items, {"val": 0.4, "test": 0.6})
    assert validate(items) == []
    by_group = {}
    for i in items:
        by_group.setdefault(i.group, set()).add(i.split)
    assert all(len(s) == 1 for s in by_group.values())
    assert {i.split for i in items} == {"val", "test"}

    again = assign_splits(load_manifest(tmp_path / "manifest.json"), {"val": 0.4, "test": 0.6})
    assert [i.split for i in again] == [i.split for i in items]


def test_leakage_is_detected(tmp_path):
    items = load_manifest(_dataset(tmp_path))
    items[0].split, items[1].split = "val", "test"  # an original and its edit
    assert split_leakage(items)
    assert any("several splits" in p for p in validate(items))


def test_manifest_round_trip_keeps_paths_and_splits(tmp_path):
    items = assign_splits(load_manifest(_dataset(tmp_path)), {"val": 0.5, "test": 0.5})
    out = tmp_path / "manifest.split.json"
    write_manifest(items, out)
    again = load_manifest(out)
    assert [(i.id, i.split, i.path, i.mask_path) for i in again] == [
        (i.id, i.split, i.path, i.mask_path) for i in items
    ]


def test_branch_targets():
    from app.services.evaluation.dataset import EvalItem

    def item(label, kind=None):
        return EvalItem(id="x", path=Path("x"), label=label, group="x", kind=kind)

    assert item("manipulated").manipulation_target is True
    assert item("ai_generated").manipulation_target is None
    assert item("real").ai_generation_target is False
    # A conventional edit is a hard negative for AI generation...
    assert item("manipulated", "splice").ai_generation_target is False
    # ...generative inpainting is ambiguous and excluded.
    assert item("manipulated", "inpainting (GLIDE)").ai_generation_target is None


# ── Metrics ───────────────────────────────────────────────────────────────────


def test_wilson_interval():
    w = wilson(8, 10)
    assert w["rate"] == 0.8 and w["ci95"][0] < 0.8 < w["ci95"][1]
    assert wilson(0, 0)["rate"] is None


def test_abstentions_are_reported_not_counted_as_correct():
    rows = (
        [{"target": True, "verdict": "SUSPICIOUS", "score": 0.9}] * 4
        + [{"target": True, "verdict": "INCONCLUSIVE", "score": None}] * 6
        + [{"target": False, "verdict": "INCONCLUSIVE", "score": None}] * 10
    )
    m = binary_metrics(rows, MANIPULATION_FLAGGED, MANIPULATION_CLEARED)
    assert m["recall"]["rate"] == 1.0 and m["recall"]["n"] == 4
    assert m["abstained_positive"]["rate"] == 0.6
    # No decided negatives: the FPR is unknown, not a perfect 0%.
    assert m["false_positive_rate"]["rate"] is None
    assert m["abstained_negative"]["rate"] == 1.0
    assert m["balanced_accuracy"] is None


def test_localisation_scores():
    truth = np.zeros((10, 10), bool)
    truth[:5] = True
    pred = np.zeros((10, 10), bool)
    pred[:5, :5] = True
    s = localisation_scores(pred, truth)
    assert s["iou"] == pytest.approx(0.5)
    assert s["f1"] == pytest.approx(2 / 3)


def _record(label, manip_verdict, score, split="test"):
    target = {"real": False, "manipulated": True}[label]
    return {
        "id": f"{label}{score}", "label": label, "split": split, "tags": [], "kind": None,
        "targets": {"manipulation": target, "ai_generation": None},
        "verdicts": {"manipulation": manip_verdict},
        "detectors": {"TruForDetector": {"execution_status": "COMPLETED", "raw_score": score,
                                         "evidential": True, "duration_ms": 10.0}},
    }


def test_compare_flags_a_recall_gain_bought_with_false_positives():
    base = compute_report(
        [_record("manipulated", "SUSPICIOUS", 0.9)] * 40
        + [_record("manipulated", "NO_SIGNIFICANT_ANOMALIES_DETECTED", 0.3)] * 10
        + [_record("real", "NO_SIGNIFICANT_ANOMALIES_DETECTED", 0.1)] * 48
        + [_record("real", "SUSPICIOUS", 0.6)] * 2
    )
    cand = compute_report(
        [_record("manipulated", "SUSPICIOUS", 0.9)] * 50
        + [_record("real", "NO_SIGNIFICANT_ANOMALIES_DETECTED", 0.1)] * 35
        + [_record("real", "SUSPICIOUS", 0.6)] * 15
    )
    result = compare(base, cand)
    assert result["manipulation"]["recall"]["verdict"] == "improved"
    assert result["manipulation"]["false_positive_rate"]["verdict"] == "regressed"
    assert "manipulation.false_positive_rate" in result["regressions"]


def test_a_failed_detector_under_a_cleared_verdict_is_a_silent_failure():
    rec = _record("real", "NO_SIGNIFICANT_ANOMALIES_DETECTED", 0.1)
    rec["detectors"]["CopyMoveDetector"] = {"execution_status": "FAILED", "evidential": True}
    assert compute_report([rec])["silent_failures"] == 1


# ── Runner uses the production decision rules ─────────────────────────────────


class _Engine:
    def __init__(self, results):
        self.results = results
        self.detectors = []

    def detect(self, path):
        return {"detections": self.results}


def _result(detector, detected, score, status=DetectorExecutionStatus.COMPLETED, regions=None, **meta):
    return DetectionResult(
        detector=detector, media_type="image", manipulation_detected=detected,
        confidence=score, manipulation_type="x", availability=DetectorAvailability.AVAILABLE,
        execution_status=status, score_semantics=ScoreSemantics.RAW_MODEL_SCORE,
        regions=regions or [], metadata=meta,
    )


def test_runner_verdicts_are_the_pipeline_verdicts(tmp_path, monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "ai_generation_advisory_detectors", [])  # UnivFD is the stub AI detector
    items = load_manifest(_dataset(tmp_path, n_groups=1))
    edited = next(i for i in items if i.label == "manipulated")
    engine = _Engine([
        _result("TruForDetector", True, 0.8,
                regions=[DetectionRegion(type="forgery_localization", confidence=0.8, bbox=[0, 0, 2, 2])]),
        _result("UnivFDDetector", False, 0.1, threshold=0.5),
        _result("CopyMoveDetector", False, 0.0, status=DetectorExecutionStatus.INCONCLUSIVE),
    ])
    rec = predict_item(edited, engine)
    assert "error" not in rec
    assert rec["verdicts"] == {"manipulation": "SUSPICIOUS", "ai_generation": "NOT_INDICATED"}
    assert rec["decision"]["rule"] == "single_family"
    assert rec["detectors"]["TruForDetector"]["raw_score"] == 0.8
    assert rec["targets"] == {"manipulation": True, "ai_generation": False}


def test_runner_records_an_error_instead_of_stopping(tmp_path):
    items = load_manifest(_dataset(tmp_path, n_groups=1))

    class Broken:
        detectors = []

        def detect(self, path):
            raise RuntimeError("boom")

    rec = predict_item(items[0], Broken())
    assert rec["error"].startswith("RuntimeError")


# ── Calibration ───────────────────────────────────────────────────────────────


def _scores(seed=0, n=200):
    rng = np.random.default_rng(seed)
    pos = np.clip(rng.beta(5, 2, n), 0.001, 0.999)
    neg = np.clip(rng.beta(2, 5, n), 0.001, 0.999)
    return list(pos) + list(neg), [True] * n + [False] * n


def test_platt_fit_improves_calibration_and_holds_the_fpr_target():
    scores, targets = _scores()
    cal = calibration.fit("D", "v1", scores, targets, target_fpr=0.05, fitted_on={"split": "val"})
    assert cal.validation["fpr"] <= 0.05
    assert cal.validation["ece_calibrated"] <= cal.validation["ece_raw"] + 1e-9
    assert cal.fitted_on == {"split": "val", "positives": 200, "negatives": 200}
    assert 0 < cal.probability(0.2) < cal.probability(0.8) < 1


def test_threshold_for_fpr():
    negatives = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    t = calibration.threshold_for_fpr(negatives, 0.1)  # one false positive allowed
    assert sum(p >= t for p in negatives) == 1


def test_calibration_applies_only_to_the_version_it_was_fitted_for(tmp_path, monkeypatch):
    scores, targets = _scores()
    cal = calibration.fit("TruForDetector", "2.0.0+w-abc", scores, targets,
                          target_fpr=0.05, fitted_on={"split": "val"})
    path = tmp_path / "calibration.json"
    calibration.save({"TruForDetector": cal}, path)
    monkeypatch.setattr(settings, "calibration_path", path)

    applied = calibration.calibrate_score("TruForDetector", "2.0.0+w-abc", 0.9)
    assert applied["applied"] is True
    assert applied["raw_score"] == 0.9
    assert applied["flagged"] == (applied["probability"] >= cal.threshold)

    other = calibration.calibrate_score("TruForDetector", "2.0.0+w-zzz", 0.9)
    assert other["applied"] is False and "different detector version" in other["reason"]
    assert calibration.calibrate_score("UnivFDDetector", "x", 0.9) is None


def test_trufor_detector_reports_calibrated_probability(tmp_path, monkeypatch):
    from app.services.detection.detectors.image.trufor_detector import TruForDetector

    class Runner:
        version_tag = "trufor-test.w-abc"

        def predict(self, path):
            return {"manipulation_detected": True, "confidence": 0.55, "regions": []}

    detector = TruForDetector(Runner(), artefacts_dir=tmp_path)
    scores, targets = _scores()
    cal = calibration.fit(detector.detector_id, detector.version, scores, targets,
                          target_fpr=0.05, fitted_on={"split": "val"})
    path = tmp_path / "calibration.json"
    calibration.save({detector.detector_id: cal}, path)
    monkeypatch.setattr(settings, "calibration_path", path)

    image = tmp_path / "x.png"
    Image.new("RGB", (8, 8)).save(image)
    result = detector.detect(str(image))

    assert result.score_semantics is ScoreSemantics.CALIBRATED_PROBABILITY
    assert result.confidence == pytest.approx(cal.probability(0.55), abs=1e-6)
    assert result.manipulation_detected == (result.confidence >= cal.threshold)
    assert result.metadata["calibration"]["raw_score"] == 0.55
    assert result.metadata["calibration"]["fitted_on"]["positives"] == 200


def test_ai_branch_reports_disagreeing_detectors_as_conflicting(monkeypatch):
    from types import SimpleNamespace

    from app.core.config import settings
    monkeypatch.setattr(settings, "ai_generation_advisory_detectors", [])

    from app.services.detection.engine import image_detection_engine as engine_module
    from app.services.orchestration.analysis_pipeline import AnalysisPipeline

    def row(name, detected, score):
        return SimpleNamespace(
            detector=name, detector_version="1", media_type="image",
            assessment="LIKELY_AI_GENERATED" if detected else "NO_SIGNIFICANT_ANOMALIES_DETECTED",
            availability="AVAILABLE", execution_status="COMPLETED",
            score_semantics="RAW_MODEL_SCORE", confidence=score,
            manipulation_detected=detected, regions=[], status_reason=None,
            detector_metadata={"threshold": 0.5},
        )

    families = dict(engine_module._SIGNAL_FAMILY, SecondGenDetector="ai_generation")
    import app.services.orchestration.analysis_pipeline as pipeline_module
    original = pipeline_module._SIGNAL_FAMILY
    pipeline_module._SIGNAL_FAMILY = families
    try:
        out = AnalysisPipeline._decide_ai_generation(
            [row("UnivFDDetector", False, 0.1), row("SecondGenDetector", True, 0.9)]
        )
    finally:
        pipeline_module._SIGNAL_FAMILY = original
    assert out["status"] == "CONFLICTING"
    assert out["detector"] == "SecondGenDetector"
    assert {d["detector"]: d["status"] for d in out["detectors"]} == {
        "UnivFDDetector": "NOT_INDICATED", "SecondGenDetector": "AI_GENERATION_INDICATED",
    }


def _ai_row(name, detected, score):
    from types import SimpleNamespace

    return SimpleNamespace(
        detector=name, detector_version="1", media_type="image",
        assessment="LIKELY_AI_GENERATED" if detected else "NO_SIGNIFICANT_ANOMALIES_DETECTED",
        availability="AVAILABLE", execution_status="COMPLETED",
        score_semantics="RAW_MODEL_SCORE", confidence=score,
        manipulation_detected=detected, regions=[], status_reason=None,
        detector_metadata={"threshold": 0.5},
    )


def test_advisory_detector_is_shown_but_does_not_vote(monkeypatch):
    from app.core.config import settings
    from app.services.orchestration.analysis_pipeline import AnalysisPipeline

    monkeypatch.setattr(settings, "ai_generation_advisory_detectors", ["UnivFDDetector"])
    out = AnalysisPipeline._decide_ai_generation(
        [_ai_row("UnivFDDetector", False, 0.1), _ai_row("BFreeDetector", True, 0.9)]
    )
    assert out["status"] == "AI_GENERATION_INDICATED"
    assert out["detector"] == "BFreeDetector"
    assert {d["detector"]: d["voting"] for d in out["detectors"]} == {
        "UnivFDDetector": False, "BFreeDetector": True,
    }
    assert "does not vote" in out["rationale"]

    # The other way round: the advisory detector cannot raise a flag either.
    out = AnalysisPipeline._decide_ai_generation(
        [_ai_row("UnivFDDetector", True, 0.9), _ai_row("BFreeDetector", False, 0.1)]
    )
    assert out["status"] == "NOT_INDICATED"


def test_advisory_detector_does_not_decide_even_when_it_is_the_only_one_scored(monkeypatch):
    from app.core.config import settings
    from app.services.orchestration.analysis_pipeline import AnalysisPipeline

    monkeypatch.setattr(settings, "ai_generation_advisory_detectors", ["UnivFDDetector"])
    out = AnalysisPipeline._decide_ai_generation([_ai_row("UnivFDDetector", True, 0.9)])
    assert out["status"] == "UNAVAILABLE"
    assert out["detectors"][0]["voting"] is False
    assert "for reference only" in out["rationale"]


def test_platt_fit_does_not_diverge_on_saturated_weak_scores():
    """UnivFD-like scores: most near 0, weakly separable. Plain Newton gave a ~ 1e10."""
    rng = np.random.default_rng(1)
    neg = list(np.clip(rng.beta(0.3, 8, 96), 1e-7, 1))
    pos = list(np.clip(rng.beta(0.4, 6, 27), 1e-7, 1)) + [0.9999999]
    a, b = calibration.fit_platt(pos + neg, [True] * len(pos) + [False] * len(neg))
    assert np.isfinite(a) and 0 < a < calibration.MAX_SLOPE


def test_a_degenerate_fit_is_refused():
    with pytest.raises(ValueError, match="degenerate"):
        # Scores that run against the labels: no monotone calibration exists.
        calibration.fit("D", "v", [0.9] * 50 + [0.1] * 50, [False] * 50 + [True] * 50,
                        target_fpr=0.05, fitted_on={})


def test_run_config_records_the_ai_branch_settings():
    """A run must say which AI detectors voted, or its numbers cannot be reproduced."""
    from app.services.orchestration.analysis_pipeline import AnalysisPipeline

    snapshot = AnalysisPipeline._config_snapshot()
    assert {"bfree_enabled", "bfree_threshold", "calibration_path",
            "video_keyframe_verdicts_enabled"} <= set(snapshot)
    assert isinstance(snapshot["ai_generation_advisory_detectors"], list)


def test_branch_auc_uses_the_deciding_detector_when_it_ran():
    from app.services.evaluation.metrics import branch_score_detector

    both = [{"detectors": {"UnivFDDetector": {"raw_score": 0.1}, "BFreeDetector": {"raw_score": 0.9}}}]
    univfd_only = [{"detectors": {"UnivFDDetector": {"raw_score": 0.1}}}]
    assert branch_score_detector(both, "ai_generation") == "BFreeDetector"
    assert branch_score_detector(univfd_only, "ai_generation") == "UnivFDDetector"
    assert branch_score_detector([], "ai_generation") is None


def test_flag_rate_calibration_for_a_detector_that_rarely_fires():
    scores = [0.0] * 60 + [0.8] * 5 + [0.0] * 68
    targets = [True] * 65 + [False] * 68
    cal = calibration.fit_flag_rate("CopyMoveDetector", "2", scores, targets,
                                    flag_threshold=0.5, fitted_on={})
    prior_odds = cal.prior / (1 - cal.prior)
    fired = cal.probability(0.8) / (1 - cal.probability(0.8)) / prior_odds
    silent = cal.probability(0.0) / (1 - cal.probability(0.0)) / prior_odds
    assert fired == pytest.approx((5.5 / 66) / (0.5 / 69))  # smoothed, finite though 0/68 fired
    assert 0.9 < silent < 1.0  # silence is barely evidence


def test_flag_rate_calibration_refuses_a_detector_that_does_not_discriminate():
    with pytest.raises(ValueError):
        calibration.fit_flag_rate("CopyMoveDetector", "2", [0.8, 0.0, 0.8, 0.0],
                                  [True, True, False, False], flag_threshold=0.5, fitted_on={})
