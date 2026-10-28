"""Evidence fusion: likelihood-ratio ranges per branch, beside the verdicts."""
import math
from types import SimpleNamespace

import pytest

from app.services.detection.calibration import Calibration
from app.services.fusion.evidence_fusion import fuse, likelihood_ratio, verbal


def _cal(detector, a=4.0, b=0.0, prior=0.5, positives=200, negatives=200, version="1"):
    return Calibration(detector=detector, detector_version=version, method="platt", a=a, b=b,
                       threshold=0.5, target_fpr=0.05, prior=prior,
                       fitted_on={"positives": positives, "negatives": negatives}, validation={})


def _row(detector, score, detected, *, version="1", execution="COMPLETED", availability="AVAILABLE"):
    return SimpleNamespace(detector=detector, detector_version=version, availability=availability,
                           execution_status=execution, confidence=score, manipulation_detected=detected,
                           detector_metadata={}, status_reason=None)


def test_likelihood_ratio_removes_the_validation_prior():
    # At prior 0.2 a calibrated probability of 0.2 is no evidence at all.
    cal = _cal("TruForDetector", a=1.0, b=math.log(0.25), prior=0.2)
    assert likelihood_ratio(cal, 0.5)["log10_lr"] == pytest.approx(0.0, abs=1e-3)


def test_likelihood_ratio_is_capped_by_the_validation_set_size():
    cal = _cal("TruForDetector", a=20.0, negatives=99)
    lr = likelihood_ratio(cal, 0.999)
    assert lr["capped"] is True and lr["log10_lr"] == pytest.approx(2.0)


def test_uncalibrated_detectors_give_direction_only():
    out = fuse([_row("TruForDetector", 0.9, True)], calibrations={})
    m = out["manipulation"]
    assert m["status"] == "NOT_ASSESSED"
    assert m["uncalibrated"] == ["TruForDetector"]
    assert m["contributions"][0]["log10_lr"] is None


def test_two_families_give_a_range_from_dependent_to_independent():
    cals = {"TruForDetector": _cal("TruForDetector"), "CopyMoveDetector": _cal("CopyMoveDetector")}
    out = fuse([_row("TruForDetector", 0.9, True), _row("CopyMoveDetector", 0.8, True)], calibrations=cals)
    m = out["manipulation"]
    low, high = m["log10_lr_range"]
    single = likelihood_ratio(cals["TruForDetector"], 0.9)["log10_lr"]
    assert low == pytest.approx(single, abs=1e-3)  # fully dependent: the strongest family
    assert high > low  # independent: the sum
    assert m["status"] == "SUPPORTS" and m["strength"] == verbal(low)
    assert m["families_counted"] == ["geometric", "learned"]


def test_same_family_is_not_counted_twice():
    cals = {"UnivFDDetector": _cal("UnivFDDetector"), "BFreeDetector": _cal("BFreeDetector")}
    out = fuse([_row("UnivFDDetector", 0.9, True), _row("BFreeDetector", 0.9, True)], calibrations=cals)
    a = out["ai_generation"]
    assert a["log10_lr_range"][0] == a["log10_lr_range"][1]
    assert sum(c.get("counted", False) for c in a["contributions"]) == 1


def test_opposite_directions_are_listed_as_conflicts():
    cals = {"TruForDetector": _cal("TruForDetector"), "CopyMoveDetector": _cal("CopyMoveDetector")}
    out = fuse([_row("TruForDetector", 0.75, True), _row("CopyMoveDetector", 0.25, False)], calibrations=cals)
    m = out["manipulation"]
    assert m["conflicts"] == [{"for": "TruForDetector", "against": "CopyMoveDetector"}]
    assert m["status"] == "CONFLICTING"
    assert m["log10_lr_range"][0] <= 0 <= m["log10_lr_range"][1]


def test_advisory_detector_is_listed_but_not_counted():
    cals = {"UnivFDDetector": _cal("UnivFDDetector"), "BFreeDetector": _cal("BFreeDetector")}
    out = fuse([_row("UnivFDDetector", 0.05, False), _row("BFreeDetector", 0.95, True)],
               calibrations=cals, advisory=["UnivFDDetector"])
    a = out["ai_generation"]
    assert a["status"] == "SUPPORTS" and a["conflicts"] == []
    assert {c["detector"]: c["advisory"] for c in a["contributions"]} == {
        "UnivFDDetector": True, "BFreeDetector": False}


def test_version_mismatch_is_not_quantified():
    cals = {"TruForDetector": _cal("TruForDetector", version="old")}
    out = fuse([_row("TruForDetector", 0.9, True, version="new")], calibrations=cals)
    assert out["manipulation"]["status"] == "NOT_ASSESSED"


def test_failed_and_non_evidential_detectors_do_not_contribute():
    out = fuse([_row("TruForDetector", 0.0, False, execution="FAILED"),
                _row("YOLOv8ObjectDetector", 0.9, True)], calibrations={})
    m = out["manipulation"]
    assert [c["status"] for c in m["contributions"]] == ["not_run"]
    assert "No detector on this branch produced an observation" in m["statement"]


def test_against_on_the_ai_branch_never_says_camera():
    cals = {"BFreeDetector": _cal("BFreeDetector", a=6.0)}
    out = fuse([_row("BFreeDetector", 0.02, False)], calibrations=cals)
    a = out["ai_generation"]
    assert a["status"] == "AGAINST"
    assert "not a finding that the image came from a camera" in a["statement"]


def test_cap_counts_distinct_originals_not_their_copies():
    cal = _cal("BFreeDetector", a=20.0, negatives=600)
    cal.fitted_on["independent_negatives"] = 99
    assert likelihood_ratio(cal, 0.999)["log10_lr"] == pytest.approx(2.0)


def _sized(row, width, height):
    row.detector_metadata = {"image_size": [width, height]}
    return row


def test_image_below_the_calibration_coverage_gives_direction_only():
    cal = _cal("BFreeDetector")
    cal.coverage = {"min_short_side": 256}
    out = fuse([_sized(_row("BFreeDetector", 0.95, True), 400, 200)], calibrations={"BFreeDetector": cal})
    a = out["ai_generation"]
    assert a["status"] == "NOT_ASSESSED" and a["outside_coverage"] == ["BFreeDetector"]
    assert a["contributions"][0]["log10_lr"] is None and a["contributions"][0]["direction"] == 1
    assert "smaller than the files the calibrations were fitted on" in a["statement"]


def test_image_at_the_coverage_floor_is_quantified():
    cal = _cal("BFreeDetector")
    cal.coverage = {"min_short_side": 256}
    out = fuse([_sized(_row("BFreeDetector", 0.95, True), 400, 256)], calibrations={"BFreeDetector": cal})
    assert out["ai_generation"]["status"] == "SUPPORTS"


def test_strength_comes_from_the_cautious_end_of_the_interval():
    cal = _cal("BFreeDetector", a=4.0)
    point = likelihood_ratio(cal, 0.95)["log10_lr"]
    # Refits with shallower slopes: the data cannot pin the ratio down.
    cal.bootstrap = [[slope, 0.0] for slope in (1.5, 2.0, 3.0, 4.0, 5.0)]
    lr = likelihood_ratio(cal, 0.95)
    low, high = lr["log10_lr_interval"]
    assert low < point <= high
    a = fuse([_row("BFreeDetector", 0.95, True)], calibrations={"BFreeDetector": cal})["ai_generation"]
    assert a["log10_lr_dependent"] == pytest.approx(point, abs=1e-3)
    assert a["log10_lr_cautious"][0] == pytest.approx(low, abs=1e-3)
    assert a["strength"] == verbal(low) != verbal(point)


def test_an_interval_that_spans_zero_is_limited():
    cal = _cal("BFreeDetector", a=4.0)
    cal.bootstrap = [[slope, 0.0] for slope in (-1.0, 2.0, 4.0, 6.0)]
    a = fuse([_row("BFreeDetector", 0.95, True)], calibrations={"BFreeDetector": cal})["ai_generation"]
    assert a["log10_lr_dependent"] > 1
    assert a["status"] == "LIMITED" and a["strength"] == "limited"
