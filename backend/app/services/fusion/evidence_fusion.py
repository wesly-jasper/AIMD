"""
Evidence fusion: how strongly the observations, taken together, bear on each
question, with the uncertainty and the disagreements left visible.

The verdict rules (``AnalysisPipeline._decide`` and ``_decide_ai_generation``)
put every analysis into one category. This module sits beside them and never
changes them. It answers a narrower question for each branch:

    How much more probable are these detector outputs if the image was
    manipulated (or generated) than if it was not?

That is a likelihood ratio (LR). It is reported as a range, never a single
"probability of manipulation": the prior, how common manipulated images are
among the files a user uploads, is not known, and an LR does not need it.

How the number is made:

  * Only a detector with a calibration fitted for its exact version contributes
    a number. Platt scaling gives P(positive | score) at the validation set's
    class balance (``prior``); dividing the odds by the prior odds gives the LR.
  * An LR is capped at what the validation data could show. With n validation
    negatives, a false-positive rate below 1/(n+1) was never observed, so an LR
    above n+1 is extrapolation. n counts distinct original images, not their
    re-encoded copies. Capped values are marked.
  * Within a signal family the strongest contribution counts, never the sum:
    detectors with the same physical basis are not independent witnesses.
  * Across families the result is a range. One end assumes the families are
    independent (log-LRs add). The other assumes they are fully dependent (only
    the strongest family counts). The truth is somewhere between, and the
    verbal strength is taken from whichever end is weaker.
  * Each LR carries a 5th-95th percentile interval from refitting the
    calibration on resamples of its data. Status and verbal strength come from
    the end of that interval nearer zero: an LR the data cannot pin down is
    not called strong.
  * A detector without a calibration is listed with the direction it points,
    never given a number. So is one whose file is smaller than the images the
    calibration was fitted on (its recorded coverage, the 5th percentile of
    their short sides): there, the fit was never tested. Coverage is about
    size only; small, heavily compressed files inside it (the 256 px COCO
    JPEGs) still mislead more often than the rest (RELIABILITY_REPORT.md §4.8).
  * Contributions that point in opposite directions are listed as conflicts.

Verbal scale on |log10 LR| of the weaker end, after the ENFSI guideline for
evaluative reporting: < 1 limited, 1-2 moderate, 2-3 moderately strong,
3-4 strong, >= 4 very strong.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Optional

import numpy as np

from app.services.detection.calibration import Calibration, calibrations_at
from app.services.detection.engine.image_detection_engine import _SIGNAL_FAMILY

BRANCHES = ("manipulation", "ai_generation")
_NOT_EVIDENCE = {"non_evidential"}
#: |log10 LR| below which a single contribution is too weak to count as a conflict.
CONFLICT_MIN = 0.3
_SCALE = ((4.0, "very strong"), (3.0, "strong"), (2.0, "moderately strong"), (1.0, "moderate"))


def _plain(value: Any) -> str:
    return value.value if hasattr(value, "value") else str(value)


def branch_of(detector: str) -> Optional[str]:
    family = _SIGNAL_FAMILY.get(detector, "unclassified")
    if family in _NOT_EVIDENCE:
        return None
    return "ai_generation" if family == "ai_generation" else "manipulation"


def verbal(log10_lr: float) -> str:
    magnitude = abs(log10_lr)
    for floor, word in _SCALE:
        if magnitude >= floor:
            return word
    return "limited"


def _raw(row: Any) -> float:
    meta = getattr(row, "detector_metadata", None) or {}
    calibration = meta.get("calibration") or {}
    if calibration.get("raw_score") is not None:
        return float(calibration["raw_score"])
    return float(row.confidence)


def _log10_lr(cal: Calibration, raw: float, params=None) -> float:
    p = min(max(cal.probability(raw, params), 1e-12), 1 - 1e-12)
    prior = min(max(cal.prior, 1e-6), 1 - 1e-6)
    return math.log10(p / (1 - p)) - math.log10(prior / (1 - prior))


def _cap(cal: Calibration, value: float) -> Optional[float]:
    fitted = cal.fitted_on or {}
    # Evidence *for* the positive class is limited by how many negatives were
    # seen, and evidence against it by how many positives were.
    side = "negatives" if value > 0 else "positives"
    # Re-encoded copies of one image are not independent, so the cap counts
    # distinct originals where the calibration recorded them.
    n = fitted.get(f"independent_{side}") or fitted.get(side)
    return math.log10(n + 1) if n else None


def _capped(cal: Calibration, value: float) -> float:
    cap = _cap(cal, value)
    return math.copysign(cap, value) if cap is not None and abs(value) > cap else value


def likelihood_ratio(cal: Calibration, raw: float) -> dict[str, Any]:
    """log10 LR of one raw score under ``cal``, capped by its validation set size,
    with a bootstrap interval when the calibration carries one."""
    value = _log10_lr(cal, raw)
    cap = _cap(cal, value)
    capped = cap is not None and abs(value) > cap
    out = {"log10_lr": round(_capped(cal, value), 3), "capped": capped, "cap": round(cap, 3) if cap else None}
    if cal.bootstrap:
        samples = sorted(_capped(cal, _log10_lr(cal, raw, ab)) for ab in cal.bootstrap)
        lo, hi = np.percentile(samples, [5, 95])
        out["log10_lr_interval"] = [round(float(lo), 3), round(float(hi), 3)]
    return out


def _conservative(entry: dict[str, Any]) -> float:
    """The end of a contribution's interval nearer zero (zero if it spans zero)."""
    lo, hi = entry.get("log10_lr_interval") or (entry["log10_lr"], entry["log10_lr"])
    if lo <= 0 <= hi:
        return 0.0
    return lo if lo > 0 else hi


def _entry(row: Any, calibrations: dict[str, Calibration], advisory: set[str]) -> Optional[dict[str, Any]]:
    branch = branch_of(row.detector)
    if branch is None:
        return None
    availability = _plain(getattr(row, "availability", "AVAILABLE"))
    execution = _plain(getattr(row, "execution_status", "COMPLETED"))
    entry: dict[str, Any] = {
        "detector": row.detector,
        "detector_version": getattr(row, "detector_version", None),
        "family": _SIGNAL_FAMILY.get(row.detector, "unclassified"),
        "branch": branch,
        "advisory": row.detector in advisory,
    }
    if availability == "UNAVAILABLE" or execution in ("FAILED", "SKIPPED"):
        return {**entry, "status": "not_run", "reason": getattr(row, "status_reason", None)}
    if execution != "COMPLETED":
        return {**entry, "status": "no_observation", "reason": getattr(row, "status_reason", None)}

    direction = 1 if bool(row.manipulation_detected) else -1
    entry.update(status="observed", direction=direction)
    cal = calibrations.get(row.detector)
    version = getattr(row, "detector_version", None)
    meta = getattr(row, "detector_metadata", None) or {}
    size = meta.get("image_size")
    floor = (cal.coverage or {}).get("min_short_side") if cal is not None else None
    if cal is not None and floor and size and min(size) < floor:
        entry.update(calibrated=False, log10_lr=None, outside_coverage=True,
                     reason=f"image is {min(size)} px on its short side, below the {floor} px "
                            "the calibration's data covers: direction only")
    elif cal is not None and cal.detector_version == version:
        raw = _raw(row)
        entry.update(calibrated=True, raw_score=round(raw, 6), **likelihood_ratio(cal, raw),
                     calibration={"prior": round(cal.prior, 4), "fitted_on": cal.fitted_on})
        if entry["log10_lr"] != 0:
            entry["direction"] = 1 if entry["log10_lr"] > 0 else -1
    else:
        entry.update(calibrated=False, log10_lr=None,
                     reason="no calibration for this detector version: direction only")
    return entry


def _fuse_branch(entries: list[dict[str, Any]]) -> dict[str, Any]:
    observed = [e for e in entries if e["status"] == "observed"]
    # Advisory detectors never count, even when they are the only ones observed.
    voting = [e for e in observed if not e["advisory"]]
    numeric = [e for e in voting if e.get("calibrated")]

    by_family: dict[str, dict[str, Any]] = {}
    for e in numeric:
        best = by_family.get(e["family"])
        if best is None or abs(e["log10_lr"]) > abs(best["log10_lr"]):
            by_family[e["family"]] = e
    for e in numeric:
        e["counted"] = by_family.get(e["family"]) is e

    conflicts = []
    pointing = [e for e in voting
                if (e.get("calibrated") and abs(e["log10_lr"]) >= CONFLICT_MIN) or not e.get("calibrated")]
    for i, a in enumerate(pointing):
        for b in pointing[i + 1:]:
            if a["direction"] != b["direction"]:
                conflicts.append({
                    "for": a["detector"] if a["direction"] > 0 else b["detector"],
                    "against": b["detector"] if a["direction"] > 0 else a["detector"],
                })

    result: dict[str, Any] = {
        "contributions": entries,
        "conflicts": conflicts,
        "families_counted": sorted(by_family),
        "uncalibrated": [e["detector"] for e in voting if not e.get("calibrated")],
        "outside_coverage": [e["detector"] for e in voting if e.get("outside_coverage")],
    }
    if not numeric:
        if not voting:
            why = "No detector on this branch produced an observation that counts."
        elif all(e.get("outside_coverage") for e in voting):
            why = ("The image is smaller than the files the calibrations were fitted on, "
                   "so the strength of the evidence is not quantified. ")
        else:
            why = ("No detector on this branch has a calibration that applies, so the "
                   "strength of the evidence is not quantified. ")
        if voting:
            why += ("The detectors' directions are listed; they disagree."
                    if conflicts else "The detectors' directions are listed.")
        result.update(status="NOT_ASSESSED", statement=why)
        return result

    values = [e["log10_lr"] for e in by_family.values()]
    independent = round(sum(values), 3)
    dependent = round(max(values, key=abs), 3)
    low, high = sorted((independent, dependent))
    # Status and verbal strength come from the cautious values: each family's
    # interval end nearer zero, combined the same two ways. If that range
    # crosses zero, the direction itself is uncertain.
    cautious = [_conservative(e) for e in by_family.values()]
    c_low, c_high = sorted((round(sum(cautious), 3), round(max(cautious, key=abs), 3)))
    if c_low <= 0 <= c_high:
        weaker = 0.0
    else:
        weaker = c_low if c_low > 0 else c_high
    result.update(log10_lr_range=[low, high], log10_lr_independent=independent,
                  log10_lr_dependent=dependent, log10_lr_cautious=[c_low, c_high],
                  strength=verbal(weaker),
                  any_capped=any(e["capped"] for e in by_family.values()))
    if conflicts and abs(weaker) < 1:
        result["status"] = "CONFLICTING"
    elif abs(weaker) < 1:
        result["status"] = "LIMITED"
    else:
        result["status"] = "SUPPORTS" if weaker > 0 else "AGAINST"
    return result


def _statement(branch: str, fused: dict[str, Any]) -> str:
    if fused["status"] == "NOT_ASSESSED":
        return fused["statement"]
    low, high = fused["log10_lr_range"]
    what = "manipulated" if branch == "manipulation" else "generated by an AI model"
    fmt = lambda v: f"{10 ** v:,.3g}"  # noqa: E731
    if fused["status"] in ("SUPPORTS", "AGAINST"):
        direction = "if" if fused["status"] == "SUPPORTS" else "if not"
        lo, hi = (low, high) if fused["status"] == "SUPPORTS" else (-high, -low)
        text = (
            f"The detector outputs are between {fmt(lo)} and {fmt(hi)} times more "
            f"probable {direction} the image was {what} than {'if not' if direction == 'if' else 'if so'} "
            f"({fused['strength']} support). "
        )
    elif fused["status"] == "CONFLICTING":
        text = "The calibrated detectors point in opposite directions; the evidence does not favour either answer. "
    else:
        text = "The detector outputs barely discriminate (limited support either way). "
    text += (
        "The range runs from treating the signal families as fully dependent to "
        "treating them as independent; the strength is graded on the cautious end "
        "of each ratio's uncertainty interval. Likelihood ratios hold for files like "
        "the validation data the calibrations were fitted on."
    )
    if fused.get("any_capped"):
        text += " At least one ratio was capped at what the validation set could show."
    if branch == "ai_generation" and fused["status"] == "AGAINST":
        text += " This is not a finding that the image came from a camera."
    return text


def fuse(rows: Iterable[Any], *, calibrations: Optional[dict[str, Calibration]] = None,
         advisory: Iterable[str] = ()) -> dict[str, Any]:
    """The fused evidence for both branches of one analysis."""
    if calibrations is None:
        from app.core.config import settings

        calibrations = calibrations_at(settings.fusion_calibration_path or settings.calibration_path)
    advisory = set(advisory)
    entries = [e for e in (_entry(r, calibrations, advisory) for r in rows) if e is not None]
    out: dict[str, Any] = {"method": "likelihood-ratio range over calibrated signal families", "version": 1}
    for branch in BRANCHES:
        fused = _fuse_branch([e for e in entries if e["branch"] == branch])
        fused["statement"] = _statement(branch, fused)
        out[branch] = fused
    return out
