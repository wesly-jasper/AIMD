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
  * A detector without a calibration is listed with the direction it points,
    never given a number.
  * Contributions that point in opposite directions are listed as conflicts.

Verbal scale on |log10 LR| of the weaker end, after the ENFSI guideline for
evaluative reporting: < 1 limited, 1-2 moderate, 2-3 moderately strong,
3-4 strong, >= 4 very strong.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Optional

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


def likelihood_ratio(cal: Calibration, raw: float) -> dict[str, Any]:
    """log10 LR of one raw score under ``cal``, capped by its validation set size."""
    p = min(max(cal.probability(raw), 1e-12), 1 - 1e-12)
    prior = min(max(cal.prior, 1e-6), 1 - 1e-6)
    value = math.log10(p / (1 - p)) - math.log10(prior / (1 - prior))
    fitted = cal.fitted_on or {}
    # Evidence *for* the positive class is limited by how many negatives were
    # seen, and evidence against it by how many positives were.
    side = "negatives" if value > 0 else "positives"
    # Re-encoded copies of one image are not independent, so the cap counts
    # distinct originals where the calibration recorded them.
    n = fitted.get(f"independent_{side}") or fitted.get(side)
    cap = math.log10((n or 0) + 1) if n else None
    capped = cap is not None and abs(value) > cap
    if capped:
        value = math.copysign(cap, value)
    return {"log10_lr": round(value, 3), "capped": capped, "cap": round(cap, 3) if cap else None}


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
    if cal is not None and cal.detector_version == version:
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
    voting = [e for e in observed if not e["advisory"]] or observed
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
    }
    if not numeric:
        result.update(
            status="NOT_ASSESSED",
            statement=(
                "No detector on this branch has a calibration for its version, so the "
                "strength of the evidence is not quantified. "
                + ("The detectors' directions are listed; they disagree."
                   if conflicts else "The detectors' directions are listed.")
                if observed else
                "No detector on this branch produced an observation."
            ),
        )
        return result

    values = [e["log10_lr"] for e in by_family.values()]
    independent = round(sum(values), 3)
    dependent = round(max(values, key=abs), 3)
    low, high = sorted((independent, dependent))
    # The verbal strength comes from the end nearer zero; if the range crosses
    # zero, the direction itself is uncertain.
    if low <= 0 <= high:
        weaker = 0.0
    else:
        weaker = low if low > 0 else high
    result.update(log10_lr_range=[low, high], log10_lr_independent=independent,
                  log10_lr_dependent=dependent, strength=verbal(weaker),
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
        "treating them as independent. Likelihood ratios hold for files like the "
        "validation data the calibrations were fitted on."
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
