"""
Metrics over saved predictions -- pure functions, no models.

Every rate is reported with its count and a Wilson 95% interval, because the
sets are small (15-60 files per group) and a bare percentage hides that. The
two branches are measured separately and never pooled: the manipulation branch
on real vs manipulated, the AI-generation branch on real (and conventionally
edited) vs generated. See ``EvalItem.manipulation_target`` and
``ai_generation_target`` for exactly which items count.

INCONCLUSIVE and FAILED outcomes are not folded into either class. They are
counted and reported: a system that abstains on everything would otherwise
score a perfect false-positive rate.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

#: Verdicts that count as "flagged" on each branch.
MANIPULATION_FLAGGED = frozenset({"SUSPICIOUS", "LIKELY_MANIPULATED"})
MANIPULATION_CLEARED = frozenset({"NO_SIGNIFICANT_ANOMALIES_DETECTED"})
AI_FLAGGED = frozenset({"AI_GENERATION_INDICATED"})
AI_CLEARED = frozenset({"NOT_INDICATED"})


def wilson(k: int, n: int, z: float = 1.96) -> dict[str, Any]:
    if n == 0:
        return {"k": 0, "n": 0, "rate": None, "ci95": None}
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return {
        "k": k, "n": n, "rate": round(p, 4),
        "ci95": [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)],
    }


def auc(pos: list[float], neg: list[float]) -> Optional[float]:
    """Probability a random positive outscores a random negative (ties count half)."""
    if not pos or not neg:
        return None
    p = np.asarray(pos, dtype=float)
    q = np.asarray(neg, dtype=float)
    wins = (p[:, None] > q[None, :]).sum() + 0.5 * (p[:, None] == q[None, :]).sum()
    return float(wins / (len(p) * len(q)))


def auc_ci(pos: list[float], neg: list[float], resamples: int = 1000, seed: int = 0) -> Optional[list[float]]:
    if not pos or not neg:
        return None
    rnd = random.Random(seed)
    values = sorted(
        auc([rnd.choice(pos) for _ in pos], [rnd.choice(neg) for _ in neg])
        for _ in range(resamples)
    )
    return [round(values[int(0.025 * resamples)], 4), round(values[int(0.975 * resamples) - 1], 4)]


def binary_metrics(rows: list[dict[str, Any]], flagged: frozenset, cleared: frozenset) -> dict[str, Any]:
    """
    ``rows``: ``{"target": bool, "verdict": str, "score": float|None}``.

    Recall and FPR are over *decided* items only (flagged or cleared); the
    abstention rate is reported beside them, per class, so neither can be
    improved by abstaining more without it showing.
    """
    pos = [r for r in rows if r["target"]]
    neg = [r for r in rows if not r["target"]]

    def count(group, verdicts):
        return sum(1 for r in group if r["verdict"] in verdicts)

    tp, fn = count(pos, flagged), count(pos, cleared)
    fp, tn = count(neg, flagged), count(neg, cleared)
    recall = wilson(tp, tp + fn)
    fpr = wilson(fp, fp + tn)
    precision = wilson(tp, tp + fp)
    balanced = (
        round(0.5 * (recall["rate"] + 1 - fpr["rate"]), 4)
        if recall["rate"] is not None and fpr["rate"] is not None else None
    )
    pos_scores = [r["score"] for r in pos if r.get("score") is not None]
    neg_scores = [r["score"] for r in neg if r.get("score") is not None]
    a = auc(pos_scores, neg_scores)
    return {
        "positives": len(pos),
        "negatives": len(neg),
        "recall": recall,
        "false_positive_rate": fpr,
        "precision": precision,
        "balanced_accuracy": balanced,
        "abstained_positive": wilson(len(pos) - tp - fn, len(pos)),
        "abstained_negative": wilson(len(neg) - fp - tn, len(neg)),
        "score_auc": round(a, 4) if a is not None else None,
        "score_auc_ci95": auc_ci(pos_scores, neg_scores),
    }


# ── Localisation ──────────────────────────────────────────────────────────────


def load_binary(path: Path, shape: Optional[tuple[int, int]] = None, threshold: float = 0.5) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as img:
        gray = img.convert("L")
        if shape is not None and gray.size != (shape[1], shape[0]):
            gray = gray.resize((shape[1], shape[0]), Image.NEAREST)
        values = np.asarray(gray, dtype=np.float32) / 255.0
    return values >= threshold


def localisation_scores(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    inter = float(np.logical_and(pred, truth).sum())
    union = float(np.logical_or(pred, truth).sum())
    p, t = float(pred.sum()), float(truth.sum())
    return {
        "iou": inter / union if union else 1.0,
        "f1": 2 * inter / (p + t) if (p + t) else 1.0,
    }


def localisation_summary(values: list[dict[str, float]]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    ious = [v["iou"] for v in values]
    f1s = [v["f1"] for v in values]
    return {
        "n": len(values),
        "mean_iou": round(float(np.mean(ious)), 4),
        "median_iou": round(float(np.median(ious)), 4),
        "mean_f1": round(float(np.mean(f1s)), 4),
        "iou_at_least_0_5": wilson(sum(i >= 0.5 for i in ious), len(ious)),
    }


# ── Whole report ──────────────────────────────────────────────────────────────


def _subgroups(record: dict[str, Any], branch: str) -> list[str]:
    keys = [f"label={record['label']}"]
    if record.get("kind"):
        keys.append(f"kind={record['kind']}")
    if record.get("generator"):
        keys.append(f"generator={record['generator']}")
    keys.extend(f"tag={t}" for t in record.get("tags") or [])
    return keys


def branch_rows(records: Iterable[dict[str, Any]], branch: str, detector: Optional[str] = None) -> list[dict[str, Any]]:
    """The records that belong to ``branch``, with target, verdict and a score."""
    rows = []
    for rec in records:
        target = rec["targets"].get(branch)
        if target is None:
            continue
        verdict = rec["verdicts"].get(branch) or "MISSING"
        score = None
        if detector:
            det = rec["detectors"].get(detector) or {}
            if det.get("execution_status") in ("COMPLETED", "INCONCLUSIVE"):
                score = det.get("raw_score")
        rows.append({**rec, "target": target, "verdict": verdict, "score": score})
    return rows


#: Detectors whose raw score may stand for each branch's AUC, in order of
#: preference: the one that decides the branch first. UnivFD is advisory once
#: B-Free runs, and the branch AUC was still reported from UnivFD's score.
BRANCH_SCORE_DETECTORS = {
    "manipulation": ("TruForDetector",),
    "ai_generation": ("BFreeDetector", "UnivFDDetector"),
}


def branch_score_detector(records: Iterable[dict[str, Any]], branch: str) -> Optional[str]:
    """The first of ``BRANCH_SCORE_DETECTORS[branch]`` that scored anything in ``records``."""
    records = list(records)
    for detector in BRANCH_SCORE_DETECTORS[branch]:
        if any((r.get("detectors") or {}).get(detector, {}).get("raw_score") is not None for r in records):
            return detector
    return None
#: Every detector that answers each branch's question, measured on its own
#: decision as well as through the branch verdict.
BRANCH_DETECTORS = {
    "manipulation": ("TruForDetector", "CopyMoveDetector"),
    "ai_generation": ("UnivFDDetector", "BFreeDetector"),
}
_OWN_FLAGGED = frozenset({"FLAGGED"})
_OWN_CLEARED = frozenset({"CLEARED"})
BRANCH_VERDICTS = {
    "manipulation": (MANIPULATION_FLAGGED, MANIPULATION_CLEARED),
    "ai_generation": (AI_FLAGGED, AI_CLEARED),
}


def compute_report(records: list[dict[str, Any]], *, split: Optional[str] = None) -> dict[str, Any]:
    if split:
        records = [r for r in records if r.get("split") == split]
    report: dict[str, Any] = {"split": split or "all", "files": len(records)}

    for branch, (flagged, cleared) in BRANCH_VERDICTS.items():
        score_detector = branch_score_detector(records, branch)
        rows = branch_rows(records, branch, score_detector)
        section = binary_metrics(rows, flagged, cleared)
        section["score_detector"] = score_detector
        # Per subgroup: recall within positive subgroups, FPR within negative ones.
        sub: dict[str, list] = defaultdict(list)
        for r in rows:
            for key in _subgroups(r, branch):
                sub[key].append(r)
        section["subgroups"] = {}
        for key, group in sorted(sub.items()):
            pos = [r for r in group if r["target"]]
            neg = [r for r in group if not r["target"]]
            entry: dict[str, Any] = {"n": len(group)}
            if pos:
                decided = [r for r in pos if r["verdict"] in flagged | cleared]
                entry["recall"] = wilson(sum(r["verdict"] in flagged for r in decided), len(decided))
            if neg:
                decided = [r for r in neg if r["verdict"] in flagged | cleared]
                entry["false_positive_rate"] = wilson(sum(r["verdict"] in flagged for r in decided), len(decided))
            section["subgroups"][key] = entry
        # Each detector on its own decision, on the same files: the fair way to
        # compare two models when the branch verdict reports their disagreements.
        section["per_detector"] = {}
        for name in BRANCH_DETECTORS[branch]:
            own = []
            for r in branch_rows(records, branch, name):
                det = r["detectors"].get(name)
                if not det:
                    continue
                status = det.get("execution_status")
                verdict = (
                    ("FLAGGED" if det.get("manipulation_detected") else "CLEARED")
                    if status == "COMPLETED" else "ABSTAINED"
                )
                own.append({**r, "verdict": verdict})
            if own:
                m = binary_metrics(own, _OWN_FLAGGED, _OWN_CLEARED)
                sub: dict[str, Any] = {}
                for r in own:
                    for key in _subgroups(r, branch):
                        sub.setdefault(key, []).append(r)
                m["subgroups"] = {
                    key: {
                        **({"recall": wilson(sum(x["verdict"] == "FLAGGED" for x in g if x["target"]),
                                             sum(1 for x in g if x["target"] and x["verdict"] != "ABSTAINED"))}
                           if any(x["target"] for x in g) else {}),
                        **({"false_positive_rate": wilson(sum(x["verdict"] == "FLAGGED" for x in g if not x["target"]),
                                                          sum(1 for x in g if not x["target"] and x["verdict"] != "ABSTAINED"))}
                           if any(not x["target"] for x in g) else {}),
                    }
                    for key, g in sorted(sub.items())
                }
                section["per_detector"][name] = m
        report[branch] = section

    report["localisation"] = localisation_summary(
        [r["localisation"] for r in records if r.get("localisation")]
    )

    # Operational reliability: every non-observation is counted, by detector.
    per_detector: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    durations: dict[str, list[float]] = defaultdict(list)
    for rec in records:
        for name, det in rec["detectors"].items():
            per_detector[name][det.get("execution_status", "MISSING")] += 1
            if det.get("timed_out"):
                per_detector[name]["TIMED_OUT"] += 1
            if det.get("duration_ms") is not None:
                durations[name].append(det["duration_ms"])
    report["detectors"] = {
        name: {
            **dict(counts),
            "p50_ms": round(float(np.percentile(durations[name], 50)), 1) if durations[name] else None,
            "p95_ms": round(float(np.percentile(durations[name], 95)), 1) if durations[name] else None,
        }
        for name, counts in sorted(per_detector.items())
    }
    report["run_errors"] = sum(1 for r in records if r.get("error"))
    # A "silent" failure is one the verdict does not reflect: a detector that
    # failed while the manipulation verdict still claimed full coverage.
    report["silent_failures"] = sum(
        1 for r in records
        if r["verdicts"].get("manipulation") in MANIPULATION_CLEARED
        and any(
            d.get("execution_status") == "FAILED" and d.get("evidential")
            for d in r["detectors"].values()
        )
    )
    return report


# ── Comparison against a baseline ─────────────────────────────────────────────


def _rate(section: dict[str, Any], key: str) -> Optional[float]:
    value = section.get(key)
    return value.get("rate") if isinstance(value, dict) else value


def compare(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """
    Per branch, the change in each headline metric, and whether it is outside
    the baseline's 95% interval. A recall gain that raises the false-positive
    rate beyond its interval is flagged as a regression, not an improvement.
    """
    out: dict[str, Any] = {}
    for branch in BRANCH_VERDICTS:
        b, c = baseline.get(branch) or {}, candidate.get(branch) or {}
        rows = {}
        for key, higher_is_better in (
            ("recall", True), ("false_positive_rate", False), ("precision", True),
            ("balanced_accuracy", True), ("score_auc", True),
        ):
            before, after = _rate(b, key), _rate(c, key)
            if before is None or after is None:
                rows[key] = {"baseline": before, "candidate": after}
                continue
            # Balanced accuracy has no interval of its own, so its change is
            # always reported as within noise; read recall and FPR for it.
            if key == "score_auc":
                ci = b.get("score_auc_ci95")
            else:
                ci = b[key].get("ci95") if isinstance(b.get(key), dict) else None
            outside = bool(ci) and not (ci[0] <= after <= ci[1])
            better = (after > before) == higher_is_better and after != before
            rows[key] = {
                "baseline": before, "candidate": after, "delta": round(after - before, 4),
                "outside_baseline_ci95": outside,
                "verdict": ("improved" if better else "regressed") if outside else "within_noise",
            }
        out[branch] = rows
    regressions = [
        f"{branch}.{key}"
        for branch, rows in out.items() for key, row in rows.items()
        if row.get("verdict") == "regressed"
    ]
    for key in ("silent_failures", "run_errors"):
        if (candidate.get(key) or 0) > (baseline.get(key) or 0):
            regressions.append(key)
    loc_b = (baseline.get("localisation") or {}).get("mean_iou")
    loc_c = (candidate.get("localisation") or {}).get("mean_iou")
    out["localisation_mean_iou"] = {"baseline": loc_b, "candidate": loc_c}
    out["regressions"] = regressions
    return out
