"""
Score calibration with a recorded procedure.

``ScoreSemantics.CALIBRATED_PROBABILITY`` is reserved for a score "validated
against a labelled dataset with a recorded calibration procedure". This module
is that procedure, and the record it writes is what entitles a detector to the
label:

  * **Fitting** (``fit``) -- Platt scaling, ``p = sigmoid(a * logit(raw) + b)``,
    fitted by maximum likelihood on the *validation* split of an evaluation run
    (``scripts/evaluate.py calibrate``). The operating threshold is chosen on
    the same split to hold the false-positive rate at or under a stated target.
  * **Record** -- the detector version (which names the exact weights), the
    dataset hash and split, class counts, the target and achieved FPR and
    recall, and expected calibration error before and after.
  * **Applying** (``calibrate_score``) -- only to the detector version it was
    fitted for. Different weights get the raw score and a stated reason, never
    a calibration that was measured on another model.

The probability is calibrated to the validation set's class balance (recorded
as ``prior``). It is "how often a file with this score was positive in a set
with this mix", not a universal likelihood; reports say so.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, dtype=float)))  # no overflow


#: A slope this steep means the fit found a near-step function: the scores do
#: not support a smooth probability, and the "calibration" would be a threshold
#: in disguise.
MAX_SLOPE = 25.0


def fit_platt(scores: Sequence[float], targets: Sequence[bool], iterations: int = 100) -> tuple[float, float]:
    """
    ``(a, b)`` maximising the likelihood of ``sigmoid(a * logit(s) + b)``.

    Newton's method with Platt's smoothed targets, which keep a perfectly
    separable validation set from driving ``a`` to infinity.
    """
    x = _logit(np.asarray(scores, dtype=float))
    y = np.asarray(targets, dtype=float)
    n_pos, n_neg = y.sum(), len(y) - y.sum()
    if n_pos == 0 or n_neg == 0:
        raise ValueError("calibration needs both positive and negative examples")
    t = np.where(y == 1, (n_pos + 1) / (n_pos + 2), 1 / (n_neg + 2))
    # A small L2 penalty on the slope plus a backtracking line search: plain
    # Newton diverged (a ~ 1e10) on UnivFD's saturated, barely separable scores.
    l2 = 1e-3

    def loss(a: float, b: float) -> float:
        z = a * x + b
        # log(1 + e^z) - t*z, computed stably
        return float(np.sum(np.logaddexp(0.0, z) - t * z) + 0.5 * l2 * a * a)

    a, b = 1.0, 0.0
    current = loss(a, b)
    for _ in range(iterations):
        p = _sigmoid(a * x + b)
        w = p * (1 - p) + 1e-12
        grad = np.array([np.sum((p - t) * x) + l2 * a, np.sum(p - t)])
        hess = np.array([[np.sum(w * x * x) + l2, np.sum(w * x)], [np.sum(w * x), np.sum(w)]])
        step = np.linalg.solve(hess + np.eye(2) * 1e-9, grad)
        scale = 1.0
        while scale > 1e-8:
            na, nb = a - scale * step[0], b - scale * step[1]
            candidate = loss(na, nb)
            if candidate <= current:
                break
            scale /= 2
        else:
            break
        moved = max(abs(na - a), abs(nb - b))
        a, b, current = na, nb, candidate
        if moved < 1e-9:
            break
    return float(a), float(b)


def threshold_for_fpr(negative_probs: Sequence[float], target_fpr: float) -> float:
    """
    The lowest threshold (``flagged = p >= threshold``) that flags at most
    ``floor(target_fpr * n)`` of the negatives.
    """
    neg = np.sort(np.asarray(negative_probs, dtype=float))[::-1]
    if len(neg) == 0:
        raise ValueError("no negatives to choose a threshold on")
    allowed = int(math.floor(target_fpr * len(neg)))
    if allowed >= len(neg):
        return 0.0
    return float(min(1.0, np.nextafter(neg[allowed], 2.0)))


def expected_calibration_error(probs: Sequence[float], targets: Sequence[bool], bins: int = 10) -> float:
    p = np.asarray(probs, dtype=float)
    y = np.asarray(targets, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if mask.any():
            total += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(total)


#: For a detector that is almost always silent and occasionally fires (the
#: copy-move detector): its evidence is whether it fired, and a sigmoid through
#: a pile of zeros and a few large values comes out nearly flat.
FLAG_RATE = "flag_rate"
#: Detectors calibrated by flag rate rather than Platt scaling. Their score is
#: an inlier-count severity that is 0 on almost every file.
FLAG_RATE_DETECTORS = frozenset({"CopyMoveDetector"})


@dataclass
class Calibration:
    detector: str
    detector_version: str
    method: str
    a: float
    b: float
    threshold: float
    target_fpr: float
    prior: float
    fitted_on: dict[str, Any]
    validation: dict[str, Any]
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    #: What the fitting data looked like; outside it, the fit was never tested.
    #: ``min_short_side``: the 5th percentile of the fitting images' shorter side.
    coverage: dict[str, Any] = field(default_factory=dict)
    #: (a, b) refitted on resamples of the fitting data (whole groups at a
    #: time), so a likelihood ratio can carry an interval, not just a value.
    bootstrap: list[list[float]] = field(default_factory=list)

    def probability(self, raw: float, params: Optional[Sequence[float]] = None) -> float:
        """P(positive | raw) at the fitting set's prior; ``params`` = (a, b) overrides the fit."""
        a, b = params if params is not None else (self.a, self.b)
        if self.method == FLAG_RATE:
            # a = P(flag | positive), b = P(flag | negative), both smoothed; the
            # score only says which side of ``threshold`` the detector came down.
            lr = a / b if raw >= self.threshold else (1 - a) / (1 - b)
            odds = lr * self.prior / (1 - self.prior)
            return float(odds / (1 + odds))
        return float(_sigmoid(a * _logit(np.asarray([raw], dtype=float)) + b)[0])

    def provenance(self) -> dict[str, Any]:
        """What a result carries so a reader can check how its number was made."""
        return {
            "method": self.method,
            "threshold": round(self.threshold, 6),
            "target_fpr": self.target_fpr,
            "prior": round(self.prior, 4),
            "fitted_on": self.fitted_on,
            "validation": self.validation,
            "created_at": self.created_at,
        }


#: Bootstrap resamples per calibration; enough for a 5th-95th percentile interval.
BOOTSTRAP = 200
#: Percentile of the fitting images' short side recorded as the coverage floor.
COVERAGE_PERCENTILE = 5


def _coverage(short_sides: Optional[Sequence[Optional[int]]]) -> dict[str, Any]:
    sides = [int(s) for s in (short_sides or []) if s]
    if not sides:
        return {}
    return {"min_short_side": int(np.percentile(sides, COVERAGE_PERCENTILE)),
            "short_side_percentile": COVERAGE_PERCENTILE, "images_with_size": len(sides)}


def _bootstrap(fit_one, scores, targets, groups, *, n: int = BOOTSTRAP, seed: int = 0) -> list[list[float]]:
    """``fit_one`` refitted on ``n`` resamples that draw whole groups, so the
    re-encoded copies of one image (one group) count as one draw."""
    rng = np.random.default_rng(seed)
    scores, targets = np.asarray(scores, dtype=float), np.asarray(targets, dtype=bool)
    groups = np.asarray(groups if groups is not None else np.arange(len(scores)))
    names = np.unique(groups)
    index = {g: np.nonzero(groups == g)[0] for g in names}
    out = []
    for _ in range(n):
        rows = np.concatenate([index[g] for g in rng.choice(names, size=len(names), replace=True)])
        y = targets[rows]
        if y.all() or not y.any():
            continue
        try:
            a, b = fit_one(scores[rows], y)
        except ValueError:
            continue
        if math.isfinite(a) and math.isfinite(b):
            out.append([round(float(a), 6), round(float(b), 6)])
    return out


def fit(
    detector: str,
    detector_version: str,
    scores: Sequence[float],
    targets: Sequence[bool],
    *,
    target_fpr: float,
    fitted_on: dict[str, Any],
    groups: Optional[Sequence[Any]] = None,
    short_sides: Optional[Sequence[Optional[int]]] = None,
    bootstrap: int = BOOTSTRAP,
) -> Calibration:
    a, b = fit_platt(scores, targets)
    if not (math.isfinite(a) and math.isfinite(b)) or abs(a) > MAX_SLOPE or a <= 0:
        raise ValueError(
            f"calibration fit is degenerate (slope {a:.3g}); the scores do not "
            "support a calibrated probability on this split"
        )
    raw = np.asarray(scores, dtype=float)
    y = np.asarray(targets, dtype=bool)
    probs = _sigmoid(a * _logit(raw) + b)
    threshold = threshold_for_fpr(probs[~y], target_fpr)
    flagged = probs >= threshold
    return Calibration(
        detector=detector,
        detector_version=detector_version,
        method="platt_logit",
        a=a,
        b=b,
        threshold=threshold,
        target_fpr=target_fpr,
        prior=float(y.mean()),
        fitted_on={**fitted_on, "positives": int(y.sum()), "negatives": int((~y).sum())},
        validation={
            # Measured on the split it was fitted on: an optimistic figure.
            # The held-out test split is what states the real performance.
            "fpr": round(float(flagged[~y].mean()), 4),
            "recall": round(float(flagged[y].mean()), 4),
            "ece_raw": round(expected_calibration_error(raw, y), 4),
            "ece_calibrated": round(expected_calibration_error(probs, y), 4),
        },
        coverage=_coverage(short_sides),
        bootstrap=_bootstrap(lambda s, t: fit_platt(s, t), scores, targets, groups, n=bootstrap),
    )


def fit_flag_rate(
    detector: str,
    detector_version: str,
    scores: Sequence[float],
    targets: Sequence[bool],
    *,
    flag_threshold: float,
    fitted_on: dict[str, Any],
    groups: Optional[Sequence[Any]] = None,
    short_sides: Optional[Sequence[Optional[int]]] = None,
    bootstrap: int = BOOTSTRAP,
) -> Calibration:
    """
    Flag rates among positives and negatives, with add-one-half smoothing, so a
    detector that never fired on the negatives gets a large but finite ratio.
    """

    def rates(s, t):
        f = np.asarray(s, dtype=float) >= flag_threshold
        t = np.asarray(t, dtype=bool)
        return (int(f[t].sum()) + 0.5) / (int(t.sum()) + 1), (int(f[~t].sum()) + 0.5) / (int((~t).sum()) + 1)

    y = np.asarray(targets, dtype=bool)
    flagged = np.asarray(scores, dtype=float) >= flag_threshold
    if y.all() or not y.any():
        raise ValueError("flag-rate calibration needs both classes")
    tp, fp = int(flagged[y].sum()), int(flagged[~y].sum())
    a = (tp + 0.5) / (int(y.sum()) + 1)
    b = (fp + 0.5) / (int((~y).sum()) + 1)
    if a <= b:
        raise ValueError(
            f"the detector fired no more often on positives ({tp}/{int(y.sum())}) than "
            f"on negatives ({fp}/{int((~y).sum())}) on this split"
        )
    return Calibration(
        detector=detector, detector_version=detector_version, method=FLAG_RATE,
        a=a, b=b, threshold=flag_threshold, target_fpr=round(b, 4), prior=float(y.mean()),
        fitted_on={**fitted_on, "positives": int(y.sum()), "negatives": int((~y).sum())},
        validation={"fpr": round(fp / max(1, int((~y).sum())), 4),
                    "recall": round(tp / max(1, int(y.sum())), 4)},
        coverage=_coverage(short_sides),
        bootstrap=_bootstrap(rates, scores, targets, groups, n=bootstrap),
    )


# ── Storage and lookup ────────────────────────────────────────────────────────


def save(calibrations: dict[str, Calibration], path: str | Path) -> None:
    Path(path).write_text(
        json.dumps({name: asdict(c) for name, c in calibrations.items()}, indent=2),
        encoding="utf-8",
    )


def load(path: str | Path) -> dict[str, Calibration]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {name: Calibration(**entry) for name, entry in data.items()}


_cache: dict[str, Any] = {"path": None, "mtime": None, "value": {}}
_cache_lock = threading.Lock()


def active_calibrations() -> dict[str, Calibration]:
    """The calibrations at ``settings.calibration_path``, reloaded when the file changes."""
    from app.core.config import settings

    return calibrations_at(settings.calibration_path)


def calibrations_at(path: Optional[str | Path]) -> dict[str, Calibration]:
    """The calibrations in the file at ``path`` (``{}`` if unset or missing), cached by mtime."""
    if not path:
        return {}
    path = Path(path)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {}
    with _cache_lock:
        if _cache["path"] != path or _cache["mtime"] != mtime:
            try:
                _cache["value"] = load(path)
            except Exception:
                logger.exception("Calibration file is unreadable; scores stay uncalibrated")
                _cache["value"] = {}
            _cache["path"], _cache["mtime"] = path, mtime
        return _cache["value"]


#: Checksums of the versioned calibration files, kept beside them.
CHECKSUM_FILE = "SHA256SUMS"


def file_status(path: Optional[str | Path]) -> dict[str, Any]:
    """
    What is at a configured calibration path: ``state`` is ``unset``,
    ``missing``, ``unreadable`` or ``ok``. When ok, also its sha256, whether
    that matches the ``SHA256SUMS`` beside it (``checksum``: ``verified``,
    ``mismatch`` or ``unlisted``) and the detector version each entry was
    fitted for. Never raises: a broken file is a status, not a crash.
    """
    if not path:
        return {"state": "unset"}
    path = Path(path)
    out: dict[str, Any] = {"path": str(path)}
    try:
        data = path.read_bytes()
    except OSError as exc:
        return {**out, "state": "missing", "error": type(exc).__name__}
    digest = hashlib.sha256(data).hexdigest()
    try:
        entries = {name: Calibration(**entry) for name, entry in json.loads(data).items()}
    except Exception as exc:  # malformed JSON or entries are the same failure to a reader
        return {**out, "state": "unreadable", "sha256": digest, "error": type(exc).__name__}
    return {**out, "state": "ok", "sha256": digest, "checksum": _checksum_state(path, digest),
            "detectors": {name: c.detector_version for name, c in entries.items()}}


def _checksum_state(path: Path, digest: str) -> str:
    try:
        lines = (path.parent / CHECKSUM_FILE).read_text(encoding="utf-8").splitlines()
    except OSError:
        return "unlisted"
    for line in lines:
        parts = line.split()
        if len(parts) == 2 and parts[1].lstrip("*") == path.name:
            return "verified" if parts[0].lower() == digest else "mismatch"
    return "unlisted"


def consistency(path: Optional[str | Path], live_versions: dict[str, Optional[str]]) -> dict[str, Any]:
    """
    ``file_status`` plus, per detector that is either running or in the file,
    whether the calibration applies to it: ``matched``, ``version_mismatch``
    (fitted for other weights or another preprocessing mode; the detector's
    scores then go uncalibrated), ``not_in_file`` or ``not_running``.
    ``consistent`` is false for a configured file that is missing, unreadable,
    fails its checksum, or does not match a running detector it covers.
    """
    status = file_status(path)
    if status["state"] != "ok":
        return {**status, "consistent": status["state"] == "unset", "detector_status": {}}
    fitted = status["detectors"]
    per: dict[str, Any] = {}
    for name in sorted(set(fitted) | set(live_versions)):
        live = live_versions.get(name)
        if name not in fitted:
            per[name] = {"status": "not_in_file", "running": live}
        elif name not in live_versions:
            per[name] = {"status": "not_running", "fitted": fitted[name]}
        elif fitted[name] != live:
            per[name] = {"status": "version_mismatch", "fitted": fitted[name], "running": live}
        else:
            per[name] = {"status": "matched", "version": live}
    consistent = status["checksum"] != "mismatch" and not any(
        p["status"] == "version_mismatch" for p in per.values())
    return {**status, "consistent": consistent, "detector_status": per}


def calibrate_score(detector: str, detector_version: Optional[str], raw: float) -> Optional[dict[str, Any]]:
    """
    ``None`` when no calibration exists for ``detector``. Otherwise a dict with
    ``applied``; when applied, the calibrated ``probability``, whether it is at
    or over the calibrated ``threshold`` (``flagged``), the ``raw_score`` it came
    from and the provenance.
    """
    calibration = active_calibrations().get(detector)
    if calibration is None:
        return None
    if calibration.detector_version != detector_version:
        return {
            "applied": False,
            "raw_score": raw,
            "reason": "calibration was fitted for a different detector version "
                      f"({calibration.detector_version}); the raw score is reported",
        }
    probability = calibration.probability(raw)
    return {
        "applied": True,
        "probability": round(probability, 6),
        # A flag-rate calibration's threshold is on the raw score (did it fire?).
        "flagged": (raw if calibration.method == FLAG_RATE else probability) >= calibration.threshold,
        "raw_score": raw,
        **calibration.provenance(),
    }
