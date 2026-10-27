"""
Running a detector so that it cannot take the analysis down with it.

The reliability evaluation (docs/RELIABILITY_REPORT.md section 4) found three
ways a detector failed without anyone being able to see it:

  * it hung, and the request hung with it -- there was no timeout;
  * it ran out of memory once and was recorded FAILED, although the same call
    succeeds after the cached memory is released;
  * it failed on every request for hours, while ``/health`` still said
    "configured", because health looked at files on disk and never at runs.

``run_detector`` bounds every call with a timeout, retries exactly once after an
out-of-memory failure, and records every outcome in ``detector_health``, which
``/health`` reports. A timed-out or failed run is AVAILABLE + FAILED: an absence
of evidence, never a negative result.
"""
from __future__ import annotations

import gc
import logging
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app.schemas.detection import (
    DetectionResult,
    DetectorAvailability,
    DetectorExecutionStatus,
    ScoreSemantics,
)

logger = logging.getLogger(__name__)

#: Fragments of out-of-memory messages from torch, CUDA, OpenCV and numpy. These
#: libraries raise generic exception types (``RuntimeError``, ``cv2.error``) for
#: allocation failures, so the message is the only thing that says what happened.
_MEMORY_MESSAGES = (
    "out of memory",
    "insufficient memory",
    "failed to allocate",
    "cannot allocate memory",
    "unable to allocate",
    "cublas_status_alloc_failed",
)


def is_memory_error(exc: BaseException) -> bool:
    """True for an allocation failure, which may succeed once memory is released."""
    if isinstance(exc, MemoryError):
        return True
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:  # torch absent or partially initialised
        pass
    message = str(exc).lower()
    return any(fragment in message for fragment in _MEMORY_MESSAGES)


def release_cached_memory() -> None:
    """Return Python garbage and torch's cached GPU blocks before a retry."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def memory_preflight(required_mb: int) -> Optional[str]:
    """
    ``None`` when at least ``required_mb`` of system memory is free, otherwise
    the reason a large model should not be loaded now.

    Loading a model into memory that is not there does not raise on Windows: it
    crashed the process inside torch with an access violation. Refusing to load
    turns that crash into an UNAVAILABLE detector with a stated reason.
    """
    if required_mb <= 0:
        return None
    try:
        import psutil
    except ImportError:
        return None  # cannot measure; the load itself is the only test left
    available_mb = psutil.virtual_memory().available // (1024 * 1024)
    if available_mb < required_mb:
        return (
            f"insufficient free memory to load the model safely "
            f"({available_mb} MB free, {required_mb} MB required)"
        )
    return None


class DetectorTimeout(TimeoutError):
    pass


# ── Health monitor ────────────────────────────────────────────────────────────


@dataclass
class _DetectorRecord:
    runs: int = 0
    completed: int = 0
    inconclusive: int = 0
    failed: int = 0
    skipped: int = 0
    timeouts: int = 0
    memory_retries: int = 0
    consecutive_failures: int = 0
    last_status: Optional[str] = None
    last_reason: Optional[str] = None
    last_run_at: Optional[str] = None
    last_success_at: Optional[str] = None
    last_duration_ms: Optional[float] = None
    #: Durations of the last runs that produced observations, for a p95.
    recent_durations_ms: list[float] = field(default_factory=list)


class DetectorHealthMonitor:
    """
    What each detector actually did since the process started.

    Configuration says whether a detector *should* work; this says whether it
    *does*. A detector that has failed on every recent request is reported
    FAILING, whatever its files on disk look like.
    """

    _DURATION_WINDOW = 50

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: dict[str, _DetectorRecord] = {}

    def reset(self) -> None:
        with self._lock:
            self._records.clear()

    def record(
        self,
        name: str,
        result: DetectionResult,
        duration_ms: float,
        *,
        timed_out: bool = False,
        memory_retry: bool = False,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        status = result.execution_status
        with self._lock:
            rec = self._records.setdefault(name, _DetectorRecord())
            rec.runs += 1
            rec.last_run_at = now
            rec.last_status = status.value
            rec.last_reason = result.status_reason
            rec.last_duration_ms = round(duration_ms, 1)
            rec.timeouts += int(timed_out)
            rec.memory_retries += int(memory_retry)
            if status is DetectorExecutionStatus.FAILED:
                rec.failed += 1
                rec.consecutive_failures += 1
                return
            if status is DetectorExecutionStatus.SKIPPED:
                # Not run: says nothing about whether the detector works, so it
                # neither breaks nor extends a run of failures.
                rec.skipped += 1
                return
            if status is DetectorExecutionStatus.COMPLETED:
                rec.completed += 1
            else:
                rec.inconclusive += 1
            rec.consecutive_failures = 0
            rec.last_success_at = now
            rec.recent_durations_ms.append(round(duration_ms, 1))
            del rec.recent_durations_ms[: -self._DURATION_WINDOW]

    def state(
        self,
        name: str,
        availability: DetectorAvailability = DetectorAvailability.AVAILABLE,
        failing_after: int = 3,
    ) -> str:
        """
        UNAVAILABLE  -- the capability is absent (says so before any run).
        UNTESTED     -- present, and not run since the process started.
        FAILING      -- the last ``failing_after`` runs all failed.
        DEGRADED     -- the most recent run failed, or one in the window timed out.
        OPERATIONAL  -- the most recent run produced observations.
        """
        if availability is DetectorAvailability.UNAVAILABLE:
            return "UNAVAILABLE"
        with self._lock:
            rec = self._records.get(name)
            if rec is None or rec.runs == rec.skipped:
                return "UNTESTED"
            if rec.consecutive_failures >= max(1, failing_after):
                return "FAILING"
            if rec.consecutive_failures > 0:
                return "DEGRADED"
            return "OPERATIONAL"

    def snapshot(self, name: str) -> dict[str, Any]:
        with self._lock:
            rec = self._records.get(name)
            if rec is None:
                return {"runs": 0}
            data = asdict(rec)
        durations = sorted(data.pop("recent_durations_ms"))
        if durations:
            data["p95_duration_ms"] = durations[min(len(durations) - 1, int(0.95 * len(durations)))]
        return data


#: The process-wide monitor. One per process, like the detectors it watches.
detector_health = DetectorHealthMonitor()


# ── Running a detector ────────────────────────────────────────────────────────


def _failure(detector: Any, reason: str, **metadata: Any) -> DetectionResult:
    """AVAILABLE + FAILED for a detector that raised or never returned."""
    failed = getattr(detector, "failed", None)
    if callable(failed):
        result = failed(reason)
    else:
        result = DetectionResult(
            detector=type(detector).__name__,
            media_type="image",
            manipulation_detected=False,
            confidence=0.0,
            manipulation_type="unknown",
            availability=DetectorAvailability.AVAILABLE,
            execution_status=DetectorExecutionStatus.FAILED,
            score_semantics=ScoreSemantics.NOT_APPLICABLE,
            status_reason=reason,
            metadata={"status": "failed", "reason": reason},
        )
    result.metadata.update(metadata)
    return result


#: Long-lived worker threads per detector. A fresh thread per call leaked: every
#: new thread that calls into torch or OpenCV gets its own native (OpenMP) worker
#: team, which outlives the Python thread -- an evaluation process reached 6,400
#: threads and 5.6 GB. Reused workers keep that bounded.
_WORKERS_PER_DETECTOR = 4
#: Weak keys: a detector discarded by a container reload takes its pool with it.
_executors: "weakref.WeakKeyDictionary[Any, ThreadPoolExecutor]" = weakref.WeakKeyDictionary()
_executors_lock = threading.Lock()


def _executor_for(detector: Any) -> ThreadPoolExecutor:
    with _executors_lock:
        executor = _executors.get(detector)
        if executor is None:
            executor = ThreadPoolExecutor(
                max_workers=_WORKERS_PER_DETECTOR,
                thread_name_prefix=f"detector-{type(detector).__name__}",
            )
            _executors[detector] = executor
        return executor


def _call_with_timeout(detector: Any, file_path: Path, timeout_s: float) -> DetectionResult:
    """
    ``detector.detect(file_path)`` on one of the detector's worker threads,
    abandoned after ``timeout_s``.

    Python cannot kill a thread, so a hung call keeps its worker until it
    returns. What the timeout guarantees is that the *analysis* does not wait
    for it. If every worker of a detector is hung, later calls queue and time
    out too -- reported as FAILED, never waited on.
    """
    future = _executor_for(detector).submit(detector.detect, file_path)
    try:
        result = future.result(timeout=timeout_s if timeout_s and timeout_s > 0 else None)
    except FutureTimeout:
        future.cancel()  # drops it if it never started
        raise DetectorTimeout(f"no result after {timeout_s:g} s") from None
    if not isinstance(result, DetectionResult):
        raise TypeError(f"Detector {type(detector).__name__} must return DetectionResult")
    return result


def _attempt(detector: Any, file_path: Path, timeout_s: float) -> tuple[DetectionResult, bool, bool]:
    """One run: ``(result, timed_out, memory_error)``. Never raises."""
    name = type(detector).__name__
    try:
        result = _call_with_timeout(detector, file_path, timeout_s)
    except DetectorTimeout as exc:
        logger.error("Detector %s timed out: %s", name, exc)
        return _failure(detector, f"timed out after {timeout_s:g} s", timed_out=True), True, False
    except Exception as exc:
        logger.warning("Detector %s failed", name, exc_info=True)
        memory = is_memory_error(exc)
        reason = f"{type(exc).__name__}: detector execution failed"
        if memory:
            reason += " (out of memory)"
        # AVAILABLE + FAILED, not UNAVAILABLE: the detector was present and was
        # invoked; it raised. Recording this as UNAVAILABLE would disguise a
        # defect in our code as an absent dependency.
        return _failure(detector, reason, memory_error=memory), False, memory
    memory = (
        result.execution_status is DetectorExecutionStatus.FAILED
        and bool(result.metadata.get("memory_error"))
    )
    return result, False, memory


def run_detector(
    detector: Any,
    file_path: str | Path,
    *,
    timeout_s: float,
    retry_on_memory_error: bool = True,
    monitor: Optional[DetectorHealthMonitor] = None,
) -> DetectionResult:
    """
    Run one detector: bounded by ``timeout_s``, retried once after running out
    of memory, recorded in ``monitor``. Always returns a ``DetectionResult``.
    """
    monitor = monitor or detector_health
    name = type(detector).__name__
    path = Path(file_path)
    started = time.perf_counter()

    result, timed_out, memory = _attempt(detector, path, timeout_s)
    retried = False
    if memory and retry_on_memory_error and not timed_out:
        logger.warning("Detector %s ran out of memory; releasing memory and retrying once", name)
        release_cached_memory()
        retried = True
        result, timed_out, _ = _attempt(detector, path, timeout_s)
        result.metadata["retried_after_memory_error"] = True

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    # Wall-clock time including any retry, as the analysis experienced it.
    result.metadata["runtime_ms"] = round(elapsed_ms, 1)
    monitor.record(name, result, elapsed_ms, timed_out=timed_out, memory_retry=retried)
    return result
