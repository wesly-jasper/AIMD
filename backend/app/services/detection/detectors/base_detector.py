"""
The detector contract.

Every detector reports two independent facts -- whether it *could* run, and what
happened when it did -- plus enough identity to reproduce the run. See
``app.schemas.detection`` for why collapsing those two facts is a forensic
misrepresentation rather than a tidiness problem.

WHY THE RESULT CONSTRUCTORS LIVE HERE
-------------------------------------
Each detector previously carried its own ``_unavailable(reason)`` helper, and all
of them routed three genuinely different situations through it:

  * a missing dependency or checkpoint      -- the capability is absent
  * an unreadable or wrong-type input       -- the capability is present, the
                                               input could not be processed
  * an exception inside the analysis        -- the capability is present and broke

All three were recorded as ``UNAVAILABLE``, so a crashing detector was
indistinguishable from one that was never installed. The first needs a
deployment fix and the other two need a code fix, and only the first is honestly
described as "we do not have this capability".

Six copies of that helper is also why the defect persisted: fixing one copy left
the others asserting the same thing. The constructors are therefore part of the
base contract, named for the situation they describe, so choosing the wrong one
requires writing the wrong word.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

from app.schemas.detection import (
    DetectionRegion,
    DetectionResult,
    DetectorAvailability,
    DetectorExecutionStatus,
    ScoreSemantics,
)


class BaseDetector(ABC):
    """
    Base class for all forensic detectors.

    Subclasses MUST override ``detector_id``, ``version``, ``manipulation_type``
    and ``supported_media_types``. ``detector_id`` is the stable identity recorded
    against every result: it names the code that actually ran, so a detector that
    falls back to a different implementation must not report the identity of the
    one that was requested.
    """

    #: Stable identity of this detector. Recorded with every result.
    detector_id: str = "UnnamedDetector"

    #: Version of the detector logic. Bump when behaviour or thresholds change,
    #: otherwise two differently-behaving runs are indistinguishable in the record.
    version: str = "0.0.0"

    #: Media types this detector can analyse. Anything else is SKIPPED -- not
    #: failed, and certainly not "no anomalies found".
    supported_media_types: frozenset[str] = frozenset({"image"})

    #: What kind of manipulation this detector looks for.
    manipulation_type: str = "unknown"

    #: How to read ``confidence`` on a result this detector produces. Defaults to
    #: the honest value for classical signal heuristics; a learned detector that
    #: emits a model output must declare RAW_MODEL_SCORE. Nothing here is entitled
    #: to CALIBRATED_PROBABILITY without a recorded calibration procedure.
    score_semantics: ScoreSemantics = ScoreSemantics.HEURISTIC_SEVERITY

    # ── Capability ────────────────────────────────────────────────────────────

    def availability(self) -> tuple[DetectorAvailability, Optional[str]]:
        """
        Whether this detector can run at all, and why not if it cannot.

        Checked BEFORE analysis, so a missing dependency or checkpoint is reported
        as an absent capability instead of surfacing later as a crash. Says
        nothing about what the detector would find.

        The default assumes a detector with no external requirements. Any detector
        that needs a model file, an optional package, or a device must override
        this -- claiming availability it has not verified is exactly the
        fabrication this contract exists to prevent.
        """
        return DetectorAvailability.AVAILABLE, None

    def supports(self, media_type: str) -> bool:
        return media_type in self.supported_media_types

    @abstractmethod
    def detect(self, file_path: str | Path) -> DetectionResult:
        """Analyse the file and return a result. Must not raise for ordinary
        failure modes: return ``self.failed(...)`` so the reason is recorded."""

    # ── Result constructors ───────────────────────────────────────────────────
    #
    # Deliberately four separate methods rather than one with a status argument:
    # the name at the call site states which of the four situations the author
    # believed they were in, and that is reviewable.

    def unavailable(
        self, reason: str, *, media_type: Optional[str] = None
    ) -> DetectionResult:
        """
        The capability is absent: no checkpoint, no dependency, no device.

        UNAVAILABLE + SKIPPED. Use only when the detector genuinely cannot run.
        A detector that ran and broke is ``failed``, not unavailable.
        """
        return self._non_result(
            DetectorAvailability.UNAVAILABLE,
            DetectorExecutionStatus.SKIPPED,
            reason,
            media_type,
        )

    def failed(
        self, reason: str, *, media_type: Optional[str] = None
    ) -> DetectionResult:
        """
        The capability is present and the run did not complete.

        AVAILABLE + FAILED. This is a defect to be fixed, and it must stay
        visible: recording it as UNAVAILABLE would hide a broken detector behind
        what looks like a configuration gap. It is an absence of evidence and
        never evidence of absence.
        """
        return self._non_result(
            DetectorAvailability.AVAILABLE,
            DetectorExecutionStatus.FAILED,
            reason,
            media_type,
        )

    def failed_from(
        self, context: str, exc: BaseException, *, media_type: Optional[str] = None
    ) -> DetectionResult:
        """
        ``failed`` for a caught exception: the exception class in the reason, and
        an out-of-memory failure marked as such so the engine can retry it once.
        The message itself is logged, not recorded -- it can carry file paths.
        """
        from app.services.detection.runtime import is_memory_error

        memory = is_memory_error(exc)
        reason = f"{context}: {type(exc).__name__}" + (" (out of memory)" if memory else "")
        result = self.failed(reason, media_type=media_type)
        result.metadata["memory_error"] = memory
        return result

    def skipped(
        self, reason: str, *, media_type: Optional[str] = None
    ) -> DetectionResult:
        """
        Not attempted, though the detector works: unsupported media type, or
        disabled by configuration.

        AVAILABLE + SKIPPED.
        """
        return self._non_result(
            DetectorAvailability.AVAILABLE,
            DetectorExecutionStatus.SKIPPED,
            reason,
            media_type,
        )

    def inconclusive(
        self,
        reason: str,
        *,
        media_type: Optional[str] = None,
        regions: Optional[list[DetectionRegion]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> DetectionResult:
        """
        It ran to completion and cannot support a determination.

        AVAILABLE + INCONCLUSIVE. A real forensic outcome, and NOT a negative
        result: "I could not tell" must never be recorded as "nothing found".
        Regions are preserved, because observations can be worth an examiner's
        attention without supporting a conclusion.
        """
        return DetectionResult(
            detector=self.detector_id,
            detector_version=self.version,
            media_type=media_type or self._default_media_type(),
            manipulation_detected=False,
            confidence=0.0,
            manipulation_type=self.manipulation_type,
            availability=DetectorAvailability.AVAILABLE,
            execution_status=DetectorExecutionStatus.INCONCLUSIVE,
            score_semantics=ScoreSemantics.NOT_APPLICABLE,
            status_reason=reason,
            regions=regions or [],
            metadata={**(metadata or {}), "status_reason": reason},
        )

    # ── Internals ─────────────────────────────────────────────────────────────

    def _default_media_type(self) -> str:
        """The single supported type, or "unknown" when a detector handles several
        and the caller did not say which was being analysed."""
        if len(self.supported_media_types) == 1:
            return next(iter(self.supported_media_types))
        return "unknown"

    def _non_result(
        self,
        availability: DetectorAvailability,
        execution_status: DetectorExecutionStatus,
        reason: str,
        media_type: Optional[str],
    ) -> DetectionResult:
        """
        A result that carries no observation.

        ``confidence`` is 0.0 with NOT_APPLICABLE semantics: the zero is a
        placeholder, not a low score. Letting it default to HEURISTIC_SEVERITY
        would let a detector that never produced an observation be read as having
        measured "very little anomaly".
        """
        return DetectionResult(
            detector=self.detector_id,
            detector_version=self.version,
            media_type=media_type or self._default_media_type(),
            manipulation_detected=False,
            confidence=0.0,
            manipulation_type=self.manipulation_type,
            availability=availability,
            execution_status=execution_status,
            score_semantics=ScoreSemantics.NOT_APPLICABLE,
            status_reason=reason,
            regions=[],
            metadata={
                "status_reason": reason,
                "availability": availability.value,
                "execution_status": execution_status.value,
            },
        )
