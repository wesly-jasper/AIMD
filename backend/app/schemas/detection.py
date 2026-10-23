"""
Detection contracts.

THE CENTRAL DISTINCTION IN THIS MODULE
--------------------------------------
A detector reports two *independent* facts, and collapsing them is a forensic
misrepresentation:

  1. Availability  -- could this detector run at all?       (AVAILABLE / UNAVAILABLE)
  2. Execution     -- what happened when it ran?            (COMPLETED / INCONCLUSIVE
                                                             / FAILED / SKIPPED)

They are independent because the interesting states are the mixed ones:

  * AVAILABLE   + FAILED        -- the model was present and the run crashed.
    Reporting this as UNAVAILABLE would hide a defect.
  * AVAILABLE   + COMPLETED + INCONCLUSIVE
                                -- it ran cleanly and still cannot support a
                                   determination. This is a real forensic outcome,
                                   not a negative result.
  * UNAVAILABLE + SKIPPED       -- it never ran. It must never be reported as
                                   having found nothing.

Neither field may be converted into a fake/real label. "The detector did not run"
and "the detector found no anomalies" are different statements about the evidence,
and only the second one is an observation at all.

A third fact is required to read ``confidence`` honestly: what KIND of number it
is. A raw model logit, a calibrated probability, and a hand-tuned heuristic
severity are not comparable, cannot be averaged, and must not be rendered to a
user as the same thing. See ``ScoreSemantics``.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator


class DetectorAvailability(str, Enum):
    """Whether a detector *could run*. Says nothing about what it found."""

    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"


class DetectorExecutionStatus(str, Enum):
    """
    What happened during the run. Independent of availability.

    COMPLETED     -- ran to completion and produced usable observations.
    INCONCLUSIVE  -- ran to completion but cannot support a determination.
                     NOT a negative result.
    FAILED        -- raised, timed out, or produced unusable output. A failure is
                     an absence of evidence, never evidence of absence.
    SKIPPED       -- never attempted (unavailable, unsupported media type, or
                     disabled by configuration).
    """

    COMPLETED = "COMPLETED"
    INCONCLUSIVE = "INCONCLUSIVE"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class ScoreSemantics(str, Enum):
    """
    How ``DetectionResult.confidence`` must be interpreted.

    Values with different semantics are NOT comparable and MUST NOT be averaged,
    summed, or presented on a shared scale.

    RAW_MODEL_SCORE         -- unitless model output. Not a probability. No
                               calibration has been performed, so 0.8 does not
                               mean "80% likely".
    CALIBRATED_PROBABILITY  -- validated against a labelled dataset with a
                               recorded calibration procedure. Nothing in this
                               codebase is currently entitled to this value.
    HEURISTIC_SEVERITY      -- a hand-tuned score from classical signal
                               measurements (ELA, FFT, noise residual). Ordinal at
                               best: higher means "more anomalous", with no
                               probabilistic meaning.
    NOT_APPLICABLE          -- no meaningful score (detector unavailable, skipped,
                               or failed). ``confidence`` is 0.0 as a placeholder
                               and carries no information.
    """

    RAW_MODEL_SCORE = "RAW_MODEL_SCORE"
    CALIBRATED_PROBABILITY = "CALIBRATED_PROBABILITY"
    HEURISTIC_SEVERITY = "HEURISTIC_SEVERITY"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class Assessment(str, Enum):
    """
    Assessment categories.

    The first five are the canonical categories and the only values permitted for
    an *evidence-level* assessment:

      NO_SIGNIFICANT_ANOMALIES_DETECTED
          The detectors that ran did not surface anomalies above their thresholds.
          This is a statement about what was looked for and not found. It is NOT a
          finding of authenticity, and it must never be rendered as "authentic",
          "genuine", "clean", or "no manipulation".
      SUSPICIOUS
          Anomalies were observed that warrant examiner review, without supporting
          a specific manipulation conclusion.
      LIKELY_MANIPULATED
          Observations support localised editing of otherwise captured media.
      LIKELY_AI_GENERATED
          Observations support wholly synthetic generation.
      INCONCLUSIVE
          Analysis cannot support any of the above. Reached when detectors were
          unavailable, failed, or returned conflicting observations. Must never be
          downgraded to NO_SIGNIFICANT_ANOMALIES_DETECTED.

    The remaining two members are DEPRECATED detector-level values retained only
    so that not-yet-migrated detectors keep type-checking during the contract
    migration. New code must not emit them:

      CLEAN        -> use NO_SIGNIFICANT_ANOMALIES_DETECTED (evidence level), or
                      ``execution_status=COMPLETED`` (detector level).
      UNAVAILABLE  -> use ``availability=UNAVAILABLE``.

    ``canonical()`` maps the deprecated values onto the canonical set.
    """

    NO_SIGNIFICANT_ANOMALIES_DETECTED = "NO_SIGNIFICANT_ANOMALIES_DETECTED"
    SUSPICIOUS = "SUSPICIOUS"
    LIKELY_MANIPULATED = "LIKELY_MANIPULATED"
    LIKELY_AI_GENERATED = "LIKELY_AI_GENERATED"
    INCONCLUSIVE = "INCONCLUSIVE"

    # --- deprecated, migration only -------------------------------------------
    CLEAN = "CLEAN"
    UNAVAILABLE = "UNAVAILABLE"

    @classmethod
    def canonical_values(cls) -> frozenset[str]:
        """The five categories permitted for an evidence-level assessment."""
        return frozenset(
            {
                cls.NO_SIGNIFICANT_ANOMALIES_DETECTED.value,
                cls.SUSPICIOUS.value,
                cls.LIKELY_MANIPULATED.value,
                cls.LIKELY_AI_GENERATED.value,
                cls.INCONCLUSIVE.value,
            }
        )

    def canonical(self) -> "Assessment":
        """
        Map a deprecated detector-level value onto the canonical category set.

        UNAVAILABLE becomes INCONCLUSIVE rather than
        NO_SIGNIFICANT_ANOMALIES_DETECTED: a detector that could not run has not
        established the absence of anomalies.
        """
        if self is Assessment.CLEAN:
            return Assessment.NO_SIGNIFICANT_ANOMALIES_DETECTED
        if self is Assessment.UNAVAILABLE:
            return Assessment.INCONCLUSIVE
        return self

    @property
    def is_deprecated(self) -> bool:
        return self in (Assessment.CLEAN, Assessment.UNAVAILABLE)


#: Assessments that must never be presented as evidence of authenticity.
#: Kept as data so UI and report code cannot drift from the rule.
NON_EXONERATING_ASSESSMENTS: frozenset[Assessment] = frozenset(
    {
        Assessment.NO_SIGNIFICANT_ANOMALIES_DETECTED,
        Assessment.INCONCLUSIVE,
    }
)


class DetectionRegion(BaseModel):
    """A localized region of forensic interest within a frame or audio segment."""

    type: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: Optional[list[float]] = None  # [x, y, width, height] in pixels
    mask_path: Optional[str] = None
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None
    start_timestamp: Optional[float] = None  # seconds
    end_timestamp: Optional[float] = None  # seconds
    extra: Optional[dict[str, Any]] = None


class DetectionResult(BaseModel):
    """
    Output from a single forensic detector.

    ``availability`` and ``execution_status`` are the authoritative fields. Read
    those, not ``manipulation_detected``, which is a lossy convenience flag:
    ``False`` means "no anomaly above threshold" AND "could not tell" AND "did not
    run", which are three different things.

    ``confidence`` is meaningless without ``score_semantics``. A result whose
    semantics are NOT_APPLICABLE has no score, and its 0.0 must not enter any
    aggregate.

    ``assessment`` is DEPRECATED at the detector level and is derived from the two
    status fields when it is not supplied. It remains only for the duration of the
    contract migration.
    """

    detector: str
    media_type: str
    manipulation_detected: bool
    confidence: float = Field(ge=0.0, le=1.0)
    manipulation_type: str

    availability: DetectorAvailability = DetectorAvailability.AVAILABLE
    execution_status: DetectorExecutionStatus = DetectorExecutionStatus.COMPLETED
    score_semantics: ScoreSemantics = ScoreSemantics.HEURISTIC_SEVERITY

    #: Why the detector is unavailable / failed / skipped. Required reading for an
    #: examiner: "no checkpoint on disk" and "CUDA out of memory" are both
    #: UNAVAILABLE but have different consequences for the investigation.
    status_reason: Optional[str] = None

    #: Identity and version of the code and weights that produced this result,
    #: so a run can be reproduced. Populated by the detector.
    detector_version: Optional[str] = None

    assessment: Assessment = Assessment.INCONCLUSIVE
    regions: list[DetectionRegion] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _reconcile_status_fields(self) -> "DetectionResult":
        """
        Bridge not-yet-migrated detectors onto the status fields.

        A detector that still passes only the deprecated ``assessment`` gets
        correct ``availability`` / ``execution_status`` derived from it, so the
        honest contract holds repo-wide before every detector has been rewritten.
        Detectors that set the status fields explicitly are left alone.
        """
        fields_set = self.model_fields_set
        set_availability = "availability" in fields_set
        set_execution = "execution_status" in fields_set
        set_assessment = "assessment" in fields_set

        if set_assessment and not (set_availability or set_execution):
            # Legacy caller: derive the authoritative fields from the old label.
            if self.assessment is Assessment.UNAVAILABLE:
                object.__setattr__(self, "availability", DetectorAvailability.UNAVAILABLE)
                object.__setattr__(
                    self, "execution_status", DetectorExecutionStatus.SKIPPED
                )
                object.__setattr__(self, "score_semantics", ScoreSemantics.NOT_APPLICABLE)
            elif self.assessment is Assessment.INCONCLUSIVE:
                object.__setattr__(
                    self, "execution_status", DetectorExecutionStatus.INCONCLUSIVE
                )
            else:
                object.__setattr__(
                    self, "execution_status", DetectorExecutionStatus.COMPLETED
                )
        elif not set_assessment:
            # Migrated caller: derive the deprecated label for consumers that
            # still read it.
            object.__setattr__(self, "assessment", self._derived_assessment())

        # A detector that did not run has no score, whatever it passed.
        if not self.produced_observations and self.confidence != 0.0:
            object.__setattr__(self, "confidence", 0.0)
            object.__setattr__(self, "score_semantics", ScoreSemantics.NOT_APPLICABLE)

        return self

    def _derived_assessment(self) -> Assessment:
        if self.availability is DetectorAvailability.UNAVAILABLE:
            return Assessment.UNAVAILABLE
        if self.execution_status is not DetectorExecutionStatus.COMPLETED:
            return Assessment.INCONCLUSIVE
        return Assessment.SUSPICIOUS if self.manipulation_detected else Assessment.CLEAN

    @property
    def produced_observations(self) -> bool:
        """
        True only when this result may contribute evidence.

        Everything else -- unavailable, failed, skipped -- is an absence of
        evidence and must be excluded from aggregation rather than counted as a
        zero score, which would read as "looked and found nothing".
        """
        return (
            self.availability is DetectorAvailability.AVAILABLE
            and self.execution_status
            in (DetectorExecutionStatus.COMPLETED, DetectorExecutionStatus.INCONCLUSIVE)
        )

    @property
    def supports_determination(self) -> bool:
        """True when this result can support a conclusion, not merely review."""
        return (
            self.availability is DetectorAvailability.AVAILABLE
            and self.execution_status is DetectorExecutionStatus.COMPLETED
        )


class DetectionRequest(BaseModel):
    """
    Request to run detection on already-ingested media.

    Addresses evidence by identity, never by location. ``media_type`` is NOT
    accepted from the client: it is read from the stored media record, where it was
    determined from the file's magic bytes at ingest. Letting a caller declare the
    media type would allow evidence to be mislabelled and routed to the wrong
    detectors.
    """

    media_id: str = Field(min_length=1)


class AnalysisRequest(BaseModel):
    """Request to run the full analysis pipeline on an already-uploaded media file."""

    media_id: str
    # Where the user says they found the image. Fetched through the SSRF guard
    # and checked like any other source; never trusted as a source by itself.
    source_url: Optional[str] = Field(default=None, max_length=2048)


class InvestigationRequest(BaseModel):
    """Optional input to POST /analysis/{id}/investigate."""
    # A URL the agent may check with verify_source_match, through the SSRF
    # guard, alongside those already in the analysis's results.
    source_url: Optional[str] = Field(
        default=None, min_length=10, max_length=2048, pattern=r"^https?://\S+$"
    )


class AnalysisResponse(BaseModel):
    """
    Response from a pipeline analysis request.

    There is deliberately no blended numeric confidence. A single number produced
    by averaging unrelated detector scores implies the detectors are independent
    measurements of one quantity, which they are not: ELA severity, copy-move
    keypoint matches, and a face-region score measure different things on
    incomparable scales, and several share the same underlying JPEG artefacts.
    Averaging them manufactures precision that the evidence does not support.

    ``evidence_strength`` is an ordinal summary of how well the observations
    support the stated assessment. It is not a probability and must not be
    rendered as a percentage.
    """

    analysis_id: str
    media_id: str
    status: str  # QUEUED | PROCESSING | COMPLETED | FAILED | PARTIAL | INCONCLUSIVE
    assessment: Optional[str] = None
    evidence_strength: Optional[str] = None  # see EvidenceStrength
    message: Optional[str] = None


class EvidenceStrength(str, Enum):
    """
    Ordinal support for an assessment. Not a probability, not averageable.

    NONE          -- no detector produced usable observations.
    WEAK          -- a single heuristic signal, or signals with benign explanations.
    MODERATE      -- multiple independent signals agree.
    STRONG        -- multiple independent signals agree and localise consistently.
    CONTRADICTORY -- usable observations point in conflicting directions. This is
                     reported, never averaged away.
    """

    NONE = "NONE"
    WEAK = "WEAK"
    MODERATE = "MODERATE"
    STRONG = "STRONG"
    CONTRADICTORY = "CONTRADICTORY"
