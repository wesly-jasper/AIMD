"""
Image Detection Engine -- aggregates results from multiple image detectors.

WHY THERE IS NO COMPOSITE CONFIDENCE NUMBER
-------------------------------------------
This engine previously computed a weighted mean of the detectors' confidence
values and used it both as the reported ``overall_confidence`` and as the
threshold driving the assessment:

    overall_confidence = sum(r.confidence * weight) / sum(weight)
    if any_suspicious and overall_confidence >= 0.35: SUSPICIOUS
    elif overall_confidence < 0.20:                   CLEAN

That is invalid for three independent reasons:

1. **Incomparable scales.** ELA severity, copy-move RANSAC inlier counts, and a
   face-region heuristic are not measurements of one quantity. Averaging them
   produces a number with no referent.
2. **Non-independence.** ForensicImageDetector, ManipulationMaskGenerator and
   CopyMoveDetector all respond to JPEG block artefacts. Averaging correlated
   signals is not corroboration -- it counts one piece of evidence several times
   and inflates apparent agreement.
3. **Absence read as evidence.** A detector that could not run contributed 0.0,
   which pulled the mean down and could push the result below the 0.20 "CLEAN"
   threshold. A detector that never ran was therefore able to help establish that
   nothing was found.

So the engine now keeps every detector's score separate, attributed, and tagged
with its semantics, and reports an ordinal ``evidence_strength`` instead. The
assessment is derived from the *structure* of the observations -- how many
independent signal families fired, whether they agree, and what was missing --
not from a scalar.

ASYMMETRY OF MISSING DETECTORS (deliberate)
-------------------------------------------
Missing detectors are treated asymmetrically, which is the forensically correct
behaviour:

  * A positive observation stands even when other detectors failed. Missing data
    cannot retract an anomaly that was actually measured.
  * An *absence* of positive observations does NOT stand when detectors are
    missing -- you cannot report "no anomalies found" for checks you never ran.
    That case is INCONCLUSIVE.
"""
import logging
from pathlib import Path
from typing import Any

from app.schemas.detection import (
    Assessment,
    DetectionResult,
    DetectorAvailability,
    DetectorExecutionStatus,
    EvidenceStrength,
    ScoreSemantics,
)

logger = logging.getLogger(__name__)

#: Signal families. Detectors in the SAME family share an underlying physical
#: basis, so their agreement is NOT independent corroboration. Corroboration is
#: counted across families, never within one.
#:
#: Rationale for the groupings:
#:   compression  -- all read JPEG quantisation / block structure, so a single
#:                   recompression makes all of them fire together.
#:   geometric    -- keypoint correspondence; independent of compression history.
#:   face         -- semantic facial-region analysis.
#:   learned      -- end-to-end learned forgery localisation.
#:
#: Scope: image detectors only. ``BaselineVideoDetector`` and
#: ``BaselineAudioAnalyzer`` are deliberately absent -- neither is ever passed to
#: this engine (``detection_container`` routes them elsewhere, and
#: ``VideoDetectionEngine`` aggregates its own frame detectors by coverage rather
#: than by family). An unlisted detector is not silently trusted: ``_family``
#: returns "unclassified", which shares one family with every other unlisted
#: detector and so cannot corroborate independently. ``test_signal_family_map``
#: additionally fails if an image detector reaches the engine unclassified.
_SIGNAL_FAMILY: dict[str, str] = {
    "ForensicImageDetector": "compression",
    "ManipulationMaskGenerator": "compression",
    "CopyMoveDetector": "geometric",
    "FaceRegionAnalyzer": "face",
    "BaselineFaceDetector": "face",
    "TruForDetector": "learned",
    # Object presence is not manipulation evidence. A detector that finds a dog in
    # a photograph has established that there is a dog in the photograph. It is
    # kept out of every family so it can never contribute to an assessment.
    "YOLOv8ObjectDetector": "non_evidential",
    "BaselineObjectDetector": "non_evidential",
}
_NON_EVIDENTIAL = "non_evidential"


class ImageDetectionEngine:

    def __init__(self, detectors=None):
        self.detectors = detectors or []

    def detect(self, file_path: str | Path) -> dict[str, Any]:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not self.detectors:
            raise RuntimeError("No image detectors are configured")

        results: list[DetectionResult] = []
        for detector in self.detectors:
            name = type(detector).__name__
            try:
                result = detector.detect(file_path)
                if not isinstance(result, DetectionResult):
                    raise TypeError(
                        f"Detector {name} must return DetectionResult"
                    )
            except Exception as exc:
                logger.warning("Detector %s failed: %s", name, exc)
                # AVAILABLE + FAILED, not UNAVAILABLE. The detector was present
                # and was invoked; it raised. Recording this as UNAVAILABLE would
                # disguise a defect in our code as an absent dependency, and the
                # two have completely different remedies.
                results.append(DetectionResult(
                    detector=name,
                    media_type="image",
                    manipulation_detected=False,
                    confidence=0.0,
                    manipulation_type="unknown",
                    availability=DetectorAvailability.AVAILABLE,
                    execution_status=DetectorExecutionStatus.FAILED,
                    score_semantics=ScoreSemantics.NOT_APPLICABLE,
                    status_reason=f"{type(exc).__name__}: detector execution failed",
                    metadata={
                        "status": "failed",
                        "reason": f"{type(exc).__name__}: detector execution failed",
                    },
                ))
                continue
            results.append(result)

        aggregate = self._aggregate(results)

        return {
            "media_type": "image",
            "detections": results,
            **aggregate,
        }

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _family(detector_name: str) -> str:
        """
        The signal family a detector belongs to.

        Unmapped detectors all share one "unclassified" family, deliberately.
        Giving each unknown name its own family would let two detectors nobody
        has assessed for independence count as two corroborating families, which
        is exactly the unearned independence claim this map exists to prevent.
        Sharing one family under-counts corroboration instead, which is the safe
        direction.

        It is still a gap, so say so: ``TruForDetector`` was mapped here while
        the detector reported itself as ``TruFor``, and the "learned" family
        silently never matched anything. A warning makes the next such mismatch
        visible instead of letting it degrade the assessment quietly.
        """
        family = _SIGNAL_FAMILY.get(detector_name)
        if family is None:
            logger.warning(
                "Detector %r has no signal family; treated as 'unclassified', so "
                "it cannot corroborate independently. Add it to _SIGNAL_FAMILY.",
                detector_name,
            )
            return "unclassified"
        return family

    def _aggregate(self, results: list[DetectionResult]) -> dict[str, Any]:
        """
        Derive the assessment from the structure of the observations.

        Returns per-detector scores rather than a composite, plus an explicit
        account of what was missing and what disagreed, so that every conclusion
        can be traced to supporting and contradicting observations.
        """
        usable = [r for r in results if r.produced_observations]
        incomplete = [r for r in results if not r.produced_observations]

        # Evidential usable results only -- a non-evidential detector must not be
        # able to influence the assessment in either direction.
        evidential = [
            r for r in usable if self._family(r.detector) != _NON_EVIDENTIAL
        ]

        positive = [r for r in evidential if r.manipulation_detected]
        negative = [
            r for r in evidential
            if not r.manipulation_detected
            and r.execution_status is DetectorExecutionStatus.COMPLETED
        ]

        positive_families = {self._family(r.detector) for r in positive}
        contradicted = bool(positive) and bool(negative)

        # Scores stay attributed and tagged. Never summed.
        detector_scores = [
            {
                "detector": r.detector,
                "score": r.confidence,
                "score_semantics": r.score_semantics.value,
                "signal_family": self._family(r.detector),
                "manipulation_detected": r.manipulation_detected,
            }
            for r in evidential
        ]

        missing = [
            {
                "detector": r.detector,
                "availability": r.availability.value,
                "execution_status": r.execution_status.value,
                "reason": r.status_reason,
            }
            for r in incomplete
        ]

        assessment, strength, rationale = self._classify(
            evidential=evidential,
            positive=positive,
            negative=negative,
            positive_families=positive_families,
            contradicted=contradicted,
            incomplete=incomplete,
        )

        all_regions: list[Any] = []
        for r in results:
            all_regions.extend(r.regions)

        return {
            "assessment": assessment.value,
            "evidence_strength": strength.value,
            "assessment_rationale": rationale,
            "supporting_observations": [r.detector for r in positive],
            "contradicting_observations": [r.detector for r in negative],
            "detector_scores": detector_scores,
            "incomplete_detectors": missing,
            "analysis_complete": not incomplete,
            # Retained for traceability: which detector reported the strongest
            # signal. This is NOT a composite and carries that detector's
            # semantics only.
            "dominant_signal": (
                max(positive, key=lambda r: r.confidence).detector if positive else None
            ),
            "all_regions": all_regions,
        }

    def _classify(
        self,
        *,
        evidential: list[DetectionResult],
        positive: list[DetectionResult],
        negative: list[DetectionResult],
        positive_families: set[str],
        contradicted: bool,
        incomplete: list[DetectionResult],
    ) -> tuple[Assessment, EvidenceStrength, str]:
        # 1. Nothing usable ran -> no determination is possible.
        if not evidential:
            return (
                Assessment.INCONCLUSIVE,
                EvidenceStrength.NONE,
                "No detector produced usable observations, so no determination "
                "about this image is supported.",
            )

        # 2. Positive observations stand regardless of what else was missing.
        if positive:
            independent = len(positive_families)
            localised = any(r.regions for r in positive)

            if contradicted:
                strength = EvidenceStrength.CONTRADICTORY
            elif independent >= 2 and localised:
                strength = EvidenceStrength.STRONG
            elif independent >= 2:
                strength = EvidenceStrength.MODERATE
            else:
                strength = EvidenceStrength.WEAK

            # Escalation beyond SUSPICIOUS requires corroboration from genuinely
            # independent signal families that agree. A single family firing --
            # however high its score -- supports review, not a conclusion.
            if independent >= 2 and not contradicted and localised:
                assessment = Assessment.LIKELY_MANIPULATED
                rationale = (
                    f"{independent} independent signal families "
                    f"({', '.join(sorted(positive_families))}) reported anomalies "
                    "that localise to specific regions."
                )
            else:
                assessment = Assessment.SUSPICIOUS
                reasons = [
                    f"{len(positive)} detector(s) reported anomalies "
                    f"across {independent} signal family/families"
                ]
                if contradicted:
                    reasons.append(
                        f"{len(negative)} detector(s) reported no anomalies, so the "
                        "observations conflict and are reported rather than reconciled"
                    )
                if independent < 2:
                    reasons.append(
                        "a single signal family is not independent corroboration, so "
                        "this supports examiner review rather than a conclusion"
                    )
                rationale = "; ".join(reasons) + "."

            if incomplete:
                rationale += (
                    f" {len(incomplete)} detector(s) did not produce observations; "
                    "this does not retract the anomalies that were measured."
                )
            return assessment, strength, rationale

        # 3. No positive observations, but coverage was incomplete. "We found
        #    nothing" cannot be reported for checks that never ran.
        if incomplete:
            names = ", ".join(sorted(r.detector for r in incomplete))
            return (
                Assessment.INCONCLUSIVE,
                EvidenceStrength.NONE,
                "No anomalies were observed by the detectors that completed, but "
                f"{len(incomplete)} detector(s) did not produce observations "
                f"({names}). Absence of anomalies cannot be reported for checks "
                "that did not run.",
            )

        # 4. No positive observations and full coverage. This is the strongest
        #    negative statement the system is entitled to make, and it is still
        #    not a finding of authenticity.
        return (
            Assessment.NO_SIGNIFICANT_ANOMALIES_DETECTED,
            EvidenceStrength.NONE,
            f"All {len(evidential)} evidential detector(s) completed and none "
            "reported anomalies above their thresholds. This records what was "
            "looked for and not found; it is not a finding of authenticity, and "
            "manipulation that these detectors cannot measure would not appear "
            "here.",
        )
