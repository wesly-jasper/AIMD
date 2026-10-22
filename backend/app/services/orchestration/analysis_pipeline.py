"""
Analysis Pipeline -- central orchestrator for the AIMD forensic pipeline.

Connects all pipeline stages under a common analysis_id:

  ingest -> metadata -> preprocess -> frames -> keyframes ->
  detection -> localization -> fingerprint -> similarity ->
  source tracing -> provenance -> evidence -> report -> persist

Design principles:
  - All stages use the same analysis_id.
  - Failures in non-critical stages mark analysis as PARTIAL, not FAILED.
  - No stage fabricates data -- all outputs must come from real computation.
  - The pipeline is synchronous (Phase 1). Designed for async swap-in:
      API -> AnalysisPipeline.run() -> persist
    Can become:
      API -> queue job -> Worker -> AnalysisPipeline.run() -> persist
    without changing the pipeline itself.
"""
import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from sqlalchemy.orm import Session

from app.core.config import settings
from app.db import repository as repo
from app.db.models import AnalysisRecord
from app.schemas.detection import Assessment, EvidenceStrength
from app.services.detection.engine.image_detection_engine import ImageDetectionEngine
from app.services.preprocessing.metadata_extractor import MetadataExtractor
from app.services.preprocessing.frame_extractor import FrameExtractor
from app.services.preprocessing.keyframe_selector import KeyframeSelector
from app.services.fingerprinting.fingerprint import FingerprintService
from app.services.similarity.similarity_engine import SimilarityEngine
from app.services.provenance.source_tracer import SourceTracer
from app.services.provenance.graph import build_provenance_graph
from app.services.evidence.evidence_service import EvidenceService
from app.services.evidence.report_generator import ReportGenerator
from app.services.detection.detection_container import DetectionContainer

logger = logging.getLogger(__name__)

_COMPLETE_NEGATIVE = {
    Assessment.CLEAN.value,
    Assessment.NO_SIGNIFICANT_ANOMALIES_DETECTED.value,
}
_SUSPICIOUS_VALUES = {
    Assessment.SUSPICIOUS.value,
    Assessment.LIKELY_MANIPULATED.value,
    Assessment.LIKELY_AI_GENERATED.value,
}

#: Settings that change what an analysis can observe, recorded on every run.
#: An allow-list, not a deny-list: a new setting holding a credential must not
#: reach the database merely because nobody remembered to exclude it.
_CONFIG_SNAPSHOT_KEYS = (
    "trufor_enabled",
    "copymove_enabled",
    "copymove_detector_type",
    "manipulation_mask_enabled",
    "face_region_analyzer_enabled",
    "face_region_use_mediapipe",
    "analysis_fps",
    "max_keyframes",
    "max_image_dimension",
    "max_video_duration_seconds",
    "max_audio_duration_seconds",
    "device",
)


def _plain_value(raw: Any) -> str:
    return raw.value if hasattr(raw, "value") else str(raw)


@dataclass(frozen=True)
class AssessmentDecision:
    """The analysis-level assessment and why it was reached."""
    category: str
    evidence_strength: str
    #: Stable identifier of the branch that decided it.
    rule: str
    rationale: str
    #: Evidential signal families with an anomaly observation.
    families: frozenset = field(default_factory=frozenset)


def observation_polarity(detection) -> Optional[str]:
    """
    What a recorded detection observed, or None if it observed nothing.

    None -- UNAVAILABLE, FAILED or SKIPPED -- is a coverage gap. It is never
    mapped to NO_ANOMALY_OBSERVED: a check that did not run found nothing only
    in the sense that it did not look.
    """
    availability = _plain_value(getattr(detection, "availability", "AVAILABLE"))
    execution = _plain_value(getattr(detection, "execution_status", "COMPLETED"))
    if availability == "UNAVAILABLE" or execution in ("FAILED", "SKIPPED"):
        return None
    label = _plain_value(detection.assessment)
    if label == "UNAVAILABLE":
        return None
    if label in _SUSPICIOUS_VALUES:
        return "ANOMALY_OBSERVED"
    if label in _COMPLETE_NEGATIVE and execution == "COMPLETED":
        return "NO_ANOMALY_OBSERVED"
    return "INDETERMINATE"


class AnalysisPipeline:
    """
    Runs the complete AIMD forensic analysis pipeline.

    Usage:
        pipeline = AnalysisPipeline()
        result = pipeline.run(media_id="...", db=db_session)
    """

    def __init__(self):
        self._metadata_extractor = MetadataExtractor()
        self._frame_extractor = FrameExtractor(output_dir=settings.frames_dir)
        self._keyframe_selector = KeyframeSelector(output_dir=settings.keyframes_dir)
        self._fingerprint_service = FingerprintService()
        self._similarity_engine = SimilarityEngine()
        self._source_tracer = SourceTracer()
        self._evidence_service = EvidenceService()
        self._report_generator = ReportGenerator()
        self._detection_container = DetectionContainer()

    def run(
        self, media_id: str, db: Session, requested_by: Optional[str] = None
    ) -> dict[str, Any]:
        """
        Run the complete pipeline for the given media_id.

        Creates an AnalysisRecord in the DB before starting, updates it
        as each stage completes, and finalizes with COMPLETED or PARTIAL.

        Returns the analysis_id and final status.
        """
        # Retrieve media
        media = repo.get_media(db, media_id)
        if not media:
            raise ValueError(f"Media not found: {media_id}")

        file_path = Path(media.file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Media file missing from storage: {file_path}")

        # Create analysis record
        analysis_id = str(uuid.uuid4())
        started = time.monotonic()
        repo.create_analysis(db, {
            "id": analysis_id,
            "media_id": media_id,
            "status": "PROCESSING",
            "requested_by": requested_by,
            "software_version": settings.app_version,
            "config_snapshot": self._config_snapshot(),
        })

        stage_failures: list[str] = []

        try:
            # ── Stage 1: Metadata extraction ──────────────────────────────────
            logger.info("[%s] Stage 1: Metadata extraction", analysis_id)
            metadata = self._run_metadata(file_path, media_id, db, stage_failures)

            # ── Stage 2: Frame extraction (video only) ────────────────────────
            frames: list[dict] = []
            keyframes: list[dict] = []
            if media.media_type == "video":
                logger.info("[%s] Stage 2: Frame extraction", analysis_id)
                frames, keyframes = self._run_frame_extraction(
                    file_path, media_id, db, stage_failures
                )

            # ── Stage 3: Detection ────────────────────────────────────────────
            logger.info("[%s] Stage 3: Detection", analysis_id)
            self._run_detection(
                file_path, media_id, analysis_id, media.media_type,
                frames, db, stage_failures
            )

            # ── Stage 4: Fingerprinting ───────────────────────────────────────
            logger.info("[%s] Stage 4: Fingerprinting", analysis_id)
            self._run_fingerprinting(file_path, media_id, analysis_id, db, stage_failures)

            # ── Stage 5: Similarity search ────────────────────────────────────
            logger.info("[%s] Stage 5: Similarity search", analysis_id)
            self._run_similarity(file_path, media_id, analysis_id, db, stage_failures)

            # ── Stage 6: Source tracing ───────────────────────────────────────
            logger.info("[%s] Stage 6: Source tracing", analysis_id)
            self._run_source_tracing(file_path, analysis_id, db, stage_failures)

            # ── Stage 7: Provenance ───────────────────────────────────────────
            logger.info("[%s] Stage 7: Provenance", analysis_id)
            self._run_provenance(media_id, analysis_id, db, stage_failures)

            # ── Stage 8: Evidence items ───────────────────────────────────────
            logger.info("[%s] Stage 8: Evidence generation", analysis_id)
            self._run_evidence(analysis_id, db, stage_failures)

            # ── Stage 9: Report ───────────────────────────────────────────────
            logger.info("[%s] Stage 9: Report generation", analysis_id)
            self._run_report(analysis_id, db, stage_failures)

            # ── Finalize ──────────────────────────────────────────────────────
            final_status = "PARTIAL" if stage_failures else "COMPLETED"
            assessment, evidence_strength = self._compute_final_assessment(
                analysis_id, db
            )
            self._run_forensic_record(analysis_id, db, stage_failures)
            if stage_failures:
                final_status = "PARTIAL"
            self._finish_run(analysis_id, db, started)

            repo.update_analysis_status(
                db, analysis_id,
                status=final_status,
                assessment=assessment,
                evidence_strength=evidence_strength,
            )

            logger.info(
                "[%s] Pipeline complete -- status=%s assessment=%s strength=%s",
                analysis_id, final_status, assessment, evidence_strength
            )

            return {
                "analysis_id": analysis_id,
                "media_id": media_id,
                "status": final_status,
                "assessment": assessment,
                "evidence_strength": evidence_strength,
                "stage_failures": stage_failures,
            }

        except Exception as exc:
            logger.error("[%s] Pipeline FAILED: %s", analysis_id, exc, exc_info=True)
            db.rollback()
            repo.update_analysis_status(
                db, analysis_id, status="FAILED", error_message=str(exc)
            )
            try:
                self._finish_run(analysis_id, db, started)
            except Exception:
                logger.exception("[%s] Could not record run provenance", analysis_id)
            return {
                "analysis_id": analysis_id,
                "media_id": media_id,
                "status": "FAILED",
                "error": str(exc),
                "stage_failures": stage_failures,
            }

    # ── Stage implementations ─────────────────────────────────────────────────

    def _run_metadata(
        self,
        file_path: Path,
        media_id: str,
        db: Session,
        failures: list,
    ) -> dict:
        try:
            metadata = self._metadata_extractor.extract(file_path)
            repo.upsert_media_metadata(db, media_id, {
                "format": metadata.get("format"),
                "width": metadata.get("width"),
                "height": metadata.get("height"),
                "duration_seconds": metadata.get("duration_seconds"),
                "fps": metadata.get("fps"),
                "frame_count": metadata.get("frame_count"),
                "codec": metadata.get("codec"),
                "sample_rate": metadata.get("sample_rate"),
                "channels": metadata.get("channels"),
                "exif_data": metadata.get("exif", {}),
                "raw_metadata": metadata,
            })
            return metadata
        except Exception as exc:
            logger.warning("Metadata extraction failed: %s", exc)
            failures.append(f"metadata: {exc}")
            return {}

    def _run_frame_extraction(
        self,
        file_path: Path,
        media_id: str,
        db: Session,
        failures: list,
    ) -> tuple[list, list]:
        try:
            result = self._frame_extractor.extract(
                file_path,
                media_id=media_id,
                target_fps=settings.analysis_fps,
            )
            frames = result.get("frames", [])
            if frames:
                repo.bulk_create_frames(db, frames)

            kf_result = self._keyframe_selector.select_from_frame_list(
                frames, media_id, max_keyframes=settings.max_keyframes
            )
            keyframes = kf_result.get("keyframes", [])
            # Mark keyframes in DB
            for kf in keyframes:
                # Update the frame record -- find it by frame_index
                frame_recs = repo.get_frames_for_media(db, media_id)
                for fr in frame_recs:
                    if fr.frame_index == kf["frame_index"]:
                        fr.is_keyframe = True
                        fr.scene_score = kf.get("scene_score")
                        db.commit()
                        break

            return frames, keyframes
        except Exception as exc:
            logger.warning("Frame extraction failed: %s", exc)
            failures.append(f"frame_extraction: {exc}")
            return [], []

    def _run_detection(
        self,
        file_path: Path,
        media_id: str,
        analysis_id: str,
        media_type: str,
        frames: list,
        db: Session,
        failures: list,
    ) -> None:
        try:
            router = self._detection_container.router
            detection_result = router.route(str(file_path), media_type)
            detections = detection_result.get("detections", [])

            for det in detections:
                if hasattr(det, "model_dump"):
                    det_data = det.model_dump()
                elif hasattr(det, "dict"):
                    det_data = det.dict()
                else:
                    det_data = det

                regions_data = [
                    {
                        "region_type": r.get("type", "unknown"),
                        "confidence": r.get("confidence", 0.0),
                        "bbox": r.get("bbox"),
                        "start_timestamp": r.get("start_timestamp"),
                        "end_timestamp": r.get("end_timestamp"),
                    }
                    for r in det_data.get("regions", [])
                ]
                meta = det_data.get("metadata", {})

                def _plain(value, default):
                    if value is None:
                        return default
                    return value.value if hasattr(value, "value") else str(value)

                assessment_val = _plain(
                    det_data.get("assessment") or meta.get("assessment"),
                    "INCONCLUSIVE",
                )
                availability = _plain(det_data.get("availability"), "AVAILABLE")
                execution_status = _plain(
                    det_data.get("execution_status"), "COMPLETED"
                )
                score_semantics = _plain(
                    det_data.get("score_semantics"), "HEURISTIC_SEVERITY"
                )
                status_reason = det_data.get("status_reason") or meta.get("reason")

                # A stage failure is recorded whenever the detector did not
                # produce usable observations. Both UNAVAILABLE (no capability)
                # and FAILED (broken capability) qualify, and they are reported
                # distinctly so the cause is not lost.
                if execution_status in ("FAILED", "SKIPPED") or meta.get(
                    "failure_messages"
                ):
                    failures.append(
                        f"detection: {det_data.get('detector', 'unknown')} incomplete "
                        f"[{availability}/{execution_status}]: "
                        f"{status_reason or meta.get('failure_messages') or 'detector did not complete'}"
                    )

                repo.create_detection(db, {
                    "analysis_id": analysis_id,
                    "detector": det_data.get("detector", "unknown"),
                    "detector_version": (
                        det_data.get("detector_version")
                        or meta.get("detector_version")
                    ),
                    "media_type": det_data.get("media_type", media_type),
                    "manipulation_type": det_data.get("manipulation_type", "unknown"),
                    "availability": availability,
                    "execution_status": execution_status,
                    "score_semantics": score_semantics,
                    "status_reason": status_reason,
                    "assessment": assessment_val,
                    "manipulation_detected": det_data.get("manipulation_detected", False),
                    "confidence": det_data.get("confidence", 0.0),
                    "detector_metadata": meta,
                    "suspicious_segments": detection_result.get("suspicious_segments"),
                    "regions": regions_data,
                })
        except Exception as exc:
            logger.warning("Detection stage failed: %s", exc)
            failures.append(f"detection: {exc}")

    def _run_fingerprinting(
        self,
        file_path: Path,
        media_id: str,
        analysis_id: str,
        db: Session,
        failures: list,
    ) -> None:
        try:
            fp_result = self._fingerprint_service.generate(
                file_path, media_id=media_id, analysis_id=analysis_id
            )
            records = fp_result.get("_records", [])
            if records:
                repo.bulk_create_fingerprints(db, records)
        except Exception as exc:
            logger.warning("Fingerprinting failed: %s", exc)
            failures.append(f"fingerprinting: {exc}")

    def _run_similarity(
        self,
        file_path: Path,
        media_id: str,
        analysis_id: str,
        db: Session,
        failures: list,
    ) -> None:
        try:
            media = repo.get_media(db, media_id)
            fingerprints = repo.get_fingerprints_for_media(db, media_id)
            fp_dict = {fp.algorithm: fp.value for fp in fingerprints}

            result = self._similarity_engine.search(
                str(file_path),
                query_fingerprints=fp_dict,
                db=db,
                exclude_media_id=media_id,
                owner=media.uploaded_by if media else None,
            )

            matches_to_save = []
            for i, m in enumerate(result.get("local_matches", [])):
                matches_to_save.append({
                    "analysis_id": analysis_id,
                    "match_type": "local",
                    "source_media_id": m.get("source_media_id"),
                    "external_url": None,
                    "similarity": m.get("similarity", 0.0),
                    "method": m.get("method", "perceptual"),
                    "algorithm": m.get("algorithm"),
                    "rank": i + 1,
                })
            for i, m in enumerate(result.get("external_matches", [])):
                matches_to_save.append({
                    "analysis_id": analysis_id,
                    "match_type": "external",
                    "source_media_id": None,
                    "external_url": m.get("external_url"),
                    "similarity": m.get("similarity", 0.0),
                    "method": "internet",
                    "algorithm": m.get("algorithm"),
                    "rank": i + 1,
                })

            if matches_to_save:
                repo.bulk_create_similarity_matches(db, matches_to_save)
        except Exception as exc:
            logger.warning("Similarity search failed: %s", exc)
            failures.append(f"similarity: {exc}")

    def _run_source_tracing(
        self,
        file_path: Path,
        analysis_id: str,
        db: Session,
        failures: list,
    ) -> None:
        try:
            trace_result = self._source_tracer.trace_source(str(file_path))
            candidates = trace_result.get("candidates", [])
            if not candidates:
                return

            sources_to_save = []
            for i, c in enumerate(candidates):
                sources_to_save.append({
                    "analysis_id": analysis_id,
                    "url": c.get("url"),
                    "title": c.get("title"),
                    "domain": c.get("domain"),
                    "similarity": c.get("similarity"),
                    "provider": c.get("provider", "unknown"),
                    "retrieval_status": "FOUND",
                    "is_earliest_known": i == 0,  # First is earliest known
                })
            if sources_to_save:
                repo.bulk_create_sources(db, sources_to_save)
        except Exception as exc:
            logger.warning("Source tracing failed: %s", exc)
            failures.append(f"source_tracing: {exc}")

    def _run_provenance(
        self,
        media_id: str,
        analysis_id: str,
        db: Session,
        failures: list,
    ) -> None:
        try:
            media = repo.get_media(db, media_id)
            sources = repo.get_sources_for_analysis(db, analysis_id)
            similarity_matches = repo.get_similarity_matches_for_analysis(db, analysis_id)

            # Build provenance graph nodes
            nodes = [
                {
                    "node_id": media_id,
                    "label": media.original_filename if media else "Uploaded Media",
                    "node_type": "uploaded",
                    "node_metadata": {"sha256": media.sha256 if media else None},
                }
            ]
            edges = []

            for src in sources:
                src_node_id = f"source_{src.id}"
                nodes.append({
                    "node_id": src_node_id,
                    "label": src.title or src.url or "Unknown Source",
                    "node_type": "source",
                    "node_metadata": {"url": src.url, "domain": src.domain},
                })
                edges.append({
                    "source_node_id": src_node_id,
                    "target_node_id": media_id,
                    "relationship": "reposted_from" if src.similarity and src.similarity > 0.95 else "visually_similar_to",
                    "confidence": src.similarity or 0.5,
                    "evidence": f"Similarity: {src.similarity}, Provider: {src.provider}",
                })

            for match in similarity_matches:
                if match.source_media_id and match.source_media_id != media_id:
                    edges.append({
                        "source_node_id": match.source_media_id,
                        "target_node_id": media_id,
                        "relationship": "visually_similar_to",
                        "confidence": match.similarity,
                        "evidence": f"Method: {match.method}, Algorithm: {match.algorithm}",
                    })

            repo.upsert_provenance_graph(db, analysis_id, nodes, edges)
        except Exception as exc:
            logger.warning("Provenance failed: %s", exc)
            failures.append(f"provenance: {exc}")

    def _run_evidence(
        self,
        analysis_id: str,
        db: Session,
        failures: list,
    ) -> None:
        """Generate structured evidence items for each pipeline finding."""
        try:
            analysis = repo.get_analysis(db, analysis_id)
            media = repo.get_media(db, analysis.media_id) if analysis else None
            detections = repo.get_detections_for_analysis(db, analysis_id)
            fingerprints = repo.get_fingerprints_for_media(db, analysis.media_id) if analysis else []

            items: list[dict] = []

            # FACT: SHA-256
            sha256_fp = next((f for f in fingerprints if f.algorithm == "sha256"), None)
            if sha256_fp and media:
                items.append({
                    "analysis_id": analysis_id,
                    "category": "FACT",
                    "description": f"SHA-256 of {media.original_filename} is {sha256_fp.value}",
                    "source_stage": "fingerprinting",
                    "detector": "HashGenerator",
                    "confidence": 1.0,
                })

            # FACT: File size
            if media:
                items.append({
                    "analysis_id": analysis_id,
                    "category": "FACT",
                    "description": f"File size: {media.size_bytes:,} bytes. Media type: {media.media_type}.",
                    "source_stage": "ingestion",
                    "confidence": 1.0,
                })

            # OBSERVATION / UNCERTAINTY: Per detector
            for det in detections:
                meta = det.detector_metadata or {}
                assessment = meta.get("assessment", "INCONCLUSIVE")

                if assessment == "SUSPICIOUS":
                    desc = (
                        f"OBSERVATION: {det.detector} detected forensic anomalies "
                        f"(type: {det.manipulation_type}, confidence: {det.confidence:.2f}). "
                        f"{meta.get('note', '')}"
                    )
                    cat = "OBSERVATION"
                elif assessment == "INCONCLUSIVE":
                    desc = (
                        f"UNCERTAINTY: {det.detector} returned INCONCLUSIVE "
                        f"(type: {det.manipulation_type}). {meta.get('note', meta.get('reason', ''))}"
                    )
                    cat = "UNCERTAINTY"
                else:  # CLEAN, UNAVAILABLE
                    desc = (
                        f"UNCERTAINTY: {det.detector} assessment: {assessment} "
                        f"(type: {det.manipulation_type}). "
                        f"{meta.get('note', meta.get('reason', ''))}"
                    )
                    cat = "UNCERTAINTY"

                items.append({
                    "analysis_id": analysis_id,
                    "category": cat,
                    "description": desc.strip(),
                    "source_stage": "detection",
                    "detector": det.detector,
                    "confidence": det.confidence,
                })

                # Suspicious segments as separate evidence items
                segments = det.suspicious_segments or []
                for seg in segments[:5]:  # Limit to 5 most relevant
                    items.append({
                        "analysis_id": analysis_id,
                        "category": "OBSERVATION",
                        "description": (
                            f"OBSERVATION: Frames {seg.get('start_frame')}–{seg.get('end_frame')} "
                            f"({seg.get('start_timestamp'):.2f}s–{seg.get('end_timestamp'):.2f}s) "
                            f"show temporal anomaly (peak score: {seg.get('peak_score', 0):.2f})."
                        ),
                        "analysis_id": analysis_id,
                        "source_stage": "video_detection",
                        "detector": det.detector,
                        "timestamp_seconds": seg.get("start_timestamp"),
                        "confidence": seg.get("peak_score", 0.0),
                    })

            if items:
                repo.bulk_create_evidence(db, items)
        except Exception as exc:
            logger.warning("Evidence generation failed: %s", exc)
            failures.append(f"evidence: {exc}")

    def _run_report(
        self,
        analysis_id: str,
        db: Session,
        failures: list,
    ) -> None:
        try:
            package = self._evidence_service.build_evidence_package(analysis_id, db)
            json_report = self._report_generator.generate_json_report(package)
            md_report = self._report_generator.generate_markdown_report(package)

            repo.upsert_report(db, analysis_id, {
                "json_report": json_report,
                "markdown_report": md_report,
                "conclusion": (
                    package.get("overall_assessment", "INCONCLUSIVE")
                ),
            })
        except Exception as exc:
            logger.warning("Report generation failed: %s", exc)
            failures.append(f"report: {exc}")

    def _compute_final_assessment(
        self, analysis_id: str, db: Session
    ) -> tuple[str, str | None]:
        """
        Derive the evidence-level assessment and an ordinal evidence strength.

        Returns ``(assessment, evidence_strength)``. It previously returned a
        numeric confidence computed as the mean of every detector's score:

            mean_conf = sum(confidences) / len(confidences)
            if mean_conf < 0.15: return "CLEAN", round(mean_conf, 4)

        That averaged incomparable scales across non-independent detectors, and
        the resulting number then gated the "CLEAN" verdict -- so adding a
        detector that happened to score low could turn an inconclusive analysis
        into a clean one. Both the average and the numeric return are gone.
        """
        decision = self._decide(repo.get_detections_for_analysis(db, analysis_id))
        return decision.category, decision.evidence_strength

    @staticmethod
    def _decide(detections: list) -> AssessmentDecision:
        """The decision rule, with the branch that fired named in ``rule``."""
        if not detections:
            return AssessmentDecision(
                Assessment.INCONCLUSIVE.value, EvidenceStrength.NONE.value,
                "no_detections", "No detector produced a result for this evidence.",
            )

        assessments = [_plain_value(d.assessment) for d in detections]

        suspicious = [
            d for d, a in zip(detections, assessments) if a in _SUSPICIOUS_VALUES
        ]
        # Anything that is not a completed, anomaly-free observation means the
        # analysis did not fully cover the evidence.
        incomplete = [
            a for a in assessments
            if a not in _COMPLETE_NEGATIVE and a not in _SUSPICIOUS_VALUES
        ]

        # Positive observations stand even when other detectors did not report:
        # missing data cannot retract an anomaly that was measured.
        if suspicious:
            families = {
                ImageDetectionEngine._family(d.detector) for d in suspicious
            }
            families.discard("non_evidential")
            if not families:
                return AssessmentDecision(
                    Assessment.INCONCLUSIVE.value, EvidenceStrength.NONE.value,
                    "non_evidential_only",
                    "Only detectors that are not evidence of manipulation (such "
                    "as object detection) reported anything.",
                )
            names = ", ".join(sorted(families))
            localised = any(getattr(d, "regions", None) for d in suspicious)
            if len(families) >= 2 and localised:
                return AssessmentDecision(
                    Assessment.LIKELY_MANIPULATED.value, EvidenceStrength.STRONG.value,
                    "multi_family_localised",
                    f"Anomalies were reported by {len(families)} independent signal "
                    f"families ({names}), with at least one localised region.",
                    frozenset(families),
                )
            if len(families) >= 2:
                return AssessmentDecision(
                    Assessment.SUSPICIOUS.value, EvidenceStrength.MODERATE.value,
                    "multi_family",
                    f"Anomalies were reported by {len(families)} independent signal "
                    f"families ({names}), without localisation.",
                    frozenset(families),
                )
            return AssessmentDecision(
                Assessment.SUSPICIOUS.value, EvidenceStrength.WEAK.value,
                "single_family",
                f"Anomalies were reported within a single signal family ({names}); "
                "detectors sharing a physical basis do not corroborate each other.",
                frozenset(families),
            )

        # No positive observations. Absence is only reportable with full coverage.
        if incomplete:
            return AssessmentDecision(
                Assessment.INCONCLUSIVE.value, EvidenceStrength.NONE.value,
                "incomplete_coverage",
                f"No anomaly was reported, but {len(incomplete)} of "
                f"{len(detections)} detector(s) did not complete an observation, "
                "so the absence of anomalies cannot be reported.",
            )

        return AssessmentDecision(
            Assessment.NO_SIGNIFICANT_ANOMALIES_DETECTED.value,
            EvidenceStrength.NONE.value,
            "full_coverage_no_anomaly",
            f"All {len(detections)} detector(s) completed and none reported an "
            "anomaly above its threshold. This is a statement about what was "
            "looked for and not found, not a determination of authenticity.",
        )

    # ── Forensic record: observations -> finding -> assessment ────────────────

    def _run_forensic_record(
        self, analysis_id: str, db: Session, failures: list
    ) -> None:
        """
        Persist the observations, the finding they bear on, and the assessment.

        Every link is explicit. A finding of anomaly cites the observations that
        support it AND the completed observations that found nothing -- the
        latter are contradicting evidence and are recorded, not discarded.
        """
        try:
            detections = repo.get_detections_for_analysis(db, analysis_id)
            decision = self._decide(detections)

            observations: list[dict] = []
            gaps: list[str] = []
            for det in detections:
                polarity = observation_polarity(det)
                if polarity is None:
                    gaps.append(det.id)
                    continue
                observations.append({
                    "id": str(uuid.uuid4()),
                    "analysis_id": analysis_id,
                    "detection_id": det.id,
                    "detector": det.detector,
                    "signal_family": ImageDetectionEngine._family(det.detector),
                    "polarity": polarity,
                    "score": det.confidence,
                    "score_semantics": det.score_semantics,
                    "localised": bool(det.regions),
                    "description": self._describe_observation(det, polarity),
                })
            if observations:
                repo.bulk_create_observations(db, observations)

            finding = self._finding_for(decision, observations)
            if finding is not None:
                statement, links = finding
                repo.create_finding(db, analysis_id, statement, links)

            repo.upsert_assessment(db, analysis_id, {
                "category": decision.category,
                "evidence_strength": decision.evidence_strength,
                "rule": decision.rule,
                "rationale": decision.rationale,
                "coverage_gaps": gaps,
            })
        except Exception as exc:
            db.rollback()
            logger.warning("Forensic record failed: %s", exc)
            failures.append(f"forensic_record: {exc}")

    @staticmethod
    def _describe_observation(det, polarity: str) -> str:
        semantics = _plain_value(det.score_semantics)
        what = {
            "ANOMALY_OBSERVED": "reported an anomaly",
            "NO_ANOMALY_OBSERVED": "completed and reported no anomaly above its threshold",
            "INDETERMINATE": "ran but reached no determination",
        }[polarity]
        regions = len(det.regions or [])
        where = f" in {regions} localised region(s)" if regions else ""
        return (
            f"{det.detector} ({det.manipulation_type}) {what}{where}; "
            f"score {det.confidence:.3f} read as {semantics}."
        )

    @staticmethod
    def _finding_for(
        decision: AssessmentDecision, observations: list[dict]
    ) -> Optional[tuple[str, list[tuple[str, str]]]]:
        evidential = [o for o in observations if o["signal_family"] != "non_evidential"]

        if decision.category in _SUSPICIOUS_VALUES:
            supports = [
                (o["id"], "SUPPORTS") for o in evidential
                if o["polarity"] == "ANOMALY_OBSERVED"
            ]
            contradicts = [
                (o["id"], "CONTRADICTS") for o in evidential
                if o["polarity"] == "NO_ANOMALY_OBSERVED"
            ]
            # The rationale already lives on the assessment; the finding states
            # the evidential relation itself, so the two do not repeat each other.
            families = ", ".join(sorted(decision.families))
            statement = (
                f"{len(supports)} observation{'s' if len(supports) != 1 else ''} "
                f"report{'' if len(supports) != 1 else 's'} an anomaly ({families})."
            )
            if contradicts:
                statement += (
                    f" {len(contradicts)} completed observation"
                    f"{'s' if len(contradicts) != 1 else ''} found none and "
                    f"contradict{'' if len(contradicts) != 1 else 's'} it."
                )
            else:
                statement += " No completed observation contradicts it."
            return statement, supports + contradicts

        if decision.category == Assessment.NO_SIGNIFICANT_ANOMALIES_DETECTED.value:
            supports = [
                (o["id"], "SUPPORTS") for o in evidential
                if o["polarity"] == "NO_ANOMALY_OBSERVED"
            ]
            return decision.rationale, supports

        # INCONCLUSIVE: there is no finding to state. The assessment's rationale
        # and coverage gaps say why.
        return None

    # ── Run provenance ────────────────────────────────────────────────────────

    @staticmethod
    def _config_snapshot() -> dict:
        snapshot = {}
        for key in _CONFIG_SNAPSHOT_KEYS:
            value = getattr(settings, key, None)
            if not isinstance(value, (bool, int, float, str, type(None))):
                value = str(value)
            snapshot[key] = value
        return snapshot

    @staticmethod
    def _detector_set_version(detections: list) -> Optional[str]:
        """sha256 over the (detector, version) pairs this run actually invoked."""
        if not detections:
            return None
        pairs = sorted({(d.detector, d.detector_version or "") for d in detections})
        return hashlib.sha256(
            json.dumps(pairs, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def _finish_run(self, analysis_id: str, db: Session, started: float) -> None:
        repo.update_analysis_run(
            db, analysis_id,
            duration_ms=int((time.monotonic() - started) * 1000),
            detector_set_version=self._detector_set_version(
                repo.get_detections_for_analysis(db, analysis_id)
            ),
        )
