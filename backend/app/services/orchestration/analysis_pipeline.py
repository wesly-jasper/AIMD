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
from urllib.parse import urlsplit

from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.safe_errors import safe_reason, scrub
from app.db import repository as repo
from app.db.models import AnalysisRecord
from app.schemas.detection import Assessment, EvidenceStrength
from app.services.detection.engine.image_detection_engine import (
    _SIGNAL_FAMILY,
    NON_MANIPULATION_FAMILIES,
    ImageDetectionEngine,
)
from app.services.preprocessing.metadata_extractor import MetadataExtractor
from app.services.preprocessing.frame_extractor import FrameExtractor
from app.services.preprocessing.keyframe_selector import KeyframeSelector
from app.services.fingerprinting.fingerprint import EMBEDDING_ALGORITHM, FingerprintService
from app.services.similarity.similarity_engine import VERIFIED_MATCH, SimilarityEngine
from app.services.provenance.source_tracer import SourceTracer
from app.services.sources import default_source_connectors
from app.services.sources.connectors import SourceQuery
from app.services.provenance.graph import build_provenance_graph
from app.services.evidence.evidence_service import EvidenceService
from app.services.evidence.report_generator import ReportGenerator
from app.services.detection.detection_container import get_detection_container
from app.services.fusion.evidence_fusion import fuse

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
    "trufor_tile_size",
    "trufor_tile_overlap",
    "univfd_enabled",
    "univfd_threshold",
    "bfree_enabled",
    "bfree_threshold",
    "ai_generation_advisory_detectors",
    "calibration_path",
    "video_keyframe_verdicts_enabled",
    "detector_timeout_seconds",
    "detector_retry_on_memory_error",
    "object_detection_enabled",
    "similarity_embedding_enabled",
    "similarity_phash_max_distance",
    "similarity_cosine_min",
    "similarity_min_inliers",
    "similarity_candidate_phash_distance",
    "similarity_candidate_cosine",
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


class EvidenceIntegrityError(RuntimeError):
    """The stored evidence does not match what was ingested."""


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
        self._source_connectors = default_source_connectors()
        self._evidence_service = EvidenceService()
        self._report_generator = ReportGenerator()
        self._detection_container = get_detection_container()

    def run(
        self, media_id: str, db: Session, requested_by: Optional[str] = None,
        source_url: Optional[str] = None,
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
            # ── Stage 0: Chain of custody ─────────────────────────────────────
            # The stored file must still be the bytes that were ingested. An
            # analysis of a file that changed after upload describes different
            # evidence, so it fails here rather than reporting on it.
            self._verify_integrity(file_path, media)

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
            self._run_source_tracing(
                file_path, media_id, analysis_id, db, stage_failures, source_url=source_url
            )

            # ── Stage 7: Provenance ───────────────────────────────────────────
            logger.info("[%s] Stage 7: Provenance", analysis_id)
            self._run_provenance(media_id, analysis_id, db, stage_failures)

            # ── Stage 8: Evidence items ───────────────────────────────────────
            logger.info("[%s] Stage 8: Evidence generation", analysis_id)
            self._run_evidence(analysis_id, db, stage_failures)

            # ── Assessment, before the report ─────────────────────────────────
            # The report reads the analysis-level assessment. It used to be
            # generated first, while that assessment was still unset, so every
            # stored report said INCONCLUSIVE whatever the analysis concluded.
            assessment, evidence_strength = self._compute_final_assessment(
                analysis_id, db
            )
            self._run_forensic_record(analysis_id, db, stage_failures)

            # ── Status, then Stage 9: Report ──────────────────────────────────
            # The report states the analysis status, so it is set first. It was
            # set after, and every stored report said PROCESSING.
            final_status = "PARTIAL" if stage_failures else "COMPLETED"
            repo.update_analysis_status(
                db, analysis_id,
                status=final_status,
                assessment=assessment,
                evidence_strength=evidence_strength,
            )
            logger.info("[%s] Stage 9: Report generation", analysis_id)
            self._run_report(analysis_id, db, stage_failures)
            if stage_failures and final_status == "COMPLETED":  # the report itself failed
                final_status = "PARTIAL"
                repo.update_analysis_status(db, analysis_id, status=final_status)

            # ── Finalize ──────────────────────────────────────────────────────
            self._finish_run(analysis_id, db, started)

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
                db, analysis_id, status="FAILED", error_message=safe_reason(exc)
            )
            try:
                self._finish_run(analysis_id, db, started)
            except Exception:
                logger.exception("[%s] Could not record run provenance", analysis_id)
            return {
                "analysis_id": analysis_id,
                "media_id": media_id,
                "status": "FAILED",
                "error": safe_reason(exc),
                "stage_failures": stage_failures,
            }

    # ── Stage implementations ─────────────────────────────────────────────────

    @staticmethod
    def _verify_integrity(file_path: Path, media) -> None:
        """Raise when the stored file no longer hashes to the SHA-256 recorded at ingest."""
        expected = (getattr(media, "sha256", None) or "").lower()
        if not expected:
            raise EvidenceIntegrityError(
                "No SHA-256 was recorded for this evidence at ingest, so its "
                "integrity cannot be verified"
            )
        digest = hashlib.sha256()
        with file_path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise EvidenceIntegrityError(
                "The stored evidence file no longer matches the SHA-256 recorded "
                "at ingest; it was not analysed"
            )

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
            db.rollback()
            failures.append(f"metadata: {safe_reason(exc)}")
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
            db.rollback()
            failures.append(f"frame_extraction: {safe_reason(exc)}")
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
                        # Relative to settings.artefacts_dir (e.g. a TruFor heatmap).
                        "mask_path": r.get("mask_path"),
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

                # A stage failure is recorded when an evidential detector did not
                # produce observations: FAILED (broken capability) always, and
                # UNAVAILABLE (no capability), reported distinctly so the cause
                # is not lost. Not a failure: AVAILABLE+SKIPPED (deliberately
                # not run, e.g. unsupported media type), or any skip of a
                # non-evidential detector -- an absent object detector used to
                # mark every image analysis PARTIAL. An absent AI-generation
                # detector is not a gap in the manipulation analysis either; its
                # own branch reports UNAVAILABLE instead.
                detector_name = det_data.get("detector", "unknown")
                outside_verdict = (
                    _SIGNAL_FAMILY.get(detector_name) in NON_MANIPULATION_FAMILIES
                )
                is_gap = execution_status == "FAILED" or (
                    execution_status == "SKIPPED"
                    and availability == "UNAVAILABLE"
                    and not outside_verdict
                )
                if is_gap or meta.get("failure_messages"):
                    # A detector's failure_messages can carry exception text.
                    reason = scrub(str(status_reason or meta.get("failure_messages") or ""))
                    failures.append(
                        f"detection: {det_data.get('detector', 'unknown')} incomplete "
                        f"[{availability}/{execution_status}]: "
                        f"{reason or 'detector did not complete'}"
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
            db.rollback()
            failures.append(f"detection: {safe_reason(exc)}")

    def _run_fingerprinting(
        self,
        file_path: Path,
        media_id: str,
        analysis_id: str,
        db: Session,
        failures: list,
    ) -> None:
        try:
            # Ingestion already stored this media's fingerprints; every analysis
            # used to write the whole set again. Only backfill what is missing,
            # such as an embedding for media ingested before embeddings were on.
            stored = {fp.algorithm for fp in repo.get_fingerprints_for_media(db, media_id)}
            fp_result = self._fingerprint_service.generate(
                file_path, media_id=media_id, analysis_id=analysis_id, skip=stored
            )
            records = fp_result.get("_records", [])
            if records:
                repo.bulk_create_fingerprints(db, records)
        except Exception as exc:
            logger.warning("Fingerprinting failed: %s", exc)
            db.rollback()
            failures.append(f"fingerprinting: {safe_reason(exc)}")

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
            embedding_version = next(
                (fp.algorithm_version for fp in fingerprints
                 if fp.algorithm == EMBEDDING_ALGORITHM), None
            )

            result = self._similarity_engine.search(
                str(file_path),
                query_fingerprints=fp_dict,
                db=db,
                exclude_media_id=media_id,
                owner=media.uploaded_by if media else None,
                embedding_version=embedding_version,
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
                    "verification_status": m.get("verification_status"),
                    "verification": m.get("verification"),
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
            db.rollback()
            failures.append(f"similarity: {safe_reason(exc)}")

    def _run_source_tracing(
        self,
        file_path: Path,
        media_id: str,
        analysis_id: str,
        db: Session,
        failures: list,
        source_url: Optional[str] = None,
    ) -> None:
        """
        Search every source connector, then any external search provider.

        Only a VERIFIED_MATCH with a publication time can be the earliest known
        occurrence. An unverified lead, or a provider result nobody checked, is
        stored as a source but never given that label.
        """
        sources_to_save: list[dict] = []
        try:
            fingerprints = repo.get_fingerprints_for_media(db, media_id)
            query = SourceQuery(
                file_path=str(file_path),
                fingerprints={fp.algorithm: fp.value for fp in fingerprints},
                embedding_version=next(
                    (fp.algorithm_version for fp in fingerprints
                     if fp.algorithm == EMBEDDING_ALGORITHM), None
                ),
                claimed_url=source_url,
            )
            searches = {}
            for connector in self._source_connectors:
                result = connector.run(query, db)
                searches[connector.name] = result.summary()
                query.prior_hits.extend(result.hits)
                for hit in result.hits:
                    sources_to_save.append({
                        "analysis_id": analysis_id,
                        "url": hit.url,
                        "title": hit.title,
                        "domain": urlsplit(hit.url).netloc or None,
                        "source_timestamp": hit.published_at,
                        "similarity": hit.similarity,
                        "provider": hit.provider,
                        "retrieval_status": "FOUND",
                        "is_earliest_known": False,
                        "verification_status": hit.verification_status,
                        "verification": hit.verification,
                        "citation": hit.citation,
                    })
            repo.update_analysis_run(db, analysis_id, source_searches=searches)

            for c in self._source_tracer.trace_source(str(file_path)).get("candidates", []):
                sources_to_save.append({
                    "analysis_id": analysis_id,
                    "url": c.get("url"),
                    "title": c.get("title"),
                    "domain": c.get("domain"),
                    "similarity": c.get("similarity"),
                    "provider": c.get("provider", "unknown"),
                    "retrieval_status": "FOUND",
                    "is_earliest_known": False,
                })

            dated = [s for s in sources_to_save
                     if s.get("verification_status") == VERIFIED_MATCH and s.get("source_timestamp")]
            if dated:
                min(dated, key=lambda s: s["source_timestamp"])["is_earliest_known"] = True
            if sources_to_save:
                repo.bulk_create_sources(db, sources_to_save)
        except Exception as exc:
            logger.warning("Source tracing failed: %s", exc)
            db.rollback()
            failures.append(f"source_tracing: {safe_reason(exc)}")

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
                # A high similarity alone used to make a source "reposted_from";
                # only verification says the source holds the same image.
                verified = src.verification_status == VERIFIED_MATCH
                edges.append({
                    "source_node_id": src_node_id,
                    "target_node_id": media_id,
                    "relationship": "same_image_as" if verified else "visually_similar_to",
                    "confidence": src.similarity or 0.5,
                    "evidence": (
                        f"Similarity: {src.similarity}, Provider: {src.provider}, "
                        f"Verification: {src.verification_status or 'NOT_VERIFIED'}"
                    ),
                })

            for match in similarity_matches:
                if match.source_media_id and match.source_media_id != media_id:
                    # Only a verified match says the two are the same image; a
                    # retrieved-but-unverified candidate stays "similar".
                    verified = match.verification_status == "VERIFIED_MATCH"
                    edges.append({
                        "source_node_id": match.source_media_id,
                        "target_node_id": media_id,
                        "relationship": "same_image_as" if verified else "visually_similar_to",
                        "confidence": match.similarity,
                        "evidence": (
                            f"Method: {match.method}, Algorithm: {match.algorithm}, "
                            f"Verification: {match.verification_status or 'NOT_VERIFIED'}"
                        ),
                    })

            repo.upsert_provenance_graph(db, analysis_id, nodes, edges)
        except Exception as exc:
            logger.warning("Provenance failed: %s", exc)
            db.rollback()
            failures.append(f"provenance: {safe_reason(exc)}")

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
                # The detection's own assessment column. Detectors stopped
                # writing ``assessment`` into their metadata, so reading it from
                # there labelled every detector INCONCLUSIVE, including ones
                # that completed with a clear result.
                assessment = _plain_value(det.assessment or "INCONCLUSIVE")
                ai_branch = _SIGNAL_FAMILY.get(det.detector) == "ai_generation"

                if ai_branch and observation_polarity(det) is not None:
                    # Generation, not editing: never worded as a forensic
                    # anomaly, and a low score is never worded as "real".
                    if det.manipulation_detected:
                        desc = (
                            f"OBSERVATION: {det.detector} scored {det.confidence:.3f} "
                            f"for AI generation, at or above its threshold of "
                            f"{meta.get('threshold')}. Uncalibrated; reported apart "
                            "from the manipulation assessment."
                        )
                    else:
                        desc = (
                            f"OBSERVATION: {det.detector} scored {det.confidence:.3f} "
                            f"for AI generation, below its threshold of "
                            f"{meta.get('threshold')}. This is not a finding that "
                            "the image was camera-captured."
                        )
                    cat = "OBSERVATION"
                elif assessment in _SUSPICIOUS_VALUES:
                    desc = (
                        f"OBSERVATION: {det.detector} detected forensic anomalies "
                        f"(type: {det.manipulation_type}, confidence: {det.confidence:.2f}). "
                        f"{meta.get('note', '')}"
                    )
                    cat = "OBSERVATION"
                elif assessment == "INCONCLUSIVE":
                    desc = (
                        f"UNCERTAINTY: {det.detector} returned INCONCLUSIVE "
                        f"(type: {det.manipulation_type}). "
                        f"{det.status_reason or meta.get('note', meta.get('reason', ''))}"
                    )
                    cat = "UNCERTAINTY"
                elif assessment in _COMPLETE_NEGATIVE:
                    # A completed check that found nothing above its threshold
                    # is an observation, not an uncertainty.
                    desc = (
                        f"OBSERVATION: {det.detector} completed and reported no "
                        f"anomalies above its threshold (type: {det.manipulation_type}). "
                        "This is not a finding of authenticity."
                    )
                    cat = "OBSERVATION"
                else:  # UNAVAILABLE: the check did not run
                    desc = (
                        f"UNCERTAINTY: {det.detector} did not produce observations "
                        f"({assessment}). "
                        f"{det.status_reason or meta.get('note', meta.get('reason', ''))}"
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
            db.rollback()
            failures.append(f"evidence: {safe_reason(exc)}")

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
            db.rollback()
            failures.append(f"report: {safe_reason(exc)}")

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
        # Non-evidential detectors (object detection) cannot leave a gap in
        # coverage: their result is not evidence either way, so neither is their
        # absence. Counting a skipped object detector here made every analysis
        # INCONCLUSIVE. The AI-generation detector is excluded for a different
        # reason: it answers another question, decided by
        # ``_decide_ai_generation``, and must not move this verdict either way.
        # Polarity, not the label alone: a FAILED run can carry an INCONCLUSIVE
        # label, and it is a gap, not a detector that ran and was undecided.
        polarities = [
            observation_polarity(d) for d in detections
            if _SIGNAL_FAMILY.get(d.detector) not in NON_MANIPULATION_FAMILIES
        ]
        negative = [p for p in polarities if p == "NO_ANOMALY_OBSERVED"]
        # Ran, but could not support a determination either way.
        undetermined = [p for p in polarities if p == "INDETERMINATE"]
        # Did not produce observations at all (unavailable, failed, skipped).
        incomplete = [p for p in polarities if p is None]

        # Positive observations stand even when other detectors did not report:
        # missing data cannot retract an anomaly that was measured.
        if suspicious:
            # An AI-generation positive is invisible here: it says nothing about
            # editing, so with nothing else firing it falls through to the
            # coverage rules below, and beside other positives it changes nothing.
            families = {
                ImageDetectionEngine._family(d.detector) for d in suspicious
            } - {"ai_generation"}
            only_non_evidential = bool(families) and families <= NON_MANIPULATION_FAMILIES
            families -= NON_MANIPULATION_FAMILIES
            if only_non_evidential:
                return AssessmentDecision(
                    Assessment.INCONCLUSIVE.value, EvidenceStrength.NONE.value,
                    "non_evidential_only",
                    "Only detectors that are not evidence of manipulation (such "
                    "as object detection) reported anything.",
                )
            suspicious = [
                d for d in suspicious
                if ImageDetectionEngine._family(d.detector) in families
            ]
        if suspicious:
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
                f"{len(polarities)} evidential detector(s) did not complete an "
                "observation, so the absence of anomalies cannot be reported.",
            )

        # Everything that ran was inconclusive: nothing found "no anomaly".
        if not negative:
            return AssessmentDecision(
                Assessment.INCONCLUSIVE.value, EvidenceStrength.NONE.value,
                "no_determination",
                f"No anomaly was reported, but none of the {len(undetermined)} "
                "evidential detector(s) that ran could support a determination "
                "either way.",
            )

        rationale = (
            f"{len(negative)} evidential detector(s) completed and none reported "
            "an anomaly above its threshold."
        )
        if undetermined:
            rationale += (
                f" {len(undetermined)} more ran but could not support a "
                "determination."
            )
        return AssessmentDecision(
            Assessment.NO_SIGNIFICANT_ANOMALIES_DETECTED.value,
            EvidenceStrength.NONE.value,
            "full_coverage_no_anomaly",
            rationale + " This is a statement about what was looked for and not "
            "found, not a determination of authenticity.",
        )

    @staticmethod
    def _decide_ai_generation(detections: list) -> dict:
        """
        The AI-generation branch, decided apart from the manipulation verdict.

        Its own question ("was this generated?") with its own detector, so it is
        reported beside ``_decide`` and never folded into it: an edited camera
        photograph and a wholly generated image are different findings, and a
        score from one must not raise or lower the other.

        ``NOT_INDICATED`` is a statement about one detector's score, not a
        finding that the image was camera-captured.
        """
        dets = [
            d for d in detections
            if _SIGNAL_FAMILY.get(d.detector) == "ai_generation"
        ]
        if not dets:
            # The detector is configured for images; videos never reach it, and
            # "no detector is configured" was untrue for them.
            if detections and all(getattr(d, "media_type", None) == "video" for d in detections):
                return {
                    "status": "UNAVAILABLE",
                    "detector": None,
                    "reason": "image_only",
                    "rationale": "The AI-generation check runs on still images. "
                                 "Video frames are not checked for AI generation yet, "
                                 "so whether this video was generated was not assessed.",
                }
            return {
                "status": "UNAVAILABLE",
                "detector": None,
                "rationale": "No AI-generation detector is configured, so whether "
                             "this image was generated was not assessed.",
            }

        # Every generation detector is read, not just the first: a second model
        # that disagrees is evidence about how far either can be trusted, and
        # it is reported rather than averaged or outvoted.
        entries = [AnalysisPipeline._ai_entry(d) for d in dets]
        scored = [e for e in entries if e["status"] in ("AI_GENERATION_INDICATED", "NOT_INDICATED")]
        undetermined = [e for e in entries if e["status"] == "INCONCLUSIVE"]
        if not scored and undetermined:
            first = undetermined[0]
            return {
                **first["base"],
                "status": "INCONCLUSIVE",
                "reason": first["reason"],
                "rationale": first["rationale"],
                "detectors": [e["public"] for e in entries],
            }
        if not scored:
            first = entries[0]
            return {
                **first["base"],
                "status": "UNAVAILABLE",
                "reason": first["reason"],
                "rationale": first["rationale"],
                "detectors": [e["public"] for e in entries],
            }

        # An advisory detector is shown but does not vote while a stronger one
        # on the branch has scored (settings.ai_generation_advisory_detectors).
        advisory = set(settings.ai_generation_advisory_detectors or ())
        voting = [e for e in scored if e["detector"] not in advisory] or scored
        for e in entries:
            e["public"]["voting"] = e in voting
        scored = voting

        indicated = [e for e in scored if e["status"] == "AI_GENERATION_INDICATED"]
        cleared = [e for e in scored if e["status"] == "NOT_INDICATED"]
        primary = max(indicated, key=lambda e: e["score"]) if indicated else scored[0]
        result = {
            **primary["base"],
            "score": primary["score"],
            "logit": primary["logit"],
            "threshold": primary["threshold"],
            "calibration": primary["calibration"],
            "detectors": [e["public"] for e in entries],
        }
        if indicated and cleared:
            return {
                **result,
                "status": "CONFLICTING",
                "rationale": (
                    f"The AI-generation detectors disagree: "
                    f"{', '.join(e['detector'] for e in indicated)} indicated generation, "
                    f"{', '.join(e['detector'] for e in cleared)} did not. The "
                    "disagreement is reported, not resolved; neither result "
                    "supports a conclusion on its own."
                ),
            }
        rationale = primary["rationale"]
        if len(scored) > 1:
            rationale += f" {len(scored)} AI-generation detectors agree."
        shown = [e["detector"] for e in entries if e not in scored and e["status"] != "UNAVAILABLE"]
        if shown:
            rationale += (
                f" {', '.join(shown)} is also shown for reference; it does not "
                "vote, being markedly less accurate on the evaluation set."
            )
        return {**result, "status": primary["status"], "rationale": rationale}

    @staticmethod
    def _with_edit_note(ai: dict, category: str) -> dict:
        """
        The branches stay separate, but an AI indication on a file the
        manipulation branch found edited needs a reader's caution: B-Free also
        fires on locally filled-in regions (4 of 20 classical object removals in
        evaluation), so the two findings may describe the same local edit.
        """
        if ai.get("status") != "AI_GENERATION_INDICATED" or category not in (
            Assessment.SUSPICIOUS.value, Assessment.LIKELY_MANIPULATED.value
        ):
            return ai
        note = (
            "The manipulation check also found signs of editing. An AI-generation "
            "detector can respond to a region that was filled in locally, so both "
            "results may describe the same local edit rather than a generated image."
        )
        return {**ai, "edit_overlap_note": note, "rationale": f"{ai.get('rationale', '')} {note}".strip()}

    @staticmethod
    def _ai_entry(det) -> dict:
        """One AI-generation detector's result, with its own wording."""
        meta = getattr(det, "detector_metadata", None) or {}
        semantics = _plain_value(det.score_semantics)
        base = {
            "detector": det.detector,
            "detector_version": getattr(det, "detector_version", None),
            "score_semantics": semantics,
            "semantics": meta.get("semantics"),
            "limitations": list(meta.get("limitations") or []),
        }
        entry = {"detector": det.detector, "base": base, "reason": det.status_reason,
                 "score": None, "logit": None, "threshold": None, "calibration": None}
        if observation_polarity(det) is not None and _plain_value(det.execution_status) == "INCONCLUSIVE":
            # Ran, and could not support a determination -- e.g. no keyframe of
            # a video flagged, which is not a finding about the frames between.
            entry.update(
                status="INCONCLUSIVE",
                rationale=f"{det.detector} ran but could not support a determination"
                          + (f": {det.status_reason}." if det.status_reason else "."),
            )
        elif observation_polarity(det) is None or _plain_value(det.execution_status) != "COMPLETED":
            entry.update(
                status="UNAVAILABLE",
                rationale=f"{det.detector} did not produce a score "
                          f"({_plain_value(det.availability)}/"
                          f"{_plain_value(det.execution_status)}), so whether "
                          "this image was generated was not assessed.",
            )
        else:
            score = float(det.confidence)
            threshold = meta.get("threshold")
            calibration = meta.get("calibration")
            calibrated = semantics == "CALIBRATED_PROBABILITY" and (calibration or {}).get("applied")
            if calibrated:
                fitted = calibration.get("fitted_on") or {}
                how = (
                    f"a calibrated probability (Platt scaling on "
                    f"{fitted.get('positives', '?')} generated and "
                    f"{fitted.get('negatives', '?')} other validation images, threshold "
                    f"set for a {calibration.get('target_fpr'):.0%} false-positive rate)"
                )
            else:
                how = "an uncalibrated score"
            indicated = bool(det.manipulation_detected)
            if indicated:
                rationale = (
                    f"{det.detector} scored {score:.3f}, at or above its threshold of "
                    f"{threshold}. This is {how}, and supports examiner review, not a "
                    "conclusion."
                )
            else:
                rationale = (
                    f"{det.detector} scored {score:.3f}, below its threshold of "
                    f"{threshold} ({how}). This is not a finding that the image was "
                    "captured by a camera: generators this detector was not trained "
                    "on can score low."
                )
            entry.update(
                status="AI_GENERATION_INDICATED" if indicated else "NOT_INDICATED",
                score=score, logit=meta.get("logit"), threshold=threshold,
                calibration=calibration, rationale=rationale,
            )
        entry["public"] = {
            "detector": det.detector,
            "status": entry["status"],
            "score": entry["score"],
            "threshold": entry["threshold"],
            "score_semantics": semantics,
            "reason": det.status_reason if entry["status"] in ("UNAVAILABLE", "INCONCLUSIVE") else None,
        }
        return entry

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
                    # A missing AI-generation detector is reported on its own
                    # branch, not as a gap in the manipulation assessment.
                    if _SIGNAL_FAMILY.get(det.detector) != "ai_generation":
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
                "ai_generation": self._with_edit_note(
                    self._decide_ai_generation(detections), decision.category
                ),
                "evidence_fusion": fuse(
                    detections, advisory=settings.ai_generation_advisory_detectors or ()
                ),
            })
        except Exception as exc:
            db.rollback()
            logger.warning("Forensic record failed: %s", exc)
            failures.append(f"forensic_record: {safe_reason(exc)}")

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
        evidential = [
            o for o in observations
            if o["signal_family"] not in NON_MANIPULATION_FAMILIES
        ]

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
            if isinstance(value, (list, tuple)):
                value = [str(v) for v in value]
            elif not isinstance(value, (bool, int, float, str, type(None))):
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
