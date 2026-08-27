"""
Evidence Service — aggregates real pipeline outputs into forensic evidence packages.

CRITICAL REQUIREMENT: This service must NEVER use hardcoded or mock values.
All evidence must be traceable to real pipeline computations.

Evidence categories:
  FACT:        Directly measured or computed — cryptographic hashes, file sizes,
               EXIF values, pixel dimensions. These can be independently verified.
  OBSERVATION: Forensic signals detected by algorithms — ELA anomalies, spectral
               discontinuities, face boundary irregularities. Real but not definitive.
  INFERENCE:   Conclusions drawn from combining multiple observations. Must clearly
               state the combination of signals and confidence level.
  UNCERTAINTY: Limitations, missing models, low-confidence conditions. Must not
               be suppressed.

Evidence is pulled from the database — not recomputed here.
"""
import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.db import repository as repo

logger = logging.getLogger(__name__)


class EvidenceService:

    def build_evidence_package(
        self,
        analysis_id: str,
        db: Session,
    ) -> dict[str, Any]:
        """
        Build a complete forensic evidence package from pipeline results stored in DB.

        All values come from the database — previously computed by the pipeline.
        No values are fabricated here.
        """
        analysis = repo.get_analysis(db, analysis_id)
        if not analysis:
            raise ValueError(f"No analysis found with id: {analysis_id}")

        media = repo.get_media(db, analysis.media_id)
        if not media:
            raise ValueError(f"No media found for analysis: {analysis_id}")

        # Retrieve all pipeline outputs
        detections = repo.get_detections_for_analysis(db, analysis_id)
        fingerprints = repo.get_fingerprints_for_media(db, analysis.media_id)
        similarity_matches = repo.get_similarity_matches_for_analysis(db, analysis_id)
        sources = repo.get_sources_for_analysis(db, analysis_id)
        provenance = repo.get_provenance_for_analysis(db, analysis_id)
        evidence_items = repo.get_evidence_for_analysis(db, analysis_id)

        # ── Case Information ──────────────────────────────────────────────────
        case_information = {
            "case_id": f"AIMD-{analysis_id[:8].upper()}",
            "analysis_id": analysis_id,
            "media_id": analysis.media_id,
            "analysis_status": analysis.status,
            "created_at": analysis.created_at.isoformat() if analysis.created_at else None,
            "completed_at": analysis.completed_at.isoformat() if analysis.completed_at else None,
        }

        # ── Media Evidence (FACT) ─────────────────────────────────────────────
        metadata_record = media.metadata_record
        media_evidence = {
            "original_filename": media.original_filename,
            "stored_filename": media.stored_filename,
            "media_type": media.media_type,
            "content_type": media.content_type,
            # FACT: cryptographic identity
            "sha256": media.sha256,
            "size_bytes": media.size_bytes,
            "evidence_category": "FACT",
            "metadata": {
                "width": metadata_record.width if metadata_record else None,
                "height": metadata_record.height if metadata_record else None,
                "duration_seconds": metadata_record.duration_seconds if metadata_record else None,
                "fps": metadata_record.fps if metadata_record else None,
                "codec": metadata_record.codec if metadata_record else None,
                "exif_data": metadata_record.exif_data if metadata_record else {},
            } if metadata_record else {},
            "ingested_at": media.created_at.isoformat() if media.created_at else None,
        }

        # ── Fingerprint Evidence ──────────────────────────────────────────────
        fp_by_algorithm: dict[str, dict] = {}
        for fp in fingerprints:
            if fp.algorithm not in fp_by_algorithm:
                fp_by_algorithm[fp.algorithm] = {
                    "algorithm": fp.algorithm,
                    "algorithm_version": fp.algorithm_version,
                    "value": fp.value,
                    "scope": fp.scope,
                    "evidence_category": "FACT" if fp.algorithm == "sha256" else "OBSERVATION",
                    "note": (
                        "Cryptographic hash — exact file identity"
                        if fp.algorithm == "sha256"
                        else "Perceptual hash — visual similarity (not cryptographic)"
                    ),
                }

        # ── Detection Evidence ────────────────────────────────────────────────
        detection_evidence = []
        for det in detections:
            assessment = det.assessment
            # Determine evidence category based on assessment
            if assessment == "SUSPICIOUS":
                category = "OBSERVATION"
            elif assessment in ("INCONCLUSIVE", "UNAVAILABLE"):
                category = "UNCERTAINTY"
            else:
                category = "OBSERVATION"

            det_dict = {
                "detector": det.detector,
                "detector_version": det.detector_version,
                "manipulation_type": det.manipulation_type,
                "assessment": assessment,
                "confidence": det.confidence,
                "evidence_category": category,
                "regions_count": len(det.regions),
                "regions": [
                    {
                        "type": r.region_type,
                        "confidence": r.confidence,
                        "bbox": r.bbox,
                        "start_timestamp": r.start_timestamp,
                        "end_timestamp": r.end_timestamp,
                    }
                    for r in det.regions[:10]  # Limit to 10 for package size
                ],
                "metadata": det.detector_metadata or {},
                "note": det.detector_metadata.get("note") if det.detector_metadata else None,
            }
            detection_evidence.append(det_dict)

        # ── Similarity Evidence ───────────────────────────────────────────────
        similarity_evidence = {
            "local_matches": [],
            "external_matches": [],
            "total_matches": 0,
            "evidence_category": "OBSERVATION",
        }
        for match in similarity_matches:
            entry = {
                "match_type": match.match_type,
                "source_media_id": match.source_media_id,
                "external_url": match.external_url,
                "similarity": match.similarity,
                "method": match.method,
                "algorithm": match.algorithm,
                "rank": match.rank,
            }
            if match.match_type == "local":
                similarity_evidence["local_matches"].append(entry)
            else:
                similarity_evidence["external_matches"].append(entry)
        similarity_evidence["total_matches"] = len(similarity_matches)

        # ── Source Evidence ───────────────────────────────────────────────────
        source_evidence = {
            "candidates": [],
            "earliest_known_occurrence": None,
            "provider_status": "UNAVAILABLE",
            "evidence_category": "INFERENCE",
            "note": (
                "INFERENCE: 'Earliest-known occurrence' means the earliest timestamp "
                "AIMD discovered — not the absolute origin. Offline and unindexed "
                "sources cannot be searched."
            ),
        }
        for src in sources:
            src_dict = {
                "url": src.url,
                "title": src.title,
                "domain": src.domain,
                "source_timestamp": src.source_timestamp.isoformat() if src.source_timestamp else None,
                "similarity": src.similarity,
                "provider": src.provider,
                "retrieval_status": src.retrieval_status,
                "is_earliest_known": src.is_earliest_known,
            }
            source_evidence["candidates"].append(src_dict)
            if src.is_earliest_known:
                source_evidence["earliest_known_occurrence"] = src_dict
            source_evidence["provider_status"] = src.retrieval_status

        # ── Provenance ────────────────────────────────────────────────────────
        provenance_evidence = {
            "nodes": provenance.get("nodes", []),
            "edges": provenance.get("edges", []),
            "edge_count": len(provenance.get("edges", [])),
            "evidence_category": "INFERENCE",
        }

        # ── Classified Evidence Items ─────────────────────────────────────────
        classified_items = {
            "FACT": [],
            "OBSERVATION": [],
            "INFERENCE": [],
            "UNCERTAINTY": [],
        }
        for item in evidence_items:
            cat = item.category if item.category in classified_items else "UNCERTAINTY"
            classified_items[cat].append({
                "description": item.description,
                "source_stage": item.source_stage,
                "detector": item.detector,
                "confidence": item.confidence,
                "frame_index": item.frame_index,
                "timestamp_seconds": item.timestamp_seconds,
                "created_at": item.created_at.isoformat() if item.created_at else None,
            })

        # ── Limitations ───────────────────────────────────────────────────────
        limitations = self._collect_limitations(detections, sources, analysis)

        return {
            "case_information": case_information,
            "media_evidence": media_evidence,
            "fingerprint_evidence": fp_by_algorithm,
            "detection_evidence": detection_evidence,
            "similarity_evidence": similarity_evidence,
            "source_evidence": source_evidence,
            "provenance_evidence": provenance_evidence,
            "classified_evidence": classified_items,
            "limitations": limitations,
            "overall_assessment": analysis.assessment,
            "overall_confidence": analysis.overall_confidence,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def _collect_limitations(self, detections, sources, analysis) -> list[str]:
        """Collect and surface all uncertainty/limitation statements."""
        limitations = []

        # Detection limitations
        for det in detections:
            meta = det.detector_metadata or {}
            if meta.get("assessment") in ("UNAVAILABLE", "INCONCLUSIVE"):
                for lim in meta.get("limitations", []):
                    limitations.append(lim)
                if note := meta.get("note"):
                    limitations.append(note)

        # Source limitations
        if not sources:
            limitations.append(
                "UNCERTAINTY: No internet source search provider is configured. "
                "Cannot determine earliest known online occurrence."
            )

        # Partial analysis
        if analysis.status == "PARTIAL":
            limitations.append(
                "UNCERTAINTY: Analysis completed only partially. Some pipeline stages failed."
            )

        # Baseline detector warning
        limitations.append(
            "UNCERTAINTY: Image forensic analysis uses heuristic methods (ELA, FFT, noise). "
            "No validated AI detection model (e.g. TruFor) is configured. "
            "Heuristic signals are forensic indicators, not proof of manipulation."
        )

        return list(dict.fromkeys(limitations))  # Deduplicate, preserve order

    # ── Legacy compatibility ──────────────────────────────────────────────────

    def generate_evidence_package(
        self,
        media_id: str,
        file_path: str,
        analysis_id: str,
    ) -> dict[str, Any]:
        """
        Legacy interface. Returns a warning that this method requires a DB session.
        Use build_evidence_package(analysis_id, db) instead.
        """
        raise NotImplementedError(
            "Use EvidenceService.build_evidence_package(analysis_id, db) — "
            "evidence requires a database session to retrieve real pipeline results."
        )
