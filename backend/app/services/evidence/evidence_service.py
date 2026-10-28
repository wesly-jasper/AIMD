"""
Evidence Service -- aggregates real pipeline outputs into forensic evidence packages.

CRITICAL REQUIREMENT: This service must NEVER use hardcoded or mock values.
All evidence must be traceable to real pipeline computations.

Evidence categories:
  FACT:        Directly measured or computed -- cryptographic hashes, file sizes,
               EXIF values, pixel dimensions. These can be independently verified.
  OBSERVATION: Forensic signals detected by algorithms -- ELA anomalies, spectral
               discontinuities, face boundary irregularities. Real but not definitive.
  INFERENCE:   Conclusions drawn from combining multiple observations. Must clearly
               state the combination of signals and confidence level.
  UNCERTAINTY: Limitations, missing models, low-confidence conditions. Must not
               be suppressed.

Evidence is pulled from the database -- not recomputed here.
"""
import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.db import repository as repo
from app.services.provenance.timeline import build_timeline

logger = logging.getLogger(__name__)


def _plain(raw: Any) -> Any:
    return raw.value if hasattr(raw, "value") else raw


_EMBEDDING_ALGORITHMS = frozenset({"clip-vit-l14"})

#: What each fingerprint is. Every non-SHA fingerprint used to be labelled a
#: "Perceptual hash", including ones that are not hashes at all.
_FINGERPRINT_NOTES = {
    "sha256": "Cryptographic hash -- exact file identity",
    "phash": "Perceptual hash (DCT) -- visual similarity, not cryptographic",
    "dhash": "Difference hash -- visual similarity, not cryptographic",
    "clip-vit-l14": (
        "CLIP ViT-L/14 image embedding -- semantic similarity used to retrieve "
        "candidates; not an identity and not cryptographic"
    ),
}


class EvidenceService:

    def build_evidence_package(
        self,
        analysis_id: str,
        db: Session,
    ) -> dict[str, Any]:
        """
        Build a complete forensic evidence package from pipeline results stored in DB.

        All values come from the database -- previously computed by the pipeline.
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
        # The real case, or none. This used to print "AIMD-<analysis id prefix>",
        # a reference that matched no case record and looked like one.
        case = repo.get_case(db, media.case_id) if media.case_id else None
        case_information = {
            "case_id": case.id if case else None,
            "case_title": case.title if case else None,
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
            # The signer's claims, with whether the signature holds and is trusted.
            "content_credentials": (metadata_record.raw_metadata or {}).get("c2pa")
            if metadata_record else None,
            "ingested_at": media.created_at.isoformat() if media.created_at else None,
        }

        # ── Fingerprint Evidence ──────────────────────────────────────────────
        fp_by_algorithm: dict[str, dict] = {}
        for fp in fingerprints:
            if fp.algorithm not in fp_by_algorithm:
                fp_by_algorithm[fp.algorithm] = {
                    "algorithm": fp.algorithm,
                    "algorithm_version": fp.algorithm_version,
                    # An embedding is 768 floats; the package carries its
                    # identity, not the vector.
                    "value": (
                        None if fp.algorithm in _EMBEDDING_ALGORITHMS else fp.value
                    ),
                    "scope": fp.scope,
                    "evidence_category": "FACT" if fp.algorithm == "sha256" else "OBSERVATION",
                    "note": _FINGERPRINT_NOTES.get(
                        fp.algorithm,
                        f"{fp.algorithm} -- visual similarity (not cryptographic)",
                    ),
                }

        # ── Detection Evidence ────────────────────────────────────────────────
        detection_evidence = []
        for det in detections:
            assessment = _plain(det.assessment)
            # Determine evidence category based on assessment
            if assessment in ("INCONCLUSIVE", "UNAVAILABLE"):
                category = "UNCERTAINTY"
            else:
                category = "OBSERVATION"

            det_dict = {
                "detector": det.detector,
                "detector_version": det.detector_version,
                "manipulation_type": det.manipulation_type,
                "assessment": assessment,
                "confidence": det.confidence,
                # What the number is, and whether there is one at all: a raw
                # model score, a calibrated probability and a heuristic severity
                # are rendered differently, and a FAILED run has no score.
                "score_semantics": _plain(det.score_semantics),
                "availability": _plain(det.availability),
                "execution_status": _plain(det.execution_status),
                "status_reason": det.status_reason,
                "evidence_category": category,
                "regions_count": len(det.regions),
                "regions": [
                    {
                        "type": r.region_type,
                        "confidence": r.confidence,
                        "bbox": r.bbox,
                        "start_timestamp": r.start_timestamp,
                        "end_timestamp": r.end_timestamp,
                        "mask_path": r.mask_path,
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
                # VERIFIED_MATCH | SIMILAR_UNVERIFIED | NO_MATCH, or None for a
                # match recorded before verification existed.
                "verification_status": match.verification_status,
                "verification": match.verification,
            }
            if match.match_type == "local":
                similarity_evidence["local_matches"].append(entry)
            else:
                similarity_evidence["external_matches"].append(entry)
        similarity_evidence["total_matches"] = len(similarity_matches)
        similarity_evidence["verified_matches"] = sum(
            1 for m in similarity_matches if m.verification_status == "VERIFIED_MATCH"
        )

        # ── AI-generation Evidence ────────────────────────────────────────────
        # Its own section: a different question from manipulation, decided by a
        # different detector, and never blended into ``overall_assessment``.
        assessment_record = repo.get_assessment_for_analysis(db, analysis_id)
        ai_generation = (
            assessment_record.ai_generation if assessment_record else None
        ) or {"status": "UNAVAILABLE", "rationale": "Not assessed for this analysis."}
        ai_generation_evidence = {
            **ai_generation,
            "evidence_category": (
                "OBSERVATION" if ai_generation.get("status") in
                ("AI_GENERATION_INDICATED", "NOT_INDICATED") else "UNCERTAINTY"
            ),
        }

        # ── Source Evidence ───────────────────────────────────────────────────
        searches = analysis.source_searches or {}
        source_evidence = {
            "candidates": [],
            "earliest_known_occurrence": None,
            "provider_status": "UNAVAILABLE",
            # Per connector: UNAVAILABLE (not searched), FAILED, or COMPLETED.
            "searches": searches,
            "evidence_category": "INFERENCE",
            "note": (
                "INFERENCE: 'Earliest-known occurrence' means the earliest timestamp "
                "AIMD discovered -- not the absolute origin. Offline and unindexed "
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
                "verification_status": src.verification_status,
                "verification": src.verification,
                "citation": src.citation,
            }
            source_evidence["candidates"].append(src_dict)
            if src.is_earliest_known:
                source_evidence["earliest_known_occurrence"] = src_dict
            source_evidence["provider_status"] = src.retrieval_status
        if any(s.get("status") == "COMPLETED" for s in searches.values()):
            source_evidence["provider_status"] = "COMPLETED"

        # ── Provenance timeline (rebuilt from the sources above) ──────────────
        provenance_timeline = build_timeline(
            sources,
            metadata_record.exif_data if metadata_record else None,
            (metadata_record.raw_metadata or {}).get("c2pa") if metadata_record else None,
        )

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
            "ai_generation_evidence": ai_generation_evidence,
            "evidence_fusion": (assessment_record.evidence_fusion if assessment_record else None),
            # How this result was produced: the software, the detector set, the
            # verdict policy and calibration files, and every detector's status.
            "run": {
                "software_version": analysis.software_version,
                "detector_set_version": analysis.detector_set_version,
                "config_snapshot": analysis.config_snapshot,
                "duration_ms": analysis.duration_ms,
            },
            "detector_statuses": [self._detector_status(d) for d in detections],
            "source_evidence": source_evidence,
            "provenance_timeline": provenance_timeline,
            "provenance_evidence": provenance_evidence,
            "classified_evidence": classified_items,
            "limitations": limitations,
            "overall_assessment": analysis.assessment,
            "evidence_strength": analysis.evidence_strength,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    def _detector_status(d) -> dict[str, Any]:
        meta = d.detector_metadata or {}
        calibration = meta.get("calibration") or {}
        return {
            "detector": d.detector,
            "detector_version": d.detector_version,
            "availability": _plain(d.availability),
            "execution_status": _plain(d.execution_status),
            "status_reason": d.status_reason,
            "score_semantics": _plain(d.score_semantics),
            "calibration": (
                {"applied": bool(calibration.get("applied")), "method": calibration.get("method"),
                 "created_at": calibration.get("created_at"), "fitted_on": calibration.get("fitted_on"),
                 "reason": calibration.get("reason")}
                if calibration else None
            ),
        }

    def _collect_limitations(self, detections, sources, analysis) -> list[str]:
        """Collect and surface all uncertainty/limitation statements."""
        limitations = []

        # Detection limitations. Read from the detection's own columns: detectors
        # no longer write ``assessment`` into metadata, so keying on
        # ``meta["assessment"]`` silently dropped every detector's limitation.
        for det in detections:
            meta = det.detector_metadata or {}
            if _plain(det.assessment) in ("UNAVAILABLE", "INCONCLUSIVE"):
                for lim in meta.get("limitations", []):
                    limitations.append(lim)
                if det.status_reason:
                    limitations.append(
                        f"UNCERTAINTY: {det.detector}: {det.status_reason}"
                    )
                elif note := meta.get("note"):
                    limitations.append(note)

        # Source limitations: say what was not searched, and that what was
        # searched is a bounded index, never "the internet".
        searches = analysis.source_searches or {}
        for name, outcome in sorted(searches.items()):
            if outcome.get("status") != "COMPLETED":
                limitations.append(
                    f"UNCERTAINTY: Source '{name}' was not searched "
                    f"({outcome.get('status')}: {outcome.get('reason')})."
                )
        if not any(s.get("status") == "COMPLETED" for s in searches.values()):
            limitations.append(
                "UNCERTAINTY: No source was searched. "
                "Cannot determine earliest known online occurrence."
            )
        elif not any(s.verification_status == "VERIFIED_MATCH" for s in sources):
            limitations.append(
                "UNCERTAINTY: No verified source was found in the indexes searched. "
                "This does not mean the image was never published elsewhere."
            )

        # Partial analysis
        if analysis.status == "PARTIAL":
            limitations.append(
                "UNCERTAINTY: Analysis completed only partially. Some pipeline stages failed."
            )

        # Learned-model warning, only when no learned model actually ran. It used
        # to be appended unconditionally, so a report said TruFor was not
        # configured even on runs where TruFor produced the result.
        learned_ran = any(
            det.detector == "TruForDetector"
            and _plain(det.availability) == "AVAILABLE"
            and _plain(det.execution_status) in ("COMPLETED", "INCONCLUSIVE")
            for det in detections
        )
        # The AI-generation detector's limitations apply to every score it
        # produced, positive or not -- a low score is where they matter most.
        for det in detections:
            if det.detector == "UnivFDDetector" and _plain(det.execution_status) == "COMPLETED":
                for lim in (det.detector_metadata or {}).get("limitations", []):
                    limitations.append(f"AI generation: {lim}")
            # TruFor's own limitations (tiled inference) qualify the result it
            # did produce; the loop above already covers its unavailable and
            # inconclusive runs.
            elif det.detector == "TruForDetector" and _plain(det.assessment) not in (
                "UNAVAILABLE", "INCONCLUSIVE"
            ):
                for lim in (det.detector_metadata or {}).get("limitations", []):
                    limitations.append(f"TruFor: {lim}")

        if not learned_ran:
            limitations.append(
                "UNCERTAINTY: No learned forgery-localisation model (TruFor) "
                "produced observations for this analysis. Image forensic analysis "
                "relied on heuristic methods (ELA, FFT, noise), which are forensic "
                "indicators, not proof of manipulation."
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
            "Use EvidenceService.build_evidence_package(analysis_id, db) -- "
            "evidence requires a database session to retrieve real pipeline results."
        )
