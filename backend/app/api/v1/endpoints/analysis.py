"""
/api/v1/analysis -- Unified analysis pipeline endpoint.

POST /api/v1/analysis  -- Run the complete forensic pipeline on a media_id.
GET  /api/v1/analysis/{analysis_id}  -- Retrieve analysis status and results.
GET  /api/v1/analysis/  -- List analyses.
"""
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session

from app.core.auth import User, get_current_user
from app.core.rate_limit import ANALYSIS_LIMIT, limiter
from app.db.database import get_db
from app.db import repository as repo
from app.schemas.detection import AnalysisRequest, AnalysisResponse
from app.services.audit.audit_log import AuditAction, record_event
from app.services.orchestration.analysis_pipeline import AnalysisPipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analysis", tags=["Analysis"])

_pipeline = AnalysisPipeline()


@router.post(
    "/",
    response_model=AnalysisResponse,
    summary="Run forensic analysis pipeline",
)
@limiter.limit(ANALYSIS_LIMIT)
def run_analysis(
    request: Request,
    body: AnalysisRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Run the complete AIMD forensic analysis pipeline on an uploaded media file.

    Stages: metadata -> frames -> detection -> fingerprinting -> similarity
            -> source tracing -> provenance -> evidence -> report

    All results are persisted to the database and associated with the returned
    analysis_id.

    The analysis runs synchronously. For large video files this may take
    several minutes.
    """
    media = repo.get_owned_media(db, body.media_id, current_user.username)
    if not media:
        raise HTTPException(status_code=404, detail="Media not found")

    record_event(
        db,
        actor=current_user.username,
        action=AuditAction.ANALYSIS_STARTED,
        case_id=media.case_id,
        media_id=media.id,
        details={"sha256": media.sha256},
    )

    # Neither exception's text is returned: both can carry the absolute storage
    # path of the evidence file.
    try:
        result = _pipeline.run(
            media_id=body.media_id, db=db, requested_by=current_user.username
        )
    except FileNotFoundError:
        logger.exception("Evidence bytes missing for media %s", body.media_id)
        raise HTTPException(status_code=404, detail="Media file missing from storage")
    except Exception:
        logger.exception("Pipeline error for media %s", body.media_id)
        raise HTTPException(status_code=500, detail="Analysis pipeline error")

    record_event(
        db,
        actor=current_user.username,
        action=AuditAction.ANALYSIS_FINISHED,
        case_id=media.case_id,
        media_id=media.id,
        analysis_id=result["analysis_id"],
        details={
            "status": result["status"],
            "assessment": result.get("assessment"),
            "evidence_strength": result.get("evidence_strength"),
        },
    )

    return AnalysisResponse(
        analysis_id=result["analysis_id"],
        media_id=result["media_id"],
        status=result["status"],
        assessment=result.get("assessment"),
        evidence_strength=result.get("evidence_strength"),
        message=f"Stage failures: {result.get('stage_failures', [])}" if result.get("stage_failures") else None,
    )


@router.get(
    "/{analysis_id}",
    summary="Retrieve analysis status and summary",
)
def get_analysis(
    analysis_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    analysis = repo.get_owned_analysis(db, analysis_id, current_user.username)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    detections = repo.get_detections_for_analysis(db, analysis_id)
    assessment_record = repo.get_assessment_for_analysis(db, analysis_id)

    media = repo.get_media(db, analysis.media_id)
    media_type = media.media_type if media else (detections[0].media_type if detections else "image")

    # Find suspicious segments from video detections
    suspicious_segments = []
    for d in detections:
        if d.suspicious_segments:
            suspicious_segments = d.suspicious_segments
            break

    return {
        "analysis_id": analysis.id,
        "id": analysis.id,
        "media_id": analysis.media_id,
        "media_type": media_type,
        "status": analysis.status,
        "assessment": analysis.assessment,
        "evidence_strength": analysis.evidence_strength,
        "suspicious_segments": suspicious_segments,
        "created_at": analysis.created_at.isoformat() if analysis.created_at else None,
        "completed_at": analysis.completed_at.isoformat() if analysis.completed_at else None,
        "error_message": analysis.error_message,
        "run": {
            "requested_by": analysis.requested_by,
            "software_version": analysis.software_version,
            "detector_set_version": analysis.detector_set_version,
            "config_snapshot": analysis.config_snapshot,
            "duration_ms": analysis.duration_ms,
        },
        # NULL for analyses recorded before the forensic data model existed.
        "assessment_detail": (
            {
                "category": assessment_record.category,
                "evidence_strength": assessment_record.evidence_strength,
                "rule": assessment_record.rule,
                "rationale": assessment_record.rationale,
                "coverage_gaps": assessment_record.coverage_gaps,
            }
            if assessment_record else None
        ),
        # Reported beside the manipulation assessment, never folded into it.
        # NULL for analyses recorded before the branch existed.
        "ai_generation": assessment_record.ai_generation if assessment_record else None,
        "similarity_matches": [
            {
                "id": m.id,
                "match_type": m.match_type,
                "source_media_id": m.source_media_id,
                "external_url": m.external_url,
                "similarity": m.similarity,
                "method": m.method,
                "algorithm": m.algorithm,
                "rank": m.rank,
                "verification_status": m.verification_status,
                "verification": m.verification,
            }
            for m in repo.get_similarity_matches_for_analysis(db, analysis_id)
        ],
        "observations": [
            {
                "id": o.id,
                "detection_id": o.detection_id,
                "detector": o.detector,
                "signal_family": o.signal_family,
                "polarity": o.polarity,
                "score": o.score,
                "score_semantics": o.score_semantics,
                "localised": o.localised,
                "description": o.description,
            }
            for o in repo.get_observations_for_analysis(db, analysis_id)
        ],
        "findings": [
            {
                "id": f.id,
                "statement": f.statement,
                "supporting_observation_ids": [
                    link.observation_id for link in f.links if link.relation == "SUPPORTS"
                ],
                "contradicting_observation_ids": [
                    link.observation_id for link in f.links if link.relation == "CONTRADICTS"
                ],
            }
            for f in repo.get_findings_for_analysis(db, analysis_id)
        ],
        "detections": [
            {
                "id": d.id,
                "detector": d.detector,
                "detector_version": d.detector_version,
                "media_type": d.media_type,
                "manipulation_type": d.manipulation_type,
                "availability": d.availability,
                "execution_status": d.execution_status,
                "status_reason": d.status_reason,
                "score_semantics": d.score_semantics,
                "assessment": d.assessment,
                "confidence": d.confidence,
                "regions": [
                    {
                        "type": r.region_type,
                        "confidence": r.confidence,
                        "bbox": r.bbox,
                        "start_timestamp": r.start_timestamp,
                        "end_timestamp": r.end_timestamp,
                    }
                    for r in d.regions
                ],
                "metadata": d.detector_metadata or {},
            }
            for d in detections
        ],
    }


@router.get("/", summary="List analyses")
def list_analyses(
    # Bounded: an unbounded ``limit`` let one request materialise every analysis
    # with its detections into a single response.
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
    media_id: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    items = repo.list_owned_analyses(
        db, current_user.username, limit=limit, offset=offset, media_id=media_id
    )
    return {
        "items": [
            {
                "id": a.id,
                "analysis_id": a.id,
                "media_id": a.media_id,
                "original_filename": a.media.original_filename if a.media else None,
                "media_type": a.media.media_type if a.media else None,
                "status": a.status,
                "assessment": a.assessment,
                "evidence_strength": a.evidence_strength,
                "created_at": a.created_at.isoformat() if a.created_at else None,
            }
            for a in items
        ],
        "count": len(items),
    }
