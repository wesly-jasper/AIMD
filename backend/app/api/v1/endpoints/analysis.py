"""
/api/v1/analysis -- Unified analysis pipeline endpoint.

POST /api/v1/analysis  -- Run the complete forensic pipeline on a media_id.
GET  /api/v1/analysis/{analysis_id}  -- Retrieve analysis status and results.
GET  /api/v1/analysis/  -- List analyses.
POST /api/v1/analysis/{analysis_id}/investigate  -- Start a Nemotron investigation.
GET  /api/v1/analysis/{analysis_id}/investigation  -- The latest investigation.
GET  /api/v1/analysis/{analysis_id}/detections/{detection_id}/artefacts/{kind}  -- A heatmap.
GET  /api/v1/analysis/{analysis_id}/report?format=json|md|html  -- The stored report.
"""
import hashlib
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from sqlalchemy.orm import Session

from app.core.auth import User, get_current_user
from app.core.config import settings
from app.core.rate_limit import ANALYSIS_LIMIT, limiter
from app.core.safe_errors import scrub
from app.db.database import get_db
from app.db import repository as repo
from app.services.agent import nemotron_agent
from app.services.agent.nemotron_agent import NemotronAgent
from app.services.provenance.timeline import build_timeline
from app.schemas.detection import AnalysisRequest, AnalysisResponse, InvestigationRequest
from app.services.audit.audit_log import AuditAction, record_event
from app.services.evidence.report_html import render_report_html
from app.services.evidence.report_investigation import investigation_json, investigation_markdown
from app.services.orchestration.analysis_pipeline import AnalysisPipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analysis", tags=["Analysis"])

_pipeline = AnalysisPipeline()
_agent = NemotronAgent()


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
        details={"sha256": media.sha256, "source_url": body.source_url},
    )

    # Neither exception's text is returned: both can carry the absolute storage
    # path of the evidence file.
    try:
        result = _pipeline.run(
            media_id=body.media_id, db=db, requested_by=current_user.username,
            source_url=body.source_url,
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
    sources = repo.get_sources_for_analysis(db, analysis_id)
    exif = media.metadata_record.exif_data if media and media.metadata_record else None
    raw = (media.metadata_record.raw_metadata or {}) if media and media.metadata_record else {}
    # NULL for media analysed before Content Credentials were checked.
    c2pa = raw.get("c2pa")

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
        # Also scrubbed on read: rows written before safe_reason hold raw text.
        "error_message": scrub(analysis.error_message),
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
        # Which source connectors ran: an empty ``sources`` list means "not
        # found in what was searched" only where the connector COMPLETED.
        "source_searches": analysis.source_searches,
        "sources": [
            {
                "id": s.id,
                "url": s.url,
                "title": s.title,
                "domain": s.domain,
                "provider": s.provider,
                "published_at": s.source_timestamp.isoformat() if s.source_timestamp else None,
                "similarity": s.similarity,
                "is_earliest_known": s.is_earliest_known,
                "verification_status": s.verification_status,
                "verification": s.verification,
                "citation": s.citation,
            }
            for s in sources
        ],
        # Rebuilt from the stored sources and metadata on every read.
        "provenance_timeline": build_timeline(sources, exif, c2pa),
        "content_credentials": c2pa,
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


# ── Investigation agent ───────────────────────────────────────────────────────

def _aware(value: datetime) -> datetime:
    # SQLite hands back naive datetimes for timezone-aware columns.
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


@router.post(
    "/{analysis_id}/investigate",
    status_code=202,
    summary="Start a Nemotron investigation of a finished analysis",
)
def start_investigation(
    analysis_id: str,
    background_tasks: BackgroundTasks,
    body: Optional[InvestigationRequest] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Ask NVIDIA Nemotron to explain this analysis using AIMD's read-only tools.

    Runs in the background; poll GET .../investigation for the result. The
    summary is INFERENCE grounded in the cited tool results, never forensic
    proof, and the analysis itself is never changed. Without a model credential
    the investigation is stored as UNAVAILABLE straight away.
    """
    analysis = repo.get_owned_analysis(db, analysis_id, current_user.username)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    if analysis.status in ("QUEUED", "PROCESSING"):
        raise HTTPException(status_code=409, detail="Analysis has not finished")

    latest = repo.get_latest_investigation(db, analysis_id)
    if latest is not None and latest.status == nemotron_agent.RUNNING:
        if _aware(latest.created_at) > nemotron_agent.stale_before():
            raise HTTPException(status_code=409, detail="An investigation of this analysis is already running")
        # Long past its timeout: the run was lost (e.g. a restart). Close it.
        repo.update_investigation(
            db, latest.id, status=nemotron_agent.FAILED,
            error_message="run lost: never finished within its time limit",
            completed_at=datetime.now(timezone.utc),
        )

    since = datetime.now(timezone.utc) - timedelta(hours=1)
    if repo.count_investigations_since(db, current_user.username, since) >= settings.agent_hourly_limit:
        raise HTTPException(
            status_code=429,
            detail=f"Hourly limit reached: {settings.agent_hourly_limit} investigations per hour",
        )

    source_url = body.source_url if body else None
    reason = _agent.unavailable_reason()
    record = repo.create_investigation(db, {
        "analysis_id": analysis_id,
        "requested_by": current_user.username,
        "status": nemotron_agent.UNAVAILABLE if reason else nemotron_agent.RUNNING,
        "model": settings.nemotron_model,
        "error_message": reason,
        "completed_at": datetime.now(timezone.utc) if reason else None,
    })
    record_event(
        db,
        actor=current_user.username,
        action=AuditAction.INVESTIGATION_STARTED,
        case_id=analysis.media.case_id if analysis.media else None,
        media_id=analysis.media_id,
        analysis_id=analysis_id,
        details={
            "investigation_id": record.id,
            "model": settings.nemotron_model,
            "status": record.status,
            "source_url": source_url,
        },
    )
    if not reason:
        background_tasks.add_task(nemotron_agent.run_investigation, record.id, _agent, source_url)
    return {
        "investigation_id": record.id,
        "analysis_id": analysis_id,
        "status": record.status,
        "error_message": reason,
    }


@router.get(
    "/{analysis_id}/investigation",
    summary="The latest investigation of an analysis",
)
def get_investigation(
    analysis_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not repo.get_owned_analysis(db, analysis_id, current_user.username):
        raise HTTPException(status_code=404, detail="Analysis not found")
    record = repo.get_latest_investigation(db, analysis_id)
    if record is None:
        raise HTTPException(status_code=404, detail="No investigation for this analysis")
    return nemotron_agent.serialise_investigation(record)


@router.get(
    "/{analysis_id}/detections/{detection_id}/artefacts/{kind}",
    summary="A detector's evidence artefact, e.g. a TruFor heatmap",
)
def get_artefact(
    analysis_id: str,
    detection_id: str,
    kind: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Only a path a detector recorded for this detection is served, and only from
    below ``settings.artefacts_dir``. The bytes are checked against the SHA-256
    recorded at analysis time, so a replaced file is refused rather than shown.
    """
    if not repo.get_owned_analysis(db, analysis_id, current_user.username):
        raise HTTPException(status_code=404, detail="Analysis not found")
    detection = next(
        (d for d in repo.get_detections_for_analysis(db, analysis_id) if d.id == detection_id),
        None,
    )
    artefacts = ((detection.detector_metadata or {}).get("artefacts") or {}) if detection else {}
    artefact = artefacts.get(kind)
    if not isinstance(artefact, dict) or not isinstance(artefact.get("path"), str):
        raise HTTPException(status_code=404, detail="No such artefact")

    root = Path(settings.artefacts_dir).resolve()
    path = (root / artefact["path"]).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise HTTPException(status_code=404, detail="Artefact missing from storage")
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if artefact.get("sha256") and digest != artefact["sha256"]:
        raise HTTPException(
            status_code=409,
            detail="Artefact bytes no longer match the hash recorded at analysis",
        )
    return Response(
        content=data,
        media_type=artefact.get("media_type") or "application/octet-stream",
        headers={"X-Content-SHA256": digest, "Cache-Control": "private, max-age=3600"},
    )


@router.get("/{analysis_id}/report", summary="The stored report as JSON, Markdown or HTML")
def get_report(
    analysis_id: str,
    format: str = Query("json", pattern="^(json|md|html)$"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    The report written when the analysis finished, with the latest investigation
    appended (see ``report_investigation``). HTML is rendered from the Markdown
    with everything escaped; its print stylesheet makes it the PDF form ("Save
    as PDF").
    """
    analysis = repo.get_owned_analysis(db, analysis_id, current_user.username)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    report = repo.get_report_for_analysis(db, analysis_id)
    if report is None:
        raise HTTPException(status_code=404, detail="No report has been written for this analysis")

    filename = f"aimd-report-{analysis_id}.{format}"
    disposition = {"Content-Disposition": f'inline; filename="{filename}"'}
    # Added on every read: investigations run after the report was written.
    investigation = repo.get_latest_investigation(db, analysis_id)
    if format == "json":
        if report.json_report is None:
            raise HTTPException(status_code=404, detail="This report has no JSON form")
        return JSONResponse(
            {**report.json_report, "investigation": investigation_json(investigation)},
            headers=disposition,
        )
    if report.markdown_report is None:
        raise HTTPException(status_code=404, detail="This report has no Markdown form")
    markdown = report.markdown_report + investigation_markdown(investigation)
    if format == "md":
        return PlainTextResponse(markdown, media_type="text/markdown; charset=utf-8",
                                 headers=disposition)
    return HTMLResponse(
        render_report_html(markdown, f"AIMD report {analysis_id}"),
        headers={
            **disposition,
            # The document is static: no script, no fetch, nothing but its own styles.
            "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'",
        },
    )
