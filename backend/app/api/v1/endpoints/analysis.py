"""
/api/v1/analysis — Unified analysis pipeline endpoint.

POST /api/v1/analysis  — Run the complete forensic pipeline on a media_id.
GET  /api/v1/analysis/{analysis_id}  — Retrieve analysis status and results.
GET  /api/v1/analysis/  — List analyses.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db import repository as repo
from app.schemas.detection import AnalysisRequest, AnalysisResponse
from app.services.orchestration.analysis_pipeline import AnalysisPipeline

router = APIRouter(prefix="/analysis", tags=["Analysis"])

_pipeline = AnalysisPipeline()


@router.post(
    "/",
    response_model=AnalysisResponse,
    summary="Run forensic analysis pipeline",
)
def run_analysis(
    request: AnalysisRequest,
    db: Session = Depends(get_db),
):
    """
    Run the complete AIMD forensic analysis pipeline on an uploaded media file.

    Stages: metadata → frames → detection → fingerprinting → similarity
            → source tracing → provenance → evidence → report

    All results are persisted to the database and associated with the returned
    analysis_id.

    The analysis runs synchronously. For large video files this may take
    several minutes.
    """
    # Verify media exists
    media = repo.get_media(db, request.media_id)
    if not media:
        raise HTTPException(
            status_code=404,
            detail=f"Media not found: {request.media_id}",
        )

    try:
        result = _pipeline.run(media_id=request.media_id, db=db)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}")

    return AnalysisResponse(
        analysis_id=result["analysis_id"],
        media_id=result["media_id"],
        status=result["status"],
        assessment=result.get("assessment"),
        overall_confidence=result.get("overall_confidence"),
        message=f"Stage failures: {result.get('stage_failures', [])}" if result.get("stage_failures") else None,
    )


@router.get(
    "/{analysis_id}",
    summary="Retrieve analysis status and summary",
)
def get_analysis(
    analysis_id: str,
    db: Session = Depends(get_db),
):
    analysis = repo.get_analysis(db, analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail=f"Analysis not found: {analysis_id}")

    detections = repo.get_detections_for_analysis(db, analysis_id)

    return {
        "analysis_id": analysis.id,
        "media_id": analysis.media_id,
        "status": analysis.status,
        "assessment": analysis.assessment,
        "overall_confidence": analysis.overall_confidence,
        "created_at": analysis.created_at.isoformat() if analysis.created_at else None,
        "completed_at": analysis.completed_at.isoformat() if analysis.completed_at else None,
        "error_message": analysis.error_message,
        "detections": [
            {
                "detector": d.detector,
                "manipulation_type": d.manipulation_type,
                "assessment": d.assessment,
                "confidence": d.confidence,
            }
            for d in detections
        ],
    }


@router.get("/", summary="List analyses")
def list_analyses(
    limit: int = 20,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    items = repo.list_analyses(db, limit=limit, offset=offset)
    return {
        "items": [
            {
                "analysis_id": a.id,
                "media_id": a.media_id,
                "status": a.status,
                "assessment": a.assessment,
                "created_at": a.created_at.isoformat() if a.created_at else None,
            }
            for a in items
        ],
        "count": len(items),
    }
