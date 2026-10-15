"""
Jobs endpoint -- polling for long-running analyses.

Spec requires:
  GET /api/v1/jobs/{job_id}

We map job_id -> analysis_id (same UUID). The pipeline is synchronous,
but this endpoint allows the frontend to poll uniformly whether async
queue is introduced later.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db import repository as repo

router = APIRouter(prefix="/jobs", tags=["Jobs"])


@router.get("/{job_id}", summary="Get job / analysis status")
def get_job(job_id: str, db: Session = Depends(get_db)):
    """
    Returns job status for polling.

    States: QUEUED | PROCESSING | ANALYZING | COMPLETED | FAILED | PARTIAL
    """
    analysis = repo.get_analysis(db, job_id)
    if not analysis:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    # Map internal status to spec states
    status_map = {
        "QUEUED": "QUEUED",
        "PROCESSING": "PROCESSING",
        "COMPLETED": "COMPLETED",
        "FAILED": "FAILED",
        "PARTIAL": "COMPLETED",  # partial still counts as completed with limitations
        "INCONCLUSIVE": "COMPLETED",
    }

    return {
        "job_id": analysis.id,
        "analysis_id": analysis.id,
        "media_id": analysis.media_id,
        "status": status_map.get(analysis.status, analysis.status),
        "raw_status": analysis.status,
        "assessment": analysis.assessment,
        "overall_confidence": analysis.overall_confidence,
        "error_message": analysis.error_message,
        "created_at": analysis.created_at.isoformat() if analysis.created_at else None,
        "completed_at": analysis.completed_at.isoformat() if analysis.completed_at else None,
        "progress": 100 if analysis.status in ("COMPLETED", "FAILED", "PARTIAL", "INCONCLUSIVE") else 50,
    }


@router.get("/", summary="List jobs / analyses")
def list_jobs(limit: int = 20, offset: int = 0, db: Session = Depends(get_db)):
    items = repo.list_analyses(db, limit=limit, offset=offset)
    return {
        "items": [
            {
                "job_id": a.id,
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
