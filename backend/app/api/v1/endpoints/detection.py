"""
/api/v1/detection — Direct detection endpoint.

Provides raw access to the detection engine without running the full pipeline.
For full forensic analysis, use POST /api/v1/analysis instead.

GET /api/v1/detection/{analysis_id} reads from DB (set by the pipeline).
POST /api/v1/detection/analyze runs detection only (no fingerprinting/provenance).
"""
import uuid
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db import repository as repo
from app.schemas.detection import DetectionRequest
from app.services.analysis.media_analysis_service import MediaAnalysisService
from app.services.detection.detection_container import create_detection_router as _create_router

router = APIRouter(prefix="/detection", tags=["Detection"])

_detection_router = _create_router()
_service = MediaAnalysisService(router=_detection_router)


@router.post("/analyze", summary="Run detection only (no full pipeline)")
def analyze_media(
    request: DetectionRequest,
    db: Session = Depends(get_db),
):
    """
    Run multimodal detection on a file path.
    Results are persisted to the database.

    NOTE: For full forensic investigation (fingerprinting, provenance, evidence),
    use POST /api/v1/analysis with a media_id instead.
    """
    try:
        result = _service.analyze(request.file_path, request.media_type)
        analysis_id = str(uuid.uuid4())

        if isinstance(result, dict):
            result["analysis_id"] = analysis_id

        return result

    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Detection error: {exc}")


@router.get("/{analysis_id}", summary="Retrieve detection results")
def get_detection_results(
    analysis_id: str,
    db: Session = Depends(get_db),
):
    """Retrieve detection results for a completed pipeline analysis."""
    analysis = repo.get_analysis(db, analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail=f"Analysis not found: {analysis_id}")

    detections = repo.get_detections_for_analysis(db, analysis_id)

    return {
        "analysis_id": analysis_id,
        "media_id": analysis.media_id,
        "status": analysis.status,
        "assessment": analysis.assessment,
        "overall_confidence": analysis.overall_confidence,
        "detections": [
            {
                "detector": d.detector,
                "media_type": d.media_type,
                "manipulation_type": d.manipulation_type,
                "assessment": d.assessment,
                "manipulation_detected": d.manipulation_detected,
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