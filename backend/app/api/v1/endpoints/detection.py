from fastapi import APIRouter, HTTPException

from app.schemas.detection import DetectionRequest

from app.services.analysis.media_analysis_service import (
    MediaAnalysisService
)

from app.services.detection.detection_container import (
    create_detection_router
)


router=APIRouter(
    prefix="/detection",
    tags=["Detection"]
)


detection_router=create_detection_router()

service=MediaAnalysisService(
    router=detection_router
)


import uuid

analysis_store = {}


@router.post("/analyze")
def analyze_media(
    request:DetectionRequest
):

    try:

        result=service.analyze(
            request.file_path,
            request.media_type
        )
        
        analysis_id = str(uuid.uuid4())
        if isinstance(result, dict):
            result["analysis_id"] = analysis_id
            analysis_store[analysis_id] = result

        return result

    except (
        FileNotFoundError,
        ValueError,
        RuntimeError
    ) as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )


@router.get("/{analysis_id}")
def get_analysis(analysis_id: str):
    if analysis_id not in analysis_store:
        raise HTTPException(
            status_code=404,
            detail="Analysis not found"
        )
    return analysis_store[analysis_id]