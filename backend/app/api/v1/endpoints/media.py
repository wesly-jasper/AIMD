from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db import repository as repo
from app.services.ingestion.media_ingestion import MediaIngestionService

router = APIRouter(prefix="/media", tags=["Media"])

_service = MediaIngestionService()


@router.post("/upload", summary="Upload media for forensic analysis")
async def upload_media(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Upload an image, video, or audio file.

    Validates the file type using magic bytes, stores it securely,
    computes SHA-256, and persists a MediaRecord to the database.

    Returns the media_id — use this with POST /api/v1/analysis to run analysis.
    """
    try:
        result = await _service.ingest(file, db=db)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion error: {exc}")


@router.get("/{media_id}", summary="Retrieve media metadata")
def get_media(
    media_id: str,
    db: Session = Depends(get_db),
):
    """
    Retrieve stored metadata and fingerprints for an uploaded media file.
    """
    media = repo.get_media(db, media_id)
    if not media:
        raise HTTPException(status_code=404, detail=f"Media not found: {media_id}")

    fingerprints = repo.get_fingerprints_for_media(db, media_id)
    meta = media.metadata_record

    return {
        "media_id": media.id,
        "original_filename": media.original_filename,
        "stored_filename": media.stored_filename,
        "content_type": media.content_type,
        "media_type": media.media_type,
        "size_bytes": media.size_bytes,
        "sha256": media.sha256,
        "created_at": media.created_at.isoformat() if media.created_at else None,
        "metadata": {
            "width": meta.width if meta else None,
            "height": meta.height if meta else None,
            "fps": meta.fps if meta else None,
            "duration_seconds": meta.duration_seconds if meta else None,
            "codec": meta.codec if meta else None,
            "exif_data": meta.exif_data if meta else {},
        } if meta else {},
        "fingerprints": [
            {
                "algorithm": fp.algorithm,
                "algorithm_version": fp.algorithm_version,
                "value": fp.value,
                "scope": fp.scope,
            }
            for fp in fingerprints
        ],
        "status": "STORED",
    }


@router.get("/", summary="List uploaded media")
def list_media(
    limit: int = 20,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    media_list = repo.list_media(db, limit=limit, offset=offset)
    return {
        "items": [
            {
                "media_id": m.id,
                "original_filename": m.original_filename,
                "media_type": m.media_type,
                "size_bytes": m.size_bytes,
                "sha256": m.sha256,
                "created_at": m.created_at.isoformat() if m.created_at else None,
            }
            for m in media_list
        ],
        "count": len(media_list),
    }