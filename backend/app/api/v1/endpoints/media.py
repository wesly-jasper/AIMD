"""
/api/v1/media -- the caller's own media.

Every route here reads only media the caller uploaded; anything else is a 404,
indistinguishable from an id that does not exist.
"""
import logging
import shutil
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.core.auth import User, get_current_user
from app.core.config import settings
from app.core.rate_limit import UPLOAD_LIMIT, limiter
from app.core.security import resolve_evidence_path
from app.db import repository as repo
from app.db.database import get_db
from app.services.audit.audit_log import AuditAction, record_event
from app.services.ingestion.media_ingestion import MediaIngestionService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/media", tags=["Media"])

_service = MediaIngestionService()


def _owned(db: Session, media_id: str, user: User):
    media = repo.get_owned_media(db, media_id, user.username)
    if media is None:
        raise HTTPException(status_code=404, detail="Media not found")
    return media


def _latest(m) -> Optional[dict]:
    """The media's most recent analysis, so a list needs no request per row."""
    if not m.analyses:
        return None
    a = max(m.analyses, key=lambda x: x.created_at)
    return {
        "analysis_id": a.id,
        "status": a.status,
        "assessment": a.assessment,
        "evidence_strength": a.evidence_strength,
        "created_at": a.created_at.isoformat() if a.created_at else None,
    }


def _summary(m) -> dict:
    return {
        "media_id": m.id,
        "original_filename": m.original_filename,
        "media_type": m.media_type,
        "content_type": m.content_type,
        "size_bytes": m.size_bytes,
        "sha256": m.sha256,
        "case_id": m.case_id,
        "uploaded_by": m.uploaded_by,
        "created_at": m.created_at.isoformat() if m.created_at else None,
    }


def _remove_stored_bytes(file_path: str, media_id: str) -> None:
    """The evidence file and the frames derived from it. Best effort, logged."""
    try:
        resolve_evidence_path(file_path).unlink(missing_ok=True)
    except Exception:
        logger.exception("Could not remove stored bytes for media %s", media_id)
    for root in (settings.frames_dir, settings.keyframes_dir):
        shutil.rmtree(root / media_id, ignore_errors=True)


@router.post("/upload", summary="Upload media for analysis")
@limiter.limit(UPLOAD_LIMIT)
async def upload_media(
    request: Request,
    file: UploadFile = File(...),
    case_id: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Upload an image, video or audio file into the caller's account.

    Validates the content by magic bytes, stores it, records its SHA-256, and
    returns its id -- then POST /api/v1/analysis runs the detectors on it.
    """
    if case_id is not None and repo.get_owned_case(db, case_id, current_user.username) is None:
        raise HTTPException(status_code=404, detail="Folder not found")

    since = datetime.now(timezone.utc) - timedelta(hours=24)
    used = repo.count_uploads_since(db, current_user.username, since)
    if used >= settings.daily_upload_limit:
        raise HTTPException(
            status_code=429,
            detail=f"Daily limit reached: {settings.daily_upload_limit} uploads in 24 hours",
        )

    try:
        result = await _service.ingest(
            file, db=db, case_id=case_id, uploaded_by=current_user.username
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        # The exception text can carry absolute storage paths; logged, not returned.
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail="Ingestion failed")

    if not result.get("persisted"):
        # The bytes were written but no record of them exists. Remove them rather
        # than report success for evidence the account can never see again.
        _remove_stored_bytes(result["file_path"], result["media_id"])
        raise HTTPException(status_code=500, detail="The upload could not be recorded")

    record_event(
        db,
        actor=current_user.username,
        action=AuditAction.EVIDENCE_INGESTED,
        case_id=case_id,
        media_id=result["media_id"],
        details={
            "sha256": result["sha256"],
            "size_bytes": result["size_bytes"],
            "original_filename": result["original_filename"],
            "media_type": result["media_type"],
        },
    )
    # Not ``result`` itself: it carries the absolute storage path and the
    # extraction internals, none of which the client needs.
    return _summary(repo.get_media(db, result["media_id"]))


@router.get("/", summary="List your media")
def list_media(
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
    case_id: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    items = repo.list_owned_media(
        db, current_user.username, limit=limit, offset=offset, case_id=case_id
    )
    return {
        "items": [{**_summary(m), "latest_analysis": _latest(m)} for m in items],
        "count": len(items),
    }


@router.get("/usage", summary="Your uploads against the daily limit")
def usage(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # Declared before /{media_id}, which would otherwise capture "usage".
    since = datetime.now(timezone.utc) - timedelta(hours=24)
    return {
        "used": repo.count_uploads_since(db, current_user.username, since),
        "limit": settings.daily_upload_limit,
        "window_hours": 24,
    }


@router.get("/{media_id}", summary="Retrieve one of your media")
def get_media(
    media_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    media = _owned(db, media_id, current_user)
    meta = media.metadata_record

    # Derived from the filesystem, not asserted.
    try:
        bytes_present = resolve_evidence_path(media.file_path).is_file()
    except HTTPException:
        bytes_present = False

    return {
        **_summary(media),
        "metadata": {
            "width": meta.width,
            "height": meta.height,
            "fps": meta.fps,
            "duration_seconds": meta.duration_seconds,
            "codec": meta.codec,
        } if meta else {},
        "storage_status": "STORED" if bytes_present else "BYTES_MISSING",
    }


@router.get("/{media_id}/download", summary="Download one of your media")
def download_media(
    media_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    The original bytes. The stored path is re-confined to the evidence roots on
    every request (defence in depth against a path persisted by any other route).
    """
    media = _owned(db, media_id, current_user)
    path = resolve_evidence_path(media.file_path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Media file missing from storage")
    return FileResponse(
        str(path),
        media_type=media.content_type or "application/octet-stream",
        filename=media.original_filename,
        # Uploads from before content sniffing hold the client's declared type;
        # never let a browser second-guess it into something renderable.
        headers={"X-Content-Type-Options": "nosniff"},
    )


@router.delete("/{media_id}", status_code=204, summary="Delete one of your media")
def delete_media(
    media_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Delete the media, its analyses and everything derived from them, and the
    stored bytes. The audit trail keeps a record that it existed and was deleted:
    audit events are not foreign keys, so they survive their subject.
    """
    media = _owned(db, media_id, current_user)
    file_path, sha256, name, case_id = media.file_path, media.sha256, media.original_filename, media.case_id

    repo.delete_media(db, media)
    _remove_stored_bytes(file_path, media_id)

    record_event(
        db,
        actor=current_user.username,
        action=AuditAction.EVIDENCE_DELETED,
        case_id=case_id,
        media_id=media_id,
        details={"sha256": sha256, "original_filename": name},
    )
