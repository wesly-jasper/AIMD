"""
/api/v1/cases -- the caller's folders.

The app calls them folders: an optional way to group uploads. Media joins one
at upload (``POST /media/upload`` with a ``case_id`` form field). A caller sees
only folders it created; anything else is a 404.
"""
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.auth import User, get_current_user
from app.db import repository as repo
from app.db.database import get_db
from app.services.audit.audit_log import AuditAction, record_event

router = APIRouter(prefix="/cases", tags=["Cases"])


class CaseCreate(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    description: Optional[str] = Field(default=None, max_length=10_000)


def _case_summary(case) -> dict:
    return {
        "case_id": case.id,
        "title": case.title,
        "description": case.description,
        "status": case.status,
        "created_by": case.created_by,
        "created_at": case.created_at.isoformat() if case.created_at else None,
    }


@router.post("/", status_code=201, summary="Open a case")
def create_case(
    body: CaseCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    case = repo.create_case(db, {
        "title": body.title.strip(),
        "description": body.description,
        "status": "OPEN",
        "created_by": current_user.username,
    })
    record_event(
        db,
        actor=current_user.username,
        action=AuditAction.CASE_CREATED,
        case_id=case.id,
        details={"title": case.title},
    )
    return _case_summary(case)


@router.get("/", summary="List cases")
def list_cases(
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    items = repo.list_cases(db, current_user.username, limit=limit, offset=offset)
    counts = repo.count_media_by_case(db, [c.id for c in items])
    return {
        "items": [
            {**_case_summary(c), "evidence_count": counts.get(c.id, 0)} for c in items
        ],
        "count": len(items),
    }


@router.get("/{case_id}", summary="Retrieve a case and its evidence")
def get_case(
    case_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    case = repo.get_owned_case(db, case_id, current_user.username)
    if case is None:
        raise HTTPException(status_code=404, detail="Folder not found")
    return {
        **_case_summary(case),
        "media": [
            {
                "media_id": m.id,
                "original_filename": m.original_filename,
                "media_type": m.media_type,
                "sha256": m.sha256,
                "uploaded_by": m.uploaded_by,
                "created_at": m.created_at.isoformat() if m.created_at else None,
            }
            for m in repo.list_media_for_case(db, case_id)
        ],
    }
