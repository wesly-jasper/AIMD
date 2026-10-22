"""
/api/v1/audit -- the hash-linked audit trail.

GET /audit/events  -- the caller's own events, filterable by case, media or analysis
GET /audit/verify  -- recompute the whole chain (administrators only: it reads
                      every event, so it is not offered to every caller)
"""
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.core.auth import User, get_current_user, require_role
from app.db.database import get_db
from app.services.audit.audit_log import events_for, verify_chain

router = APIRouter(prefix="/audit", tags=["Audit"])


@router.get("/events", summary="List audit events")
def list_events(
    case_id: Optional[str] = None,
    media_id: Optional[str] = None,
    analysis_id: Optional[str] = None,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # Only events the caller performed: another account's actions -- and the
    # names of its files -- are not this caller's to read.
    events = events_for(
        db, case_id=case_id, media_id=media_id, analysis_id=analysis_id,
        actor=current_user.username, limit=limit, offset=offset,
    )
    return {
        "items": [
            {
                "sequence": e.sequence,
                "occurred_at": e.occurred_at,
                "actor": e.actor,
                "action": e.action,
                "case_id": e.case_id,
                "media_id": e.media_id,
                "analysis_id": e.analysis_id,
                "details": e.details,
                "previous_event_hash": e.previous_event_hash,
                "event_hash": e.event_hash,
            }
            for e in events
        ],
        "count": len(events),
    }


@router.get(
    "/verify",
    summary="Verify the audit chain",
    dependencies=[Depends(require_role("admin"))],
)
def verify(db: Session = Depends(get_db)):
    result = verify_chain(db)
    return {
        "intact": result.intact,
        "events_checked": result.events_checked,
        "head_hash": result.head_hash,
        "first_broken_sequence": result.first_broken_sequence,
        "reason": result.reason,
        # Stated with the result so it cannot be read as more than it is.
        "scope": (
            "Internal consistency of the stored trail. Detects edits, deletions "
            "and reordering, but not a rewrite by someone able to recompute every "
            "later hash; compare head_hash against an externally recorded value "
            "for that."
        ),
    }
