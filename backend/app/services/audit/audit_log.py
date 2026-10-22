"""
Hash-linked, append-only audit trail.

Each event's hash covers its own fields and the previous event's hash:

    event_hash = sha256(canonical_json({sequence, occurred_at, actor, action,
                                        case_id, media_id, analysis_id, details,
                                        previous_event_hash}))

so modifying, deleting, inserting or reordering any event changes the hash that
every later event committed to, and ``verify_chain`` reports the first break.

WHAT THIS PROVES, AND WHAT IT DOES NOT
--------------------------------------
It proves the stored trail is internally consistent: no recorded event has been
altered or removed *without the attacker also recomputing every later hash*.
Anyone with write access to the database can do exactly that, so this is tamper
*evidence* against accidental damage and naive edits, not tamper *proof* against
a database administrator. Anchoring the head hash outside the database (a
signed export, a printed report) is what closes that gap; ``chain_head`` exists
for that purpose.

CONCURRENCY
-----------
Appending reads the chain head and writes the next event. Two writers reading
the same head would produce a fork; the unique constraints on ``sequence`` and
``previous_event_hash`` make the second insert fail instead, and it is retried
against the new head.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.models import AuditEventRecord

logger = logging.getLogger(__name__)

#: ``previous_event_hash`` of the first event. A fixed value rather than NULL so
#: the unique constraint also forbids a second genesis event (SQL treats NULLs as
#: distinct, so a NULL genesis would allow any number of them).
GENESIS_HASH = "0" * 64

_MAX_APPEND_ATTEMPTS = 5


class AuditAction:
    USER_CREATED = "USER_CREATED"
    CASE_CREATED = "CASE_CREATED"
    EVIDENCE_INGESTED = "EVIDENCE_INGESTED"
    EVIDENCE_DELETED = "EVIDENCE_DELETED"
    ANALYSIS_STARTED = "ANALYSIS_STARTED"
    ANALYSIS_FINISHED = "ANALYSIS_FINISHED"


def _canonical(payload: dict[str, Any]) -> bytes:
    # sort_keys + fixed separators make the serialisation independent of dict
    # order and whitespace. No ``default=``: a value json cannot represent is an
    # error at write time, not a str() that might not round-trip at verify time.
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def compute_event_hash(
    *,
    sequence: int,
    occurred_at: str,
    actor: str,
    action: str,
    case_id: Optional[str],
    media_id: Optional[str],
    analysis_id: Optional[str],
    details: dict,
    previous_event_hash: str,
) -> str:
    return hashlib.sha256(_canonical({
        "sequence": sequence,
        "occurred_at": occurred_at,
        "actor": actor,
        "action": action,
        "case_id": case_id,
        "media_id": media_id,
        "analysis_id": analysis_id,
        "details": details,
        "previous_event_hash": previous_event_hash,
    })).hexdigest()


def _hash_of(row: AuditEventRecord) -> str:
    return compute_event_hash(
        sequence=row.sequence,
        occurred_at=row.occurred_at,
        actor=row.actor,
        action=row.action,
        case_id=row.case_id,
        media_id=row.media_id,
        analysis_id=row.analysis_id,
        details=row.details or {},
        previous_event_hash=row.previous_event_hash,
    )


def chain_head(db: Session) -> Optional[AuditEventRecord]:
    return (
        db.query(AuditEventRecord)
        .order_by(AuditEventRecord.sequence.desc())
        .first()
    )


def record_event(
    db: Session,
    *,
    actor: str,
    action: str,
    case_id: Optional[str] = None,
    media_id: Optional[str] = None,
    analysis_id: Optional[str] = None,
    details: Optional[dict] = None,
) -> AuditEventRecord:
    """
    Append one event and commit it.

    Commits the session: an audit event that is rolled back with some later,
    unrelated failure never happened as far as the trail is concerned, which is
    the opposite of what a trail is for. Callers must therefore not have pending
    work in ``db`` that they do not also intend to commit.
    """
    details = dict(details or {})
    # Fail before touching the database if details cannot be hashed faithfully.
    _canonical(details)

    for attempt in range(1, _MAX_APPEND_ATTEMPTS + 1):
        head = chain_head(db)
        sequence = head.sequence + 1 if head else 1
        previous = head.event_hash if head else GENESIS_HASH
        occurred_at = datetime.now(timezone.utc).isoformat()

        row = AuditEventRecord(
            sequence=sequence,
            occurred_at=occurred_at,
            actor=actor,
            action=action,
            case_id=case_id,
            media_id=media_id,
            analysis_id=analysis_id,
            details=details,
            previous_event_hash=previous,
            event_hash=compute_event_hash(
                sequence=sequence,
                occurred_at=occurred_at,
                actor=actor,
                action=action,
                case_id=case_id,
                media_id=media_id,
                analysis_id=analysis_id,
                details=details,
                previous_event_hash=previous,
            ),
        )
        db.add(row)
        try:
            db.commit()
        except IntegrityError:
            # Another writer extended the chain first. Retry on the new head.
            db.rollback()
            logger.info("Audit append raced at sequence %d (attempt %d)", sequence, attempt)
            continue
        db.refresh(row)
        return row

    raise RuntimeError(
        f"Could not append audit event {action!r} after {_MAX_APPEND_ATTEMPTS} attempts"
    )


@dataclass(frozen=True)
class ChainVerification:
    intact: bool
    events_checked: int
    head_hash: Optional[str]
    #: Sequence number of the first event that does not verify, if any.
    first_broken_sequence: Optional[int] = None
    reason: Optional[str] = None


def verify_chain(db: Session) -> ChainVerification:
    """
    Recompute every hash from genesis and check every link.

    Detects: an edited event (its recomputed hash differs), a deleted or
    inserted event (a sequence gap, or a link to a hash that is not its
    predecessor's), and reordering (both).
    """
    expected_previous = GENESIS_HASH
    expected_sequence = 1
    checked = 0
    last_hash: Optional[str] = None

    rows = db.query(AuditEventRecord).order_by(AuditEventRecord.sequence.asc())
    for row in rows.yield_per(500):
        def broken(reason: str) -> ChainVerification:
            return ChainVerification(
                intact=False,
                events_checked=checked,
                head_hash=last_hash,
                first_broken_sequence=row.sequence,
                reason=reason,
            )

        if row.sequence != expected_sequence:
            return broken(
                f"expected sequence {expected_sequence}, found {row.sequence}"
            )
        if row.previous_event_hash != expected_previous:
            return broken("link does not match the preceding event's hash")
        if _hash_of(row) != row.event_hash:
            return broken("event contents do not match its recorded hash")

        checked += 1
        expected_sequence += 1
        expected_previous = row.event_hash
        last_hash = row.event_hash

    return ChainVerification(intact=True, events_checked=checked, head_hash=last_hash)


def events_for(
    db: Session,
    *,
    case_id: Optional[str] = None,
    media_id: Optional[str] = None,
    analysis_id: Optional[str] = None,
    actor: Optional[str] = None,
    limit: int = 200,
    offset: int = 0,
) -> list[AuditEventRecord]:
    query = db.query(AuditEventRecord)
    if actor is not None:
        query = query.filter(AuditEventRecord.actor == actor)
    if case_id is not None:
        query = query.filter(AuditEventRecord.case_id == case_id)
    if media_id is not None:
        query = query.filter(AuditEventRecord.media_id == media_id)
    if analysis_id is not None:
        query = query.filter(AuditEventRecord.analysis_id == analysis_id)
    return (
        query.order_by(AuditEventRecord.sequence.asc())
        .offset(offset)
        .limit(limit)
        .all()
    )
