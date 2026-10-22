"""
The audit trail is hash-linked: editing, deleting or reordering an event is
detected, and concurrent writers cannot fork the chain.
"""
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from app.db.models import AuditEventRecord, Base
from app.services.audit import audit_log
from app.services.audit.audit_log import (
    GENESIS_HASH,
    chain_head,
    record_event,
    verify_chain,
)


@pytest.fixture
def session_factory(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'audit.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    yield factory
    engine.dispose()


@pytest.fixture
def db(session_factory):
    with session_factory() as session:
        yield session


def _three_events(db):
    record_event(db, actor="alice", action="CASE_CREATED", case_id="c1",
                 details={"title": "Case one"})
    record_event(db, actor="alice", action="EVIDENCE_INGESTED", case_id="c1",
                 media_id="m1", details={"sha256": "ab" * 32, "size_bytes": 10})
    record_event(db, actor="bob", action="ANALYSIS_FINISHED", media_id="m1",
                 analysis_id="a1", details={"status": "COMPLETED", "score": 0.125})


def test_events_link_from_genesis(db):
    _three_events(db)
    rows = db.query(AuditEventRecord).order_by(AuditEventRecord.sequence).all()

    assert [r.sequence for r in rows] == [1, 2, 3]
    assert rows[0].previous_event_hash == GENESIS_HASH
    assert rows[1].previous_event_hash == rows[0].event_hash
    assert rows[2].previous_event_hash == rows[1].event_hash


def test_an_untampered_chain_verifies(db):
    _three_events(db)
    result = verify_chain(db)

    assert result.intact
    assert result.events_checked == 3
    assert result.head_hash == chain_head(db).event_hash


def test_an_empty_chain_is_intact(db):
    result = verify_chain(db)
    assert result.intact and result.events_checked == 0 and result.head_hash is None


def test_editing_an_event_is_detected(db):
    _three_events(db)
    # Change history the way someone with database access would: in SQL,
    # bypassing the ORM entirely.
    db.execute(text(
        "UPDATE audit_events SET actor = 'mallory' WHERE sequence = 2"
    ))
    db.commit()

    result = verify_chain(db)
    assert not result.intact
    assert result.first_broken_sequence == 2
    assert "contents" in result.reason


def test_editing_details_is_detected(db):
    _three_events(db)
    db.execute(text(
        "UPDATE audit_events SET details = '{\"sha256\": \"forged\", \"size_bytes\": 10}'"
        " WHERE sequence = 2"
    ))
    db.commit()

    assert verify_chain(db).first_broken_sequence == 2


def test_deleting_an_event_is_detected(db):
    _three_events(db)
    db.execute(text("DELETE FROM audit_events WHERE sequence = 2"))
    db.commit()

    result = verify_chain(db)
    assert not result.intact
    assert result.first_broken_sequence == 3


def test_deleting_the_newest_event_cannot_be_detected_internally(db):
    """
    The documented limit: truncating the tail leaves a consistent shorter chain.
    Only an externally recorded head hash reveals it -- which is why
    verify_chain reports ``head_hash``.
    """
    _three_events(db)
    head_before = verify_chain(db).head_hash
    db.execute(text("DELETE FROM audit_events WHERE sequence = 3"))
    db.commit()

    result = verify_chain(db)
    assert result.intact
    assert result.head_hash != head_before


def test_a_second_genesis_event_is_refused(db):
    record_event(db, actor="alice", action="CASE_CREATED")
    db.add(AuditEventRecord(
        sequence=99, occurred_at="2026-01-01T00:00:00+00:00", actor="mallory",
        action="FORGED", details={}, previous_event_hash=GENESIS_HASH,
        event_hash="f" * 64,
    ))
    with pytest.raises(IntegrityError):
        db.commit()


def test_details_that_cannot_be_hashed_faithfully_are_refused(db):
    with pytest.raises(TypeError):
        record_event(db, actor="alice", action="X", details={"when": object()})
    assert db.query(AuditEventRecord).count() == 0


def test_a_racing_writer_is_retried_instead_of_forking(session_factory, monkeypatch):
    """
    Writer A reads the head, then writer B appends before A commits. A's first
    insert must fail on the unique constraints and be retried on B's event.
    """
    with session_factory() as a_db, session_factory() as b_db:
        record_event(a_db, actor="alice", action="FIRST")

        real_head = audit_log.chain_head
        calls = {"n": 0}

        def stale_head_once(session):
            head = real_head(session)
            calls["n"] += 1
            if calls["n"] == 1:
                # B extends the chain after A has read the head.
                record_event(b_db, actor="bob", action="INTERLEAVED")
            return head

        monkeypatch.setattr(audit_log, "chain_head", stale_head_once)
        record_event(a_db, actor="alice", action="SECOND")
        monkeypatch.setattr(audit_log, "chain_head", real_head)

        actions = [
            r.action for r in
            a_db.query(AuditEventRecord).order_by(AuditEventRecord.sequence)
        ]
        assert actions == ["FIRST", "INTERLEAVED", "SECOND"]
        assert verify_chain(a_db).intact
