"""forensic data model: users, cases, observations, findings, assessments, audit

New tables
----------
users                 accounts, previously held in process memory only
cases                 an investigation grouping evidence
observations          one measurement by one detector that actually ran
findings              a statement about the evidence
finding_observations  each observation's bearing on a finding: SUPPORTS | CONTRADICTS
assessments           the analysis-level category with its rule, rationale and
                      coverage gaps -- previously two bare columns on ``analyses``
audit_events          hash-linked append-only trail

Changed tables
--------------
media     + case_id (FK cases, RESTRICT), uploaded_by
analyses  + requested_by, software_version, detector_set_version,
            config_snapshot, duration_ms
          - overall_confidence

WHY overall_confidence IS DROPPED, NOT KEPT
-------------------------------------------
It held a weighted mean of every detector's score. Those scores are on
incomparable scales (a raw model score, a heuristic severity) from detectors
that are not independent, and a detector that never ran contributed 0.0. The
mean had no referent, and it gated the "clean" verdict. Nothing has written it
since a1b2c3d4e5f6. Leaving the column in place invites a future reader to
display it; any historical values are of the same meaningless kind, so nothing
of evidential value is lost by removing them.

SQLITE TABLE REBUILD
--------------------
SQLite cannot drop a column or add a foreign key in place, so ``analyses`` and
``media`` are rebuilt via ``batch_alter_table`` (create copy, copy rows, drop
original, rename). The drop is the dangerous step: with ``PRAGMA foreign_keys=ON``
dropping ``analyses`` would fire ON DELETE CASCADE and wipe every detection,
fingerprint and report. ``alembic/env.py`` therefore disables foreign-key
enforcement for the migration and runs ``PRAGMA foreign_key_check`` afterwards.
``tests/integration/test_migrations.py`` seeds a legacy database with child rows
and asserts they survive this revision.

Idempotent like its predecessors: databases built by ``create_all`` may already
hold any subset of these tables and columns.

Revision ID: b7e1f0c2d9a4
Revises: a1b2c3d4e5f6
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "b7e1f0c2d9a4"
down_revision: Union[str, Sequence[str], None] = "a1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _tz() -> sa.DateTime:
    return sa.DateTime(timezone=True)


def _inspector() -> sa.engine.reflection.Inspector:
    return sa.inspect(op.get_bind())


def _columns(table: str) -> set[str]:
    return {col["name"] for col in _inspector().get_columns(table)}


def _new_tables() -> list[tuple[str, list]]:
    return [
        ("users", [
            sa.Column("username", sa.String(150), primary_key=True),
            sa.Column("email", sa.String(254)),
            sa.Column("hashed_password", sa.String(), nullable=False),
            sa.Column("role", sa.String(), nullable=False),
            sa.Column("is_active", sa.Boolean(), nullable=False),
            sa.Column("created_at", _tz(), nullable=False),
        ]),
        ("cases", [
            sa.Column("id", sa.String(), primary_key=True),
            sa.Column("title", sa.String(300), nullable=False),
            sa.Column("description", sa.Text()),
            sa.Column("status", sa.String(), nullable=False),
            sa.Column("created_by", sa.String(150)),
            sa.Column("created_at", _tz(), nullable=False),
        ]),
        ("observations", [
            sa.Column("id", sa.String(), primary_key=True),
            sa.Column("analysis_id", sa.String(),
                      sa.ForeignKey("analyses.id", ondelete="CASCADE"),
                      nullable=False, index=True),
            sa.Column("detection_id", sa.String(),
                      sa.ForeignKey("detections.id", ondelete="CASCADE"),
                      nullable=False),
            sa.Column("detector", sa.String(), nullable=False),
            sa.Column("signal_family", sa.String(), nullable=False),
            sa.Column("polarity", sa.String(), nullable=False),
            sa.Column("score", sa.Float()),
            sa.Column("score_semantics", sa.String(), nullable=False),
            sa.Column("localised", sa.Boolean(), nullable=False),
            sa.Column("description", sa.Text(), nullable=False),
            sa.Column("created_at", _tz(), nullable=False),
        ]),
        ("findings", [
            sa.Column("id", sa.String(), primary_key=True),
            sa.Column("analysis_id", sa.String(),
                      sa.ForeignKey("analyses.id", ondelete="CASCADE"),
                      nullable=False, index=True),
            sa.Column("statement", sa.Text(), nullable=False),
            sa.Column("created_at", _tz(), nullable=False),
        ]),
        ("finding_observations", [
            sa.Column("finding_id", sa.String(),
                      sa.ForeignKey("findings.id", ondelete="CASCADE"),
                      primary_key=True),
            sa.Column("observation_id", sa.String(),
                      sa.ForeignKey("observations.id", ondelete="CASCADE"),
                      primary_key=True),
            sa.Column("relation", sa.String(), nullable=False),
        ]),
        ("assessments", [
            sa.Column("id", sa.String(), primary_key=True),
            sa.Column("analysis_id", sa.String(),
                      sa.ForeignKey("analyses.id", ondelete="CASCADE"),
                      nullable=False, unique=True),
            sa.Column("category", sa.String(), nullable=False),
            sa.Column("evidence_strength", sa.String(), nullable=False),
            sa.Column("rule", sa.String(), nullable=False),
            sa.Column("rationale", sa.Text(), nullable=False),
            sa.Column("coverage_gaps", sa.JSON(), nullable=False),
            sa.Column("created_at", _tz(), nullable=False),
        ]),
        ("audit_events", [
            sa.Column("id", sa.String(), primary_key=True),
            sa.Column("sequence", sa.Integer(), nullable=False),
            sa.Column("occurred_at", sa.String(40), nullable=False),
            sa.Column("actor", sa.String(150), nullable=False),
            sa.Column("action", sa.String(), nullable=False),
            sa.Column("case_id", sa.String(), index=True),
            sa.Column("media_id", sa.String(), index=True),
            sa.Column("analysis_id", sa.String(), index=True),
            sa.Column("details", sa.JSON(), nullable=False),
            sa.Column("previous_event_hash", sa.String(64), nullable=False),
            sa.Column("event_hash", sa.String(64), nullable=False, unique=True),
            sa.UniqueConstraint("sequence", name="uq_audit_events_sequence"),
            sa.UniqueConstraint("previous_event_hash",
                                name="uq_audit_events_previous_hash"),
        ]),
    ]


_ANALYSIS_RUN_COLUMNS = [
    ("requested_by", lambda: sa.Column("requested_by", sa.String(150))),
    ("software_version", lambda: sa.Column("software_version", sa.String())),
    ("detector_set_version", lambda: sa.Column("detector_set_version", sa.String(64))),
    ("config_snapshot", lambda: sa.Column("config_snapshot", sa.JSON())),
    ("duration_ms", lambda: sa.Column("duration_ms", sa.Integer())),
]


def upgrade() -> None:
    existing = set(_inspector().get_table_names())
    for name, columns in _new_tables():
        if name not in existing:
            op.create_table(name, *columns)

    # ── media ─────────────────────────────────────────────────────────────────
    media_cols = _columns("media")
    media_fks = {
        tuple(fk["constrained_columns"]) for fk in _inspector().get_foreign_keys("media")
    }
    media_indexes = {ix["name"] for ix in _inspector().get_indexes("media")}
    with op.batch_alter_table("media") as batch:
        if "case_id" not in media_cols:
            batch.add_column(sa.Column("case_id", sa.String()))
        if ("case_id",) not in media_fks:
            batch.create_foreign_key(
                "fk_media_case_id_cases", "cases", ["case_id"], ["id"],
                ondelete="RESTRICT",
            )
        if "ix_media_case_id" not in media_indexes:
            batch.create_index("ix_media_case_id", ["case_id"])
        if "uploaded_by" not in media_cols:
            batch.add_column(sa.Column("uploaded_by", sa.String(150)))

    # ── analyses ──────────────────────────────────────────────────────────────
    analysis_cols = _columns("analyses")
    with op.batch_alter_table("analyses") as batch:
        for name, column in _ANALYSIS_RUN_COLUMNS:
            if name not in analysis_cols:
                batch.add_column(column())
        if "overall_confidence" in analysis_cols:
            batch.drop_column("overall_confidence")


def downgrade() -> None:
    with op.batch_alter_table("analyses") as batch:
        batch.add_column(sa.Column("overall_confidence", sa.Float()))
        for name, _ in reversed(_ANALYSIS_RUN_COLUMNS):
            batch.drop_column(name)

    with op.batch_alter_table("media") as batch:
        batch.drop_column("uploaded_by")
        batch.drop_index("ix_media_case_id")
        batch.drop_constraint("fk_media_case_id_cases", type_="foreignkey")
        batch.drop_column("case_id")

    for name, _ in reversed(_new_tables()):
        op.drop_table(name)
