"""detector status contract: separate availability from execution

Adds the columns required to persist the detector result contract, replacing the
single collapsed ``assessment`` label and the blended numeric confidence.

On ``detections``:
  availability      AVAILABLE | UNAVAILABLE                -- could it run?
  execution_status  COMPLETED | INCONCLUSIVE | FAILED | SKIPPED -- what happened?
  score_semantics   how ``confidence`` must be interpreted
  status_reason     why it was unavailable / failed / skipped

On ``analyses``:
  evidence_strength NONE | WEAK | MODERATE | STRONG | CONTRADICTORY

``analyses.overall_confidence`` is deliberately NOT dropped here. It held a
weighted mean of incomparable detector scores, and nothing writes it any more,
but dropping a column requires a table rebuild on SQLite and would discard
existing rows' values. It is left in place, unwritten, for removal in the Phase 4
data-model migration once the forensic entities land.

BACKFILL
--------
Existing rows are backfilled from the deprecated ``assessment`` label, which is
the only information those rows carry:

  assessment = 'UNAVAILABLE'  -> UNAVAILABLE + SKIPPED      (it never ran)
  assessment = 'INCONCLUSIVE' -> AVAILABLE   + INCONCLUSIVE (ran, no determination)
  anything else               -> AVAILABLE   + COMPLETED

Historical rows cannot be distinguished between UNAVAILABLE and AVAILABLE+FAILED,
because the old schema could not express that difference -- the very defect this
migration fixes. Backfilled rows are therefore marked in ``status_reason`` so an
examiner can see the value was inferred rather than recorded at run time.

Revision ID: a1b2c3d4e5f6
Revises: 04e00c1a868b
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, Sequence[str], None] = "04e00c1a868b"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_BACKFILL_NOTE = "backfilled from legacy assessment label; not recorded at run time"


def _columns(table: str) -> set[str]:
    """Existing column names, so the migration is idempotent.

    The previous migration was a no-op while ``Base.metadata.create_all`` created
    the tables, so a given database may already have some of these columns
    depending on when it was first created.
    """
    inspector = sa.inspect(op.get_bind())
    return {col["name"] for col in inspector.get_columns(table)}


def upgrade() -> None:
    detection_cols = _columns("detections")

    if "availability" not in detection_cols:
        op.add_column(
            "detections",
            sa.Column(
                "availability",
                sa.String(),
                nullable=False,
                server_default="AVAILABLE",
            ),
        )
    if "execution_status" not in detection_cols:
        op.add_column(
            "detections",
            sa.Column(
                "execution_status",
                sa.String(),
                nullable=False,
                server_default="COMPLETED",
            ),
        )
    if "score_semantics" not in detection_cols:
        op.add_column(
            "detections",
            sa.Column(
                "score_semantics",
                sa.String(),
                nullable=False,
                server_default="HEURISTIC_SEVERITY",
            ),
        )
    if "status_reason" not in detection_cols:
        op.add_column("detections", sa.Column("status_reason", sa.Text(), nullable=True))

    if "evidence_strength" not in _columns("analyses"):
        op.add_column(
            "analyses", sa.Column("evidence_strength", sa.String(), nullable=True)
        )

    # ── Backfill ────────────────────────────────────────────────────────────────
    # Only when this revision created the columns. If they already existed, a
    # create_all-built database has been recording real availability/execution
    # values, and overwriting them with values inferred from the legacy label
    # would replace recorded facts with guesses.
    if {"availability", "execution_status"} <= detection_cols:
        return

    bind = op.get_bind()

    bind.execute(
        sa.text(
            """
            UPDATE detections
               SET availability     = 'UNAVAILABLE',
                   execution_status = 'SKIPPED',
                   score_semantics  = 'NOT_APPLICABLE',
                   status_reason    = :note
             WHERE assessment = 'UNAVAILABLE'
            """
        ),
        {"note": _BACKFILL_NOTE},
    )

    bind.execute(
        sa.text(
            """
            UPDATE detections
               SET availability     = 'AVAILABLE',
                   execution_status = 'INCONCLUSIVE',
                   status_reason    = :note
             WHERE assessment = 'INCONCLUSIVE'
            """
        ),
        {"note": _BACKFILL_NOTE},
    )

    bind.execute(
        sa.text(
            """
            UPDATE detections
               SET availability     = 'AVAILABLE',
                   execution_status = 'COMPLETED'
             WHERE assessment NOT IN ('UNAVAILABLE', 'INCONCLUSIVE')
            """
        )
    )

    # Historical analyses have no recorded evidence strength. Leaving it NULL is
    # correct: inventing one from the stored mean confidence would reintroduce the
    # fabricated precision this migration exists to remove.


def downgrade() -> None:
    op.drop_column("analyses", "evidence_strength")
    op.drop_column("detections", "status_reason")
    op.drop_column("detections", "score_semantics")
    op.drop_column("detections", "execution_status")
    op.drop_column("detections", "availability")
