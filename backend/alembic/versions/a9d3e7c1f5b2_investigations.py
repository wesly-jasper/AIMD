"""investigations

Runs of the Nemotron investigation agent. A table of their own: an
investigation reads an analysis and never changes it, and its summary is model
output (INFERENCE), not a result of the forensic pipeline.

Revision ID: a9d3e7c1f5b2
Revises: f2c6a9d4e8b1
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "a9d3e7c1f5b2"
down_revision: Union[str, Sequence[str], None] = "f2c6a9d4e8b1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _tables() -> set[str]:
    """Existing table names: a create_all-built database may already have them."""
    return set(sa.inspect(op.get_bind()).get_table_names())


def upgrade() -> None:
    if "investigations" in _tables():
        return
    op.create_table(
        "investigations",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("analysis_id", sa.String(), nullable=False),
        sa.Column("requested_by", sa.String(length=150), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("model", sa.String(), nullable=True),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("summary_category", sa.String(), nullable=False),
        sa.Column("citations", sa.JSON(), nullable=True),
        sa.Column("flags", sa.JSON(), nullable=True),
        sa.Column("tool_calls", sa.JSON(), nullable=True),
        sa.Column("steps", sa.Integer(), nullable=True),
        sa.Column("usage", sa.JSON(), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["analysis_id"], ["analyses.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_investigations_analysis_id", "investigations", ["analysis_id"])
    op.create_index("ix_investigations_requested_by", "investigations", ["requested_by"])


def downgrade() -> None:
    op.drop_table("investigations")
