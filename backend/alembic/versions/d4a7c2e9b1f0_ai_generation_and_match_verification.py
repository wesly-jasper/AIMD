"""ai-generation branch and similarity match verification

The AI-generation detector (UnivFD) is reported beside the manipulation verdict,
never folded into it, so the assessment gets its own ``ai_generation`` record.
Similarity matches gain the outcome of geometric verification: a status
(VERIFIED_MATCH / SIMILAR_UNVERIFIED / NO_MATCH) and the measurements behind it.

Existing rows are left NULL: matches recorded before verification existed were
never verified, and saying otherwise would be fabricated.

Revision ID: d4a7c2e9b1f0
Revises: c9d2e4f1a7b3
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "d4a7c2e9b1f0"
down_revision: Union[str, Sequence[str], None] = "c9d2e4f1a7b3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _columns(table: str) -> set[str]:
    """Existing column names: a create_all-built database may already have them."""
    return {col["name"] for col in sa.inspect(op.get_bind()).get_columns(table)}


def upgrade() -> None:
    match_cols = _columns("similarity_matches")
    if "verification_status" not in match_cols:
        op.add_column("similarity_matches",
                      sa.Column("verification_status", sa.String(), nullable=True))
    if "verification" not in match_cols:
        op.add_column("similarity_matches",
                      sa.Column("verification", sa.JSON(), nullable=True))
    if "ai_generation" not in _columns("assessments"):
        op.add_column("assessments", sa.Column("ai_generation", sa.JSON(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("assessments") as batch:
        batch.drop_column("ai_generation")
    with op.batch_alter_table("similarity_matches") as batch:
        batch.drop_column("verification")
        batch.drop_column("verification_status")
