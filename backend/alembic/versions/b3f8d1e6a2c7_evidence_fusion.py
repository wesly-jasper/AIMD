"""evidence fusion on the assessment

Likelihood-ratio ranges per branch, stored beside the assessment's categories
(``services/fusion/evidence_fusion``). Existing rows stay NULL: they were never
fused, and a value computed now would describe today's calibrations, not the
ones in force when they were assessed.

Revision ID: b3f8d1e6a2c7
Revises: a9d3e7c1f5b2
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "b3f8d1e6a2c7"
down_revision: Union[str, Sequence[str], None] = "a9d3e7c1f5b2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    columns = {col["name"] for col in sa.inspect(op.get_bind()).get_columns("assessments")}
    if "evidence_fusion" not in columns:
        op.add_column("assessments", sa.Column("evidence_fusion", sa.JSON(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("assessments") as batch:
        batch.drop_column("evidence_fusion")
