"""source verification and connector outcomes

A discovered source now carries the outcome of the same verification a local
match gets, plus the citation the source itself provides. The analysis records
each source connector's outcome (UNAVAILABLE / FAILED / COMPLETED), so an empty
source list can be told apart from a search that never ran.

Existing rows are left NULL: sources recorded before this were never verified,
and which connectors ran for old analyses is unknown.

Revision ID: f2c6a9d4e8b1
Revises: e5b8f3a1c2d6
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "f2c6a9d4e8b1"
down_revision: Union[str, Sequence[str], None] = "e5b8f3a1c2d6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _columns(table: str) -> set[str]:
    """Existing column names: a create_all-built database may already have them."""
    return {col["name"] for col in sa.inspect(op.get_bind()).get_columns(table)}


def upgrade() -> None:
    source_cols = _columns("sources")
    for name, type_ in (("verification_status", sa.String()),
                        ("verification", sa.JSON()),
                        ("citation", sa.JSON())):
        if name not in source_cols:
            op.add_column("sources", sa.Column(name, type_, nullable=True))
    if "source_searches" not in _columns("analyses"):
        op.add_column("analyses", sa.Column("source_searches", sa.JSON(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("analyses") as batch:
        batch.drop_column("source_searches")
    with op.batch_alter_table("sources") as batch:
        batch.drop_column("citation")
        batch.drop_column("verification")
        batch.drop_column("verification_status")
