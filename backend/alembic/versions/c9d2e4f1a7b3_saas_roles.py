"""saas roles: investigator -> user

The product became self-service: every account is a ``user`` that sees only its
own media, and ``admin`` exists only through the bootstrap environment. Accounts
recorded with the retired ``investigator`` role are renamed; their permissions
are unchanged (neither role could do more than the other outside admin routes).

Revision ID: c9d2e4f1a7b3
Revises: b7e1f0c2d9a4
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "c9d2e4f1a7b3"
down_revision: Union[str, Sequence[str], None] = "b7e1f0c2d9a4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.get_bind().execute(sa.text("UPDATE users SET role = 'user' WHERE role = 'investigator'"))


def downgrade() -> None:
    op.get_bind().execute(sa.text("UPDATE users SET role = 'investigator' WHERE role = 'user'"))
