"""reference corpus

Public images (Wikimedia Commons first) indexed so an upload can be traced to
where it was published. They get their own tables rather than rows in ``media``:
they belong to no account, must never surface through the owner-scoped media
search, and each carries the citation a source claim rests on.

Revision ID: e5b8f3a1c2d6
Revises: d4a7c2e9b1f0
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "e5b8f3a1c2d6"
down_revision: Union[str, Sequence[str], None] = "d4a7c2e9b1f0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _tables() -> set[str]:
    """Existing table names: a create_all-built database may already have them."""
    return set(sa.inspect(op.get_bind()).get_table_names())


def upgrade() -> None:
    tables = _tables()
    if "reference_images" not in tables:
        op.create_table(
            "reference_images",
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("corpus", sa.String(), nullable=False),
            sa.Column("external_id", sa.String(), nullable=False),
            sa.Column("page_url", sa.String(), nullable=False),
            sa.Column("file_url", sa.String(), nullable=False),
            sa.Column("download_url", sa.String(), nullable=False),
            sa.Column("uploader", sa.String(), nullable=True),
            sa.Column("uploaded_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("licence", sa.String(), nullable=True),
            sa.Column("licence_url", sa.String(), nullable=True),
            sa.Column("artist", sa.Text(), nullable=True),
            sa.Column("original_sha1", sa.String(length=40), nullable=True),
            sa.Column("original_width", sa.Integer(), nullable=True),
            sa.Column("original_height", sa.Integer(), nullable=True),
            sa.Column("original_mime", sa.String(), nullable=True),
            sa.Column("file_path", sa.String(), nullable=False),
            sa.Column("sha256", sa.String(length=64), nullable=False),
            sa.Column("size_bytes", sa.Integer(), nullable=False),
            sa.Column("width", sa.Integer(), nullable=True),
            sa.Column("height", sa.Integer(), nullable=True),
            sa.Column("categories", sa.JSON(), nullable=False),
            sa.Column("source_metadata", sa.JSON(), nullable=True),
            sa.Column("retrieved_at", sa.DateTime(timezone=True), nullable=False),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("corpus", "external_id",
                                name="uq_reference_images_corpus_external_id"),
        )
        op.create_index("ix_reference_images_sha256", "reference_images", ["sha256"])
    if "reference_fingerprints" not in tables:
        op.create_table(
            "reference_fingerprints",
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("reference_id", sa.String(), nullable=False),
            sa.Column("algorithm", sa.String(), nullable=False),
            sa.Column("algorithm_version", sa.String(), nullable=True),
            sa.Column("value", sa.Text(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(["reference_id"], ["reference_images.id"],
                                    ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_reference_fingerprints_reference_id",
                        "reference_fingerprints", ["reference_id"])
        op.create_index("ix_reference_fingerprints_algorithm",
                        "reference_fingerprints", ["algorithm"])


def downgrade() -> None:
    op.drop_table("reference_fingerprints")
    op.drop_table("reference_images")
