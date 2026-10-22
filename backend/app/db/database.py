"""
SQLAlchemy database engine and session management.

Supports SQLite (local dev/testing) and PostgreSQL (production)
via DATABASE_URL environment variable. Application code is
agnostic to the backend -- only the URL changes.
"""
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from app.core.config import settings


# ── Engine ────────────────────────────────────────────────────────────────────
connect_args = {}
if settings.database_url.startswith("sqlite"):
    # SQLite requires check_same_thread=False for FastAPI's threading model
    connect_args = {"check_same_thread": False}

engine = create_engine(
    settings.database_url,
    connect_args=connect_args,
    # Connection pool settings (ignored by SQLite)
    pool_pre_ping=True,
)

# Enable WAL mode for SQLite to improve concurrent read performance
if settings.database_url.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


# ── Session ───────────────────────────────────────────────────────────────────
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


# ── Base class for all ORM models ─────────────────────────────────────────────
class Base(DeclarativeBase):
    pass


# ── FastAPI dependency ─────────────────────────────────────────────────────────
def get_db():
    """Yield a database session and ensure it is closed after use."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


_BACKEND_DIR = Path(__file__).resolve().parents[2]


def alembic_config(database_url: str | None = None):
    """
    An Alembic config for this application, without reading ``alembic.ini``.

    Not reading the ini file is deliberate: env.py configures logging from it,
    which would replace the application's own logging setup at startup.
    """
    from alembic.config import Config

    cfg = Config()
    cfg.set_main_option("script_location", str(_BACKEND_DIR / "alembic"))
    cfg.set_main_option("sqlalchemy.url", database_url or settings.database_url)
    return cfg


def init_db(database_url: str | None = None):
    """
    Bring the schema to the latest migration. Called on application startup.

    This used to be ``Base.metadata.create_all``, which creates missing tables
    but never alters existing ones: a database created before a column was added
    simply lacked it, and the first query touching that column failed. Every
    schema change now goes through a migration, and this applies them.

    Each revision is idempotent against databases that ``create_all`` built, so
    a legacy database with no ``alembic_version`` table is upgraded in place.
    """
    from alembic import command

    command.upgrade(alembic_config(database_url), "head")
