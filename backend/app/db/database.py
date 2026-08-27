"""
SQLAlchemy database engine and session management.

Supports SQLite (local dev/testing) and PostgreSQL (production)
via DATABASE_URL environment variable. Application code is
agnostic to the backend — only the URL changes.
"""
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


def init_db():
    """Create all tables. Called on application startup."""
    # Models are imported at module level by repository.py when it is first used.
    # We do NOT re-import them here to avoid double-registration on the Base metadata.
    # If the app starts without repository being imported first, import models now.
    try:
        from app.db import models  # noqa: F401 — registers tables with Base
    except ImportError:
        pass
    Base.metadata.create_all(bind=engine)
