"""
Migrations build the schema from nothing and upgrade legacy databases without
losing evidence.

Three defects are guarded here:

1. The initial revision was a no-op, so ``alembic upgrade head`` on an empty
   database created no tables and the next revision crashed. Startup relied on
   ``create_all`` instead, which never alters an existing table.
2. Dropping ``analyses.overall_confidence`` on SQLite rebuilds the table, and
   with foreign keys enforced the DROP of the original would cascade-delete every
   detection, fingerprint and report hanging off it.
3. The detector-status backfill ran unconditionally, so on a database that had
   recorded real availability/execution values it would overwrite them with
   values guessed from the legacy label.

Every test migrates its own scratch file; none touches the suite's database.
"""
import sqlite3

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import create_engine, inspect

from app.db.database import alembic_config
from app.db.models import Base

HEAD = "c9d2e4f1a7b3"


@pytest.fixture
def scratch_db(tmp_path):
    path = tmp_path / "scratch.db"
    return path, f"sqlite:///{path}"


def _upgrade(url: str, target: str = "head") -> None:
    command.upgrade(alembic_config(url), target)


def _rows(path, sql, params=()):
    with sqlite3.connect(path) as conn:
        return conn.execute(sql, params).fetchall()


def _seed_legacy_evidence(path) -> None:
    """One media item with an analysis and every kind of child row."""
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            INSERT INTO media (id, original_filename, stored_filename, file_path,
                               content_type, media_type, size_bytes, sha256, created_at)
            VALUES ('m1', 'exhibit.jpg', 'm1.jpg', 'uploads/m1.jpg', 'image/jpeg',
                    'image', 1234, 'ab' || substr(hex(zeroblob(31)), 1, 62),
                    '2026-09-01 10:00:00');
            INSERT INTO analyses (id, media_id, status, assessment, overall_confidence,
                                  created_at)
            VALUES ('a1', 'm1', 'COMPLETED', 'SUSPICIOUS', 0.42, '2026-09-01 10:01:00');
            INSERT INTO detections (id, analysis_id, detector, media_type,
                                    manipulation_type, assessment, confidence, created_at)
            VALUES ('d1', 'a1', 'ForensicImageDetector', 'image', 'splice',
                    'SUSPICIOUS', 0.7, '2026-09-01 10:01:01'),
                   ('d2', 'a1', 'TruForDetector', 'image', 'forgery',
                    'UNAVAILABLE', 0.0, '2026-09-01 10:01:02');
            INSERT INTO detection_regions (id, detection_id, region_type, confidence, bbox)
            VALUES ('r1', 'd1', 'splice', 0.7, '[1, 2, 3, 4]');
            INSERT INTO fingerprints (id, media_id, analysis_id, algorithm, value, scope,
                                      created_at)
            VALUES ('f1', 'm1', 'a1', 'sha256', 'abc', 'full', '2026-09-01 10:01:03');
            INSERT INTO evidence_items (id, analysis_id, category, description, created_at)
            VALUES ('e1', 'a1', 'OBSERVATION', 'legacy observation', '2026-09-01 10:01:04');
            INSERT INTO reports (id, analysis_id, markdown_report, generated_at)
            VALUES ('rep1', 'a1', '# legacy', '2026-09-01 10:01:05');
            """
        )


# ── Empty database ────────────────────────────────────────────────────────────

def test_upgrade_from_empty_matches_the_models_exactly(scratch_db):
    path, url = scratch_db
    _upgrade(url)

    engine = create_engine(url)
    with engine.connect() as conn:
        context = MigrationContext.configure(conn, opts={"compare_type": True})
        diff = compare_metadata(context, Base.metadata)
        version = context.get_current_revision()
    engine.dispose()

    assert diff == [], f"migrated schema diverges from the models: {diff}"
    assert version == HEAD


def test_initial_revision_alone_creates_tables(scratch_db):
    """It used to be ``pass``: no tables, and the next revision crashed."""
    path, url = scratch_db
    _upgrade(url, "04e00c1a868b")

    tables = set(inspect(create_engine(url)).get_table_names())
    assert {"media", "analyses", "detections", "reports"} <= tables


# ── Legacy data ───────────────────────────────────────────────────────────────

def test_legacy_evidence_survives_the_table_rebuild(scratch_db):
    path, url = scratch_db
    _upgrade(url, "04e00c1a868b")
    _seed_legacy_evidence(path)

    _upgrade(url)

    # Nothing cascaded away when ``analyses`` was rebuilt.
    for table, expected in [
        ("media", 1), ("analyses", 1), ("detections", 2), ("detection_regions", 1),
        ("fingerprints", 1), ("evidence_items", 1), ("reports", 1),
    ]:
        assert _rows(path, f"SELECT COUNT(*) FROM {table}")[0][0] == expected, table

    assert _rows(path, "SELECT status, assessment FROM analyses WHERE id='a1'") == [
        ("COMPLETED", "SUSPICIOUS")
    ]
    assert _rows(path, "SELECT markdown_report FROM reports")[0][0] == "# legacy"
    assert _rows(path, "PRAGMA foreign_key_check") == []


def test_overall_confidence_is_gone_and_run_columns_exist(scratch_db):
    path, url = scratch_db
    _upgrade(url, "04e00c1a868b")
    _seed_legacy_evidence(path)
    _upgrade(url)

    columns = {row[1] for row in _rows(path, "PRAGMA table_info(analyses)")}
    assert "overall_confidence" not in columns
    assert {"requested_by", "software_version", "detector_set_version",
            "config_snapshot", "duration_ms"} <= columns

    # Not backfilled: the values in force when the legacy run happened are
    # unknown, and inventing them would be fabricated provenance.
    assert _rows(path, "SELECT software_version, duration_ms FROM analyses") == [
        (None, None)
    ]


def test_legacy_detection_labels_are_backfilled_and_marked(scratch_db):
    path, url = scratch_db
    _upgrade(url, "04e00c1a868b")
    _seed_legacy_evidence(path)
    _upgrade(url)

    rows = dict(
        (r[0], r[1:]) for r in _rows(
            path,
            "SELECT id, availability, execution_status, status_reason FROM detections",
        )
    )
    assert rows["d2"][:2] == ("UNAVAILABLE", "SKIPPED")
    assert "backfilled" in rows["d2"][2]
    assert rows["d1"][:2] == ("AVAILABLE", "COMPLETED")


def test_backfill_does_not_overwrite_recorded_statuses(scratch_db):
    """
    An unstamped database whose detections table already has the status
    columns has been recording real values. The backfill must leave them be.
    """
    path, url = scratch_db
    _upgrade(url, "a1b2c3d4e5f6")
    _seed_legacy_evidence(path)
    with sqlite3.connect(path) as conn:
        conn.execute(
            "UPDATE detections SET availability='AVAILABLE', execution_status='FAILED',"
            " status_reason='recorded at run time' WHERE id='d1'"
        )
        # As if create_all built it: tables and columns present, no version row.
        conn.execute("DROP TABLE alembic_version")

    _upgrade(url)

    assert _rows(
        path, "SELECT execution_status, status_reason FROM detections WHERE id='d1'"
    ) == [("FAILED", "recorded at run time")]


def test_an_unstamped_create_all_database_is_upgraded_in_place(scratch_db):
    """The live database predates migrations: tables exist, no version row."""
    path, url = scratch_db
    _upgrade(url, "04e00c1a868b")
    _seed_legacy_evidence(path)
    with sqlite3.connect(path) as conn:
        conn.execute("DROP TABLE alembic_version")

    _upgrade(url)

    assert _rows(path, "SELECT version_num FROM alembic_version") == [(HEAD,)]
    assert _rows(path, "SELECT COUNT(*) FROM detections")[0][0] == 2


def test_upgrade_is_repeatable(scratch_db):
    path, url = scratch_db
    _upgrade(url)
    _upgrade(url)
    assert _rows(path, "SELECT version_num FROM alembic_version") == [(HEAD,)]


def test_a_case_with_evidence_cannot_be_deleted(scratch_db):
    """RESTRICT, not CASCADE: closing out a case must not delete its evidence."""
    path, url = scratch_db
    _upgrade(url)
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute(
            "INSERT INTO cases (id, title, status, created_at)"
            " VALUES ('c1', 'Case', 'OPEN', '2026-10-01')"
        )
        conn.execute(
            "INSERT INTO media (id, case_id, original_filename, stored_filename,"
            " file_path, content_type, media_type, size_bytes, sha256, created_at)"
            " VALUES ('m1', 'c1', 'x.jpg', 'm1.jpg', 'p', 'image/jpeg', 'image', 1,"
            " 'h', '2026-10-01')"
        )
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("DELETE FROM cases WHERE id='c1'")
