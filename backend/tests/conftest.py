"""
Test isolation: keep the suite away from real evidence.

WHY THIS FILE EXISTS
--------------------
``app.core.config.Settings`` defaults ``database_url`` to ``sqlite:///<repo>/aimd.db``
and the storage directories to ``<repo>/uploads``, ``<repo>/processed`` and
``<repo>/reports``. ``app.db.database`` builds its engine at *import* time from
those settings, and the integration tests import ``app.main`` and drive a real
``TestClient``. With no conftest, every test run therefore wrote media rows,
analysis rows, detection rows and uploaded files into the operator's live
evidence store, interleaving synthetic fixtures with real case records.

That is a forensic integrity failure, not just untidiness: once test media and
genuine evidence share a database, no later query can tell them apart.

HOW IT IS FIXED
---------------
The environment is redirected *before* ``app`` is imported anywhere. pytest loads
conftest modules ahead of test modules, and nothing above this line imports
``app``, so the ``Settings()`` instance that every module shares is constructed
from these values. Environment variables take precedence over ``backend/.env``
in pydantic-settings, so a developer's local ``.env`` cannot pull the suite back
onto the real database.

The redirect is process-wide and deliberately not a fixture: a fixture runs too
late, after the engine has already been bound.
"""
import os
import shutil
import tempfile
from pathlib import Path

# ── Redirect storage and database BEFORE app.core.config is imported ──────────

# Resolved eagerly: on Windows ``mkdtemp`` can return an 8.3 short path
# (``C:\Users\WESLYJ~1\...``) which ``Path.resolve()`` later expands to the long
# form, so an unresolved root never compares equal to a resolved child.
_TEST_ROOT = Path(tempfile.mkdtemp(prefix="aimd-tests-")).resolve()

os.environ["DATABASE_URL"] = f"sqlite:///{_TEST_ROOT / 'test.db'}"
os.environ["UPLOAD_DIR"] = str(_TEST_ROOT / "uploads")
os.environ["PROCESSED_DIR"] = str(_TEST_ROOT / "processed")
os.environ["FRAMES_DIR"] = str(_TEST_ROOT / "processed" / "frames")
os.environ["KEYFRAMES_DIR"] = str(_TEST_ROOT / "processed" / "keyframes")
os.environ["REPORT_DIR"] = str(_TEST_ROOT / "reports")
os.environ["LOGS_DIR"] = str(_TEST_ROOT / "logs")

# The suite must never reach a provider over the network, and must never pick up
# a real credential from the developer's environment or .env file.
os.environ["GOOGLE_VISION_API_KEY"] = ""
os.environ["TINEYE_API_KEY"] = ""
os.environ["SOURCE_WAYBACK_ENABLED"] = "false"
# The investigation agent's key is moved aside, not lost: the one live smoke
# test (marked ``live``) reads it from AIMD_LIVE_NEBIUS_API_KEY and skips
# without it. Everything else sees no key and must answer UNAVAILABLE.
os.environ.setdefault("AIMD_LIVE_NEBIUS_API_KEY", os.environ.get("NEBIUS_API_KEY", ""))
os.environ["NEBIUS_API_KEY"] = ""

for _sub in ("uploads", "processed", "processed/frames", "processed/keyframes",
             "reports", "logs"):
    (_TEST_ROOT / _sub).mkdir(parents=True, exist_ok=True)

import pytest  # noqa: E402  -- must follow the environment redirect

#: The identity ``authenticated_by_default`` supplies. Rows a test creates
#: directly must carry it as their owner to be visible through the API.
TEST_USERNAME = "pytest_user@example.invalid"


# ── Guard ─────────────────────────────────────────────────────────────────────

def _repo_database_path() -> Path:
    """The real database this suite must never touch."""
    from app.core.config import BASE_DIR
    return (BASE_DIR / "aimd.db").resolve()


@pytest.fixture(scope="session", autouse=True)
def assert_not_running_against_real_evidence():
    """
    Fail the whole session rather than write to the operator's evidence store.

    This is a guard, not a formality. The suite previously ran against
    ``backend/aimd.db`` and added rows to it; if a future change to settings
    loading, ``.env`` precedence or import order reintroduces that, this fixture
    stops the run before any test executes instead of letting the damage happen
    quietly and be discovered later.
    """
    from app.core.config import settings
    from app.db.database import engine

    real_db = _repo_database_path()

    for label, url in (("settings", settings.database_url),
                       ("engine", str(engine.url))):
        if url.startswith("sqlite") and ":memory:" not in url:
            target = Path(url.split("sqlite:///", 1)[-1]).resolve()
            assert target != real_db, (
                f"{label} database is the live evidence store ({real_db}). "
                "Refusing to run: the suite would write test fixtures into real "
                "case records."
            )

    for name in ("upload_dir", "processed_dir", "frames_dir", "keyframes_dir",
                 "report_dir"):
        configured = Path(getattr(settings, name)).resolve()
        assert configured == _TEST_ROOT or _TEST_ROOT in configured.parents, (
            f"{name} is {configured}, outside the test sandbox {_TEST_ROOT}. "
            "Test fixtures would be written into real evidence storage."
        )

    yield

    # Temp artefacts only; nothing here is user data.
    shutil.rmtree(_TEST_ROOT, ignore_errors=True)


@pytest.fixture(scope="session", autouse=True)
def initialise_test_schema(assert_not_running_against_real_evidence):
    """
    Build the sandboxed database by running every migration from empty.

    ``init_db`` is ``alembic upgrade head``, the same path the application takes
    at startup, so the whole suite runs against the migrated schema rather than
    one ``create_all`` produced -- a migration that diverges from the models
    fails here, not in deployment.
    """
    from app.db.database import init_db
    init_db()
    yield


# ── Authentication ────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def authenticated_by_default(request):
    """
    Supply an authenticated identity to every test that is not about auth.

    Routes are protected at the router level in ``app.main``, so without this
    every existing integration test would assert 401 instead of exercising its
    actual subject. The override is applied per test and removed afterwards.

    The override means these tests CANNOT detect the accidental removal of a
    router's auth dependency -- so that is guarded separately and explicitly by
    ``tests/integration/test_route_authentication.py``, which walks the real
    route table with no token at all. A test that needs the genuine dependency
    declares ``@pytest.mark.real_auth`` and this fixture steps aside.
    """
    if "real_auth" in request.keywords:
        yield
        return

    from app.core.auth import User, get_current_user
    from app.main import app

    def _test_identity() -> User:
        # An ordinary user, not an admin: a test that happens to hit an
        # admin-gated route should fail rather than quietly pass with elevated
        # rights it would not have in production.
        return User(
            username=TEST_USERNAME,
            email=TEST_USERNAME,
            role="user",
            is_active=True,
        )

    app.dependency_overrides[get_current_user] = _test_identity
    try:
        yield
    finally:
        app.dependency_overrides.pop(get_current_user, None)


# ── Rate limiting ─────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def rate_limits_off_by_default(request):
    """
    Disable the limiter for tests that are not about it.

    Every ``TestClient`` request comes from the same address, so the suite would
    exhaust the login and upload limits within a few files and then fail on 429s
    that say nothing about the code under test. Tests marked ``rate_limited``
    get the limiter on, with counters reset before and after so no test inherits
    another's spent budget.
    """
    from app.core.rate_limit import limiter

    enabled = "rate_limited" in request.keywords
    previous = limiter.enabled
    limiter.enabled = enabled
    if enabled:
        limiter.reset()
    try:
        yield
    finally:
        if enabled:
            limiter.reset()
        limiter.enabled = previous
