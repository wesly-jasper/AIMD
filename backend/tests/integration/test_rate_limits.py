"""
Rate limits are enforced, answer on the standard envelope, and fail closed on
misconfiguration.

The limiter existed before this file did: ``app.main`` built one with
``default_limits=["120/minute"]`` and registered its handler, but never
installed ``SlowAPIMiddleware`` -- the component that enforces default limits --
and decorated no route. So nothing was limited, login included.

The conftest disables the limiter for every other test, because all TestClient
requests share one client address; this file opts back in with
``rate_limited``, and the counters are reset around every test.
"""
import pytest
from fastapi.testclient import TestClient
from limits import parse_many
from slowapi.middleware import SlowAPIMiddleware

from app.core import rate_limit
from app.core.auth import user_store
from app.core.config import settings
from app.core.rate_limit import (
    ANALYSIS_LIMIT,
    SIGNUP_LIMIT,
    DEFAULT_LIMIT,
    LOGIN_LIMIT,
    UPLOAD_LIMIT,
    limiter,
)
from app.main import app

pytestmark = pytest.mark.rate_limited

client = TestClient(app)


def _allowance(limit: str) -> int:
    """Requests permitted before the tightest window in ``limit`` is exhausted."""
    return min(item.amount for item in parse_many(limit))


def _assert_rate_limited(response):
    assert response.status_code == 429, response.text
    body = response.json()
    # The application's envelope, not slowapi's own ``{"error": "..."}``.
    assert body["success"] is False
    assert body["error"]["code"] == "RATE_LIMITED"
    assert body["request_id"] == response.headers["X-Request-ID"]

    retry_after = int(response.headers["Retry-After"])
    assert retry_after >= 1
    assert str(retry_after) in body["error"]["message"]
    # The client's own address is the limiter key; it is not echoed back.
    assert "testclient" not in response.text


def _failed_login():
    return client.post(
        "/api/v1/auth/login",
        json={"username": "nobody", "password": "not-the-password"},
    )


# ── The limiter is actually wired ─────────────────────────────────────────────

def test_the_enforcing_middleware_is_installed():
    """Without it, default limits are configuration that nothing reads."""
    installed = [m.cls for m in app.user_middleware]
    assert SlowAPIMiddleware in installed
    assert app.state.limiter is limiter


def test_the_rate_limit_handler_is_the_enveloped_one():
    from slowapi.errors import RateLimitExceeded

    assert app.exception_handlers[RateLimitExceeded] is rate_limit.rate_limit_exceeded_handler


# ── Login ─────────────────────────────────────────────────────────────────────

def test_login_is_limited_per_client():
    for attempt in range(_allowance(LOGIN_LIMIT)):
        response = _failed_login()
        assert response.status_code == 401, f"attempt {attempt}: {response.text}"

    _assert_rate_limited(_failed_login())


def test_a_correct_password_does_not_bypass_an_exhausted_login_limit():
    """
    Otherwise the limit would only slow wrong guesses, and the right one -- the
    one that matters -- would always be let through.
    """
    user_store.create("limited_user", "a-correct-long-password", "user")
    try:
        for _ in range(_allowance(LOGIN_LIMIT)):
            _failed_login()

        response = client.post(
            "/api/v1/auth/login",
            json={"username": "limited_user", "password": "a-correct-long-password"},
        )
        _assert_rate_limited(response)
        assert "access_token" not in response.text
    finally:
        from app.db.database import SessionLocal
        from app.db.models import UserRecord

        with SessionLocal() as db:
            db.query(UserRecord).filter_by(username="limited_user").delete()
            db.commit()


# ── Expensive routes ──────────────────────────────────────────────────────────

def test_analysis_is_limited():
    payload = {"media_id": "00000000-0000-0000-0000-000000000000"}
    for attempt in range(_allowance(ANALYSIS_LIMIT)):
        response = client.post("/api/v1/analysis/", json=payload)
        assert response.status_code == 404, f"attempt {attempt}: {response.text}"

    _assert_rate_limited(client.post("/api/v1/analysis/", json=payload))


@pytest.mark.parametrize(
    "endpoint,expected",
    [
        ("app.api.v1.endpoints.auth.login", LOGIN_LIMIT),
        ("app.api.v1.endpoints.media.upload_media", UPLOAD_LIMIT),
        ("app.api.v1.endpoints.analysis.run_analysis", ANALYSIS_LIMIT),
        ("app.api.v1.endpoints.auth.signup", SIGNUP_LIMIT),
    ],
)
def test_expensive_routes_carry_their_own_limit(endpoint, expected):
    """
    Registered with the limit configured for them.

    Read from the limiter's registry (a private attribute) because exhausting
    each route over HTTP would ingest real files or run the pipeline dozens of
    times; login and analysis above are the behavioural proof that a registered
    limit is enforced.
    """
    registered = limiter._route_limits.get(endpoint)
    assert registered, f"{endpoint} has no route limit"
    assert {str(lim.limit) for lim in registered} == {
        str(item) for item in parse_many(expected)
    }


# ── Everything else ───────────────────────────────────────────────────────────

def test_undecorated_routes_get_the_default_limit():
    """
    Every request uses a different path parameter, and they still share one
    budget. slowapi's default keys on the concrete URL, under which each of these
    would have been the first request to its own bucket and none would ever be
    refused.
    """
    for attempt in range(_allowance(DEFAULT_LIMIT)):
        response = client.get(f"/api/v1/cases/no-such-folder-{attempt}")
        assert response.status_code == 404, f"attempt {attempt}: {response.text}"

    _assert_rate_limited(client.get("/api/v1/cases/yet-another-folder"))


def test_the_limit_does_not_reset_on_a_window_boundary():
    """Moving window: a fixed window would admit a second full burst at :00."""
    assert type(limiter.limiter).__name__ == "MovingWindowRateLimiter"


def test_counters_are_per_endpoint():
    """Exhausting login does not lock the same client out of the rest of the API."""
    for _ in range(_allowance(LOGIN_LIMIT) + 1):
        _failed_login()

    assert client.get("/health").status_code == 200


def test_a_429_is_readable_by_the_browser():
    """
    The rate limiter sits inside CORS. If it were outside, the 429 would leave
    without CORS headers and the frontend would see a network error, not a
    status it can tell the investigator about.
    """
    origin = settings.cors_origins[0]
    for _ in range(_allowance(LOGIN_LIMIT)):
        _failed_login()

    response = client.post(
        "/api/v1/auth/login",
        json={"username": "nobody", "password": "not-the-password"},
        headers={"Origin": origin},
    )
    assert response.status_code == 429
    assert response.headers.get("access-control-allow-origin") == origin


# ── Misconfiguration fails closed ─────────────────────────────────────────────

@pytest.mark.parametrize("bad", ["lots", "10/fortnight", "", "ten/minute"])
def test_an_unparseable_limit_is_fatal(bad):
    """
    slowapi logs an unparseable limit and installs none -- a typo in
    RATE_LIMIT_LOGIN would silently remove brute-force protection.
    """
    with pytest.raises(RuntimeError):
        rate_limit._validated("RATE_LIMIT_LOGIN", bad)


def test_a_valid_limit_is_accepted():
    assert rate_limit._validated("RATE_LIMIT_LOGIN", "5/minute;100/day") == "5/minute;100/day"
