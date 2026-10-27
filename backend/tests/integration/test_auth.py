"""
The authentication layer fails closed, and a token is not self-describing.

Four defects are guarded here, each of which on its own opened every route the
dependency was applied to:

1. ``get_current_user`` returned ``anonymous_analyst`` when no token was
   presented, so omitting the Authorization header granted the investigator role.
2. The signing key was a literal in committed source, so anyone with the
   repository could forge a token for any user and role.
3. The role was read from the token's own ``role`` claim, so it could not be
   revoked before the token expired.
4. ``/register`` was unauthenticated and took ``role`` from the request body, so
   one POST created an administrator. It is gone; ``/signup`` creates only
   ordinary ``user`` accounts.
"""
import os

import pytest
from fastapi.testclient import TestClient
from jose import jwt

from app.core import auth as auth_module
from app.core.auth import (
    ALGORITHM,
    ROLES,
    create_access_token,
    get_password_hash,
    signing_key,
    user_store,
    verify_password,
)
from app.main import app

# The conftest fixture supplies an authenticated investigator to every test that
# is not about auth. This whole file is about auth, so it opts out and drives the
# real dependency -- otherwise every assertion below would be made against an
# override and the file would pass no matter how broken the real thing was.
pytestmark = pytest.mark.real_auth

client = TestClient(app)

ADMIN_PASSWORD = "test-admin-password-01"
ANALYST_PASSWORD = "test-analyst-password-01"


def _clear_users() -> None:
    from app.db.database import SessionLocal
    from app.db.models import UserRecord

    with SessionLocal() as db:
        db.query(UserRecord).delete()
        db.commit()


@pytest.fixture(autouse=True)
def _accounts():
    """A known admin and user, and no one else, for each test."""
    _clear_users()
    user_store.create("case_admin", ADMIN_PASSWORD, "admin", "admin@example.test")
    user_store.create("case_analyst", ANALYST_PASSWORD, "user")
    yield
    _clear_users()


def _token(username: str, password: str) -> str:
    response = client.post(
        "/api/v1/auth/login", json={"username": username, "password": password}
    )
    assert response.status_code == 200, response.text
    return response.json()["access_token"]


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ── Fail closed ───────────────────────────────────────────────────────────────

def test_no_token_is_rejected_not_treated_as_an_analyst():
    response = client.get("/api/v1/auth/me")

    assert response.status_code == 401
    assert "anonymous_analyst" not in response.text


def test_garbage_token_is_rejected():
    for value in ("", "Bearer", "Bearer ", "Bearer not-a-jwt", "Basic abc"):
        response = client.get("/api/v1/auth/me", headers={"Authorization": value})
        assert response.status_code == 401, f"accepted Authorization={value!r}"


def test_login_then_me_returns_the_real_identity():
    token = _token("case_analyst", ANALYST_PASSWORD)
    response = client.get("/api/v1/auth/me", headers=_auth(token))

    assert response.status_code == 200, response.text
    assert response.json()["username"] == "case_analyst"
    assert response.json()["role"] == "user"


# ── The signing key ───────────────────────────────────────────────────────────

def test_the_committed_placeholder_key_is_refused(monkeypatch):
    """
    The key that was committed is public and must never be accepted again, even
    from a ``.env`` that still carries it.
    """
    monkeypatch.setenv(
        "AIMD_JWT_SECRET",
        "aimd_forensic_platform_secret_key_change_in_production",
    )
    monkeypatch.setattr(auth_module, "_SIGNING_KEY", None)

    with pytest.raises(RuntimeError, match="public"):
        auth_module.signing_key()


def test_a_short_key_is_refused(monkeypatch):
    monkeypatch.setenv("AIMD_JWT_SECRET", "too-short")
    monkeypatch.setattr(auth_module, "_SIGNING_KEY", None)

    with pytest.raises(RuntimeError, match="at least"):
        auth_module.signing_key()


def test_a_missing_key_is_fatal_outside_development(monkeypatch):
    """
    A generated per-process key would differ between workers and reject tokens
    at random, so production refuses to run without one.
    """
    monkeypatch.delenv("AIMD_JWT_SECRET", raising=False)
    monkeypatch.delenv("SECRET_KEY", raising=False)
    monkeypatch.setattr(auth_module.settings, "app_env", "production")
    monkeypatch.setattr(auth_module, "_SIGNING_KEY", None)

    with pytest.raises(RuntimeError, match="Refusing to start"):
        auth_module.signing_key()


def test_a_token_signed_with_another_key_is_rejected():
    """A forged token is what a key in committed source made possible."""
    forged = jwt.encode(
        {"sub": "case_admin", "role": "admin"},
        "an-attacker-chosen-key-of-sufficient-length",
        algorithm=ALGORITHM,
    )
    response = client.get("/api/v1/auth/me", headers=_auth(forged))
    assert response.status_code == 401


def test_the_signing_key_is_not_a_value_from_source():
    """
    Whatever the process is signing with, it is not the published placeholder.
    """
    assert signing_key() != "aimd_forensic_platform_secret_key_change_in_production"
    assert len(signing_key()) >= 32


# ── The role comes from the store, not the token ──────────────────────────────

def test_an_elevated_role_claim_in_a_valid_token_is_ignored():
    """
    The token is signed by us, so it decodes -- but ``role`` is advisory. The
    role previously came straight from this claim.
    """
    token = create_access_token({"sub": "case_analyst", "role": "admin"})

    response = client.get("/api/v1/auth/me", headers=_auth(token))
    assert response.status_code == 200
    assert response.json()["role"] == "user", (
        "the role was taken from the token claim instead of the user store"
    )


def test_deactivating_an_account_invalidates_its_live_token():
    token = _token("case_analyst", ANALYST_PASSWORD)
    assert client.get("/api/v1/auth/me", headers=_auth(token)).status_code == 200

    user_store.set_active("case_analyst", False)

    # The token is still cryptographically valid and unexpired; it is refused on
    # the strength of the store alone.
    assert client.get("/api/v1/auth/me", headers=_auth(token)).status_code == 401


def test_a_token_for_a_user_who_does_not_exist_is_rejected():
    token = create_access_token({"sub": "never_registered", "role": "admin"})
    assert client.get("/api/v1/auth/me", headers=_auth(token)).status_code == 401


def test_a_token_without_a_subject_is_rejected():
    for payload in ({}, {"sub": ""}, {"sub": None}, {"role": "admin"}):
        token = create_access_token(payload)
        response = client.get("/api/v1/auth/me", headers=_auth(token))
        assert response.status_code == 401, f"accepted payload {payload!r}"


# ── Sign-up ───────────────────────────────────────────────────────────────────

def _signup(email: str, password: str = "a-perfectly-long-password", **extra):
    return client.post("/api/v1/auth/signup", json={"email": email, "password": password, **extra})


def test_signup_creates_an_ordinary_account_and_signs_it_in():
    response = _signup("New.Person@Example.test")

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["role"] == "user"
    # Stored lower-cased, so the address logs in however it is typed.
    assert body["username"] == "new.person@example.test"
    me = client.get("/api/v1/auth/me", headers=_auth(body["access_token"]))
    assert me.json()["username"] == "new.person@example.test"
    assert _token("NEW.PERSON@example.test", "a-perfectly-long-password")


def test_signup_cannot_create_an_administrator():
    """The old /register took ``role`` from the body. Sign-up ignores it."""
    response = _signup("climber@example.test", role="admin")

    assert response.status_code == 201
    assert response.json()["role"] == "user"
    assert user_store.get("climber@example.test")["role"] == "user"


def test_signup_does_not_overwrite_an_existing_account():
    _signup("taken@example.test", "the-first-password-01")
    original = user_store.get("taken@example.test")["hashed_password"]

    response = _signup("taken@example.test", "an-attacker-chosen-pass")

    assert response.status_code == 409
    assert user_store.get("taken@example.test")["hashed_password"] == original
    assert _token("taken@example.test", "the-first-password-01")


def test_signup_refuses_a_short_password():
    response = _signup("weak@example.test", "short")
    assert response.status_code == 422
    assert not user_store.exists("weak@example.test")


def test_signup_refuses_something_that_is_not_an_email():
    for value in ("no-at-sign", "two@@example.test", "spaces in@example.test", "x@nodot"):
        response = _signup(value)
        assert response.status_code == 422, f"accepted {value!r}"


def test_signup_needs_no_token_but_me_does():
    assert _signup("open@example.test").status_code == 201
    assert client.get("/api/v1/auth/me").status_code == 401


def test_register_no_longer_exists():
    """Accounts are self-service; the admin-only creation route is gone."""
    token = _token("case_admin", ADMIN_PASSWORD)
    response = client.post(
        "/api/v1/auth/register",
        json={"username": "someone", "password": "a-perfectly-long-password", "role": "admin"},
        headers=_auth(token),
    )
    assert response.status_code in (404, 405)
    assert not user_store.exists("someone")


# ── Login behaviour ───────────────────────────────────────────────────────────

def test_wrong_password_and_unknown_user_are_indistinguishable():
    wrong = client.post(
        "/api/v1/auth/login",
        json={"username": "case_analyst", "password": "not-the-password"},
    )
    unknown = client.post(
        "/api/v1/auth/login",
        json={"username": "no_such_person", "password": "not-the-password"},
    )

    assert wrong.status_code == unknown.status_code == 401
    # Identical bodies, or the response enumerates registered usernames.
    assert wrong.json()["error"]["message"] == unknown.json()["error"]["message"]


def test_a_deactivated_account_cannot_log_in():
    user_store.set_active("case_analyst", False)
    response = client.post(
        "/api/v1/auth/login",
        json={"username": "case_analyst", "password": ANALYST_PASSWORD},
    )
    assert response.status_code == 401


def test_no_credential_material_appears_in_a_login_response():
    response = client.post(
        "/api/v1/auth/login",
        json={"username": "case_admin", "password": ADMIN_PASSWORD},
    )
    assert response.status_code == 200
    body = response.text
    assert ADMIN_PASSWORD not in body
    assert "hashed_password" not in body
    assert "$2b$" not in body, "a bcrypt hash was serialised to the client"
    assert signing_key() not in body


# ── Password hashing ──────────────────────────────────────────────────────────

def test_hashes_are_salted_and_verify():
    first = get_password_hash("the-same-password")
    second = get_password_hash("the-same-password")

    assert first != second, "identical passwords produced identical hashes"
    assert verify_password("the-same-password", first)
    assert verify_password("the-same-password", second)
    assert not verify_password("a-different-password", first)


def test_a_malformed_stored_hash_is_a_failure_not_an_admission():
    assert verify_password("anything", "not-a-bcrypt-hash") is False
    assert verify_password("anything", "") is False


# ── The store seeds nothing by default ────────────────────────────────────────

def test_no_credentials_are_seeded_from_source():
    """
    ``admin``/``admin123`` and ``investigator``/``investigator123`` were hardcoded
    in ``DEMO_USERS_DB``. Published credentials are a backdoor, not a demo.
    """
    saved_env = {
        key: os.environ.pop(key, None)
        for key in ("AIMD_BOOTSTRAP_ADMIN_USERNAME", "AIMD_BOOTSTRAP_ADMIN_PASSWORD")
    }
    _clear_users()
    fresh = auth_module.UserStore()
    try:
        fresh.bootstrap_from_env()
    finally:
        for key, value in saved_env.items():
            if value is not None:
                os.environ[key] = value

    assert fresh.count() == 0
    for username in ("admin", "investigator", "anonymous_analyst"):
        assert not fresh.exists(username)


def test_a_bootstrap_admin_comes_only_from_the_environment(monkeypatch):
    monkeypatch.setenv("AIMD_BOOTSTRAP_ADMIN_USERNAME", "operator")
    monkeypatch.setenv("AIMD_BOOTSTRAP_ADMIN_PASSWORD", "a-sufficiently-long-pass")

    fresh = auth_module.UserStore()
    fresh.bootstrap_from_env()

    assert fresh.get("operator")["role"] == "admin"
    assert fresh.authenticate("operator", "a-sufficiently-long-pass") is not None
    assert fresh.authenticate("operator", "wrong") is None


def test_a_weak_bootstrap_password_is_refused(monkeypatch):
    monkeypatch.setenv("AIMD_BOOTSTRAP_ADMIN_USERNAME", "operator")
    monkeypatch.setenv("AIMD_BOOTSTRAP_ADMIN_PASSWORD", "short")

    with pytest.raises(RuntimeError, match="12 characters"):
        auth_module.UserStore().bootstrap_from_env()
    assert not user_store.exists("operator")


# ── Persistence ───────────────────────────────────────────────────────────────

def test_a_registered_user_survives_a_new_store_instance():
    """
    Users were held in a dict on the store instance, so a restart -- modelled
    here as a fresh instance -- forgot every account registered at runtime.
    """
    assert _signup("durable@example.test", "a-durable-password-01").status_code == 201

    restarted = auth_module.UserStore()
    assert restarted.authenticate("durable@example.test", "a-durable-password-01") is not None


def test_bootstrap_does_not_reset_an_existing_admin(monkeypatch):
    """
    With persistent users the bootstrap runs on every restart against an
    account that already exists. Re-applying the environment would undo a
    password change, a demotion or a deactivation.
    """
    monkeypatch.setenv("AIMD_BOOTSTRAP_ADMIN_USERNAME", "case_analyst")
    monkeypatch.setenv("AIMD_BOOTSTRAP_ADMIN_PASSWORD", "an-environment-password")
    user_store.set_active("case_analyst", False)

    auth_module.UserStore().bootstrap_from_env()

    record = user_store.get("case_analyst")
    assert record["role"] == "user"
    assert record["is_active"] is False
    assert verify_password(ANALYST_PASSWORD, record["hashed_password"])


def test_get_returns_a_snapshot_not_a_live_record():
    """Mutating the returned dict must not be a back door into the store."""
    user_store.get("case_analyst")["role"] = "admin"
    assert user_store.get("case_analyst")["role"] == "user"


def test_signup_is_audited():
    from app.db.database import SessionLocal
    from app.services.audit.audit_log import events_for

    assert _signup("audited@example.test", "an-audited-password-1").status_code == 201

    with SessionLocal() as db:
        created = [
            e for e in events_for(db, limit=500)
            if e.action == "USER_CREATED" and e.details.get("username") == "audited@example.test"
        ]
    assert len(created) == 1
    assert created[0].actor == "audited@example.test"
    assert created[0].details["via"] == "signup"
    assert "password" not in str(created[0].details).lower()


def test_login_accepts_the_field_signup_uses():
    """Signup takes ``email``; login took only ``username`` for the same value."""
    for body in ({"email": "case_analyst", "password": ANALYST_PASSWORD},
                 {"username": "CASE_ANALYST", "password": ANALYST_PASSWORD},
                 {"username": "case_analyst", "email": "case_analyst", "password": ANALYST_PASSWORD}):
        r = client.post("/api/v1/auth/login", json=body)
        assert r.status_code == 200, (body, r.text)
        assert r.json()["username"] == "case_analyst"


def test_login_needs_one_account_name():
    missing = client.post("/api/v1/auth/login", json={"password": ANALYST_PASSWORD})
    assert missing.status_code == 422
    both = client.post("/api/v1/auth/login", json={
        "username": "case_analyst", "email": "someone_else", "password": ANALYST_PASSWORD})
    assert both.status_code == 422
