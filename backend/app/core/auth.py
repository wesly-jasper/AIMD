"""
Authentication primitives: signing key, password hashing, user store, identity.

WHAT WAS WRONG HERE
-------------------
Four defects, each sufficient on its own to make every protected route open:

1. **A signing key committed to the repository.**
   ``SECRET_KEY = "aimd_forensic_platform_secret_key_change_in_production"``
   was a module-level literal, and ``settings.secret_key`` held a second copy of
   the same string. Anyone with read access to the repository could mint a valid
   token for any username with any role. That string is now in git history
   permanently, so it is treated as compromised forever: ``_load_signing_key``
   rejects it outright rather than letting a deployment inherit it from a
   ``.env``.

2. **A fail-open identity dependency.** ``get_current_user`` returned
   ``User(username="anonymous_analyst", role="investigator")`` when no token was
   presented. Omitting the Authorization header was therefore the easiest way to
   be treated as an investigator. ``auto_error=False`` on the bearer scheme is
   what allowed the ``token is None`` branch to exist at all.

3. **The role was taken from the token.** A token carried ``role`` and that claim
   was trusted verbatim, so a role could not be revoked before the token's
   24-hour expiry. The role is now read from the user store on every request;
   the claim is ignored.

4. **Published credentials.** ``DEMO_USERS_DB`` seeded ``admin``/``admin123`` and
   ``investigator``/``investigator123`` in committed source. Those are not demo
   credentials once the file is public, they are a backdoor. The store now seeds
   only from the environment.

WHAT CHANGED SINCE
------------------
Users were held in memory, so an account registered at runtime vanished on
restart and existed on one worker only. They are now persisted in the ``users``
table (``UserStore`` below).
"""
from __future__ import annotations

import logging
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from pydantic import BaseModel

from app.core.config import settings

logger = logging.getLogger(__name__)

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 12

#: Roles the application recognises. A role arriving from a request body or a
#: token claim that is not in this set is refused rather than stored, so a typo
#: cannot create a role that no authorisation check will ever match.
#: ``user`` is every self-service account; ``admin`` exists only through the
#: bootstrap environment variables and additionally may verify the audit chain.
ROLES = frozenset({"admin", "user"})

#: The key that was committed to this repository. It is public, so it must never
#: be accepted again even if a deployment still carries it in a ``.env``.
_COMPROMISED_KEYS = frozenset({
    "aimd_forensic_platform_secret_key_change_in_production",
    "change_in_production",
    "secret",
    "changeme",
})

_MIN_KEY_LENGTH = 32

# auto_error=True: a missing or malformed Authorization header is a 401 raised by
# the scheme itself. The previous auto_error=False handed ``None`` to the
# dependency, which is what made an unauthenticated caller representable as a
# user at all.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=True)


def _is_development() -> bool:
    return settings.app_env.strip().lower() in {"development", "dev", "local", "test"}


def _load_signing_key() -> str:
    """
    The JWT signing key, from the environment only.

    There is deliberately no fallback value in source. In development an
    ephemeral random key is generated per process -- unforgeable, but every
    restart invalidates outstanding tokens, which is announced rather than
    hidden. Outside development a missing key is fatal: generating one silently
    would give each worker in a multi-worker deployment a different key and
    reject tokens at random, which is far harder to diagnose than a refusal at
    startup.
    """
    key = os.getenv("AIMD_JWT_SECRET") or os.getenv("SECRET_KEY") or ""
    key = key.strip()

    if key and key.lower() in _COMPROMISED_KEYS:
        raise RuntimeError(
            "The configured signing key is the placeholder that was committed to "
            "this repository's history. It is public and anyone could forge "
            "tokens with it. Set AIMD_JWT_SECRET to a fresh random value, e.g. "
            "`python -c \"import secrets; print(secrets.token_urlsafe(48))\"`."
        )

    if key and len(key) < _MIN_KEY_LENGTH:
        raise RuntimeError(
            f"AIMD_JWT_SECRET is {len(key)} characters; at least "
            f"{_MIN_KEY_LENGTH} are required for an HMAC-SHA256 signing key."
        )

    if key:
        return key

    if not _is_development():
        raise RuntimeError(
            f"AIMD_JWT_SECRET is not set and APP_ENV is {settings.app_env!r}. "
            "Refusing to start without a signing key: a generated per-process "
            "key would be inconsistent across workers."
        )

    logger.warning(
        "AIMD_JWT_SECRET is not set. Generated an ephemeral signing key for this "
        "process -- every access token becomes invalid when it restarts, and "
        "separate workers will not accept each other's tokens. Set "
        "AIMD_JWT_SECRET for anything beyond a single local process."
    )
    return secrets.token_urlsafe(48)


_SIGNING_KEY: Optional[str] = None


def signing_key() -> str:
    """Resolved once per process, on first use rather than at import."""
    global _SIGNING_KEY
    if _SIGNING_KEY is None:
        _SIGNING_KEY = _load_signing_key()
    return _SIGNING_KEY


class User(BaseModel):
    username: str
    email: Optional[str] = None
    role: str
    is_active: bool = True


def get_password_hash(password: str) -> str:
    # bcrypt reads at most 72 bytes and newer builds raise rather than truncate,
    # so the slice is explicit. Characters past 72 bytes do not contribute.
    return bcrypt.hashpw(password.encode("utf-8")[:72], bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8")[:72],
            hashed_password.encode("utf-8"),
        )
    except (ValueError, TypeError) as exc:
        # A malformed stored hash, not a wrong password. Log the shape of the
        # problem and never the password or the hash.
        logger.error("Stored password hash is unusable: %s", type(exc).__name__)
        return False


#: A real bcrypt hash of a random value, used to spend the same time verifying a
#: password for a username that does not exist as for one that does. Without it,
#: response latency reveals which usernames are registered.
_DUMMY_HASH = get_password_hash(secrets.token_urlsafe(32))


class UserStore:
    """
    User records, persisted in the ``users`` table.

    Each method opens its own short session, so the store is usable from the
    auth dependency without sharing a request's session. ``get`` returns a
    detached snapshot: mutating it changes nothing -- use ``set_active``.

    The bootstrap administrator is seeded only from the environment, by
    ``bootstrap_from_env`` at application startup (after migrations, since the
    table must exist). With neither variable set the store stays empty and every
    login fails, which is the correct default for a deployment whose operator has
    not yet created an account.
    """

    def __init__(self, session_factory=None) -> None:
        self._session_factory = session_factory

    def _session(self):
        if self._session_factory is not None:
            return self._session_factory()
        # Resolved late so that importing this module does not bind a database.
        from app.db.database import SessionLocal
        return SessionLocal()

    @staticmethod
    def _snapshot(row) -> dict:
        return {
            "username": row.username,
            "email": row.email,
            "hashed_password": row.hashed_password,
            "role": row.role,
            "is_active": bool(row.is_active),
        }

    def bootstrap_from_env(self) -> None:
        username = (os.getenv("AIMD_BOOTSTRAP_ADMIN_USERNAME") or "").strip().lower()
        password = os.getenv("AIMD_BOOTSTRAP_ADMIN_PASSWORD") or ""

        if not username or not password:
            logger.info(
                "No bootstrap admin configured. Set AIMD_BOOTSTRAP_ADMIN_USERNAME "
                "and AIMD_BOOTSTRAP_ADMIN_PASSWORD to create the first account."
            )
            return

        if len(password) < 12:
            raise RuntimeError(
                "AIMD_BOOTSTRAP_ADMIN_PASSWORD is shorter than 12 characters. "
                "Refusing to create the bootstrap administrator."
            )

        if self.exists(username):
            # Users now persist, so on every restart after the first this account
            # already exists. It is left exactly as it is: re-hashing the
            # environment password would silently undo a password change, and
            # re-asserting the admin role would undo a deliberate demotion or
            # deactivation.
            logger.info("Bootstrap administrator %r already exists; unchanged.", username)
            return

        self.create(username=username, password=password, role="admin", email=None)
        # The username identifies the account; the password is never logged.
        logger.info("Bootstrap administrator %r created from the environment.", username)

    def get(self, username: str) -> Optional[dict]:
        from app.db.models import UserRecord

        with self._session() as db:
            row = db.get(UserRecord, username)
            return self._snapshot(row) if row else None

    def exists(self, username: str) -> bool:
        return self.get(username) is not None

    def count(self) -> int:
        from app.db.models import UserRecord

        with self._session() as db:
            return db.query(UserRecord).count()

    def create(
        self,
        username: str,
        password: str,
        role: str,
        email: Optional[str] = None,
    ) -> dict:
        """
        Create an account. Raises ``ValueError`` for an unknown role and
        ``UserExistsError`` for a taken username -- it never overwrites.
        """
        from sqlalchemy.exc import IntegrityError
        from app.db.models import UserRecord

        if role not in ROLES:
            raise ValueError(f"Unknown role {role!r}; expected one of {sorted(ROLES)}")

        with self._session() as db:
            row = UserRecord(
                username=username,
                email=email,
                hashed_password=get_password_hash(password),
                role=role,
                is_active=True,
            )
            db.add(row)
            try:
                db.commit()
            except IntegrityError:
                # The primary key, so a concurrent registration of the same name
                # cannot replace the first one's password.
                db.rollback()
                raise UserExistsError(username)
            return self._snapshot(row)

    def set_active(self, username: str, active: bool) -> bool:
        """Activate or deactivate an account. False if no such account."""
        from app.db.models import UserRecord

        with self._session() as db:
            row = db.get(UserRecord, username)
            if row is None:
                return False
            row.is_active = active
            db.commit()
            return True

    def authenticate(self, username: str, password: str) -> Optional[dict]:
        """
        The record for these credentials, or None.

        A wrong password, an unknown username and a deactivated account are
        deliberately indistinguishable to the caller, and all three cost one
        bcrypt verification so they are indistinguishable in latency too.
        """
        record = self.get(username)
        hashed = record["hashed_password"] if record else _DUMMY_HASH

        if not verify_password(password, hashed):
            return None
        if record is None or not record.get("is_active", False):
            return None
        return record


class UserExistsError(Exception):
    """The username is already registered."""


user_store = UserStore()


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Mint an access token.

    ``sub`` is the only claim authorisation depends on. A ``role`` claim may be
    included for a client's own display purposes, but ``get_current_user``
    ignores it and reads the role from the store -- so revoking a role or
    deactivating an account takes effect on the next request rather than at token
    expiry.
    """
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode["exp"] = expire
    return jwt.encode(to_encode, signing_key(), algorithm=ALGORITHM)


_CREDENTIALS_EXCEPTION = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Could not validate credentials",
    headers={"WWW-Authenticate": "Bearer"},
)


def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """
    The authenticated user, or 401. There is no anonymous outcome.

    The return type is ``User``, not ``Optional[User]``: a caller that receives a
    value from this dependency has an authenticated identity. The previous
    signature returned ``Optional[User]`` and supplied ``anonymous_analyst`` for
    an absent token, so every route that depended on it was open.
    """
    try:
        payload = jwt.decode(token, signing_key(), algorithms=[ALGORITHM])
    except JWTError:
        raise _CREDENTIALS_EXCEPTION

    username = payload.get("sub")
    if not username or not isinstance(username, str):
        raise _CREDENTIALS_EXCEPTION

    record = user_store.get(username)
    if record is None or not record.get("is_active", False):
        # A token that decodes correctly but names no active account. This is the
        # revocation path: the token is still cryptographically valid and is
        # still refused.
        raise _CREDENTIALS_EXCEPTION

    return User(
        username=record["username"],
        email=record.get("email"),
        # From the store, never from payload["role"].
        role=record["role"],
        is_active=True,
    )


def require_role(*allowed: str):
    """
    Dependency factory restricting a route to the given roles.

    Returns 403 for an authenticated user whose role is not permitted, which is
    distinct from the 401 for no valid identity -- the caller needs to know
    whether to re-authenticate or to stop.
    """
    unknown = set(allowed) - ROLES
    if unknown:
        raise ValueError(f"require_role called with unknown roles: {sorted(unknown)}")

    allowed_set = frozenset(allowed)

    def _dependency(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in allowed_set:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="This action requires a different role",
            )
        return current_user

    return _dependency
