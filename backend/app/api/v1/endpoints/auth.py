"""
Authentication routes.

Accounts are self-service: ``/signup`` creates an ordinary ``user`` account and
signs it in. The account name is the email address, lower-cased.

What the old ``/register`` got wrong is kept wrong-proof here: the role is never
taken from the request. A signup can only ever create a ``user``; administrators
come solely from the bootstrap environment variables (app.core.auth).
"""
import re
from datetime import timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field, field_validator, model_validator
from sqlalchemy.orm import Session

from app.core.auth import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    User,
    UserExistsError,
    create_access_token,
    get_current_user,
    user_store,
)
from app.core.rate_limit import LOGIN_LIMIT, SIGNUP_LIMIT, limiter
from app.db.database import get_db
from app.services.audit.audit_log import AuditAction, record_event

router = APIRouter(prefix="/auth", tags=["Authentication"])

# Deliberately loose: one "@", something either side, a dot in the domain. Full
# RFC 5322 validation rejects real addresses and proves nothing about delivery.
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str
    role: str
    expires_in_seconds: int


class LoginRequest(BaseModel):
    # Signup names the account ``email``; login said ``username`` for the same
    # value. Both are accepted, so the pair can be used symmetrically.
    username: Optional[str] = Field(default=None, min_length=1, max_length=254)
    email: Optional[str] = Field(default=None, min_length=1, max_length=254)
    password: str = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def _account(self) -> "LoginRequest":
        if not (self.username or self.email):
            raise ValueError("Enter your email address")
        if self.username and self.email and self.username.strip().lower() != self.email.strip().lower():
            raise ValueError("username and email name different accounts")
        return self

    @property
    def account(self) -> str:
        return (self.email or self.username or "").strip().lower()


class SignupRequest(BaseModel):
    email: str = Field(min_length=3, max_length=150)
    # The same 12-character floor the bootstrap administrator is held to.
    password: str = Field(min_length=12, max_length=256)

    @field_validator("email")
    @classmethod
    def _email(cls, value: str) -> str:
        value = value.strip().lower()
        if not _EMAIL.match(value):
            raise ValueError("Enter a valid email address")
        return value


def _token_for(record: dict) -> TokenResponse:
    expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    return TokenResponse(
        access_token=create_access_token(
            data={"sub": record["username"], "role": record["role"]},
            expires_delta=expires,
        ),
        username=record["username"],
        role=record["role"],
        expires_in_seconds=int(expires.total_seconds()),
    )


@router.post("/login", response_model=TokenResponse)
@limiter.limit(LOGIN_LIMIT)
def login(request: Request, credentials: LoginRequest):
    # ``request`` is the HTTP request, which slowapi requires by that name to key
    # the limit; the body is ``credentials``. Emails are stored lower-cased.
    record = user_store.authenticate(credentials.account, credentials.password)
    if record is None:
        # One message for a wrong password, an unknown account and a deactivated
        # one: distinguishing them would confirm which accounts exist.
        # ``authenticate`` also spends the same time on all three.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return _token_for(record)


@router.post("/signup", response_model=TokenResponse, status_code=201)
@limiter.limit(SIGNUP_LIMIT)
def signup(request: Request, body: SignupRequest, db: Session = Depends(get_db)):
    """
    Create an ordinary account and sign it in.

    A taken email returns 409. That does confirm the address has an account --
    the usual trade-off for a sign-up form that can tell people why it failed;
    the per-address rate limit bounds how fast it can be probed.
    """
    try:
        record = user_store.create(
            username=body.email, password=body.password, role="user", email=body.email,
        )
    except UserExistsError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists",
        )

    record_event(
        db,
        actor=record["username"],
        action=AuditAction.USER_CREATED,
        details={"username": record["username"], "role": record["role"], "via": "signup"},
    )
    return _token_for(record)


@router.get("/me", response_model=User)
def get_me(current_user: User = Depends(get_current_user)):
    return current_user
