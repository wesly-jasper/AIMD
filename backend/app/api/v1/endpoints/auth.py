from datetime import timedelta
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel

from app.core.auth import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    DEMO_USERS_DB,
    User,
    create_access_token,
    get_current_user,
    get_password_hash,
    verify_password,
)

router = APIRouter(prefix="/auth", tags=["Authentication"])


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str
    role: str


class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    password: str
    email: Optional[str] = None
    role: Optional[str] = "investigator"


@router.post("/login", response_model=TokenResponse)
def login(request: LoginRequest):
    user_dict = DEMO_USERS_DB.get(request.username)
    if not user_dict or not verify_password(request.password, user_dict["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user_dict["username"], "role": user_dict["role"]},
        expires_delta=access_token_expires,
    )
    return TokenResponse(
        access_token=access_token,
        username=user_dict["username"],
        role=user_dict["role"],
    )


@router.post("/register", response_model=TokenResponse)
def register(request: RegisterRequest):
    if request.username in DEMO_USERS_DB:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered",
        )

    hashed_pw = get_password_hash(request.password)
    user_entry = {
        "username": request.username,
        "email": request.email,
        "hashed_password": hashed_pw,
        "role": request.role or "investigator",
        "is_active": True,
    }
    DEMO_USERS_DB[request.username] = user_entry

    access_token = create_access_token(
        data={"sub": request.username, "role": user_entry["role"]}
    )
    return TokenResponse(
        access_token=access_token,
        username=request.username,
        role=user_entry["role"],
    )


@router.get("/me", response_model=User)
def get_me(current_user: User = Depends(get_current_user)):
    return current_user
