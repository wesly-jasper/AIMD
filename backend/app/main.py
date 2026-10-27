import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.auth import get_current_user, user_store
from app.core.config import settings
from app.core.exceptions import (
    AIMDException,
    RequestIdMiddleware,
    aimd_exception_handler,
    generic_exception_handler,
    http_exception_handler,
    validation_exception_handler,
)
from app.core.health import build_health_report
from app.core.logging import setup_logging
from app.core.rate_limit import limiter, rate_limit_exceeded_handler
from app.db.database import init_db
from app.api.v1.endpoints.media import router as media_router
from app.api.v1.endpoints.analysis import router as analysis_router
from app.api.v1.endpoints.auth import router as auth_router
from app.api.v1.endpoints.cases import router as cases_router
from app.api.v1.endpoints.audit import router as audit_router
from app.api.v1.endpoints.system import router as system_router

setup_logging(settings.log_level)
logger = logging.getLogger("aimd")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Migrate the database, then seed the bootstrap admin (needs the table)."""
    logger.info("AIMD starting -- applying database migrations")
    init_db()
    user_store.bootstrap_from_env()
    logger.info("Database ready")
    yield
    logger.info("AIMD shutting down")


app = FastAPI(
    title="AIMD -- AI Media Detection & Digital Forensics Platform",
    version=settings.app_version,
    description=(
        "Forensic platform for analysing digital media and determining "
        "whether it is AI-generated or manipulated, with full evidence traceability."
    ),
    lifespan=lifespan,
)

# Rate limiting -- see app.core.rate_limit. The limiter used to be constructed
# and registered here without its middleware, which is what enforces default
# limits, so no request was ever limited.
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

# Error handling.
#
# These handlers were defined in app.core.exceptions but never attached, so an
# unhandled exception produced an unstructured response and AIMDException
# subclasses fell through to a bare 500.
#
# All four are attached now. Previously only the AIMD and catch-all handlers
# were, which left HTTPException and RequestValidationError on FastAPI's default
# {"detail": ...} -- the API returned two different error shapes depending on
# which exception type a route happened to raise.
app.add_exception_handler(AIMDException, aimd_exception_handler)
app.add_exception_handler(StarletteHTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)

# Middleware order. ``add_middleware`` prepends, so the stack on the way in is
# RequestId -> CORS -> SlowAPI -> routes:
#   * SlowAPI is innermost so a 429 still passes back out through CORS. Outside
#     it, a browser would receive a 429 with no CORS headers and report an opaque
#     network error instead of a status the frontend can act on. It also means
#     CORS preflights are answered before they can consume the caller's budget.
#   * RequestId is outermost so a 429 carries the id it was logged under.
app.add_middleware(SlowAPIMiddleware)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request correlation.
#
# ``add_middleware`` prepends, so the last one added is the outermost: this is
# registered after CORS so that every response leaving the application carries
# the id its log lines were written under. Nothing set ``request.state
# .request_id`` before this, so the ``request_id`` in every error body was a
# freshly minted value that appeared in no log line -- see app.core.exceptions.
app.add_middleware(RequestIdMiddleware)

# Routers
#
# ACCESS CONTROL
# --------------
# Authentication is applied here, per router, rather than route by route. Every
# route in a router carrying ``authenticated`` is protected, including routes
# added later -- so the default for new work is "protected", and exempting
# something requires an explicit decision in this file.
#
# Before this, ``get_current_user`` was consumed by exactly one route
# (``/auth/me``) and returned ``anonymous_analyst`` when no token was presented.
# The practical effect was that evidence upload, evidence download, analysis and
# every report format were reachable with no credentials at all.
#
# The auth router is deliberately NOT blanket-protected: ``/auth/login`` and
# ``/auth/signup`` must be reachable without a token, while ``/auth/me`` is
# authenticated at the route itself.
#
# OWNERSHIP
# ---------
# Authentication says who is calling; every router below additionally scopes
# what it reads to that caller (app.db.repository "Ownership"). The standalone
# detection, fingerprint, similarity, source, provenance, evidence, report and
# jobs routers were removed with the move to a self-service product: the app
# does not use them, and each read records by id with no notion of an owner.
authenticated = [Depends(get_current_user)]

app.include_router(auth_router, prefix="/api/v1")
app.include_router(media_router, prefix="/api/v1", dependencies=authenticated)
app.include_router(analysis_router, prefix="/api/v1", dependencies=authenticated)
app.include_router(cases_router, prefix="/api/v1", dependencies=authenticated)
app.include_router(audit_router, prefix="/api/v1", dependencies=authenticated)
app.include_router(system_router, prefix="/api/v1", dependencies=authenticated)


@app.get("/health", tags=["System"])
@app.get("/api/v1/health", tags=["System"])
def health():
    # Implemented in app.core.health so this route and the v1 health router cannot
    # drift apart. They previously held independent copies of the same hardcoded
    # detector map, and fixing one left the other asserting availability it had
    # never checked.
    return build_health_report()
